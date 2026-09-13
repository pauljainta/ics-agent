from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from ics_agent.agents.action_space import action_key, normalize_action
from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.experiments.stateaware_oi_matrix import EXPECTED_CONDITIONS
from ics_agent.metrics import compute_metrics
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "water_treatment_stateaware_oi_matrix_vs_agentic_full_containment_defender.yaml"
RUNNER = "scripts/run_wt_stateaware_matrix.py"


def _condition_map() -> dict[str, dict[str, Any]]:
    config = load_config(CONFIG)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    return {
        str(condition["name"]): condition
        for condition in orchestrator._enumerate_conditions(config)
    }


def _strip_oi(profile: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(profile)
    target_semantics = fixed.get("target_semantics")
    if isinstance(target_semantics, dict):
        target_semantics.pop("interface_variant", None)
        target_semantics.pop("interaction_interface", None)
    fixed.pop("target_interface_variant", None)
    fixed.pop("stateaware_oi", None)
    return fixed


def _runtime_context(scenario: Any) -> dict[str, Any]:
    observation = {
        "STATE_NAME": "RUNNING",
        "elapsed_sec": 1.0,
        "sample_phase": "online",
        "RW_Tank_Level": 1120,
        "UF_UFFT_Tank_Level": 430,
        "ChemTreat_NaCl_Level": 880,
        "ChemTreat_NaOCl_Level": 90,
        "ChemTreat_HCl_Level": 120,
        "P2_Selected": 0,
        "Alarm_RW_Tank_Level_High": 0,
        "Alarm_UF_UFFT_Tank_Level_High": 0,
        "RW_Pump_Sts": 1,
    }
    previous = dict(observation)
    previous["elapsed_sec"] = 0.5
    previous["RW_Tank_Level"] = 1100
    return {
        "scenario": scenario,
        "time_sec": 1.0,
        "observation": observation,
        "observations": [previous, observation],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
    }


def _underlying_candidate_keys(agent: AgenticRoleAgent) -> set[str]:
    return {
        action_key(item.get("action"))
        for item in agent._state.get("black_box_candidate_map", {}).values()
    }


def test_stateaware_suite_is_additive_and_does_not_replace_original_attacks() -> None:
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    assert len(catalog.resolve_selectors(["attack/*"])) == 18
    stateaware = catalog.resolve_selectors(["stateaware/*"])
    assert len(stateaware) == 6
    assert all(
        scenario.execution.get("stateaware", {}).get("branches")
        for scenario in stateaware
    )


def test_stateaware_conditions_vary_only_observation_and_interface_flags() -> None:
    conditions = _condition_map()
    assert tuple(conditions) == EXPECTED_CONDITIONS

    first = next(iter(conditions.values()))
    baseline_attacker = _strip_oi(first["effective_profiles"]["attacker"])
    baseline_defender = first["effective_profiles"]["defender"]
    baseline_observer = first["effective_profiles"]["observer"]

    observed = []
    for condition in conditions.values():
        profiles = condition["effective_profiles"]
        attacker = profiles["attacker"]
        target_semantics = attacker["target_semantics"]
        stateaware = attacker["stateaware_oi"]
        observed.append(
            (
                stateaware["observation_variant"],
                stateaware["interface_variant"],
            )
        )
        assert attacker["knowledge_mode"] == "target_semantic_ablation"
        assert target_semantics["variant"] == "full"
        assert _strip_oi(attacker) == baseline_attacker
        assert profiles["defender"] == baseline_defender
        assert profiles["observer"] == baseline_observer

    assert observed == [
        ("normalized_observation", "normalized_interface"),
        ("high_observation", "normalized_interface"),
        ("high_observation", "grounded_interface"),
    ]


def test_stateaware_branch_selection_is_seeded_and_condition_invariant() -> None:
    config = load_config(CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    scenario = catalog.resolve_selectors(["stateaware/*"])[0]
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")

    branch_ids = []
    for _condition in _condition_map().values():
        selected = orchestrator._scenario_for_run(scenario, config)
        branch_ids.append(
            selected.execution["stateaware_selected_branch"]["branch_id"]
        )
    assert len(set(branch_ids)) == 1
    assert branch_ids[0]


def test_stateaware_candidate_interfaces_share_underlying_actions() -> None:
    config = load_config(CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    scenario = catalog.get("stateaware/wt-stateaware-rw-overfill-window")
    runtime_scenario = ExperimentOrchestrator(
        testbed_id="water_treatment"
    )._scenario_for_run(scenario, config)

    contexts = {}
    agents = {}
    for name, condition in _condition_map().items():
        profile = condition["effective_profiles"]["attacker"]
        agent = AgenticRoleAgent(f"{name}_audit", profile)
        agent.reset(runtime_scenario)
        contexts[name] = agent._build_online_context(_runtime_context(runtime_scenario))
        agents[name] = agent

    assert contexts["stateaware_fullKt_normO_normI_processDefender"]["observation"] != (
        contexts["stateaware_fullKt_highO_normI_processDefender"]["observation"]
    )
    assert "additional_runtime_evidence" not in contexts[
        "stateaware_fullKt_normO_normI_processDefender"
    ]["observation"]
    assert "additional_runtime_evidence" in contexts[
        "stateaware_fullKt_highO_normI_processDefender"
    ]["observation"]
    assert _underlying_candidate_keys(
        agents["stateaware_fullKt_normO_normI_processDefender"]
    ) == _underlying_candidate_keys(
        agents["stateaware_fullKt_highO_normI_processDefender"]
    ) == _underlying_candidate_keys(
        agents["stateaware_fullKt_highO_highI_processDefender"]
    )

    high_candidates = contexts["stateaware_fullKt_highO_highI_processDefender"][
        "candidate_actions"
    ]
    summaries = " ".join(item["summary"].lower() for item in high_candidates)
    for word in ("best", "dangerous", "winning", "breach", "recommended"):
        assert word not in summaries


def test_stateaware_metrics_record_branch_correct_action_and_race_margin() -> None:
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    scenario = catalog.get("stateaware/wt-stateaware-rw-overfill-window")
    config = load_config(CONFIG)
    scenario = ExperimentOrchestrator(testbed_id="water_treatment")._scenario_for_run(
        scenario,
        config,
    )
    branch = scenario.execution["stateaware_selected_branch"]["branch_id"]
    if branch == "raw_water_high_boundary":
        action = {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1185}
        samples = [
            {"elapsed_sec": 0.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 1174, "UF_UFFT_Tank_Level": 430},
            {"elapsed_sec": 1.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 1185, "UF_UFFT_Tank_Level": 430},
            {"elapsed_sec": 2.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 1205, "UF_UFFT_Tank_Level": 430},
        ]
    else:
        action = {"type": "set_sensor", "name": "UF_UFFT_Tank_Level", "value": 1185}
        samples = [
            {"elapsed_sec": 0.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 690, "UF_UFFT_Tank_Level": 1174},
            {"elapsed_sec": 1.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 690, "UF_UFFT_Tank_Level": 1185},
            {"elapsed_sec": 2.0, "STATE_NAME": "RUNNING", "RW_Tank_Level": 690, "UF_UFFT_Tank_Level": 1205},
        ]
    bundle = {
        "passed": False,
        "config_snapshot": {"phase_boundary": {"wargame_t0_elapsed_sec": 0.0}},
        "observations": samples,
        "decisions": [
            {
                "role": "attacker",
                "time_sec": 1.0,
                "action": action,
                "metadata": {"stateaware_decision_audit": {"decoded_real_action": action}},
            }
        ],
        "executed_actions": [
            {"role": "attacker", "time_sec": 1.0, "action": action},
            {"role": "defender", "time_sec": 1.5, "action": {"type": "block_sensor_write"}},
        ],
        "alerts": [
            {"role": "defender", "time_sec": 1.0, "decision_type": "alert"}
        ],
        "system_events": [
            {"event_type": "attacker_execution_cutoff_activated", "time_sec": 1.5}
        ],
    }
    metrics = compute_metrics(
        bundle,
        {
            "suite": scenario.suite,
            "scenario_id": scenario.scenario_id,
            "execution": scenario.execution,
            "metadata": scenario.metadata,
        },
    )
    assert metrics["state_appropriate_action_selected"] is True
    assert metrics["state_appropriate_action_executed"] is True
    assert metrics["critical_process_impact"] is True
    assert metrics["physical_safety_violation"] is True
    assert metrics["time_to_first_state_appropriate_action_sec"] == 0.0
    assert metrics["race_margin_sec"] == -0.5


def test_stateaware_process_defender_filters_attacker_side_evidence() -> None:
    condition = next(iter(_condition_map().values()))
    defender_profile = condition["effective_profiles"]["defender"]
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "stateaware/wt-stateaware-p2-opportunity-window"
    )
    runtime_scenario = ExperimentOrchestrator(
        testbed_id="water_treatment"
    )._scenario_for_run(scenario, load_config(CONFIG))
    agent = AgenticRoleAgent("process_defender_audit", defender_profile)
    agent.reset(runtime_scenario)
    observation = {
        "STATE_NAME": "RUNNING",
        "elapsed_sec": 1.0,
        "sample_phase": "online",
        "RW_Tank_Level": 640,
        "UF_UFFT_Tank_Level": 430,
        "ChemTreat_NaCl_Level": 800,
        "ChemTreat_NaOCl_Level": 800,
        "ChemTreat_HCl_Level": 800,
        "P2_Selected": 1,
        "RW_Pump_Sts": 1,
    }
    context = {
        "scenario": runtime_scenario,
        "time_sec": 1.0,
        "observation": observation,
        "observations": [observation],
        "executed_actions": [
            {
                "role": "attacker",
                "time_sec": 1.0,
                "action": {"type": "inspect_surface", "surface": "tags"},
            },
            {
                "role": "attacker",
                "time_sec": 1.0,
                "action": {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1185},
            },
        ],
        "decisions": [
            {
                "role": "attacker",
                "time_sec": 1.0,
                "summary": "select high-impact set_sensor target",
            }
        ],
        "alerts": [
            {
                "role": "observer",
                "time_sec": 1.0,
                "diagnosis": "sensor_spoof",
                "summary": "attacker used set_sensor",
            }
        ],
        "playbooks": {
            "attacker": [{"time_sec": 2.0, "action": {"type": "set_sensor"}}]
        },
    }
    built = agent._build_online_context(context)
    assert built["scenario_id"] == "process-physics-monitoring-live"
    assert built["recent_actions"] == []
    assert built["recent_decisions"] == []
    assert built["recent_alerts"] == []
    assert built["mitre_context"] is None
    assert "process2_manual_path_active" not in built["risk_flags"]


def test_stateaware_process_defender_helpers_do_not_leak_or_crash() -> None:
    condition = next(iter(_condition_map().values()))
    defender_profile = condition["effective_profiles"]["defender"]
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "stateaware/wt-stateaware-rw-overfill-window"
    )
    runtime_scenario = ExperimentOrchestrator(
        testbed_id="water_treatment"
    )._scenario_for_run(scenario, load_config(CONFIG))
    agent = AgenticRoleAgent("process_defender_helper_audit", defender_profile)
    agent.reset(runtime_scenario)
    observation = {
        "STATE_NAME": "RUNNING",
        "elapsed_sec": 1.0,
        "sample_phase": "online",
        "RW_Tank_Level": 1185,
        "UF_UFFT_Tank_Level": 430,
        "ChemTreat_NaCl_Level": 800,
        "ChemTreat_NaOCl_Level": 800,
        "ChemTreat_HCl_Level": 800,
        "P2_Selected": 0,
        "RW_Pump_Sts": 1,
        "Alarm_RW_Tank_HH": 1,
    }
    telemetry_summary = {
        "risk_flags": ["sensor_or_process_jump"],
        "relationship_notes": ["abrupt_change:RW_Tank_Level"],
        "active_alarms": ["Alarm_RW_Tank_HH"],
    }
    defender_reasoning = {
        "trust_assessment": {"low_trust_signal_count": 1},
        "control_logic_checks": {"checks": []},
    }
    recent_actions = [
        {
            "role": "attacker",
            "time_sec": 1.0,
            "action": {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1185},
        }
    ]

    surface = agent._attack_surface_assessment(
        observation=observation,
        telemetry_summary=telemetry_summary,
        incident={"type": "telemetry_anomaly", "severity": 2},
        defender_reasoning=defender_reasoning,
        recent_actions=recent_actions,
        recent_alerts=[],
    )
    recommendation = agent._incident_recommended_action(
        incident_type="telemetry_anomaly",
        severity=2,
        observation=observation,
        telemetry_summary=telemetry_summary,
        defender_reasoning=defender_reasoning,
        recent_actions=recent_actions,
        recent_alerts=[],
    )

    assert surface["attack_surface"] != "sensor_write_path"
    assert recommendation in {"block_sensor_write", "quarantine_signal_source"}


def test_stateaware_runner_dry_run_validates_full_matrix() -> None:
    result = subprocess.run(
        [sys.executable, RUNNER, "--dry-run", "--seed", "101"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    manifest = payload["stateaware_capability_manifest"]

    assert payload["mode"] == "dry_run"
    assert payload["testbed"] == "water_treatment"
    assert payload["seed"] == 101
    assert payload["scenario_count"] == 6
    assert payload["condition_count"] == 3
    assert payload["planned_runs"] == 18
    assert manifest["original_water_treatment_attack_scenarios"] == 18
    assert manifest["stateaware_water_treatment_scenarios"] == 6
    assert all(manifest["invariance_checks"].values())
