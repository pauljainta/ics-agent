from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ics_agent.experiments.stateaware_oi_matrix as stateaware
from ics_agent.agents.action_space import action_key, build_candidate_actions
from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.experiments.stateaware_oi_matrix import WD_EXPECTED_CONDITIONS
from ics_agent.experiments.stateaware_oi_matrix import _load_observations
from ics_agent.experiments.stateaware_oi_matrix import _preflight_mrg_auth
from ics_agent.experiments.stateaware_oi_matrix import _remove_failed_or_incomplete_trials
from ics_agent.metrics import compute_metrics
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "water_distribution_stateaware_oi_matrix_vs_agentic_full_containment_defender.yaml"
RUNNER = "scripts/run_wd_stateaware_matrix.py"


def _condition_map() -> dict[str, dict[str, Any]]:
    config = load_config(CONFIG)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
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
        "Supply_Tank_Level": 1160,
        "Supply_NaOCl_Level": 800,
        "Supply_NH4Cl_Level": 800,
        "Supply_Pump_Flow": 120,
        "Grid_Elev_Tank_Level": 1180,
        "Grid_Consum_Tank_Level": 1000,
        "RWS_Tank_Level": 500,
        "Supply_Pump_Sts": 1,
        "Supply_Tank_Valve_Sts": 1,
        "Grid_Elev_Valve_Sts": 0,
        "RWS_Pump_Sts": 0,
        "RWS_Tank_Valve_Sts": 0,
        "Alarm_Supply_Tank_HH": 0,
    }
    previous = dict(observation)
    previous["elapsed_sec"] = 0.5
    previous["Supply_Tank_Level"] = 1140
    previous["Grid_Elev_Tank_Level"] = 1190
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


def test_distribution_stateaware_suite_is_additive() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    assert len(catalog.resolve_selectors(["attack/*"])) == 11
    stateaware = catalog.resolve_selectors(["stateaware/*"])
    assert len(stateaware) == 4
    assert all(
        scenario.execution.get("stateaware", {}).get("branches")
        for scenario in stateaware
    )
    for scenario in stateaware:
        branches = scenario.execution.get("stateaware", {}).get("branches") or []
        for branch in branches:
            assert branch.get("decisive_actions")
            critical = branch.get("critical_impact")
            assert isinstance(critical, dict)
            assert critical.get("requires_decisive_action") is True
            assert critical.get("conditions")


def test_distribution_stateaware_conditions_vary_only_observation_and_interface() -> None:
    conditions = _condition_map()
    assert tuple(conditions) == WD_EXPECTED_CONDITIONS

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
        assert profiles["defender"]["process_physics_only"] is True
        assert profiles["defender"]["defender_observation_mode"] == "process_physics_only"
        assert profiles["defender"] == baseline_defender
        assert profiles["observer"] == baseline_observer

    assert observed == [
        ("normalized_observation", "normalized_interface"),
        ("normalized_observation", "grounded_interface"),
        ("high_observation", "grounded_interface"),
    ]


def test_distribution_branch_selection_is_seeded_and_condition_invariant() -> None:
    config = load_config(CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.resolve_selectors(["stateaware/*"])[0]
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")

    branch_ids = []
    fingerprints = []
    for _condition in _condition_map().values():
        selected = orchestrator._scenario_for_run(scenario, config)
        branch = selected.execution["stateaware_selected_branch"]
        branch_ids.append(branch["branch_id"])
        fingerprints.append(json.dumps(branch["initial_conditions"], sort_keys=True))
    assert len(set(branch_ids)) == 1
    assert len(set(fingerprints)) == 1
    assert branch_ids[0]


def test_distribution_candidate_interfaces_share_underlying_actions() -> None:
    config = load_config(CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-supply-conditional")
    runtime_scenario = ExperimentOrchestrator(
        testbed_id="water_distribution"
    )._scenario_for_run(scenario, config)

    contexts = {}
    agents = {}
    for name, condition in _condition_map().items():
        profile = condition["effective_profiles"]["attacker"]
        agent = AgenticRoleAgent(f"{name}_audit", profile)
        agent.reset(runtime_scenario)
        contexts[name] = agent._build_online_context(_runtime_context(runtime_scenario))
        agents[name] = agent

    assert contexts["wd_stateaware_full_semantics_normO_normI"]["observation"] == (
        contexts["wd_stateaware_full_semantics_normO_highI"]["observation"]
    )
    assert "additional_runtime_evidence" not in contexts[
        "wd_stateaware_full_semantics_normO_normI"
    ]["observation"]
    assert "additional_runtime_evidence" in contexts[
        "wd_stateaware_full_semantics_highO_highI"
    ]["observation"]
    assert _underlying_candidate_keys(agents["wd_stateaware_full_semantics_normO_normI"]) == (
        _underlying_candidate_keys(agents["wd_stateaware_full_semantics_normO_highI"])
    ) == _underlying_candidate_keys(agents["wd_stateaware_full_semantics_highO_highI"])
    assert contexts["wd_stateaware_full_semantics_normO_normI"]["policy_scaffold"] == (
        contexts["wd_stateaware_full_semantics_normO_highI"]["policy_scaffold"]
    ) == contexts["wd_stateaware_full_semantics_highO_highI"]["policy_scaffold"]
    assert contexts["wd_stateaware_full_semantics_normO_normI"]["policy_scaffold"][
        "version"
    ] == "stateaware_generic_policy_v1"

    high_candidates = contexts["wd_stateaware_full_semantics_normO_highI"][
        "candidate_actions"
    ]
    summaries = " ".join(item["summary"].lower() for item in high_candidates)
    for word in ("best", "dangerous", "winning", "breach", "recommended"):
        assert word not in summaries
    assert "supply_pump_speed" in summaries
    assert all("mechanism" in item for item in high_candidates)


def test_distribution_stateaware_prompt_does_not_embed_winning_values() -> None:
    config = load_config(CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-adaptive-crosszone")
    runtime_scenario = ExperimentOrchestrator(
        testbed_id="water_distribution"
    )._scenario_for_run(scenario, config)
    condition = _condition_map()["wd_stateaware_full_semantics_highO_highI"]
    agent = AgenticRoleAgent("wd_stateaware_prompt_audit", condition["effective_profiles"]["attacker"])
    agent.reset(runtime_scenario)

    context = agent._build_online_context(_runtime_context(runtime_scenario))
    scaffold_text = json.dumps(context["policy_scaffold"], sort_keys=True)

    for forbidden in ("270", "360", "1380", "1840", "branch_id", "grid_low"):
        assert forbidden not in scaffold_text


def test_distribution_process_only_defender_candidates_ignore_attacker_history() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-supply-conditional")
    allowed = [
        "hmi_stop",
        "hmi_estop",
        "disable_hmi_session",
        "isolate_engineering_host",
        "block_sensor_write",
        "block_actuator_write",
        "quarantine_signal_source",
        "noop",
    ]
    observation = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 900,
        "Grid_Elev_Tank_Level": 1200,
        "RWS_Tank_Level": 500,
        "Supply_Pump_Sts": 0,
        "Alarm_Supply_Tank_HH": 0,
    }
    recent_actions = [
        {
            "role": "attacker",
            "time_sec": 1.0,
            "action": {"type": "set_sensor", "name": "Supply_Tank_Level", "value": 1380},
        }
    ]

    leaked = build_candidate_actions(
        role="defender",
        allowed_action_types=allowed,
        scenario=scenario,
        observation=observation,
        risk_flags=[],
        active_alarms=[],
        relationship_notes=[],
        recent_alerts=[],
        recent_actions=recent_actions,
        incident={"type": "none", "status": "none"},
        remaining_playbook=[],
        now=1.0,
        playbook_anticipation_sec=0.0,
        defender_process_only=False,
    )
    process_only = build_candidate_actions(
        role="defender",
        allowed_action_types=allowed,
        scenario=scenario,
        observation=observation,
        risk_flags=[],
        active_alarms=[],
        relationship_notes=[],
        recent_alerts=[],
        recent_actions=recent_actions,
        incident={"type": "none", "status": "none"},
        remaining_playbook=[],
        now=1.0,
        playbook_anticipation_sec=0.0,
        defender_process_only=True,
    )

    leaked_types = {item["action"]["type"] for item in leaked}
    process_only_types = {item["action"]["type"] for item in process_only}
    assert "block_sensor_write" in leaked_types
    assert "quarantine_signal_source" in leaked_types
    assert "block_sensor_write" not in process_only_types
    assert "quarantine_signal_source" not in process_only_types


def test_distribution_process_only_defender_does_not_treat_startup_band_as_cyber_evidence() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-supply-conditional")
    allowed = [
        "hmi_stop",
        "hmi_estop",
        "disable_hmi_session",
        "isolate_engineering_host",
        "block_sensor_write",
        "block_actuator_write",
        "quarantine_signal_source",
        "noop",
    ]
    observation = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 1160,
        "Grid_Elev_Tank_Level": 1200,
        "RWS_Tank_Level": 500,
        "Supply_Pump_Sts": 1,
        "Alarm_Supply_Tank_HH": 0,
    }

    process_only = build_candidate_actions(
        role="defender",
        allowed_action_types=allowed,
        scenario=scenario,
        observation=observation,
        risk_flags=["supply_high_high_band"],
        active_alarms=[],
        relationship_notes=[],
        recent_alerts=[],
        recent_actions=[],
        incident={"type": "none", "status": "none"},
        remaining_playbook=[],
        now=0.0,
        playbook_anticipation_sec=0.0,
        defender_process_only=True,
    )

    process_only_types = {item["action"]["type"] for item in process_only}
    assert "block_sensor_write" not in process_only_types
    assert "quarantine_signal_source" not in process_only_types


def test_distribution_pre_effect_containment_guard_defers_until_attacker_effect() -> None:
    profile = {"defer_containment_until_attacker_effect": True}
    bundle = SimpleNamespace(executed_actions=[])
    action = {"type": "block_sensor_write"}

    assert ExperimentOrchestrator._should_defer_pre_effect_containment(
        role="defender",
        action=action,
        profile=profile,
        bundle=bundle,
        elapsed=0.0,
    )

    bundle.executed_actions.append(
        {"role": "attacker", "time_sec": 0.5, "action": {"type": "inspect_surface"}}
    )
    assert ExperimentOrchestrator._should_defer_pre_effect_containment(
        role="defender",
        action=action,
        profile=profile,
        bundle=bundle,
        elapsed=0.5,
    )

    bundle.executed_actions.append(
        {
            "role": "attacker",
            "time_sec": 1.0,
            "action": {
                "type": "set_sensor",
                "name": "Supply_Tank_Level",
                "value": 1380,
            },
        }
    )
    assert not ExperimentOrchestrator._should_defer_pre_effect_containment(
        role="defender",
        action=action,
        profile=profile,
        bundle=bundle,
        elapsed=1.0,
    )


def test_distribution_metrics_flag_pre_attack_cutoff_without_negative_t_contain() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-supply-conditional")
    config = load_config(CONFIG)
    scenario = ExperimentOrchestrator(testbed_id="water_distribution")._scenario_for_run(
        scenario,
        config,
    )
    action = {"type": "set_actuator", "name": "Supply_Pump_Speed", "value": 100}
    bundle = {
        "passed": False,
        "config_snapshot": {"phase_boundary": {"wargame_t0_elapsed_sec": 0.0}},
        "observations": [
            {
                "elapsed_sec": 0.0,
                "sample_phase": "pre_action",
                "STATE_NAME": "RUNNING",
                "Supply_Tank_Level": 1160,
                "Supply_Pump_Speed": 0,
            },
            {
                "elapsed_sec": 1.0,
                "sample_phase": "post_action",
                "STATE_NAME": "RUNNING",
                "Supply_Tank_Level": 1185,
                "Supply_Pump_Speed": 100,
            },
        ],
        "decisions": [
            {
                "role": "attacker",
                "time_sec": 1.0,
                "action": action,
                "metadata": {"stateaware_decision_audit": {"decoded_real_action": action}},
            }
        ],
        "executed_actions": [{"role": "attacker", "time_sec": 1.0, "action": action}],
        "alerts": [],
        "system_events": [
            {"event_type": "attacker_execution_cutoff_activated", "time_sec": 0.0}
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

    assert metrics["invalid_precontained_run"] is True
    assert metrics["runtime_containment"]["attacker_execution_cutoff_pre_attack_start"] is True
    assert metrics["runtime_containment"]["attacker_execution_cutoff_activation_time_sec"] is None
    assert metrics["state_aware"]["canonical_event_times"]["T_contain"] is None


def test_distribution_stateaware_metrics_record_branch_correct_action() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("stateaware/wd-stateaware-grid-valve-conditional")
    config = load_config(CONFIG)
    scenario = ExperimentOrchestrator(testbed_id="water_distribution")._scenario_for_run(
        scenario,
        config,
    )
    branch = scenario.execution["stateaware_selected_branch"]["branch_id"]
    action = (
        {"type": "set_sensor", "name": "Grid_Elev_Tank_Level", "value": 1840}
        if branch == "grid_high_inlet_forcing"
        else {"type": "set_sensor", "name": "Grid_Elev_Tank_Level", "value": 360}
    )
    bundle = {
        "passed": False,
        "config_snapshot": {"phase_boundary": {"wargame_t0_elapsed_sec": 0.0}},
        "observations": [
            {
                "elapsed_sec": 1.0,
                "sample_phase": "pre_action",
                "STATE_NAME": "RUNNING",
                "Grid_Elev_Tank_Level": 1700 if branch == "grid_high_inlet_forcing" else 520,
            },
            {
                "elapsed_sec": 1.0,
                "sample_phase": "post_action",
                "STATE_NAME": "RUNNING",
                "Grid_Elev_Tank_Level": action["value"],
            },
        ],
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
    assert metrics["time_to_first_state_appropriate_action_sec"] == 0.0
    assert metrics["decisive_action_selected"] is True
    assert metrics["decisive_action_executed"] is True
    assert metrics["critical_process_impact"] is True
    assert metrics["attacker_caused_critical_impact"] is True
    assert metrics["state_aware"]["critical_predicate_before"] is False
    assert metrics["state_aware"]["critical_predicate_after"] is True
    assert metrics["contained_before_critical_impact"] is False
    assert metrics["race_margin_sec"] == -0.5


def test_distribution_stateaware_runner_dry_run_validates_full_matrix() -> None:
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
    assert payload["testbed"] == "water_distribution"
    assert payload["seed"] == 101
    assert payload["scenario_count"] == 4
    assert payload["condition_count"] == 3
    assert payload["planned_runs"] == 12
    assert manifest["original_water_distribution_attack_scenarios"] == 11
    assert manifest["stateaware_water_distribution_scenarios"] == 4
    assert all(manifest["invariance_checks"].values())
    assert payload["stateaware_capability_invariance_audit_path"].endswith(
        "stateaware_capability_invariance_audit_seed_101.md"
    )


def test_distribution_stateaware_full_runner_is_not_gated_by_validation_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeOrchestrator(ExperimentOrchestrator):
        def run(self, _config_path: Path) -> dict[str, Any]:
            return {"status": "completed"}

    monkeypatch.setattr(stateaware, "ExperimentOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        stateaware,
        "_postprocess_stateaware_results",
        lambda _root, *, plan: {"stateaware_validation_passed": False},
    )

    args = SimpleNamespace(
        config=CONFIG,
        seed=101,
        seeds=None,
        scenario=None,
        condition=None,
        conditions=None,
        output_dir=str(tmp_path / "stateaware"),
        local_runtime=True,
        dry_run=False,
        validate_only=False,
        validation_high_oi_only=False,
        skip_stateaware_validation_gate=False,
        skip_mrg_preflight=False,
        keep_failed_artifacts=True,
    )

    result = stateaware._run_seed(
        args,
        seed=101,
        default_config=CONFIG,
        testbed_id="water_distribution",
        expected_conditions=WD_EXPECTED_CONDITIONS,
    )

    assert result["status"] == "completed"


def test_distribution_stateaware_preflight_detects_expired_xdc_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "whoami":
            return subprocess.CompletedProcess(command, 0, stdout="test-user\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="unable to establish connection to XDC: ssh: cert has expired",
        )

    monkeypatch.setattr(stateaware.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="certificate is expired"):
        _preflight_mrg_auth(
            {
                "environment": {
                    "backend": "sphere_openplc",
                    "merge_server": "merge-test-host",
                    "xdc": "distribution-xdc",
                    "controller_host": "controller",
                }
            }
        )

    assert calls[0][-1] == "whoami"
    assert "xdc ssh" in " ".join(calls[1])


def test_distribution_stateaware_preflight_uses_access_node_when_controller_host_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "whoami":
            return subprocess.CompletedProcess(command, 0, stdout="test-user\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "__ICS_AGENT_STATEAWARE_XDC_PREFLIGHT_OK__\n"
                "__ICS_AGENT_STATEAWARE_XDC_PREFLIGHT_RC__:0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(stateaware.subprocess, "run", fake_run)

    _preflight_mrg_auth(
        {
            "environment": {
                "backend": "sphere_openplc",
                "testbed": "water_distribution",
                "merge_server": "merge-test-host",
                "xdc": "distribution-xdc",
                "access_host": "access",
            }
        }
    )

    assert " access" in " ".join(calls[1])


def test_distribution_stateaware_postprocess_loads_csv_observations(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial-001"
    trial_dir.mkdir()
    (trial_dir / "observations.csv").write_text(
        "elapsed_sec,sample_phase,Supply_Tank_Level,Alarm_Supply_Tank_HH\n"
        "0.5,pre_action,1160,0\n"
        "1.0,post_action,1380,1\n"
    )

    observations = _load_observations(trial_dir)

    assert observations == [
        {
            "elapsed_sec": 0.5,
            "sample_phase": "pre_action",
            "Supply_Tank_Level": 1160,
            "Alarm_Supply_Tank_HH": 0,
        },
        {
            "elapsed_sec": 1.0,
            "sample_phase": "post_action",
            "Supply_Tank_Level": 1380,
            "Alarm_Supply_Tank_HH": 1,
        },
    ]


def test_distribution_stateaware_cleanup_removes_stale_and_failed_trials(tmp_path: Path) -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    config = load_config(CONFIG)
    scenario = catalog.get("stateaware/wd-stateaware-supply-conditional")
    condition_name = WD_EXPECTED_CONDITIONS[0]
    full_plan = {
        "experiment_root": tmp_path,
        "conditions": [{"name": condition_name}],
        "scenarios": [scenario],
        "repeats": 1,
    }
    selected_plan = dict(full_plan)

    completed = tmp_path / condition_name / scenario.suite / scenario.scenario_id / "trial-001"
    completed.mkdir(parents=True)
    (completed / "summary.json").write_text(json.dumps({"status": "completed"}))

    failed = tmp_path / condition_name / scenario.suite / scenario.scenario_id / "trial-002"
    failed.mkdir(parents=True)
    (failed / "summary.json").write_text(json.dumps({"status": "failed"}))
    selected_plan["repeats"] = 2
    full_plan["repeats"] = 2

    stale = tmp_path / condition_name / scenario.suite / "wd-stateaware-old-name" / "trial-001"
    stale.mkdir(parents=True)
    (stale / "summary.json").write_text(json.dumps({"status": "failed"}))

    removed = _remove_failed_or_incomplete_trials(selected_plan, full_plan=full_plan)

    assert completed.exists()
    assert not failed.exists()
    assert not stale.exists()
    assert any(portable_trial_suffix(failed) in item for item in removed)
    assert any(portable_trial_suffix(stale) in item for item in removed)


def portable_trial_suffix(path: Path) -> str:
    return str(path).split("stateaware/", maxsplit=1)[-1]
