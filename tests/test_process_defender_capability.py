from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.experiments.process_defender_capability import (
    _attacker_trace_rows,
    _condition_aggregate_rows,
    _defender_trace_rows,
    _event_trace_rows,
    _legacy_critical_audit_row,
    _postprocess_results,
    main as process_defender_capability_main,
    _write_generated_config,
)
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_process_defender_capability_rerun.py"


def _run_validate(*args: str) -> dict[str, object]:
    env = dict(os.environ)
    env["ICS_AGENT_WD_PROCESS_DEFENDER_XDC"] = "wd-xdc-for-test"
    result = subprocess.run(
        [sys.executable, str(RUNNER), *args, "--validate-only", "--skip-mrg-preflight"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    decoder = json.JSONDecoder()
    payloads = []
    text = result.stdout.lstrip()
    while text:
        payload, end = decoder.raw_decode(text)
        payloads.append(payload)
        text = text[end:].lstrip()
    assert payloads
    return payloads[-1]


def _condition_map(config_name: str, testbed_id: str) -> dict[str, dict[str, object]]:
    config = load_config(config_name)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    return {
        str(condition["name"]): condition
        for condition in orchestrator._enumerate_conditions(config)
    }


def _treatment_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "elapsed_sec": 18.0,
        "RW_Tank_Level": 600.0,
        "UF_UFFT_Tank_Level": 400.0,
        "ChemTreat_NaCl_Level": 800.0,
        "ChemTreat_NaOCl_Level": 800.0,
        "ChemTreat_HCl_Level": 800.0,
        "RW_Pump_Sts": 1.0,
        "RW_Pump_Flow": 100.0,
        "RW_Tank_PR_Valve_Sts": 0.0,
        "P2_Selected": 0.0,
        "ChemTreat_NaCl_Valve_Sts": 0.0,
        "ChemTreat_NaOCl_Valve_Sts": 0.0,
        "ChemTreat_HCl_Valve_Sts": 0.0,
        "Alarm_RW_Tank_LL": 0.0,
        "Alarm_RW_Tank_HH": 0.0,
    }
    observation.update(overrides)
    return observation


def test_generated_wt_kt_config_preserves_attacker_profiles_and_patches_defender() -> None:
    source = (
        "water_treatment_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
    )
    generated_path = _write_generated_config(
        source_config=source,
        testbed_id="water_treatment",
        suite="kt",
        seed=101,
        scenario=None,
        only_condition=None,
    )
    base_conditions = _condition_map(source, "water_treatment")
    generated_conditions = _condition_map(str(generated_path), "water_treatment")
    assert base_conditions.keys() == generated_conditions.keys()
    for name, condition in generated_conditions.items():
        assert (
            condition["effective_profiles"]["attacker"]
            == base_conditions[name]["effective_profiles"]["attacker"]
        )
        defender = condition["effective_profiles"]["defender"]
        assert defender["process_physics_only"] is True
        assert defender["defender_observation_mode"] == "process_physics_only"
        assert defender["defer_containment_until_attacker_effect"] is True


def test_generated_wt_observation_config_varies_only_attacker_observation_and_patches_defender() -> None:
    source = (
        "water_treatment_target_observation_ablation_vs_agentic_full_containment_defender.yaml"
    )
    generated_path = _write_generated_config(
        source_config=source,
        testbed_id="water_treatment",
        suite="observation",
        seed=101,
        scenario=None,
        only_condition=None,
    )
    generated_conditions = _condition_map(str(generated_path), "water_treatment")
    assert sorted(generated_conditions) == [
        "full_target_semantics_highO_highI__agentic_defender",
        "full_target_semantics_normO_highI__agentic_defender",
    ]
    attacker_profiles = [
        condition["effective_profiles"]["attacker"]
        for condition in generated_conditions.values()
    ]
    stripped_profiles = []
    observed_o_variants = set()
    observed_target_interfaces = set()
    observed_stateaware_interfaces = set()
    for profile in attacker_profiles:
        observed_target_interfaces.add(profile["target_semantics"]["interface_variant"])
        stateaware_oi = dict(profile["stateaware_oi"])
        observed_o_variants.add(stateaware_oi["observation_variant"])
        observed_stateaware_interfaces.add(stateaware_oi["interface_variant"])
        stateaware_oi.pop("condition_name", None)
        stateaware_oi.pop("observation_variant", None)
        stripped = dict(profile)
        stripped["stateaware_oi"] = stateaware_oi
        stripped_profiles.append(stripped)
    assert observed_o_variants == {"normalized_observation", "high_observation"}
    assert observed_target_interfaces == {"legacy_white_interface"}
    assert observed_stateaware_interfaces == {"legacy_white_interface"}
    assert stripped_profiles[0] == stripped_profiles[1]
    for condition in generated_conditions.values():
        defender = condition["effective_profiles"]["defender"]
        assert defender["process_physics_only"] is True
        assert defender["defender_observation_mode"] == "process_physics_only"
        assert defender["defer_containment_until_attacker_effect"] is True


def test_wt_observation_ablation_high_o_adds_runtime_evidence_on_old_attack_scenario() -> None:
    source = (
        "water_treatment_target_observation_ablation_vs_agentic_full_containment_defender.yaml"
    )
    generated_path = _write_generated_config(
        source_config=source,
        testbed_id="water_treatment",
        suite="observation",
        seed=101,
        scenario=None,
        only_condition=None,
    )
    conditions = _condition_map(str(generated_path), "water_treatment")
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "attack/wt-attack-sensor-rw-refill-spoof"
    )
    previous_observation = _treatment_observation(elapsed_sec=17.5, RW_Tank_Level=590.0)
    current_observation = _treatment_observation(elapsed_sec=18.0, RW_Tank_Level=600.0)
    context = {
        "scenario": scenario,
        "observation": current_observation,
        "observations": [previous_observation, current_observation],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "remaining_playbook": [],
        "playbooks": {},
        "time_sec": 18.0,
    }
    built_contexts = {}
    for name, condition in conditions.items():
        agent = AgenticRoleAgent(name, condition["effective_profiles"]["attacker"])
        agent.reset(scenario)
        built_contexts[name] = agent._build_online_context(context)

    norm_context = built_contexts["full_target_semantics_normO_highI__agentic_defender"]
    high_context = built_contexts["full_target_semantics_highO_highI__agentic_defender"]
    assert "additional_runtime_evidence" not in norm_context["observation"]
    assert "additional_runtime_evidence" in high_context["observation"]
    assert all(
        "additional_runtime_evidence" not in recent_observation
        for recent_observation in high_context["recent_observations"]
    )
    assert {
        item.get("source") for item in high_context["candidate_actions"]
    } == {"legacy_white_interface"}
    assert norm_context["system_knowledge"]["target_semantics"] == (
        high_context["system_knowledge"]["target_semantics"]
    )


def test_validate_only_wt_all_capability_reports_real_legacy_suites() -> None:
    payload = _run_validate("--testbed", "wt", "--suite", "all-capability", "--seed", "101")
    assert payload["status"] == "completed"
    assert [item["status"] for item in payload["summaries"]] == [
        "validated",
        "validated",
        "validated",
    ]
    kt_payload = _run_validate("--testbed", "wt", "--suite", "kt", "--seed", "101")
    assert kt_payload["status"] == "validated"
    plan = kt_payload["plan"]
    assert plan["scenario_count"] == 18
    assert plan["condition_count"] == 8
    assert plan["planned_runs"] == 144
    assert "process-defender-critical-impact-v2" in plan["resolved_experiment_root"]
    observation_payload = _run_validate(
        "--testbed",
        "wt",
        "--suite",
        "observation",
        "--seed",
        "101",
    )
    assert observation_payload["status"] == "validated"
    observation_plan = observation_payload["plan"]
    assert observation_plan["scenario_count"] == 18
    assert observation_plan["condition_count"] == 2
    assert observation_plan["planned_runs"] == 36
    assert "target-observation-ablation" in observation_plan["resolved_experiment_root"]


def test_dry_run_wt_all_capability_does_not_execute_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute the experiment orchestrator")

    monkeypatch.setattr(ExperimentOrchestrator, "run", fail_run)
    rc = process_defender_capability_main(
        [
            "--testbed",
            "wt",
            "--suite",
            "all-capability",
            "--seed",
            "101",
            "--dry-run",
            "--skip-mrg-preflight",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    first_plan = payload["summaries"][0]["plan"]
    assert first_plan["mode"] == "dry_run"
    assert first_plan["defender_mode"] == "process-aware"
    assert first_plan["seeds"] == [101]


def test_wd_dry_run_requires_external_xdc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ICS_AGENT_WD_PROCESS_DEFENDER_XDC", raising=False)
    monkeypatch.delenv("ICS_AGENT_SPHERE_XDC_OVERRIDE", raising=False)
    with pytest.raises(RuntimeError, match="require an explicit XDC"):
        process_defender_capability_main(
            [
                "--testbed",
                "wd",
                "--suite",
                "kt",
                "--seed",
                "101",
                "--dry-run",
                "--skip-mrg-preflight",
            ]
        )


def test_smoke_mode_for_wt_kt_selects_first_scenario_and_first_condition() -> None:
    payload = _run_validate(
        "--testbed",
        "wt",
        "--suite",
        "kt",
        "--seed",
        "101",
        "--smoke",
        "--skip-mrg-preflight",
    )
    assert payload["status"] == "validated"
    plan = payload["plan"]
    assert plan["scenario_count"] == 1
    assert plan["condition_count"] == 1
    assert plan["planned_runs"] == 1


def test_validate_only_wd_interface_requires_explicit_xdc_and_reports_22_runs() -> None:
    payload = _run_validate(
        "--testbed",
        "wd",
        "--suite",
        "interface",
        "--seed",
        "101",
        "--xdc",
        "wd-xdc-for-test",
    )
    assert payload["status"] == "validated"
    plan = payload["plan"]
    assert plan["scenario_count"] == 11
    assert plan["condition_count"] == 2
    assert plan["planned_runs"] == 22


def test_focused_condition_preserves_full_suite_manifest() -> None:
    payload = _run_validate(
        "--testbed",
        "wt",
        "--suite",
        "interface",
        "--only-condition",
        "full_target_semantics_normalized_interface__agentic_defender",
        "--scenario",
        "wt-attack-sensor-rw-refill-spoof",
    )
    assert payload["status"] == "validated"
    plan = payload["plan"]
    assert plan["scenario_count"] == 1
    assert plan["condition_count"] == 1
    assert plan["planned_runs"] == 1
    assert plan["full_suite_plan"]["planned_runs"] == 36


def test_legacy_critical_audit_marks_control_integrity_without_safety_oracle() -> None:
    scenario = ScenarioCatalog(testbed_id="water_distribution").get(
        "attack/wd-attack-grid-valve-force-open"
    )
    row = _legacy_critical_audit_row(
        condition="full_target_semantics__agentic_defender",
        scenario_id="wd-attack-grid-valve-force-open",
        trial="trial-001",
        summary={
            "status": "completed",
            "run_id": "attack-wd-attack-grid-valve-force-open-trial-001",
        },
        metrics={
            "safety_violation_count": 0,
            "attacker_action_count": 1,
            "attacker": {"caused_safety_violation": False},
            "runtime_containment": {},
        },
        decisions=[
            {
                "role": "attacker",
                "time_sec": 18.0,
                "action": {
                    "type": "set_actuator",
                    "name": "Grid_Elev_Valve",
                    "value": True,
                },
            }
        ],
        actions=[
            {
                "role": "attacker",
                "time_sec": 18.0,
                "action": {
                    "type": "set_actuator",
                    "name": "Grid_Elev_Valve",
                    "value": True,
                },
            }
        ],
        observations=[
            {"elapsed_sec": 17.5, "sample_phase": "pre_action", "Grid_Elev_Tank_Level": 1550},
            {"elapsed_sec": 18.0, "sample_phase": "post_action", "Grid_Elev_Tank_Level": 1550},
        ],
        system_events=[],
        scenario=scenario,
        seed=101,
        testbed_id="water_distribution",
        suite="kt",
    )
    assert row["critical_impact_contract_status"] == "legacy_derived_audit"
    assert row["physical_safety_impact"] is False
    assert row["control_integrity_impact"] is True
    assert row["attacker_caused_critical_impact"] is True
    assert row["decisive_action_executed"] is True


def test_condition_aggregate_reports_requested_critical_impact_columns() -> None:
    rows = [
        {
            "condition": "c1",
            "status": "completed",
            "state_appropriate_action_selected": True,
            "state_appropriate_action_executed": True,
            "decisive_action_selected": True,
            "decisive_action_issued": True,
            "decisive_action_executed": True,
            "critical_impact_evaluated": True,
            "attacker_caused_critical_impact": True,
            "physical_safety_impact": True,
            "availability_impact": False,
            "control_integrity_impact": False,
            "defender_prevention": False,
            "defender_mitigation": True,
            "defender_recovery": False,
            "invalid_precontained_run": False,
            "T_critical": 2.0,
            "T_detect": 1.0,
            "T_contain": 3.0,
            "attacker_attempted_action_count": 2,
            "attacker_executed_action_count": 1,
            "attacker_effect_capable_action_count": 1,
            "attacker_ineffective_action_count": 1,
            "blocked_attacker_action_count": 1,
        },
        {"condition": "c1", "status": "failed"},
    ]
    aggregate = _condition_aggregate_rows(rows)[0]
    assert aggregate["attack_run_count"] == 2
    assert aggregate["completed_run_count"] == 1
    assert aggregate["failed_run_count"] == 1
    assert aggregate["decisive_issued_run_count"] == 1
    assert aggregate["attacker_caused_critical_impact_run_count"] == 1
    assert aggregate["defender_mitigation_run_count"] == 1
    assert aggregate["avg_T_critical"] == 2.0
    assert aggregate["avg_attempted_actions"] == 2.0
    assert aggregate["avg_effect_capable_actions"] == 1.0
    assert aggregate["avg_blocked_actions"] == 1.0


def test_condition_aggregate_null_safe_critical_impact_counts_and_rates() -> None:
    rows = [
        {
            "condition": "c1",
            "status": "completed",
            "critical_impact_evaluated": None,
            "attacker_caused_critical_impact": None,
            "physical_safety_impact": None,
            "availability_impact": None,
            "control_integrity_impact": None,
        },
        {
            "condition": "c1",
            "status": "completed",
            "critical_impact_evaluated": True,
            "attacker_caused_critical_impact": True,
            "physical_safety_impact": True,
            "availability_impact": False,
            "control_integrity_impact": False,
        },
    ]
    aggregate = _condition_aggregate_rows(rows)[0]
    assert aggregate["critical_impact_evaluated_run_count"] == 1
    assert aggregate["critical_impact_unknown_run_count"] == 1
    assert aggregate["attacker_caused_critical_impact_run_count"] == 1
    assert aggregate["attacker_caused_critical_impact_false_run_count"] == 0
    assert aggregate["attacker_caused_critical_impact_rate"] == 1.0


def test_process_defender_trace_helpers_explain_selected_blocked_and_contained() -> None:
    decisions = [
        {
            "role": "attacker",
            "time_sec": 1.0,
            "action": {"type": "set_sensor", "name": "Tank", "value": 11},
            "metadata": {
                "candidate_actions": [
                    {"id": "candidate_01", "action": {"type": "inspect_surface"}},
                    {
                        "id": "candidate_02",
                        "action": {"type": "set_sensor", "name": "Tank", "value": 11},
                    },
                ],
                "selected_candidate_id": "candidate_02",
            },
        },
        {
            "role": "attacker",
            "time_sec": 2.0,
            "action": {"type": "set_sensor", "name": "Tank", "value": 12},
            "metadata": {"selected_candidate_id": "candidate_03"},
        },
        {
            "role": "defender",
            "time_sec": 1.5,
            "action": {"type": "block_sensor_write", "name": "Tank"},
            "diagnosis": "Tank value exceeded process envelope.",
        },
    ]
    actions = [
        {
            "role": "attacker",
            "time_sec": 1.0,
            "action": {"type": "set_sensor", "name": "Tank", "value": 11},
        },
        {
            "role": "defender",
            "time_sec": 1.5,
            "action": {"type": "block_sensor_write", "name": "Tank"},
            "containment": {"is_containment_action": True},
        },
    ]
    observations = [
        {"elapsed_sec": 1.0, "sample_phase": "pre_action", "Tank": 8},
        {"elapsed_sec": 1.0, "sample_phase": "post_action", "Tank": 11},
        {"elapsed_sec": 1.5, "sample_phase": "pre_action", "Tank": 11},
    ]
    system_events = [
        {
            "event_type": "attacker_execution_cutoff_activated",
            "role": "defender",
            "time_sec": 1.5,
        },
        {
            "event_type": "attacker_action_blocked",
            "time_sec": 2.0,
            "attempted_action": {"type": "set_sensor", "name": "Tank", "value": 12},
            "reason": "runtime containment cutoff",
        },
    ]

    attacker_rows = _attacker_trace_rows(
        condition="c1",
        scenario_id="s1",
        trial="trial-001",
        decisions=decisions,
        actions=actions,
        observations=observations,
        system_events=system_events,
    )
    defender_rows = _defender_trace_rows(
        condition="c1",
        scenario_id="s1",
        trial="trial-001",
        decisions=decisions,
        actions=actions,
        observations=observations,
        system_events=system_events,
    )
    event_rows = _event_trace_rows(
        condition="c1",
        scenario_id="s1",
        trial="trial-001",
        decisions=decisions,
        actions=actions,
        observations=observations,
        system_events=system_events,
    )

    assert attacker_rows[0]["selected_candidate_id"] == "candidate_02"
    assert attacker_rows[0]["issued"] is True
    assert attacker_rows[0]["executed"] is True
    assert attacker_rows[0]["effect_capable_action"] is True
    assert attacker_rows[1]["blocked"] is True
    assert attacker_rows[1]["block_reason"] == "runtime containment cutoff"
    assert defender_rows[0]["containment_requested"] is True
    assert defender_rows[0]["containment_effective"] is True
    assert defender_rows[0]["containment_effective_time_sec"] == 1.5
    assert [row["event_index"] for row in event_rows] == list(
        range(1, len(event_rows) + 1)
    )


def test_process_defender_postprocess_writes_forensic_trace_files(tmp_path: Path) -> None:
    class Scenario:
        catalog_id = "attack/s1"
        scenario_id = "s1"
        suite = "attack"
        metadata = {"safety_bounds": {"Tank": {"max": 10}}}
        execution = {
            "initial_conditions": {"Tank": 8},
            "attack": {
                "steps": [
                    {"action": {"type": "set_sensor", "name": "Tank", "value": 11}}
                ]
            },
        }

    experiment_root = tmp_path / "experiment"
    trial_dir = experiment_root / "c1" / "attack" / "s1" / "trial-001"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(
        json.dumps({"status": "completed", "run_id": "r1"}) + "\n"
    )
    (trial_dir / "metrics.json").write_text(
        json.dumps(
            {
                "safety_violation_count": 1,
                "attacker_action_count": 1,
                "attacker": {"caused_safety_violation": True},
                "runtime_containment": {
                    "attacker_execution_cutoff_activated": True,
                    "blocked_attacker_action_count": 1,
                },
                "defender_intervention": {
                    "contained_before_safety_violation": False,
                    "time_to_first_action_sec": 1.5,
                },
            }
        )
        + "\n"
    )
    (trial_dir / "decisions.json").write_text(
        json.dumps(
            [
                {
                    "role": "attacker",
                    "time_sec": 1.0,
                    "action": {"type": "set_sensor", "name": "Tank", "value": 11},
                },
                {
                    "role": "defender",
                    "time_sec": 1.5,
                    "action": {"type": "block_sensor_write", "name": "Tank"},
                    "diagnosis": "Tank exceeded bound.",
                },
            ]
        )
        + "\n"
    )
    (trial_dir / "actions.json").write_text(
        json.dumps(
            [
                {
                    "role": "attacker",
                    "time_sec": 1.0,
                    "action": {"type": "set_sensor", "name": "Tank", "value": 11},
                },
                {
                    "role": "defender",
                    "time_sec": 1.5,
                    "action": {"type": "block_sensor_write", "name": "Tank"},
                    "containment": {"is_containment_action": True},
                },
            ]
        )
        + "\n"
    )
    (trial_dir / "system_events.json").write_text(
        json.dumps(
            [
                {
                    "event_type": "attacker_execution_cutoff_activated",
                    "role": "defender",
                    "time_sec": 1.5,
                }
            ]
        )
        + "\n"
    )
    (trial_dir / "observations.json").write_text(
        json.dumps(
            [
                {"elapsed_sec": 1.0, "sample_phase": "pre_action", "Tank": 8},
                {"elapsed_sec": 1.0, "sample_phase": "post_action", "Tank": 11},
                {"elapsed_sec": 1.5, "sample_phase": "pre_action", "Tank": 11},
            ]
        )
        + "\n"
    )

    postprocess = _postprocess_results(
        experiment_root,
        plan={
            "conditions": [{"name": "c1"}],
            "scenarios": [Scenario()],
            "repeats": 1,
            "config": {"reproducibility": {"seed": 101}},
            "testbed_id": "water_treatment",
            "suite": "kt",
        },
    )

    assert postprocess["runs_postprocessed"] == 1
    assert postprocess["attacker_trace_events"] == 1
    assert postprocess["defender_trace_events"] == 1
    assert postprocess["event_trace_events"] > 1
    for name in (
        "process_defender_critical_impact_audit.json",
        "process_defender_condition_comparison.json",
        "process_defender_attacker_trace.json",
        "process_defender_defender_trace.json",
        "process_defender_event_trace.json",
        "process_defender_validation_report.json",
    ):
        assert (experiment_root / name).exists()
