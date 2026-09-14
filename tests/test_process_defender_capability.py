from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ics_agent.config import load_config
from ics_agent.experiments.process_defender_capability import (
    _legacy_critical_audit_row,
    _condition_aggregate_rows,
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


def test_validate_only_wt_all_capability_reports_real_legacy_suites_and_missing_observation() -> None:
    payload = _run_validate("--testbed", "wt", "--suite", "all-capability", "--seed", "101")
    assert payload["status"] == "completed"
    assert [item["status"] for item in payload["summaries"]] == [
        "validated",
        "validated",
        "not_discovered",
    ]
    kt_payload = _run_validate("--testbed", "wt", "--suite", "kt", "--seed", "101")
    assert kt_payload["status"] == "validated"
    plan = kt_payload["plan"]
    assert plan["scenario_count"] == 18
    assert plan["condition_count"] == 8
    assert plan["planned_runs"] == 144
    assert "process-defender-critical-impact-v2" in plan["resolved_experiment_root"]


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
