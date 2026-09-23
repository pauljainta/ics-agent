from __future__ import annotations

import json
from pathlib import Path

import pytest

from ics_agent.config import load_config
from ics_agent.experiments import generic_reconnaissance_runner as recon


def test_manifest_enumerates_expected_39_scenarios() -> None:
    entries = recon.build_manifest_entries(
        testbeds=["water_treatment", "water_distribution"],
        suites=["attack", "stateaware"],
        scenario_id=None,
    )
    validation = recon._validate_manifest_counts(entries)

    assert validation["full_39_scenario_manifest"] is True
    assert validation["actual_total"] == 39
    assert validation["counts"]["water_treatment::attack"] == 18
    assert validation["counts"]["water_treatment::stateaware"] == 6
    assert validation["counts"]["water_distribution::attack"] == 11
    assert validation["counts"]["water_distribution::stateaware"] == 4
    assert len({entry["canonical_id"] for entry in entries}) == 39


def test_prepare_configs_uses_generic_attacker_and_silent_defender(tmp_path: Path) -> None:
    plans, entries, manifest_path = recon.prepare_run_plans(
        testbeds=["water_treatment"],
        suites=["attack"],
        scenario_id=None,
        seed=101,
        output_root=tmp_path,
        write_configs=True,
    )

    assert manifest_path.exists()
    assert len(entries) == 18
    assert len(plans) == 1
    config = load_config(plans[0].config_path)
    condition = config["conditions"][0]
    overrides = condition["profile_overrides"]

    assert condition["name"] == recon.CONDITION_NAME
    assert condition["profiles"]["attacker"] == "target_semantic_ablation_attacker"
    assert condition["profiles"]["defender"] == "silent_defender"
    assert condition["profiles"]["observer"] == "silent_observer"
    assert overrides["attacker"]["target_semantics"]["variant"] == "no_target_semantics"
    assert overrides["attacker"]["target_semantics"]["interface_variant"] == "normalized_interface"
    assert overrides["attacker"]["knowledge_files"] == [
        "configs/knowledge/shared/high_cyber_ics_methodology_manual.md"
    ]
    assert overrides["defender"]["allowed_action_types"] == ["noop"]
    assert overrides["observer"]["allowed_action_types"] == ["noop"]
    assert config["study_metadata"]["expected_xdc"] == "example-xdc-09"


def test_fixed_xdc_environment_overrides_private_config(tmp_path: Path) -> None:
    plans, _entries, _manifest_path = recon.prepare_run_plans(
        testbeds=["water_distribution"],
        suites=["stateaware"],
        scenario_id=None,
        seed=101,
        output_root=tmp_path,
        write_configs=True,
    )

    with recon._fixed_xdc_environment("water_distribution"):
        config = load_config(plans[0].config_path)

    assert config["environment"]["xdc"] == "example-xdc-06"
    assert config["environment"].get("allow_xdc_discovery") is False


def test_default_main_is_dry_run_and_does_not_preflight_or_execute(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = {"preflight": 0, "execute": 0}

    def fake_preflight(_plan):
        called["preflight"] += 1

    def fake_execute(_plan):
        called["execute"] += 1

    monkeypatch.setattr(recon, "preflight_plan", fake_preflight)
    monkeypatch.setattr(recon, "execute_plan", fake_execute)

    rc = recon.main(
        [
            "--testbed",
            "wt",
            "--family",
            "legacy",
            "--output-root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["mode"] == "dry-run"
    assert called == {"preflight": 0, "execute": 0}
    assert payload["plans"][0]["scenario_count"] == 18


def test_enrich_trial_artifacts_writes_separated_recon_logs(tmp_path: Path) -> None:
    trial_dir = tmp_path / "condition" / "attack" / "scenario-a" / "trial-001"
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.json").write_text(
        json.dumps(
            {
                "scenario": "attack/scenario-a",
                "scenario_id": "scenario-a",
                "testbed": "water_treatment",
                "suite": "attack",
                "condition_name": recon.CONDITION_NAME,
                "status": "completed",
                "timestamp_start_utc": "2026-01-01T00:00:00+00:00",
                "timestamp_end_utc": "2026-01-01T00:00:10+00:00",
            }
        )
    )
    (trial_dir / "metrics.json").write_text(
        json.dumps(
            {
                "attacker": {"attacker_caused_safety_violation": True},
                "defender_detection": {"time_to_detection_sec": None},
            }
        )
    )
    (trial_dir / "actions.json").write_text(
        json.dumps(
            [
                {
                    "role": "attacker",
                    "time_sec": 0.5,
                    "timestamp_utc": "2026-01-01T00:00:01+00:00",
                    "action": {"type": "inspect_surface", "surface": "surf_01"},
                    "result": {"surface": "surf_01"},
                },
                {
                    "role": "attacker",
                    "time_sec": 1.0,
                    "timestamp_utc": "2026-01-01T00:00:02+00:00",
                    "action": {"type": "set_sensor", "name": "sig_01", "value": 999},
                    "result": {"accepted": True},
                },
            ]
        )
    )
    (trial_dir / "decisions.json").write_text(
        json.dumps(
            [
                {
                    "role": "attacker",
                    "time_sec": 1.0,
                    "timestamp_utc": "2026-01-01T00:00:02+00:00",
                    "decision_type": "action",
                    "summary": "Probe then drive a signal.",
                    "action": {"type": "set_sensor", "name": "sig_01", "value": 999},
                    "metadata": {"reasoning_summary": "Drive effect on observed process state."},
                }
            ]
        )
    )
    (trial_dir / "system_events.json").write_text("[]")

    audit = recon.enrich_trial_artifacts(trial_dir, {"canonical_id": "attack/scenario-a"})

    assert audit["critical_impact_evidence"]["attacker_caused_critical_impact"] is True
    assert audit["action_accounting"]["attacker_recon_action_count"] == 1
    assert audit["action_accounting"]["attacker_effect_capable_action_count"] == 1
    assert (trial_dir / "attacker_visible_trace.jsonl").exists()
    assert (trial_dir / "evaluator_private_trace.jsonl").exists()
    assert (trial_dir / "recon_audit_summary.json").exists()
