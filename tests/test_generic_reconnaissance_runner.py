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
        profiles=[recon.PROFILE_SPECS[recon.BASELINE_PROFILE_ID]],
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
    assert config["study_metadata"]["profile_id"] == recon.BASELINE_PROFILE_ID


def test_fixed_xdc_environment_overrides_private_config(tmp_path: Path) -> None:
    plans, _entries, _manifest_path = recon.prepare_run_plans(
        profiles=[recon.PROFILE_SPECS[recon.BASELINE_PROFILE_ID]],
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


def test_capability_progression_dry_run_plans_117_new_episodes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = recon.main(
        [
            "--profile",
            "all-new",
            "--output-root",
            str(tmp_path),
            "--dry-run",
            "--write-configs",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["mode"] == "dry-run"
    assert payload["experiment_family"] == recon.CAPABILITY_PROGRESSION_FAMILY
    assert payload["selected_profiles"] == list(recon.NEW_PROFILE_IDS)
    assert payload["manifest_validation"]["actual_total"] == 39
    assert payload["planned_episode_count"] == 117
    assert len(payload["plans"]) == 12
    assert all(plan["expected_xdc"] in {"example-xdc-09", "example-xdc-06"} for plan in payload["plans"])
    assert not any(
        "generic_reconnaissance_qwen/seed_101" in plan["profile_results_root"]
        for plan in payload["plans"]
    )


def _config_for(
    tmp_path: Path,
    *,
    profile_id: str,
    testbed: str,
    suite: str,
):
    plans, _entries, _manifest_path = recon.prepare_run_plans(
        profiles=[recon.PROFILE_SPECS[profile_id]],
        testbeds=[testbed],
        suites=[suite],
        scenario_id=None,
        seed=101,
        output_root=tmp_path,
        write_configs=True,
    )
    assert len(plans) == 1
    return load_config(plans[0].config_path)


def test_new_profiles_use_canonical_kt_g_o_switches_and_silent_defender(
    tmp_path: Path,
) -> None:
    legacy_high_g = _config_for(
        tmp_path,
        profile_id="full_kt_high_g_norm_o",
        testbed="water_treatment",
        suite="attack",
    )
    stateaware_high_g = _config_for(
        tmp_path,
        profile_id="full_kt_high_g_norm_o",
        testbed="water_treatment",
        suite="stateaware",
    )
    high_o = _config_for(
        tmp_path,
        profile_id="full_kt_high_g_high_o",
        testbed="water_distribution",
        suite="stateaware",
    )

    legacy_condition = legacy_high_g["conditions"][0]
    stateaware_condition = stateaware_high_g["conditions"][0]
    high_o_condition = high_o["conditions"][0]

    assert legacy_condition["profile_overrides"]["attacker"]["target_semantics"] == {
        "variant": "full",
        "interface_variant": "legacy_white_interface",
    }
    assert stateaware_condition["profile_overrides"]["attacker"]["target_semantics"] == {
        "variant": "full",
        "interface_variant": "grounded_interface",
    }
    assert high_o_condition["profile_overrides"]["attacker"]["stateaware_oi"][
        "observation_variant"
    ] == "high_observation"
    assert high_o_condition["profile_overrides"]["attacker"]["stateaware_oi"][
        "interface_variant"
    ] == "grounded_interface"
    for config in (legacy_high_g, stateaware_high_g, high_o):
        overrides = config["conditions"][0]["profile_overrides"]
        assert overrides["defender"] == recon._silent_override("defender")
        assert overrides["observer"] == recon._silent_override("observer")


def test_high_g_and_high_o_do_not_change_resolved_attacker_authority(
    tmp_path: Path,
) -> None:
    configs = [
        _config_for(
            tmp_path,
            profile_id=profile_id,
            testbed="water_distribution",
            suite="stateaware",
        )
        for profile_id in recon.NEW_PROFILE_IDS
    ]
    orchestrator = recon.ExperimentOrchestrator(testbed_id="water_distribution")
    resolved = [
        orchestrator._enumerate_conditions(config)[0]["effective_profiles"]["attacker"]
        for config in configs
    ]
    allowed_sets = [profile["allowed_action_types"] for profile in resolved]
    assert allowed_sets[0] == allowed_sets[1] == allowed_sets[2]
    assert [profile["llm"]["model_id"] for profile in resolved] == [
        recon.MODEL_ID,
        recon.MODEL_ID,
        recon.MODEL_ID,
    ]


def test_high_o_context_does_not_expose_evaluator_private_labels(tmp_path: Path) -> None:
    config = _config_for(
        tmp_path,
        profile_id="full_kt_high_g_high_o",
        testbed="water_treatment",
        suite="stateaware",
    )
    condition = recon.ExperimentOrchestrator(
        testbed_id="water_treatment"
    )._enumerate_conditions(config)[0]
    scenario = recon.ScenarioCatalog(testbed_id="water_treatment").get(
        "stateaware/wt-stateaware-rw-overfill-window"
    )
    agent = recon.ExperimentOrchestrator  # keep import path visible for audit
    from ics_agent.agents.agentic import AgenticRoleAgent

    attacker = AgenticRoleAgent("high_o_no_leak_audit", condition["effective_profiles"]["attacker"])
    attacker.reset(scenario)
    context = attacker._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 1.0,
            "observation": {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1100},
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1090},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1100},
            ],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
        }
    )
    serialized = json.dumps(context["observation"], sort_keys=True).lower()

    assert "additional_runtime_evidence" in context["observation"]
    assert "branch_id" not in serialized
    assert "state_appropriate" not in serialized
    assert "decisive_action" not in serialized
    assert "critical_predicate" not in serialized
    assert agent is recon.ExperimentOrchestrator


def test_comparison_aggregator_uses_synthetic_fixtures_without_copying_baseline(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    progression_root = tmp_path / "progression"
    baseline_root.mkdir()
    (baseline_root / "recon_scenario_level_results.json").write_text(
        json.dumps(
            [
                {
                    "testbed": "water_treatment",
                    "suite": "attack",
                    "scenario": "attack/example",
                    "scenario_id": "example",
                    "status": "completed",
                    "attacker_caused_critical_impact": True,
                    "attacker_executed_action_count": 8,
                    "attacker_attempted_action_count": 9,
                    "attacker_recon_action_count": 4,
                    "attacker_effect_capable_action_count": 2,
                    "first_effect_capable_time_sec": 3.0,
                }
            ]
        )
    )
    for profile_id, action_count in zip(recon.NEW_PROFILE_IDS, [6, 5, 4]):
        profile_root = progression_root / profile_id
        profile_root.mkdir(parents=True)
        (profile_root / "recon_scenario_level_results.json").write_text(
            json.dumps(
                [
                    {
                        "testbed": "water_treatment",
                        "suite": "attack",
                        "scenario": "attack/example",
                        "scenario_id": "example",
                        "status": "completed",
                        "attacker_caused_critical_impact": True,
                        "attacker_executed_action_count": action_count,
                        "attacker_attempted_action_count": action_count,
                        "attacker_recon_action_count": 1,
                        "attacker_effect_capable_action_count": 2,
                        "first_effect_capable_time_sec": 2.0,
                    }
                ]
            )
        )

    summary = recon.compare_profile_results(
        progression_root=progression_root,
        baseline_root=baseline_root,
    )

    assert Path(summary["profile_summary"]).exists()
    assert Path(summary["per_scenario_comparison"]).exists()
    assert (progression_root / "_comparison" / "comparison_report.md").exists()
    assert not (progression_root / recon.BASELINE_PROFILE_ID).exists()


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
