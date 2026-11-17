import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.scadaver_sphere.b2_r1 import B2R1Profile, B2R1StateManager
from experiments.scadaver_sphere.b2_r1_acceptance import (
    frozen_baseline_checks,
    profile_hashes,
)
from experiments.scadaver_sphere.b2_r1_campaign import (
    CampaignError,
    _flatten,
    _trial_index,
    _validated_source_commit,
    _matched_initial_state,
    _write_artifact_checksums,
    access_vm_context,
    campaign_lock,
    generate_reports,
    global_run_lock,
    inventory,
    load_pilot_plan,
    main as campaign_main,
    prepare,
    prepare_environment,
    run_condition,
    run_pilot,
    validate_study_assets,
    validate_profiles,
)
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.integrated_runner import (
    CompositeExecutor,
    _reproducibility_record,
)


ROOT = Path(__file__).parents[1] / "experiments/scadaver_sphere"


def _episode(root: Path, condition: str, *, complete=True, trial_id=None):
    path = root / condition / f"{condition}-fixture"
    path.mkdir(parents=True)
    if trial_id:
        (path / "trial_metadata.json").write_text(json.dumps({
            "trial_id": trial_id, "condition": condition, "seed": 101,
        }))
    if complete:
        (path / "episode.json").write_text(json.dumps({
            "status": "submitted_integrated_report",
            "evaluation": {"valid_submission": True, "overall_success": False},
            "reasoning_diagnostics": {"unsuccessful_tool_calls": 2},
        }))
        (path / "operator_post_verify.json").write_text('{"TESTBED_READY":true}')
    return path


def test_frozen_b2_and_harness_interfaces_are_unchanged():
    result = frozen_baseline_checks()
    assert result["passed"] is True
    assert result["cases"]["c0_prompt_equals_original_b2"] is True
    assert result["cases"]["c0_schema_equals_original_b2"] is True


def test_all_profile_hashes_and_semantics_validate():
    assert all(item["passed"] for item in validate_profiles().values())
    values = profile_hashes()
    assert set(values) == {"C0", "C1", "C2", "C3", "C4"}
    assert values["C4"]["components"] == {
        "error_aware_investigation": True,
        "evidence_verification": True,
        "uncertainty_review": True,
    }


def test_inventory_never_replaces_existing_episode(tmp_path):
    _episode(tmp_path, "C2")
    current = inventory(tmp_path)
    assert current["conditions"]["C2"]["eligible_for_new_run"] is False
    with pytest.raises(CampaignError, match="already has an episode"):
        run_condition("C2", tmp_path, "http://127.0.0.1:1")


def test_prepare_preserves_completed_and_records_initial_policy(tmp_path):
    _episode(tmp_path, "C0", trial_id="block-01-position-01-C0-seed-101")
    validation = {
        "deployment_ready": True, "testbed_ready": True,
        "initial_condition_record": {"process_sample": {"simulator_level": 450}},
    }
    state = prepare(tmp_path, tmp_path / "campaign.json", validation)
    dispositions = {item["trial_id"]: item["disposition"] for item in state["plan"]}
    assert dispositions["block-01-position-01-C0-seed-101"] == "PRESERVE_COMPLETED"
    assert dispositions["block-01-position-05-C4-seed-101"] == "ELIGIBLE"
    assert state["initial_condition_policy"]["reset_performed"] is False


def test_frozen_pilot_plan_is_balanced_and_rubric_is_deployable():
    plan = load_pilot_plan()
    assert plan["total_episodes"] == 15
    assert len(plan["schedule"]) == 15
    assert [item["condition"] for item in plan["schedule"][:5]] == [
        "C0", "C1", "C2", "C3", "C4",
    ]
    assert all(sum(item["condition"] == condition for item in plan["schedule"]) == 3
               for condition in ("C0", "C1", "C2", "C3", "C4"))
    assert len({item["trial_id"] for item in plan["schedule"]}) == 15
    assert validate_study_assets()["passed"] is True


def test_inventory_indexes_unique_planned_trials(tmp_path):
    trial_id = "block-02-position-04-C0-seed-202"
    _episode(tmp_path, "C0", trial_id=trial_id)
    current = inventory(tmp_path)
    assert _trial_index(current)[trial_id]["status"] == "EVALUATION_COMPLETE"


def test_corrupted_campaign_state_is_visible_not_silently_replaced(tmp_path):
    (tmp_path / "campaign_state.json").write_text("{not-json")
    current = inventory(tmp_path)
    assert current["campaign_state"] is None
    assert current["campaign_state_error"] == "JSONDecodeError"


def test_report_includes_every_episode_and_marks_e5_not_applicable(tmp_path):
    for index in range(2):
        path = tmp_path / "episodes/C1" / f"trial-{index}"
        path.mkdir(parents=True)
        (path / "trial_metadata.json").write_text(json.dumps({
            "trial_id": f"trial-{index}", "condition": "C1", "seed": index,
        }))
        (path / "episode.json").write_text(json.dumps({
            "status": "submitted_integrated_report",
            "evaluation": {"valid_submission": True, "e5_execution": {"required": False}},
            "b2_r1_readonly_evaluation": {
                "read_only_core_success": index == 0,
                "e1_discovery": {"passed": True},
                "e2_identification": {"passed": True},
                "e3_understanding": {"passed": index == 0},
                "read_only_planning": {"present": True, "citations_valid": True},
            },
        }))
        (path / "operator_post_verify.json").write_text('{"TESTBED_READY":true}')
    report = generate_reports(tmp_path, tmp_path / "comparison")
    assert report["aggregates"]["C1"]["episodes"] == 2
    assert report["aggregates"]["C1"]["read_only_core_successes"] == 1
    assert "does not evaluate E5" in (tmp_path / "comparison/C0_C4_comparison.md").read_text()


def test_artifact_checksums_exclude_their_own_file(tmp_path):
    (tmp_path / "episode.json").write_text("{}")
    checksums = _write_artifact_checksums(tmp_path)
    assert set(checksums) == {"episode.json"}
    recorded = json.loads((tmp_path / "artifact_checksums.json").read_text())
    assert recorded["files"] == checksums


def test_access_vm_context_distinguishes_wrong_host(monkeypatch):
    def fake_run(command, **kwargs):
        if command[:3] == ["ip", "-json", "address"]:
            return SimpleNamespace(stdout='[{"addr_info":[{"local":"192.0.2.10"}]}]')
        return SimpleNamespace(stdout="")
    monkeypatch.setattr("experiments.scadaver_sphere.b2_r1_campaign.subprocess.run", fake_run)
    result = access_vm_context()
    assert result["passed"] is False
    assert result["failure_layer"] == "wrong_execution_host_or_industrial_route"


def test_prepare_environment_recovers_only_when_status_is_not_ready(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "experiments.scadaver_sphere.b2_r1_campaign.access_vm_context",
        lambda: {"passed": True},
    )
    def fake_operator(operation, **kwargs):
        calls.append(operation)
        return {"TESTBED_READY": operation == "recover"}
    monkeypatch.setattr("experiments.scadaver_sphere.b2_r1_campaign._operator", fake_operator)
    monkeypatch.setattr(
        "experiments.scadaver_sphere.b2_r1_campaign.validate",
        lambda *args, **kwargs: {"CAMPAIGN_RUNNER_READY": True},
    )
    result = prepare_environment("http://worker", tmp_path)
    assert calls == ["status", "recover"]
    assert result["CAMPAIGN_RUNNER_READY"] is True


def test_campaign_and_global_locks_reject_duplicates(tmp_path):
    with campaign_lock(tmp_path):
        with pytest.raises(CampaignError, match="holds the lock"):
            with campaign_lock(tmp_path):
                pass
    with global_run_lock():
        with pytest.raises(CampaignError, match="another live"):
            with global_run_lock():
                pass


def test_cli_run_requires_explicit_live_acknowledgement(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", [
        "b2_r1_campaign", "run", "--results-root", str(tmp_path),
        "--condition", "C0", "--seed", "101",
    ])
    assert campaign_main() == 2
    assert "explicit --execute-live" in capsys.readouterr().out


def test_cli_full_plan_requires_matched_band(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", [
        "b2_r1_campaign", "run", "--results-root", str(tmp_path),
        "--all", "--execute-live",
    ])
    assert campaign_main() == 2
    assert "requires a matched initial level band" in capsys.readouterr().out


def test_cli_single_condition_requires_seed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", [
        "b2_r1_campaign", "run", "--results-root", str(tmp_path),
        "--condition", "C3", "--execute-live",
    ])
    assert campaign_main() == 2
    assert "requires --seed" in capsys.readouterr().out


def test_run_pilot_uses_frozen_balanced_schedule_without_retry(monkeypatch, tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text("{}")
    monkeypatch.setenv("ICS_AGENT_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(
        "experiments.scadaver_sphere.b2_r1_campaign.prepare_environment",
        lambda *args: {"validation": {"acceptance_record": str(acceptance)}},
    )
    calls = []
    def fake_run(condition, results_root, worker_url, seed, *args, **kwargs):
        calls.append((condition, seed, kwargs["trial_id"]))
        return {"status": "COMPLETED", "condition": condition,
                "trial_id": kwargs["trial_id"]}
    monkeypatch.setattr(
        "experiments.scadaver_sphere.b2_r1_campaign.run_condition", fake_run,
    )
    result = run_pilot(tmp_path, "http://worker", 400, 500, max_episodes=5)
    assert [item[:2] for item in calls] == [
        ("C0", 101), ("C1", 101), ("C2", 101), ("C3", 101), ("C4", 101),
    ]
    assert len({item[2] for item in calls}) == 5
    assert result["status"] == "COMPLETED"


def test_matched_state_band_is_read_only_and_fail_closed():
    record = {"process_sample": {"simulator_level": 450}}
    accepted = _matched_initial_state(record, 440, 460)
    assert accepted["matched"] is True
    assert accepted["record"] == record
    with pytest.raises(CampaignError, match="outside matched band"):
        _matched_initial_state(record, 300, 400)
    with pytest.raises(CampaignError, match="requires both"):
        _matched_initial_state(record, 400, None)


def test_composite_executor_registers_current_episode_observation():
    class Transport:
        def execute(self, name, arguments):
            return {"tool": name, "status": "ok", "observation_id": "obs-1",
                    "time_utc": "2026-11-16T00:00:00Z"}

    profile = B2R1Profile.from_json(ROOT / "config/b2_r1_v1/C1.json")
    manager = B2R1StateManager(profile)
    executor = CompositeExecutor(Transport(), None, None, manager)
    result = executor.execute("discover_hosts", {"cidr": "192.0.2.0/24"})
    assert result["observation_id"] == "obs-1"
    assert manager.snapshot()["observation_registry"]["obs-1"]["tool"] == "discover_hosts"


def test_b2_r1_adds_no_action_authority_or_action_state():
    class Transport:
        def execute(self, name, arguments):
            raise AssertionError("transport should not receive action")

    profile = B2R1Profile.from_json(ROOT / "config/b2_r1_v1/C3.json")
    manager = B2R1StateManager(profile)
    executor = CompositeExecutor(Transport(), None, None, manager)
    selected = {"ip": "192.0.2.9", "port": 1500, "address": 17,
                "active": True, "duration_seconds": 1.0}
    result = executor.execute("bounded_binary_output", selected)
    assert result["error"] == "actions_not_authorized"
    snapshot = manager.snapshot()
    assert "action_requests" not in snapshot
    assert "decision_readiness" not in snapshot


def test_read_only_campaign_never_reports_e5_or_full_system_success():
    episode = {
        "status": "EVALUATION_COMPLETE",
        "evaluation": {
            "overall_success": False,
            "read_only_workflow_success": True,
            "e1_discovery": {"passed": True},
            "e2_service_protocol_identification": {"passed": True},
            "e3_understanding": {"passed": True},
            "e4_planning": {"passed": True},
            "e5_execution": {"required": False, "passed": True},
        },
        "b2_r1_readonly_evaluation": {
            "read_only_core_success": True,
            "e1_discovery": {"passed": True},
            "e2_identification": {"passed": True},
            "e3_understanding": {"passed": True},
            "read_only_planning": {"present": True, "citations_valid": True},
        },
        "statistics": {"usage": {}},
        "reasoning_diagnostics": {},
    }
    row = _flatten("C4", episode)
    assert row["read_only_core_success"] is True
    assert row["full_system_overall_success"] is None
    assert row["e5_status"] == "NOT_APPLICABLE_READ_ONLY"
    assert row["e5_pass"] is None


def test_episode_reproducibility_record_freezes_prompts_tools_and_seed_state():
    config_path = ROOT / "config/integrated_e1_e5_first_qwen.json"
    profile_path = ROOT / "config/b2_r1_v1/C2.json"
    args = SimpleNamespace(
        out_dir=Path("/tmp/C2-fixture"), seed=202, objective="fixture objective",
        config=config_path, reasoning_config=profile_path,
    )
    record = _reproducibility_record(
        args, ScenarioConfig.from_json(config_path), [{"function": {"name": "fixture"}}],
        "system", "task", {"manifest_sha256": "a" * 64},
        [{"provider": "fixture", "model": "qwen/qwen3.5-9b",
          "seed_acknowledgement": "not_reported"}],
    )
    assert record["episode_id"] == "C2-fixture"
    assert record["seed_requested"] == 202
    assert record["provider_seed_acknowledgement"] == ["not_reported"]
    assert record["providers_reported"] == ["fixture"]
    assert record["reasoning_profile_sha256"]
    assert record["effective_tool_schema_sha256"]
    assert record["source"]["commit_source"] in {"git", "environment", "unavailable"}


def test_live_campaign_requires_an_exact_source_commit(monkeypatch):
    monkeypatch.delenv("ICS_AGENT_SOURCE_COMMIT", raising=False)
    with pytest.raises(CampaignError, match="ICS_AGENT_SOURCE_COMMIT"):
        _validated_source_commit()
    monkeypatch.setenv("ICS_AGENT_SOURCE_COMMIT", "a" * 40)
    assert _validated_source_commit() == "a" * 40
