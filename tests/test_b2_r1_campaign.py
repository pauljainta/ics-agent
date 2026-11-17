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
    _validated_source_commit,
    _matched_initial_state,
    inventory,
    prepare,
    run_condition,
    validate_profiles,
)
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.integrated_runner import (
    CompositeExecutor,
    _reproducibility_record,
)


ROOT = Path(__file__).parents[1] / "experiments/scadaver_sphere"


def _episode(root: Path, condition: str, *, complete=True):
    path = root / condition / f"{condition}-fixture"
    path.mkdir(parents=True)
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
    _episode(tmp_path, "C0")
    validation = {
        "deployment_ready": True, "testbed_ready": True,
        "initial_condition_record": {"process_sample": {"simulator_level": 450}},
    }
    state = prepare(tmp_path, tmp_path / "campaign.json", validation)
    dispositions = {item["condition"]: item["disposition"] for item in state["plan"]}
    assert dispositions["C0"] == "PRESERVE_COMPLETED"
    assert dispositions["C4"] == "ELIGIBLE"
    assert state["initial_condition_policy"]["reset_performed"] is False


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
