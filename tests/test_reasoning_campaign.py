import json
from pathlib import Path

import pytest

from experiments.scadaver_sphere.reasoning_campaign import (
    CampaignError,
    campaign_lock,
    generate_reports,
    inspect_episode,
    inventory,
    prepare,
    run_condition,
    validate_profiles,
)


def _episode(root: Path, condition: str, *, complete=True, recovered=True):
    path = root / condition / f"{condition}-fixture"
    path.mkdir(parents=True)
    if complete:
        (path / "episode.json").write_text(json.dumps({
            "status": "submitted_integrated_report", "report": {},
            "evaluation": {"valid_submission": True, "overall_success": False,
                           "e1_discovery": {"passed": True},
                           "e2_service_protocol_identification": {"passed": False},
                           "e3_understanding": {"passed": False},
                           "e4_planning": {"passed": True},
                           "e5_execution": {"passed": False, "causal_attribution": "OBJECTIVE_NOT_ACHIEVED"}},
            "statistics": {"usage": {"model_calls": 1, "tool_calls": 2}},
        }))
    if recovered:
        (path / "operator_post_verify.json").write_text('{"TESTBED_READY":true}')
    (path / "SHA256SUMS").write_text("fixture\n")
    return path


def test_frozen_profile_hashes_validate():
    assert all(value["passed"] for value in validate_profiles().values())


def test_inventory_preserves_completed_and_marks_missing(tmp_path):
    _episode(tmp_path, "B1")
    state = inventory(tmp_path)
    assert state["conditions"]["B1"]["latest"]["status"] == "EVALUATION_COMPLETE"
    assert state["conditions"]["B1"]["eligible_for_new_run"] is False
    assert state["conditions"]["B4"]["eligible_for_new_run"] is True


def test_interrupted_episode_is_not_eligible_for_automatic_replacement(tmp_path):
    path = _episode(tmp_path, "B3", complete=False)
    assert inspect_episode(path)["status"] == "INTERRUPTED"
    state = inventory(tmp_path)
    assert state["conditions"]["B3"]["eligible_for_new_run"] is False


def test_prepare_requires_operator_decision_for_interrupted_episode(tmp_path):
    _episode(tmp_path, "B1")
    _episode(tmp_path, "B3", complete=False)
    state = prepare(tmp_path, tmp_path / "campaign.json",
                    {"deployment_ready": True, "testbed_ready": True})
    dispositions = {item["condition"]: item["disposition"] for item in state["plan"]}
    assert dispositions == {"B1": "PRESERVE_COMPLETED", "B2": "ELIGIBLE",
                            "B3": "OPERATOR_DECISION_REQUIRED", "B4": "ELIGIBLE"}


def test_campaign_lock_rejects_second_process_handle(tmp_path):
    with campaign_lock(tmp_path):
        with pytest.raises(CampaignError, match="holds the lock"):
            with campaign_lock(tmp_path):
                pass


def test_report_generation_keeps_missing_values_unknown(tmp_path):
    _episode(tmp_path, "B1")
    result = generate_reports(tmp_path, tmp_path / "reports")
    b4 = next(item for item in result["rows"] if item["condition"] == "B4")
    assert b4["status"] == "NOT_AVAILABLE"
    assert (tmp_path / "reports/B0_B4_comparison.csv").exists()


def test_report_generation_uses_separate_corrected_b0_evaluation(tmp_path):
    _episode(tmp_path, "B0")
    corrected = tmp_path / "comparisons/B0_corrected_causal_evaluation.json"
    corrected.parent.mkdir()
    corrected.write_text(json.dumps({
        "corrected_e5": {"passed": False, "causal_attribution": "OBJECTIVE_NOT_ACHIEVED"},
        "corrected_evaluator_version": "integrated-e1-e5-causal-v3",
        "corrected_overall_success": False,
    }))
    result = generate_reports(tmp_path, tmp_path / "comparisons")
    b0 = next(item for item in result["rows"] if item["condition"] == "B0")
    assert b0["e5_pass"] is False
    assert b0["causal_attribution"] == "OBJECTIVE_NOT_ACHIEVED"


def test_run_refuses_to_replace_existing_episode(tmp_path):
    _episode(tmp_path, "B4")
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"authorization_status": "AUTHORIZED", "condition": "B4"}))
    with pytest.raises(CampaignError, match="already has an episode"):
        run_condition("B4", tmp_path, "http://127.0.0.1:1", authorization)


def test_run_does_not_require_manually_created_authorization_for_completed_condition(tmp_path):
    _episode(tmp_path, "B4")
    with pytest.raises(CampaignError, match="already has an episode"):
        run_condition("B4", tmp_path, "http://127.0.0.1:1")


def test_run_refuses_wrong_condition_authorization(tmp_path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"authorization_status": "AUTHORIZED", "condition": "B3"}))
    with pytest.raises(CampaignError, match="not active"):
        run_condition("B4", tmp_path, "http://127.0.0.1:1", authorization)
