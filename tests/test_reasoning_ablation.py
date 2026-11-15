import json
from pathlib import Path

import pytest

from experiments.scadaver_sphere.agent import run_agent
from experiments.scadaver_sphere.reasoning_ablation import (
    FLAGS,
    ReasoningProfile,
    ReasoningStateManager,
    load_prompt,
    reasoning_tool_schema,
)


ROOT = Path(__file__).parents[1] / "experiments/scadaver_sphere"
CONFIG = ROOT / "config/reasoning_ablation_v1"


@pytest.mark.parametrize("condition", ["B0", "B1", "B2", "B3", "B4"])
def test_profiles_are_cumulative_and_parse(condition):
    profile = ReasoningProfile.from_json(CONFIG / f"{condition}.json")
    expected = int(condition[1:])
    assert [getattr(profile, name) for name in FLAGS] == [index < expected for index in range(4)]
    assert (CONFIG / profile.base_config).resolve() == (ROOT / "config/integrated_e1_e5_first_qwen.json").resolve()


def test_b0_adds_no_prompt_or_tool():
    profile = ReasoningProfile.from_json(CONFIG / "B0.json")
    assert load_prompt(profile, ROOT) == ""
    assert reasoning_tool_schema(profile) is None


@pytest.mark.parametrize("condition", ["B1", "B2", "B3", "B4"])
def test_reasoning_prompts_have_no_deployment_answers(condition):
    profile = ReasoningProfile.from_json(CONFIG / f"{condition}.json")
    prompt = load_prompt(profile, ROOT).lower()
    for forbidden in ("10.100.", "openplc", "simulator", "1502", "1503", "register 0"):
        assert forbidden not in prompt


def _state(**extra):
    value = {
        "phase": "investigation",
        "evidence_ledger": [{"claim": "endpoint behavior is uncertain", "observation_ids": ["obs-1"], "status": "uncertain", "confidence": 0.4}],
        "unresolved_questions": ["which observation discriminates alternatives"],
        "next_observations": ["collect a bounded temporal sample"],
    }
    value.update(extra)
    return value


def test_manager_records_only_model_supplied_state():
    profile = ReasoningProfile.from_json(CONFIG / "B1.json")
    manager = ReasoningStateManager(profile)
    supplied = _state()
    result = manager.execute(supplied)
    assert result["status"] == "ok"
    assert manager.snapshot()["current"]["evidence_ledger"] == supplied["evidence_ledger"]
    assert manager.snapshot()["current"]["reasoning_state_id"] == "reasoning-0001"


def test_reasoning_state_is_isolated_per_episode():
    profile = ReasoningProfile.from_json(CONFIG / "B1.json")
    first = ReasoningStateManager(profile)
    second = ReasoningStateManager(profile)
    first.execute(_state())
    assert len(first.snapshot()["updates"]) == 1
    assert second.snapshot()["updates"] == []


def test_b4_schema_requires_all_cumulative_sections():
    profile = ReasoningProfile.from_json(CONFIG / "B4.json")
    required = reasoning_tool_schema(profile)["function"]["parameters"]["required"]
    assert {"evidence_ledger", "hypotheses", "pre_action_review", "information_gain_selection"} <= set(required)


def test_invalid_non_cumulative_profile_is_rejected(tmp_path):
    value = json.loads((CONFIG / "B2.json").read_text())
    value["evidence_ledger"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cumulative"):
        ReasoningProfile.from_json(path)


def test_authorization_templates_remain_pending_and_unique():
    values = [json.loads((CONFIG / "authorizations" / f"B{i}.template.json").read_text()) for i in range(5)]
    assert all(value["authorization_status"] == "PENDING" for value in values)
    assert len({value["authorization_id"] for value in values}) == 5
    assert all(value["maximum_action_count"] == 1 for value in values)


def test_integrated_submission_has_unambiguous_stop_reason():
    class Client:
        def chat(self, messages, tools):
            report = {
                "run_id": "fixture", "objective": "fixture", "systems": [],
                "relationships": [], "permission_assessment": [],
                "objective_plan": {"summary": "fixture", "steps": [],
                                   "supporting_observation_ids": [],
                                   "requires_action": False, "limitations": []},
                "action_attempted": False, "outcome": "fixture",
                "outcome_observation_ids": [], "overall_assessment": "fixture",
            }
            return {"tool_calls": [{"id": "final", "function": {
                "name": "submit_integrated_report",
                "arguments": json.dumps({"report": report}),
            }}]}

    class Executor:
        def execute(self, name, arguments):
            return {"tool": name, "status": "accepted", "report": arguments["report"]}

    outcome = run_agent(
        Client(), Executor(), allowed_cidr="192.0.2.0/24", max_steps=1,
        tool_schemas=[{"type": "function", "function": {
            "name": "submit_integrated_report", "parameters": {"type": "object"}}}],
        submission_tool_name="submit_integrated_report",
        report_validator=lambda value: value,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
