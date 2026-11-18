from __future__ import annotations

import json
from pathlib import Path

from experiments.scadaver_sphere.budgets import EpisodeBudget, EpisodeLimits
from experiments.scadaver_sphere.integrated_runner import validate_report
from experiments.scadaver_sphere.n3 import EventJournal, N3Executor, tool_schemas
from experiments.scadaver_sphere.n3_scaffolded import CONFIG, ScaffoldOptions
from experiments.scadaver_sphere.n3_scaffolded_agent import (
    PROGRESS_SCHEMA,
    PROGRESS_TOOL_NAME,
    run_scaffolded_agent,
)


def _progress(decision: str) -> dict:
    return {
        "current_phase": "E1" if decision == "observe" else "E4",
        "established_claims": [],
        "unresolved_questions": ["What evidence is still needed?"] if decision == "observe" else [],
        "next_information_need": "bounded discovery" if decision == "observe" else "none",
        "stage_assessment": {"E1": "incomplete" if decision == "observe" else "supported",
                             "E2": "uncertain", "E3": "uncertain", "E4": "uncertain"},
        "action_checkpoint": {
            "target_supported": False, "expected_effect_supported": False,
            "permission_supported": False, "verification_available": False,
            "blocking_uncertainties": ["fixture"],
        },
        "next_decision": decision,
        "decision_rationale": "fixture",
    }


def _report() -> dict:
    return {
        "run_id": "n3-s1-fixture", "objective": "fixture", "systems": [],
        "relationships": [], "permission_assessment": [],
        "objective_plan": {"summary": "insufficient evidence", "steps": [],
                           "supporting_observation_ids": [], "requires_action": False,
                           "limitations": ["fixture"]},
        "action_attempted": False, "outcome": "not attempted",
        "outcome_observation_ids": [], "overall_assessment": "fixture complete",
    }


def test_scaffold_profile_is_separate_and_bounded() -> None:
    options = ScaffoldOptions.from_json(CONFIG)
    assert options.planner_max_output_tokens == 4096
    assert options.selector_max_output_tokens == 2048
    text = CONFIG.read_text(encoding="utf-8")
    assert "scaffolded-v1" in text
    assert "198.18.4.20" not in text


def test_progress_schema_contains_no_target_answer() -> None:
    rendered = json.dumps(PROGRESS_SCHEMA, sort_keys=True)
    assert PROGRESS_TOOL_NAME in rendered
    assert "10.100.0." not in rendered
    assert "coil" not in rendered.lower()


def test_scaffolded_loop_plans_selects_one_tool_then_finalizes(tmp_path: Path) -> None:
    report = _report()

    class Client:
        max_tokens = 8192
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, _messages, tools, **kwargs):
            self.calls.append({"tools": tools, **kwargs})
            index = len(self.calls)
            if index == 1:
                name, arguments = PROGRESS_TOOL_NAME, _progress("observe")
            elif index == 2:
                name, arguments = "get_skill", {"skill_name": "ot-fundamentals"}
            elif index == 3:
                name, arguments = PROGRESS_TOOL_NAME, _progress("finalize")
            else:
                name, arguments = "submit_integrated_report", {"report": report}
            return {"tool_calls": [{"id": f"call-{index}", "function": {
                "name": name, "arguments": json.dumps(arguments)}}],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        def execute(self, _name, _arguments):
            raise AssertionError("only the local skill and submission tools are expected")

    client = Client()
    outcome, states = run_scaffolded_agent(
        client, N3Executor(Transport(), EventJournal(tmp_path), False), max_cycles=4,
        system_prompt="system", mission_prompt="mission", scaffold_prompt="scaffold",
        initial_context="foothold", tool_schemas=tool_schemas(False, 1.25),
        report_validator=validate_report, planner_max_tokens=4096,
        selector_max_tokens=2048, final_submission_max_tokens=16384,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(
            max_model_calls=12, max_tool_calls=8, max_input_tokens=1_000_000,
            max_output_tokens=100_000, max_api_cost_usd=25)),
        max_context_chars=100_000, context_tail_groups=12,
        event_sink=EventJournal(tmp_path).record,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    assert outcome.report == report
    assert [state["next_decision"] for state in states] == ["observe", "finalize"]
    selector_names = {item["function"]["name"] for item in client.calls[1]["tools"]}
    assert "bounded_binary_output" not in selector_names
    assert "submit_integrated_report" not in selector_names
    final_names = {item["function"]["name"] for item in client.calls[-1]["tools"]}
    assert final_names == {"submit_integrated_report"}
    assert client.calls[0]["tool_choice"] == {
        "type": "function", "function": {"name": PROGRESS_TOOL_NAME},
    }
    assert client.calls[1]["tool_choice"] == "required"


def test_act_checkpoint_exposes_action_and_abstention_only(tmp_path: Path) -> None:
    report = _report()

    class Client:
        max_tokens = 8192
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, _messages, tools, **kwargs):
            self.calls.append({"tools": tools, **kwargs})
            index = len(self.calls)
            if index == 1:
                state = _progress("act")
                state["current_phase"] = "E5"
                state["action_checkpoint"] = {
                    "target_supported": True, "expected_effect_supported": True,
                    "permission_supported": True, "verification_available": True,
                    "blocking_uncertainties": [],
                }
                state["stage_assessment"] = {
                    "E1": "supported", "E2": "supported",
                    "E3": "supported", "E4": "supported",
                }
                name, arguments = PROGRESS_TOOL_NAME, state
            elif index == 2:
                name, arguments = "abstain_action", {
                    "reason": "fixture abstention", "blocking_uncertainties": ["fixture"]}
            else:
                name, arguments = "submit_integrated_report", {"report": report}
            return {"tool_calls": [{"id": f"call-{index}", "function": {
                "name": name, "arguments": json.dumps(arguments)}}],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        pass

    client = Client()
    outcome, _states = run_scaffolded_agent(
        client, N3Executor(Transport(), EventJournal(tmp_path), True), max_cycles=2,
        system_prompt="system", mission_prompt="mission", scaffold_prompt="scaffold",
        initial_context="foothold", tool_schemas=tool_schemas(True, 1.25),
        report_validator=validate_report, planner_max_tokens=4096,
        selector_max_tokens=2048, final_submission_max_tokens=16384,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(
            max_model_calls=8, max_tool_calls=4, max_input_tokens=1_000_000,
            max_output_tokens=100_000, max_api_cost_usd=25)),
        max_context_chars=100_000, context_tail_groups=12,
        event_sink=lambda *_args: None,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    names = {item["function"]["name"] for item in client.calls[1]["tools"]}
    assert names == {"bounded_binary_output", "abstain_action"}


def test_multiple_selector_calls_are_paired_with_rejections_before_retry(tmp_path: Path) -> None:
    report = _report()

    class Client:
        max_tokens = 8192
        request_records: list[dict] = []

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, _messages, _tools, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                calls = [(PROGRESS_TOOL_NAME, _progress("observe"))]
            elif self.calls == 2:
                calls = [("get_skill", {"skill_name": "ot-fundamentals"}),
                         ("get_skill", {"skill_name": "network-discovery"})]
            elif self.calls == 3:
                calls = [("get_skill", {"skill_name": "network-discovery"})]
            elif self.calls == 4:
                calls = [(PROGRESS_TOOL_NAME, _progress("finalize"))]
            else:
                calls = [("submit_integrated_report", {"report": report})]
            return {"tool_calls": [{"id": f"call-{self.calls}-{index}", "function": {
                "name": name, "arguments": json.dumps(arguments)}}
                for index, (name, arguments) in enumerate(calls)],
                "_scadaver_response_metadata": {"finish_reason": "tool_calls"}}

        def mark_termination(self, _reason):
            return None

    class Transport:
        pass

    outcome, _states = run_scaffolded_agent(
        Client(), N3Executor(Transport(), EventJournal(tmp_path), False), max_cycles=3,
        system_prompt="system", mission_prompt="mission", scaffold_prompt="scaffold",
        initial_context="foothold", tool_schemas=tool_schemas(False, 1.25),
        report_validator=validate_report, planner_max_tokens=4096,
        selector_max_tokens=2048, final_submission_max_tokens=16384,
        episode_deadline_seconds=30,
        budget=EpisodeBudget(EpisodeLimits(
            max_model_calls=12, max_tool_calls=8, max_input_tokens=1_000_000,
            max_output_tokens=100_000, max_api_cost_usd=25)),
        max_context_chars=100_000, context_tail_groups=12,
        event_sink=lambda *_args: None,
    )
    assert outcome.stop_reason == "submitted_integrated_report"
    rejected = [json.loads(message["content"]) for message in outcome.messages
                if message.get("role") == "tool"
                and "exactly_one_tool_call_required" in message.get("content", "")]
    assert len(rejected) == 2
