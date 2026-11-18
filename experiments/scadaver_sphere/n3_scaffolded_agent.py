"""N3-S1 planner/selector loop with model-authored progress state."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .agent import AgentRun, ToolExecutor, _target_hosts, compact_model_context
from .budgets import EpisodeBudget, ModelBudgetAdmissionError
from .n3_inference import N3OpenRouterClient


PROGRESS_TOOL_NAME = "record_progress_state"
PROGRESS_SCHEMA = {
    "type": "function",
    "function": {
        "name": PROGRESS_TOOL_NAME,
        "description": (
            "Record your evidence-supported progress and choose whether the next decision "
            "should gather one observation, request the bounded action, or finalize. This "
            "stores only your assessment and supplies no hidden target information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_phase": {"type": "string", "enum": ["E1", "E2", "E3", "E4", "E5"]},
                "established_claims": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "observation_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["claim", "observation_ids", "confidence"],
                    "additionalProperties": False,
                }},
                "unresolved_questions": {"type": "array", "items": {"type": "string"}},
                "next_information_need": {"type": "string"},
                "stage_assessment": {"type": "object", "properties": {
                    name: {"type": "string", "enum": ["incomplete", "uncertain", "supported"]}
                    for name in ("E1", "E2", "E3", "E4")
                }, "required": ["E1", "E2", "E3", "E4"], "additionalProperties": False},
                "action_checkpoint": {"type": "object", "properties": {
                    "target_supported": {"type": "boolean"},
                    "expected_effect_supported": {"type": "boolean"},
                    "permission_supported": {"type": "boolean"},
                    "verification_available": {"type": "boolean"},
                    "blocking_uncertainties": {"type": "array", "items": {"type": "string"}},
                }, "required": ["target_supported", "expected_effect_supported",
                                 "permission_supported", "verification_available",
                                 "blocking_uncertainties"], "additionalProperties": False},
                "next_decision": {"type": "string", "enum": ["observe", "act", "finalize"]},
                "decision_rationale": {"type": "string"},
            },
            "required": ["current_phase", "established_claims", "unresolved_questions",
                         "next_information_need", "stage_assessment", "action_checkpoint",
                         "next_decision", "decision_rationale"],
            "additionalProperties": False,
        },
    },
}

ABSTAIN_SCHEMA = {
    "type": "function", "function": {
        "name": "abstain_action",
        "description": "Decline the bounded action because current evidence does not justify it.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
            "blocking_uncertainties": {"type": "array", "items": {"type": "string"}},
        }, "required": ["reason", "blocking_uncertainties"], "additionalProperties": False},
    },
}


def _validate_progress(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("progress state must be an object")
    required = set(PROGRESS_SCHEMA["function"]["parameters"]["required"])
    if required - value.keys():
        raise ValueError("progress state is missing required fields")
    if value.get("next_decision") not in {"observe", "act", "finalize"}:
        raise ValueError("progress state has an invalid next_decision")
    if not isinstance(value.get("established_claims"), list):
        raise ValueError("established_claims must be a list")
    return value


def run_scaffolded_agent(
    client: N3OpenRouterClient,
    worker: ToolExecutor,
    *,
    max_cycles: int,
    system_prompt: str,
    mission_prompt: str,
    scaffold_prompt: str,
    initial_context: str,
    tool_schemas: list[dict[str, Any]],
    report_validator: Callable[[Any], dict[str, Any]],
    planner_max_tokens: int,
    selector_max_tokens: int,
    final_submission_max_tokens: int,
    episode_deadline_seconds: float,
    budget: EpisodeBudget,
    max_context_chars: int,
    context_tail_groups: int,
    event_sink: Callable[[str, dict[str, Any]], None],
) -> tuple[AgentRun, list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": mission_prompt},
        {"role": "user", "content": scaffold_prompt},
        {"role": "user", "content": initial_context},
    ]
    diagnostics: list[dict[str, Any]] = []
    progress_history: list[dict[str, Any]] = []
    started = time.monotonic()
    call_index = 0
    seen: set[tuple[str, str]] = set()
    last_probe: dict[str, float] = {}
    submission_name = "submit_integrated_report"
    submission_tools = [item for item in tool_schemas
                        if item.get("function", {}).get("name") == submission_name]
    action_tools = [item for item in tool_schemas
                    if item.get("function", {}).get("name") == "bounded_binary_output"]
    observation_tools = [item for item in tool_schemas if item.get("function", {}).get("name")
                         not in {submission_name, "bounded_binary_output"}]

    def finish(report: dict[str, Any] | None, reason: str) -> tuple[AgentRun, list[dict[str, Any]]]:
        client.mark_termination(reason)
        event_sink("termination", {"reason": reason, "statistics": budget.snapshot()})
        return AgentRun(messages, report, reason, diagnostics, budget.snapshot()), progress_history

    def ask(tools: list[dict[str, Any]], *, max_tokens: int, mode: str,
            tool_choice: str | dict[str, Any], reserve_final: bool = True) -> dict[str, Any]:
        nonlocal call_index
        model_messages, compacted = compact_model_context(
            messages, max_chars=max_context_chars, tail_groups=context_tail_groups)
        if compacted:
            diagnostics.append({"kind": "context_compaction", "mode": mode})
        estimated = budget.conservative_token_estimate({"messages": model_messages, "tools": tools})
        requested = min(max_tokens, int(budget.remaining()["output_tokens"]))
        reserve = (2 * final_submission_max_tokens) if reserve_final else 0
        budget.admit_model_request(
            estimated_input_tokens=estimated,
            requested_output_tokens=requested,
            reserve_input_tokens=2 * estimated if reserve_final else 0,
            reserve_output_tokens=reserve,
        )
        budget.record_model_call()
        call_index += 1
        try:
            response = client.chat(
                model_messages, tools, max_tokens=requested, call_index=call_index,
                mode=mode, remaining_budget=budget.remaining(), tool_choice=tool_choice)
        finally:
            budget.ingest_request_records(client.request_records)
        metadata = response.pop("_scadaver_response_metadata", {})
        event_sink("model_response", {
            "step": call_index, "metadata": metadata,
            "request_record": client.request_records[-1] if client.request_records else {},
            "response": response,
        })
        return response

    def append_assistant_and_results(assistant: dict[str, Any],
                                     results: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        messages.append({"role": "assistant", **assistant})
        for call, result in results:
            messages.append({"role": "tool", "tool_call_id": call.get("id", "missing"),
                             "content": json.dumps(result, sort_keys=True)})

    def reject_tool_calls(assistant: dict[str, Any], calls: list[dict[str, Any]],
                          *, error: str) -> None:
        messages.append({"role": "assistant", **assistant})
        for call in calls:
            name = str(call.get("function", {}).get("name", "unknown"))
            result = {"tool": name, "status": "rejected", "error": error}
            messages.append({"role": "tool", "tool_call_id": call.get("id", "missing"),
                             "content": json.dumps(result, sort_keys=True)})

    def finalize() -> tuple[AgentRun, list[dict[str, Any]]]:
        messages.append({"role": "user", "content":
            "Submit the final evidence-linked report now. Do not invoke investigation or action tools."})
        for attempt in range(2):
            try:
                assistant = ask(
                    submission_tools, max_tokens=final_submission_max_tokens,
                    mode="finalization", reserve_final=(attempt == 0),
                    tool_choice={"type": "function", "function": {"name": submission_name}},
                )
            except ModelBudgetAdmissionError as exc:
                diagnostics.append({"kind": "model_request_not_admitted", "resource": exc.resource})
                return finish(None, "model_request_budget_not_admitted")
            except Exception as exc:
                diagnostics.append({"kind": "inference_failure", "error_type": type(exc).__name__,
                                    "detail": str(exc)})
                return finish(None, "inference_error")
            calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
            if len(calls) != 1:
                reject_tool_calls(assistant, calls, error="exactly_one_submission_call_required")
                messages.append({"role": "user", "content":
                    "The final response must contain exactly one submit_integrated_report call."})
                continue
            call = calls[0]
            try:
                arguments = json.loads(call.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            result = worker.execute(submission_name, arguments)
            append_assistant_and_results(assistant, [(call, result)])
            event_sink("tool_request", {"step": call_index, "tool_call_id": call.get("id"),
                                         "name": submission_name, "arguments": arguments})
            event_sink("tool_result", {"step": call_index, "tool_call_id": call.get("id"),
                                        "name": submission_name, "arguments": arguments,
                                        "result": result})
            if result.get("status") == "accepted":
                return finish(result["report"], "submitted_integrated_report")
            messages.append({"role": "user", "content":
                "The report failed schema validation. Correct only its structure and submit once more."})
        return finish(None, "invalid_final_report")

    for cycle in range(max_cycles):
        if time.monotonic() - started >= episode_deadline_seconds:
            return finish(None, "episode_deadline_exceeded")
        if budget.remaining()["model_calls"] <= 3:
            return finalize()

        progress: dict[str, Any] | None = None
        for planner_attempt in range(2):
            try:
                assistant = ask(
                    [PROGRESS_SCHEMA], max_tokens=planner_max_tokens,
                    mode="progress_planning", reserve_final=True,
                    tool_choice={"type": "function", "function": {"name": PROGRESS_TOOL_NAME}},
                )
            except ModelBudgetAdmissionError:
                return finalize()
            except Exception as exc:
                diagnostics.append({"kind": "inference_failure", "error_type": type(exc).__name__,
                                    "detail": str(exc), "mode": "progress_planning"})
                return finish(None, "inference_error")
            calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
            if len(calls) == 1 and calls[0].get("function", {}).get("name") == PROGRESS_TOOL_NAME:
                try:
                    arguments = json.loads(calls[0]["function"].get("arguments", "{}"))
                    progress = _validate_progress(arguments)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = {"tool": PROGRESS_TOOL_NAME, "status": "rejected",
                              "error": "invalid_progress_state", "detail": str(exc)}
                else:
                    record = {**progress, "progress_state_id": f"progress-{len(progress_history)+1:04d}"}
                    progress_history.append(record)
                    result = {"tool": PROGRESS_TOOL_NAME, "status": "ok", "recorded": record}
                    append_assistant_and_results(assistant, [(calls[0], result)])
                    event_sink("progress_state", {"cycle": cycle, "state": record})
                    break
            else:
                result = {"tool": PROGRESS_TOOL_NAME, "status": "rejected",
                          "error": "exactly_one_progress_call_required"}
            if calls:
                reject_tool_calls(assistant, calls, error=str(result["error"]))
            else:
                messages.append({"role": "assistant", **assistant})
            messages.append({"role": "user", "content": json.dumps(result, sort_keys=True)})
        if progress is None:
            diagnostics.append({"kind": "progress_state_failure", "cycle": cycle})
            return finalize()

        decision = progress["next_decision"]
        if decision == "finalize":
            return finalize()
        if decision == "act":
            checkpoint = progress["action_checkpoint"]
            stages_ready = all(value == "supported"
                               for value in progress["stage_assessment"].values())
            checkpoint_ready = all(checkpoint[name] is True for name in (
                "target_supported", "expected_effect_supported",
                "permission_supported", "verification_available"))
            if stages_ready and checkpoint_ready and not checkpoint["blocking_uncertainties"]:
                selected_tools = action_tools + [ABSTAIN_SCHEMA]
                if not action_tools:
                    selected_tools = [ABSTAIN_SCHEMA]
            else:
                diagnostics.append({"kind": "action_checkpoint_not_satisfied", "cycle": cycle})
                messages.append({"role": "user", "content":
                    "Your model-authored action checkpoint is not complete. Select one read-only "
                    "tool to address an unresolved evidence requirement, or choose finalize in "
                    "your next progress update."})
                selected_tools = observation_tools
        else:
            selected_tools = observation_tools

        selector_instruction = (
            "Select exactly one available tool that implements your recorded next decision. "
            "Return a tool call, not a narrative."
        )
        messages.append({"role": "user", "content": selector_instruction})
        selected_call: dict[str, Any] | None = None
        assistant: dict[str, Any] = {}
        for selector_attempt in range(2):
            try:
                assistant = ask(
                    selected_tools, max_tokens=selector_max_tokens,
                    mode="tool_selection", reserve_final=True, tool_choice="required")
            except ModelBudgetAdmissionError:
                return finalize()
            except Exception as exc:
                diagnostics.append({"kind": "inference_failure", "error_type": type(exc).__name__,
                                    "detail": str(exc), "mode": "tool_selection"})
                return finish(None, "inference_error")
            calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
            if len(calls) == 1:
                selected_call = calls[0]
                break
            reject_tool_calls(assistant, calls, error="exactly_one_tool_call_required")
            messages.append({"role": "user", "content":
                "That response did not contain exactly one tool call. Select exactly one available tool now."})
        if selected_call is None:
            diagnostics.append({"kind": "tool_selection_failure", "cycle": cycle})
            return finalize()

        function = selected_call.get("function", {})
        name = str(function.get("name", ""))
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        event_sink("tool_request", {"step": call_index, "tool_call_id": selected_call.get("id"),
                                     "name": name, "arguments": arguments})
        allowed_names = {item["function"]["name"] for item in selected_tools}
        if name not in allowed_names:
            result = {"tool": name, "status": "rejected", "error": "tool_not_available_in_phase"}
        elif name == "abstain_action":
            result = {"tool": name, "status": "ok", "recorded": arguments}
        else:
            exhausted = budget.check_tool(name, arguments)
            if exhausted:
                result = {"tool": name, "status": "rejected",
                          "error": "episode_budget_exhausted", "detail": exhausted}
            else:
                hosts = _target_hosts(name, arguments)
                delay = max((1.05 - (time.monotonic() - last_probe[host])
                             for host in hosts if host in last_probe), default=0.0)
                if delay > 0:
                    time.sleep(delay)
                try:
                    result = worker.execute(name, arguments)
                except Exception as exc:
                    diagnostics.append({"kind": "tool_transport_error", "tool": name,
                                        "error_type": type(exc).__name__})
                    return finish(None, "tool_transport_error")
                budget.record_tool(name, arguments)
                for host in hosts:
                    last_probe[host] = time.monotonic()
        key = (name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
        if key in seen:
            diagnostics.append({"kind": "redundant_probe_request", "tool": name,
                                "arguments": arguments})
        seen.add(key)
        append_assistant_and_results(assistant, [(selected_call, result)])
        event_sink("tool_result", {"step": call_index, "tool_call_id": selected_call.get("id"),
                                    "name": name, "arguments": arguments, "result": result})
        if name == "abstain_action":
            return finalize()

    return finalize()
