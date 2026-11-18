"""N3-isolated continuous tool-calling loop with incremental event logging."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .agent import AgentRun, ToolExecutor, _target_hosts, compact_model_context
from .budgets import EpisodeBudget, ModelBudgetAdmissionError
from .n3_inference import N3OpenRouterClient


def run_n3_agent(client: N3OpenRouterClient, worker: ToolExecutor, *, max_steps: int,
                 system_prompt: str, mission_prompt: str,
                 initial_context: str, tool_schemas: list[dict[str, Any]],
                 report_validator: Callable[[Any], dict[str, Any]],
                 final_submission_max_tokens: int, reserved_completion_steps: int,
                 episode_deadline_seconds: float, budget: EpisodeBudget,
                 max_context_chars: int, context_tail_groups: int,
                 decision_recovery_max_output_tokens: int,
                 enable_investigation_truncation_recovery: bool,
                 force_final_submission_tool_choice: bool,
                 event_sink: Callable[[str, dict[str, Any]], None]) -> AgentRun:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": mission_prompt},
        {"role": "user", "content": initial_context},
    ]
    diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    regular_steps = max(0, max_steps - reserved_completion_steps)
    submission_name = "submit_integrated_report"
    submission_tools = [item for item in tool_schemas
                        if item.get("function", {}).get("name") == submission_name]
    investigation_tools = [item for item in tool_schemas
                           if item.get("function", {}).get("name") != submission_name]
    seen: set[tuple[str, str]] = set()
    last_probe: dict[str, float] = {}
    recovery_used = False
    investigation_recovery_used = False
    investigation_recovery_pending = False
    submission_schema_recovery_used = False

    def finish(report: dict[str, Any] | None, reason: str) -> AgentRun:
        client.mark_termination(reason)
        event_sink("termination", {"reason": reason, "statistics": budget.snapshot()})
        return AgentRun(messages, report, reason, diagnostics, budget.snapshot())

    for step in range(max_steps):
        if time.monotonic() - started >= episode_deadline_seconds:
            return finish(None, "episode_deadline_exceeded")
        finalization = step >= regular_steps or budget.should_finalize()
        remaining_before = budget.remaining()
        # Preserve capacity for the initial structured submission plus its one
        # permitted correction.  Transition to finalization before an
        # investigation request could consume that reserve.
        if (not finalization
                and int(remaining_before["output_tokens"])
                < client.max_tokens + (2 * final_submission_max_tokens)):
            finalization = True
        selected_tools = (
            submission_tools if finalization
            else investigation_tools if investigation_recovery_pending
            else tool_schemas
        )
        if finalization and (not messages or "reserved for final submission" not in str(messages[-1].get("content"))):
            messages.append({"role": "user", "content":
                "The remaining decision budget is reserved for final submission. Do not perform more observations. Call submit_integrated_report now with the best evidence-linked report. Unknown or insufficient evidence is valid."})
        model_messages, compacted = compact_model_context(
            messages, max_chars=max_context_chars, tail_groups=context_tail_groups)
        if compacted:
            diagnostics.append({"kind": "context_compaction", "step": step})
        estimated = budget.conservative_token_estimate({"messages": model_messages,
                                                         "tools": selected_tools})
        requested = min(
            final_submission_max_tokens if finalization else (
                decision_recovery_max_output_tokens if investigation_recovery_pending
                else client.max_tokens
            ),
            int(budget.remaining()["output_tokens"]),
        )
        try:
            correction_pending = recovery_used or submission_schema_recovery_used
            budget.admit_model_request(
                estimated_input_tokens=estimated, requested_output_tokens=requested,
                reserve_input_tokens=(
                    0 if finalization and correction_pending
                    else estimated if finalization else 2 * estimated
                ),
                reserve_output_tokens=(
                    0 if finalization and correction_pending
                    else final_submission_max_tokens if finalization
                    else 2 * final_submission_max_tokens
                ),
            )
            budget.record_model_call()
            assistant = client.chat(
                model_messages, selected_tools, max_tokens=requested,
                call_index=step + 1, mode="finalization" if finalization else "investigation",
                remaining_budget=budget.remaining(),
                tool_choice=(
                    {"type": "function", "function": {"name": submission_name}}
                    if finalization and force_final_submission_tool_choice
                    else "required" if investigation_recovery_pending else "auto"
                ),
            )
            investigation_recovery_pending = False
            budget.ingest_request_records(client.request_records)
        except ModelBudgetAdmissionError as exc:
            diagnostics.append({"kind": "model_request_not_admitted", "resource": exc.resource})
            return finish(None, "model_request_budget_not_admitted")
        except Exception as exc:
            diagnostics.append({"kind": "inference_failure", "error_type": type(exc).__name__,
                                "detail": str(exc)})
            return finish(None, "inference_error")
        metadata = assistant.pop("_scadaver_response_metadata", {})
        event_sink("model_response", {
            "step": step, "metadata": metadata,
            "request_record": client.request_records[-1] if client.request_records else {},
            "response": assistant,
        })
        messages.append({"role": "assistant", **assistant})
        calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
        if not calls:
            content = assistant.get("content")
            truncated = metadata.get("finish_reason") == "length"
            if (truncated and not finalization
                    and enable_investigation_truncation_recovery
                    and not investigation_recovery_used):
                investigation_recovery_used = True
                investigation_recovery_pending = True
                messages.append({"role": "user", "content":
                    "Your prior decision was truncated before it selected a tool. Continue the investigation. "
                    "Return exactly one available tool call now; do not summarize or finalize."})
                event_sink("investigation_recovery", {"step": step,
                    "reason": "length_truncated_without_tool_call",
                    "requested_max_tokens": decision_recovery_max_output_tokens})
                continue
            if (truncated or not isinstance(content, str) or not content.strip()) and not recovery_used:
                recovery_used = True
                messages.append({"role": "user", "content":
                    "Your response was empty or length-truncated. Do not make more observations or request an action. Call submit_integrated_report once with your best evidence-linked report."})
                regular_steps = step + 1
                continue
            if not isinstance(content, str):
                return finish(None, "invalid_final_report")
            try:
                return finish(report_validator(json.loads(content)), "final_report")
            except (ValueError, TypeError, json.JSONDecodeError):
                return finish(None, "invalid_final_report")
        for call in calls:
            function = call.get("function", {})
            name = str(function.get("name", ""))
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            event_sink("tool_request", {"step": step, "tool_call_id": call.get("id"),
                                         "name": name, "arguments": arguments})
            key = (name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
            if key in seen:
                diagnostics.append({"kind": "redundant_probe_request", "tool": name,
                                    "arguments": arguments})
            seen.add(key)
            if finalization and name != submission_name:
                result = {"tool": name, "status": "rejected",
                          "error": "completion_budget_reserved"}
            else:
                exhausted = budget.check_tool(name, arguments) if name != submission_name else None
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
                    if name != submission_name:
                        budget.record_tool(name, arguments)
                    for host in hosts:
                        last_probe[host] = time.monotonic()
            messages.append({"role": "tool", "tool_call_id": call.get("id", "missing"),
                             "content": json.dumps(result, sort_keys=True)})
            event_sink("tool_result", {"step": step, "tool_call_id": call.get("id"),
                                        "name": name, "arguments": arguments, "result": result})
            if name == submission_name and result.get("status") == "accepted":
                return finish(result["report"], "submitted_integrated_report")
            if (name == submission_name
                    and result.get("error") == "invalid_report_schema"):
                if submission_schema_recovery_used:
                    return finish(None, "invalid_final_report")
                submission_schema_recovery_used = True
                # Reserve the next call for schema-only correction.  The
                # correction receives the full finalization budget, never
                # the small investigation-recovery budget.
                regular_steps = step + 1
                messages.append({"role": "user", "content":
                    "The report was rejected for schema validation: "
                    f"{result.get('detail', 'invalid report')}. "
                    "Call submit_integrated_report once more with a JSON object that "
                    "conforms to the supplied schema. Do not invoke any other tool."})
                event_sink("submission_schema_recovery", {
                    "step": step,
                    "error": result.get("error"),
                    "requested_max_tokens": final_submission_max_tokens,
                })
                break
    return finish(None, "max_steps")
