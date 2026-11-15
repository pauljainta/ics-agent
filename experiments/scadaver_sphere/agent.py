"""Bounded client-side OpenRouter tool-call loop for the SCADAver experiment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .openrouter import OpenRouterClient
from .budgets import EpisodeBudget
from .schemas import SYSTEM_PROMPT, TOOL_SCHEMAS, validate_inventory


class ToolExecutor(Protocol):
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AgentRun:
    messages: list[dict[str, Any]]
    report: dict[str, Any] | None
    stop_reason: str
    diagnostics: list[dict[str, Any]]
    statistics: dict[str, Any] | None = None


def run_agent(client: OpenRouterClient, worker: ToolExecutor, *, allowed_cidr: str, max_steps: int,
              system_prompt: str = SYSTEM_PROMPT, task_prompt: str | None = None,
              episode_deadline_seconds: float | None = None,
              permitted_ports: tuple[int, ...] | None = None,
              tool_schemas: list[dict[str, Any]] | None = None,
              report_validator: Callable[[Any], dict[str, Any]] = validate_inventory,
              submission_tool_name: str = "submit_inventory",
              final_submission_max_tokens: int = 2048,
              reserved_completion_steps: int = 2,
              same_host_probe_interval_seconds: float = 1.05,
              budget: EpisodeBudget | None = None,
              max_context_chars: int | None = None,
              context_tail_groups: int = 24) -> AgentRun:
    scope = f"\nAuthorized lab CIDR: {allowed_cidr}."
    if permitted_ports:
        scope += "\nAuthorized read-only TCP ports: " + ", ".join(str(port) for port in permitted_ports) + "."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (task_prompt or "Build an evidence-linked inventory.") + scope},
    ]
    started = time.monotonic()
    diagnostics: list[dict[str, Any]] = []
    seen_requests: set[tuple[str, str]] = set()
    last_probe_at: dict[str, float] = {}
    finalization_notice_sent = False
    recovery_attempted = False
    submission_recovery_active = False
    regular_steps = max(0, max_steps - max(0, reserved_completion_steps))
    model_call_index = 0
    submission_tools = [
        item for item in (tool_schemas or TOOL_SCHEMAS)
        if item.get("function", {}).get("name") == submission_tool_name
    ]

    def finish(report: dict[str, Any] | None, reason: str) -> AgentRun:
        if isinstance(client, OpenRouterClient):
            client.mark_termination(reason)
        return AgentRun(messages, report, reason, diagnostics, budget.snapshot() if budget else None)

    def remaining_budget_snapshot() -> dict[str, int | float] | None:
        if not budget:
            return None
        remaining = budget.remaining()
        remaining["steps"] = max(0, max_steps - step - 1)
        if episode_deadline_seconds is not None:
            remaining["episode_seconds"] = max(
                0.0, episode_deadline_seconds - (time.monotonic() - started)
            )
        return remaining

    def ask(*, finalization: bool = False,
            recovery: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal model_call_index
        if budget and not budget.can_model_call():
            raise RuntimeError("model_call_budget_exhausted")
        if budget:
            budget.record_model_call()
        model_call_index += 1
        model_messages, compacted = compact_model_context(
            messages, max_chars=max_context_chars, tail_groups=context_tail_groups
        )
        if compacted:
            diagnostics.append({
                "kind": "context_compaction",
                "raw_message_count": len(messages),
                "model_message_count": len(model_messages),
                "raw_chars": sum(len(json.dumps(item, sort_keys=True)) for item in messages),
                "model_chars": sum(len(json.dumps(item, sort_keys=True)) for item in model_messages),
            })
        if isinstance(client, OpenRouterClient):
            requested_max = final_submission_max_tokens if finalization else None
            if finalization and budget:
                requested_max = min(final_submission_max_tokens,
                                    int(budget.remaining()["output_tokens"]))
            response = client.chat(
                model_messages,
                submission_tools if finalization else (tool_schemas or TOOL_SCHEMAS),
                max_tokens=requested_max,
                call_index=model_call_index,
                mode="submission_recovery" if recovery else (
                    "finalization" if finalization else "investigation"
                ),
                remaining_budget=remaining_budget_snapshot(),
            )
            if budget:
                budget.ingest_request_records(client.request_records)
                client.update_last_request(remaining_budget_after=remaining_budget_snapshot())
        else:
            response = client.chat(
                model_messages,
                submission_tools if finalization else (tool_schemas or TOOL_SCHEMAS),
            )
        metadata = response.pop("_scadaver_response_metadata", {})
        return response, metadata if isinstance(metadata, dict) else {}

    def recovery_budget_blocker() -> str | None:
        if not budget:
            return None
        if not budget.can_model_call():
            return "model_calls"
        for name in ("input_tokens", "output_tokens", "api_cost_usd"):
            if name in budget.exhausted or budget.remaining()[name] <= 0:
                return name
        return None

    def add_finalization_notice(*, retry: bool = False) -> None:
        nonlocal finalization_notice_sent
        if retry:
            text = (
                "Your prior response was empty or truncated before it supplied a final report. "
                f"Do not perform more network probes. Call {submission_tool_name} now with your "
                "best evidence-linked report. Unknown and insufficient-evidence classifications are valid."
            )
        else:
            text = (
                "The reconnaissance decision budget is now reserved for final submission. Do not "
                "perform further network probes. Call "
                f"{submission_tool_name} now with your evidence-linked report. Unknown and "
                "insufficient-evidence classifications are valid."
            )
        messages.append({"role": "user", "content": text})
        finalization_notice_sent = True

    for step in range(max_steps):
        if episode_deadline_seconds is not None and time.monotonic() - started >= episode_deadline_seconds:
            return finish(None, "episode_deadline_exceeded")
        finalization = step >= regular_steps or submission_recovery_active or bool(budget and budget.should_finalize())
        recovery_turn = bool(recovery_attempted and submission_recovery_active)
        if finalization and not finalization_notice_sent:
            add_finalization_notice()
        try:
            assistant, response_metadata = ask(finalization=finalization, recovery=recovery_turn)
        except Exception as exc:
            diagnostics.append({"kind": "inference_failure", "error_type": type(exc).__name__, "detail": str(exc)})
            return finish(None, "inference_error")
        messages.append({"role": "assistant", **assistant})
        calls = assistant.get("tool_calls", [])
        if not calls:
            content = assistant.get("content")
            empty = not isinstance(content, str) or not content.strip()
            length_truncated = response_metadata.get("finish_reason") == "length"
            if recovery_turn:
                return finish(None, "submission_recovery_truncated" if length_truncated
                              else "submission_recovery_invalid")
            # Preserve the complete failed response and offer one model-authored,
            # submission-only turn. Never synthesize or repair substantive content.
            if ((empty or length_truncated) and not recovery_attempted and step + 1 < max_steps):
                blocker = recovery_budget_blocker()
                if blocker:
                    diagnostics.append({"kind": "submission_recovery_blocked", "budget": blocker})
                    return finish(None, "submission_recovery_budget_exhausted")
                recovery_attempted = True
                submission_recovery_active = True
                add_finalization_notice(retry=True)
                continue
            try:
                if not isinstance(content, str):
                    return finish(None, "invalid_final_report")
                report = report_validator(json.loads(content))
            except (TypeError, ValueError, json.JSONDecodeError):
                return finish(None, "invalid_final_report")
            return finish(report, "final_report")
        # Serial execution preserves every model-selected call and its result.
        for call in calls:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            name = str(function.get("name", ""))
            request_key = (name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
            if request_key in seen_requests:
                diagnostics.append({"kind": "redundant_probe_request", "tool": name, "arguments": arguments})
            seen_requests.add(request_key)
            if finalization and name != submission_tool_name:
                result = {
                    "tool": name,
                    "status": "rejected",
                    "error": "completion_budget_reserved",
                    "detail": f"No further network operations are allowed during finalization; call {submission_tool_name} with the report.",
                }
            else:
                exhausted = budget.check_tool(name, arguments) if budget and name != submission_tool_name else None
                if exhausted:
                    result = {
                        "tool": name,
                        "status": "rejected",
                        "error": "episode_budget_exhausted",
                        "detail": f"The global {exhausted} budget is exhausted; submit the final report.",
                    }
                    if exhausted not in budget.exhausted:
                        budget.exhausted.append(exhausted)
                    messages.append({"role": "tool", "tool_call_id": call.get("id", "missing"),
                                     "content": json.dumps(result, sort_keys=True)})
                    diagnostics.append({"kind": "episode_budget_exhausted", "budget": exhausted, "tool": name})
                    submission_recovery_active = True
                    continue
                hosts = _target_hosts(name, arguments)
                wait_seconds = max((same_host_probe_interval_seconds - (time.monotonic() - last_probe_at[host])
                                    for host in hosts if host in last_probe_at), default=0.0)
                if wait_seconds > 0:
                    diagnostics.append({"kind": "same_host_pacing", "tool": name, "hosts": sorted(hosts), "wait_seconds": round(wait_seconds, 3)})
                    time.sleep(wait_seconds)
                try:
                    result = worker.execute(name, arguments)
                    if budget and name != submission_tool_name:
                        budget.record_tool(name, arguments)
                except Exception as exc:
                    diagnostics.append({"kind": "tool_transport_error", "tool": name, "error_type": type(exc).__name__})
                    return finish(None, "tool_transport_error")
                for host in hosts:
                    last_probe_at[host] = time.monotonic()
            if result.get("status") in {"error", "rejected", "unsupported", "disabled"}:
                diagnostics.append({"kind": "unsuccessful_tool_call", "tool": name, "status": result.get("status"), "error": result.get("error"), "detail": result.get("detail")})
            messages.append({"role": "tool", "tool_call_id": call.get("id", "missing"),
                             "content": json.dumps(result, sort_keys=True)})
            if name == submission_tool_name and result.get("status") == "accepted":
                reason = ("submitted_integrated_report"
                          if submission_tool_name == "submit_integrated_report"
                          else "submitted_inventory")
                return finish(result["report"], reason)
        if recovery_turn:
            return finish(None, "submission_recovery_invalid")
    return finish(None, "max_steps")


def compact_model_context(messages: list[dict[str, Any]], *, max_chars: int | None,
                          tail_groups: int = 24) -> tuple[list[dict[str, Any]], bool]:
    """Build a bounded model view while preserving the complete raw trace.

    Only model-visible messages are summarized. The initial system/objective
    messages remain verbatim, observation IDs are retained, and no evaluator
    or deployment-private data is introduced.
    """
    size = sum(len(json.dumps(item, sort_keys=True)) for item in messages)
    if not max_chars or size <= max_chars or len(messages) <= 2:
        return messages, False
    base = messages[:2]
    groups: list[list[dict[str, Any]]] = []
    for message in messages[2:]:
        if message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    kept = groups[-max(1, tail_groups):]
    old = groups[:-len(kept)]
    summaries: list[dict[str, Any]] = []
    for group in old:
        for message in group:
            role = message.get("role")
            if role == "assistant":
                calls = []
                for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    calls.append({"name": function.get("name"), "arguments": function.get("arguments")})
                summaries.append({"role": "assistant", "tool_calls": calls,
                                  "content": str(message.get("content") or "")[:500]})
            elif role == "tool":
                try:
                    value = json.loads(message.get("content", "{}"))
                except (TypeError, json.JSONDecodeError):
                    value = {"status": "unparseable_tool_result"}
                if isinstance(value, dict):
                    summaries.append({key: value.get(key) for key in (
                        "observation_id", "tool", "status", "target", "error", "detail"
                    ) if key in value})
            elif role == "user":
                summaries.append({"role": "user", "content": str(message.get("content") or "")[:500]})
    # Keep the most recent compacted evidence if the summary itself is large.
    summary_budget = max(10_000, max_chars // 3)
    while summaries and len(json.dumps(summaries, sort_keys=True)) > summary_budget:
        summaries.pop(0)
    compacted_notice = {
        "role": "user",
        "content": (
            "Recorded context compaction. This summary contains only prior agent-visible "
            "decisions and tool metadata; the full raw trace remains available to the operator. "
            "Treat summarized claims as prior reasoning, not new evidence.\n" +
            json.dumps(summaries, sort_keys=True)
        ),
    }
    model_messages = base + [compacted_notice] + [item for group in kept for item in group]
    return model_messages, True


def _target_hosts(name: str, arguments: dict[str, Any]) -> set[str]:
    """Extract model-supplied network targets solely for client-side pacing."""
    if name in {"get_previous_observations", "submit_inventory", "submit_fingerprint_assessment"}:
        return set()
    hosts: set[str] = set()
    ip = arguments.get("ip")
    if isinstance(ip, str):
        hosts.add(ip)
    targets = arguments.get("targets")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and isinstance(target.get("ip"), str):
                hosts.add(target["ip"])
    return hosts
