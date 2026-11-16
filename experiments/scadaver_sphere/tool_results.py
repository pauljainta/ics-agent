"""Versioned, model-visible result metadata shared by read-only tools."""

from __future__ import annotations

from typing import Any


CONTRACT_VERSION = "scadaver-tool-result-v2"

_RETRYABLE = {
    "connection_timeout", "connection_failure", "tool_execution_timeout",
    "worker_unavailable", "rate_limit_exceeded",
}


def policy_error_category(detail: str) -> str:
    lowered = detail.lower()
    if "rate" in lowered:
        return "rate_limit_exceeded"
    if "budget" in lowered or "sampling limits" in lowered:
        return "resource_budget_exhausted"
    if any(word in lowered for word in ("must", "invalid", "required", "outside")):
        return "invalid_arguments"
    return "policy_rejected"


def normalize_tool_result(value: dict[str, Any], *, tool: str,
                          attempted: bool | None = None,
                          budget_consumed: bool | None = None) -> dict[str, Any]:
    """Add operational metadata without changing evidence or legacy error codes."""
    result = dict(value)
    result.setdefault("tool", tool)
    status = str(result.get("status", "error"))
    error = result.get("error")
    if attempted is None:
        attempted = status not in {"rejected", "disabled"}
    if budget_consumed is None:
        budget_consumed = attempted
    category = result.get("error_category")
    if category is None and error:
        category = {
            "timeout_or_connection_failure": "connection_failure",
            "observation_budget_exhausted": "resource_budget_exhausted",
            "probe_budget_exhausted": "resource_budget_exhausted",
            "policy_denied": "policy_rejected",
            "command_failed": "tool_execution_error",
            "modbus_exception": "protocol_unsupported",
        }.get(str(error), str(error))
    result.setdefault("result_contract", CONTRACT_VERSION)
    result.setdefault("operation_attempted", bool(attempted))
    result.setdefault("operation_executed", status in {"ok", "authentication_required", "redirect"})
    result.setdefault("retryable", category in _RETRYABLE)
    result.setdefault("budget_consumed", bool(budget_consumed))
    result.setdefault("state_change_possible", False)
    if category is not None:
        result.setdefault("error_category", category)
    return result
