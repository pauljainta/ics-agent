"""V2 tool composition; read-only worker and operator broker stay separated."""

from __future__ import annotations

from typing import Any, Protocol

from ..n3_worker import N3_WORKER_TOOL_SCHEMAS
from .schemas import FINAL_SCHEMA, action_schema, validate_report
from .skills import SKILL_SCHEMA, load_skill


class ReadOnlyTransport(Protocol):
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class ActionTransport(Protocol):
    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


def tool_schemas(max_duration_seconds: float = 1.25, *,
                 action_enabled: bool = True) -> list[dict[str, Any]]:
    tools = [*N3_WORKER_TOOL_SCHEMAS, SKILL_SCHEMA]
    if action_enabled:
        tools.append(action_schema(max_duration_seconds))
    tools.append(FINAL_SCHEMA)
    return tools


def capability_manifest(action_enabled: bool, action_requests_remaining: int = 1) -> str:
    """Describe active capabilities without disclosing targets or credentials."""
    state = "available" if action_enabled else "unavailable"
    remaining = action_requests_remaining if action_enabled else 0
    return (
        "Runtime capability state (generated from registered tools):\n"
        "- Read-only investigation tools: available.\n"
        f"- Controller-mediated bounded action: {state}.\n"
        f"- Action requests remaining in this episode: {remaining}.\n"
        "- Authorization and safety are enforced outside the model. "
        "A rejected request consumes the one-request allowance."
    )


class CompositeExecutorV2:
    def __init__(self, readonly: ReadOnlyTransport, action: ActionTransport | None) -> None:
        self.readonly = readonly
        self.action = action
        self.action_requests = 0

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_skill":
            try:
                content = load_skill(str(arguments.get("skill_name")))
            except ValueError as exc:
                return {"tool": name, "status": "rejected", "error": str(exc)}
            return {"tool": name, "status": "ok",
                    "skill_name": arguments["skill_name"], "content": content}
        if name == "bounded_controller_output":
            self.action_requests += 1
            if self.action_requests > 1:
                return {"tool": name, "status": "rejected",
                        "error": "action_request_budget_exhausted"}
            if self.action is None:
                return {"tool": name, "status": "rejected",
                        "error": "actions_not_authorized"}
            return self.action.execute(arguments)
        if name == "submit_integrated_report":
            try:
                report = validate_report(arguments.get("report"))
            except ValueError as exc:
                return {"tool": name, "status": "rejected",
                        "error": "invalid_report_schema", "detail": str(exc)}
            return {"tool": name, "status": "accepted", "report": report}
        return self.readonly.execute(name, arguments)
