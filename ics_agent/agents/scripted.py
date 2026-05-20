from __future__ import annotations

from typing import Any

from .base import RoleAgent
from ..types import RoleDecision


class ScriptedRoleAgent(RoleAgent):
    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.mode == "offline":
            return []
        if self.role == "planner":
            return self._step_planner(context)
        return self._step_scripted_plan(context)

    def finalize(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.role != "offline_analysis":
            return []
        metrics = context.get("metrics", {})
        summary = (
            f"Offline scripted analysis: pass={context.get('passed')} "
            f"ttd={metrics.get('time_to_detection_sec')} "
            f"ttm={metrics.get('time_to_mitigation_sec')}"
        )
        return [
            self.decision(
                decision_type="analysis",
                summary=summary,
                time_sec=float(context.get("duration_sec", 0.0)),
                confidence=1.0,
                diagnosis=context.get("scenario")
                .metadata.get("ground_truth", {})
                .get("expected_diagnosis"),
                labels=["offline", "scripted"],
                metadata={"metrics": metrics},
            )
        ]

    def _step_scripted_plan(self, context: dict[str, Any]) -> list[RoleDecision]:
        now = float(context["time_sec"])
        plan = self._resolve_plan(context)
        cursor = int(self._state.get("cursor", 0))
        decisions: list[RoleDecision] = []

        while cursor < len(plan) and now >= float(plan[cursor]["time_sec"]):
            item = plan[cursor]
            action = item.get("action")
            summary = str(item.get("summary") or self._default_summary(action))
            decision_type = str(item.get("decision_type", "action"))

            decisions.append(
                self.decision(
                    decision_type=decision_type,
                    summary=summary,
                    time_sec=now,
                    action=(
                        action
                        if isinstance(action, dict) or decision_type != "action"
                        else {"type": action}
                    )
                    if decision_type == "action"
                    else None,
                    confidence=float(item.get("confidence", 1.0)),
                    diagnosis=item.get("diagnosis"),
                    labels=["scripted", "scenario-plan"],
                    metadata={"plan_index": cursor},
                )
            )
            cursor += 1

        self._state["cursor"] = cursor
        return decisions

    def _resolve_plan(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        profile_plan = self.profile.get("scripted_plan")
        if isinstance(profile_plan, list):
            return [item for item in profile_plan if isinstance(item, dict)]
        planned_events = context.get("planned_events", {})
        return list(planned_events.get(self.role, []))

    def _default_summary(self, action: Any) -> str:
        if isinstance(action, dict):
            action_type = action.get("type", "unknown")
        else:
            action_type = str(action or "unknown")
        return f"Execute scripted {self.role} action `{action_type}`"

    def _step_planner(self, context: dict[str, Any]) -> list[RoleDecision]:
        if "planner-note" in self._state["emitted_notes"]:
            return []
        scenario = context["scenario"]
        self._state["emitted_notes"].add("planner-note")
        return [
            self.decision(
                decision_type="note",
                summary=f"Scripted planner briefing: {scenario.metadata['intent']}",
                time_sec=float(context["time_sec"]),
                confidence=1.0,
                labels=["scripted", "planner"],
                metadata={
                    "expected_outcomes": scenario.metadata.get("expected_outcomes", [])
                },
            )
        ]
