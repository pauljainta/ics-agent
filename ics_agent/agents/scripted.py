from __future__ import annotations

from typing import Any

from .base import RoleAgent
from ..types import RoleDecision


def _read_value(observation: dict[str, Any], key: str) -> Any:
    if key in observation:
        return observation[key]
    for group in (
        "levels",
        "status",
        "commands",
        "states",
        "alarms",
        "hmi",
        "process_controls",
    ):
        nested = observation.get(group)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def _match_clause(clause: dict[str, Any], context: dict[str, Any]) -> bool:
    observation = context.get("observation", {})
    field = clause.get("field")
    actual = _read_value(observation, field) if field else None
    if "equals" in clause:
        return actual == clause["equals"]
    if "above" in clause:
        return actual is not None and actual > clause["above"]
    if "below" in clause:
        return actual is not None and actual < clause["below"]
    if "state_is" in clause:
        return (
            observation.get("STATE_NAME") == clause["state_is"]
            or observation.get("state_name") == clause["state_is"]
        )
    if "role_alerted" in clause:
        target = clause["role_alerted"]
        return any(item.get("role") == target for item in context.get("alerts", []))
    if "time_after_sec" in clause:
        return float(context["time_sec"]) >= float(clause["time_after_sec"])
    return False


def _match_rule(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    when = rule.get("when", {})
    if "all" in when:
        return all(_match_clause(clause, context) for clause in when["all"])
    if "any" in when:
        return any(_match_clause(clause, context) for clause in when["any"])
    return bool(when) and _match_clause(when, context)


class ScriptedRoleAgent(RoleAgent):
    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.mode == "offline":
            return []
        if self.role in {"operator", "attacker"}:
            return self._step_scenario_plan(context)
        if self.role in {"defender", "observer"}:
            return self._step_rules(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

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

    def _step_scenario_plan(self, context: dict[str, Any]) -> list[RoleDecision]:
        plan = context.get("planned_events", {}).get(self.role, [])
        cursor = int(self._state.get("cursor", 0))
        now = float(context["time_sec"])
        decisions: list[RoleDecision] = []

        while cursor < len(plan) and now >= float(plan[cursor]["time_sec"]):
            action = plan[cursor]["action"]
            action_type = (
                action if isinstance(action, str) else action.get("type", "unknown")
            )
            decisions.append(
                self.decision(
                    decision_type="action",
                    summary=f"Execute scripted {self.role} action `{action_type}`",
                    time_sec=now,
                    action=action if isinstance(action, dict) else {"type": action},
                    confidence=1.0,
                    labels=["scripted", "scenario-plan"],
                    metadata={"plan_index": cursor},
                )
            )
            cursor += 1

        self._state["cursor"] = cursor
        return decisions

    def _step_rules(self, context: dict[str, Any]) -> list[RoleDecision]:
        decisions: list[RoleDecision] = []
        triggered_rules: set[str] = self._state.setdefault("triggered_rules", set())
        now = float(context["time_sec"])

        for rule in self.profile.get("rules", []):
            rule_id = str(rule.get("id", "rule"))
            if rule.get("once_per_run", True) and rule_id in triggered_rules:
                continue
            if not _match_rule(rule, context):
                continue

            diagnosis = rule.get("diagnosis")
            alert_text = rule.get("alert")
            labels = ["scripted", "rule-based"]
            if alert_text:
                decisions.append(
                    self.decision(
                        decision_type="alert",
                        summary=str(alert_text),
                        time_sec=now,
                        confidence=float(rule.get("confidence", 1.0)),
                        diagnosis=diagnosis,
                        labels=labels,
                        metadata={"rule_id": rule_id},
                    )
                )
            action = rule.get("action")
            if action:
                decisions.append(
                    self.decision(
                        decision_type="action",
                        summary=str(
                            rule.get("action_summary", f"{self.role} mitigation action")
                        ),
                        time_sec=now,
                        action=action,
                        confidence=float(rule.get("confidence", 1.0)),
                        diagnosis=diagnosis,
                        labels=labels,
                        metadata={"rule_id": rule_id},
                    )
                )
            triggered_rules.add(rule_id)
        return decisions

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
