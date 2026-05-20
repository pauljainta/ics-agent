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


def _normalize_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return action
    if isinstance(action, str):
        return {"type": action}
    return {}


def _action_matches(actual: Any, expected: Any) -> bool:
    actual_payload = _normalize_action(actual)
    expected_payload = _normalize_action(expected)
    if not actual_payload or not expected_payload:
        return False
    return all(actual_payload.get(key) == value for key, value in expected_payload.items())


def _has_planned_role_action(spec: Any, context: dict[str, Any]) -> bool:
    if not isinstance(spec, dict):
        return False
    role = str(spec.get("role", "")).strip()
    expected_action = spec.get("action")
    if not role or expected_action is None:
        return False

    planned_events = context.get("planned_events") or context.get("playbooks") or {}
    role_plan = planned_events.get(role, [])
    return any(
        _action_matches(item.get("action"), expected_action)
        for item in role_plan
        if isinstance(item, dict)
    )


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
    if "planned_role_action" in clause:
        return _has_planned_role_action(clause["planned_role_action"], context)
    if "planned_role_action_absent" in clause:
        return not _has_planned_role_action(
            clause["planned_role_action_absent"], context
        )
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


class RuleBasedRoleAgent(RoleAgent):
    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.mode == "offline":
            return []
        if self.role == "attacker":
            return self._step_attacker(context)
        if self.role in {"defender", "observer"}:
            return self._step_rules(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

    def _step_attacker(self, context: dict[str, Any]) -> list[RoleDecision]:
        explicit_rules = self.profile.get("rules")
        if isinstance(explicit_rules, list) and explicit_rules:
            return self._step_rules(context)

        plan = context.get("planned_events", {}).get(self.role, [])
        cursor = int(self._state.get("cursor", 0))
        now = float(context["time_sec"])
        decisions: list[RoleDecision] = []

        while cursor < len(plan):
            event = plan[cursor]
            if not self._planned_attack_event_ready(event, context):
                break
            action = event.get("action")
            action_type = action if isinstance(action, str) else action.get("type", "unknown")
            decisions.append(
                self.decision(
                    decision_type="action",
                    summary=f"Trigger rule-based attacker action `{action_type}` when conditions are met.",
                    time_sec=now,
                    action=action if isinstance(action, dict) else {"type": action},
                    confidence=1.0,
                    labels=["rule-based", "derived-plan"],
                    metadata={"plan_index": cursor, "scheduled_time_sec": event.get("time_sec")},
                )
            )
            cursor += 1

        self._state["cursor"] = cursor
        return decisions

    def _planned_attack_event_ready(
        self, event: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        now = float(context["time_sec"])
        event_time = float(event.get("time_sec", 0.0))
        if now < event_time:
            return False

        action = _normalize_action(event.get("action"))
        if not action:
            return False

        observation = context.get("observation", {})
        state_name = observation.get("STATE_NAME") or observation.get("state_name")
        startup_time = self._operator_start_time(context)
        pre_start_attack = startup_time is not None and event_time < startup_time
        action_type = str(action.get("type", "unknown"))

        if action_type == "noop":
            return True
        if action_type == "hmi_start":
            return state_name in {"IDLE", "SHUTDOWN"}
        if action_type in {"hmi_stop", "hmi_estop"}:
            return state_name == "RUNNING"
        if action_type == "set_sensor":
            return pre_start_attack or state_name == "RUNNING"
        if action_type == "select_process":
            desired_state = bool(action.get("active", True))
            current_state = int(observation.get("P2_Selected", 0) or 0)
            if desired_state:
                return state_name == "RUNNING" and current_state == 0
            return current_state == 1 or now >= event_time
        if action_type == "set_manual_valve":
            if pre_start_attack:
                return False
            if state_name != "RUNNING":
                return False
            return int(observation.get("P2_Selected", 0) or 0) == 1
        return True

    def _operator_start_time(self, context: dict[str, Any]) -> float | None:
        operator_plan = context.get("planned_events", {}).get("operator", [])
        for item in operator_plan:
            action = _normalize_action(item.get("action"))
            if action.get("type") == "hmi_start":
                return float(item.get("time_sec", 0.0))
        return None

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
            labels = ["rule-based", self.role]
            alert_text = rule.get("alert")
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
                            rule.get("action_summary", f"{self.role} rule-based action")
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
                summary=f"Rule-based planner briefing: {scenario.metadata['intent']}",
                time_sec=float(context["time_sec"]),
                confidence=1.0,
                labels=["rule-based", "planner"],
                metadata={
                    "expected_outcomes": scenario.metadata.get("expected_outcomes", [])
                },
            )
        ]
