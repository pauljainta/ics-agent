from __future__ import annotations

from typing import Any

from ..action_semantics import attacker_surface
from .base import RoleAgent
from ..mitre import (
    attacker_manipulation_started,
    attacker_recon_count,
    context_phase_events,
    normalize_action,
    optionb_mitre_enabled,
    phase_metadata,
    phase_reached,
    process2_valve_open,
    read_observation_value,
    rw_high_high,
    rw_low_or_spoof,
    scenario_attack_profile,
    unexpected_process2_selection,
    uf_high,
)
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


def _rule_based_defender_response_mode(profile: dict[str, Any]) -> str:
    behavior = profile.get("behavior_model")
    if not isinstance(behavior, dict):
        return ""
    return str(behavior.get("response_mode", "")).strip()


def _is_rule_based_defender_policy(profile: dict[str, Any]) -> bool:
    behavior = profile.get("behavior_model")
    if not isinstance(behavior, dict):
        return False
    return str(behavior.get("policy", "")).strip() == "rule_based_defender"


def _containment_action_for_attacker_action(action: dict[str, Any] | None) -> dict[str, Any]:
    surface = attacker_surface(action)
    if surface == "sensor_write_path":
        return {"type": "block_sensor_write"}
    if surface == "actuator_write_path":
        return {"type": "block_actuator_write"}
    if surface == "process2_manual_path":
        return {"type": "block_process2_manual_path"}
    if surface == "hmi_session":
        return {"type": "disable_hmi_session"}
    return {"type": "isolate_engineering_host"}


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
        if optionb_mitre_enabled(self.profile):
            return self._step_optionb_mitre(context)
        if self.role == "attacker":
            return self._step_attacker(context)
        if self.role in {"defender", "observer"}:
            return self._step_rules(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

    def _step_optionb_mitre(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.role == "attacker":
            return self._step_optionb_mitre_attacker(context)
        if self.role in {"observer", "defender"}:
            return self._step_optionb_mitre_blue_team(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

    def _step_optionb_mitre_attacker(
        self,
        context: dict[str, Any],
    ) -> list[RoleDecision]:
        now = float(context["time_sec"])
        attack_profile = scenario_attack_profile(context)
        schedule = dict(attack_profile.get("phase_schedule") or {})
        emitted: set[str] = self._state.setdefault("mitre_phases_emitted", set())
        observation = context.get("observation", {})
        state_name = str(observation.get("STATE_NAME") or observation.get("state_name") or "")

        if (
            "staged_entry_context" not in emitted
            and now >= float(schedule.get("staged_entry_context", 0.0))
        ):
            emitted.add("staged_entry_context")
            return [
                self.decision(
                    decision_type="note",
                    summary="Adaptive attacker established staged Option B access context before touching plant-facing controls.",
                    time_sec=now,
                    confidence=1.0,
                    labels=["rule-based", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="staged_entry_context",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            "discovery" not in emitted
            and now >= float(schedule.get("discovery", 0.0))
        ):
            emitted.add("discovery")
            return [
                self.decision(
                    decision_type="action",
                    summary="Adaptively inspect shared process tags before choosing a manipulation surface.",
                    time_sec=now,
                    action={
                        "type": "inspect_surface",
                        "surface": attack_profile["discovery_surface"],
                    },
                    confidence=1.0,
                    labels=["rule-based", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="discovery",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                        discovery_surface=attack_profile["discovery_surface"],
                    ),
                )
            ]

        if "process_understanding" not in emitted and attacker_recon_count(context) >= 1:
            emitted.add("process_understanding")
            return [
                self.decision(
                    decision_type="note",
                    summary=f"Adaptive attacker chose the `{attack_profile['family']}` tactic family after observing process context.",
                    time_sec=now,
                    confidence=1.0,
                    labels=["rule-based", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="process_understanding",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        qualification_ready = not attack_profile["runtime_required"] or state_name == "RUNNING"
        if "control_path_qualification" not in emitted and qualification_ready:
            emitted.add("control_path_qualification")
            return [
                self.decision(
                    decision_type="action",
                    summary="Adaptively qualify the selected process-facing surface before manipulation.",
                    time_sec=now,
                    action={
                        "type": "inspect_surface",
                        "surface": attack_profile["qualification_surface"],
                    },
                    confidence=1.0,
                    labels=["rule-based", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="control_path_qualification",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                        qualification_surface=attack_profile["qualification_surface"],
                        runtime_ready=qualification_ready,
                    ),
                )
            ]

        plan = context.get("planned_events", {}).get(self.role, [])
        cursor = int(self._state.get("cursor", 0))
        decisions: list[RoleDecision] = []
        while cursor < len(plan):
            event = plan[cursor]
            if not self._planned_attack_event_ready(event, context):
                break
            action = normalize_action(event.get("action"))
            if not action:
                cursor += 1
                continue
            action_type = action.get("type", "unknown")
            decisions.append(
                self.decision(
                    decision_type="action",
                    summary=f"Adaptively execute staged manipulation `{action_type}` when the process state permits it.",
                    time_sec=now,
                    action=action,
                    confidence=1.0,
                    labels=["rule-based", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="manipulation",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                        plan_index=cursor,
                        scheduled_time_sec=float(event.get("time_sec", now)),
                    ),
                )
            )
            cursor += 1

        self._state["cursor"] = cursor
        return decisions

    def _step_optionb_mitre_blue_team(
        self,
        context: dict[str, Any],
    ) -> list[RoleDecision]:
        now = float(context["time_sec"])
        observation = context.get("observation", {})
        attack_profile = scenario_attack_profile(context)
        emitted: set[str] = self._state.setdefault("mitre_phases_emitted", set())
        manipulation_started = attacker_manipulation_started(context)
        state_name = str(observation.get("STATE_NAME") or observation.get("state_name") or "")
        latest_attacker_action = None
        for item in context.get("executed_actions", []) or []:
            if item.get("role") == "attacker":
                latest_attacker_action = normalize_action(item.get("action"))

        if (
            self.role == "defender"
            and _is_rule_based_defender_policy(self.profile)
            and manipulation_started
            and latest_attacker_action
            and "containment_mitigation" not in emitted
        ):
            emitted.add("containment_mitigation")
            action = _containment_action_for_attacker_action(latest_attacker_action)
            surface = attacker_surface(latest_attacker_action)
            metadata = phase_metadata(
                phase_family="defender",
                phase="containment_mitigation",
                policy_mode="rule_based",
                attack_profile=attack_profile,
                evidence="observed_attacker_manipulation",
                attacker_action_type=latest_attacker_action.get("type"),
                attacker_surface=surface,
                response_mode=_rule_based_defender_response_mode(self.profile),
            )
            labels = ["rule-based", "optionb-mitre", "defender"]
            if _rule_based_defender_response_mode(self.profile) == "advisory_response":
                return [
                    self.decision(
                        decision_type="alert",
                        summary=(
                            "Rule-based defender issued an advisory response after "
                            f"observing manipulation on `{surface}`; recommended `{action['type']}`."
                        ),
                        time_sec=now,
                        confidence=1.0,
                        diagnosis=str(attack_profile["family"]),
                        labels=[*labels, "advisory-response"],
                        metadata=metadata,
                    )
                ]
            return [
                self.decision(
                    decision_type="action",
                    summary=(
                        "Rule-based defender applied deterministic containment "
                        f"`{action['type']}` after observing manipulation on `{surface}`."
                    ),
                    time_sec=now,
                    action=action,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=labels,
                    metadata=metadata,
                )
            ]

        if (
            self.role == "observer"
            and "access_presence_suspicion" not in emitted
            and attacker_recon_count(context) >= 1
        ):
            emitted.add("access_presence_suspicion")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Observer raised early suspicion from staged-path discovery activity.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["rule-based", "optionb-mitre", "observer"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="access_presence_suspicion",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            self.role == "observer"
            and "discovery_recognition" not in emitted
            and phase_reached(
                context,
                role="attacker",
                phase_family="attacker",
                phase="control_path_qualification",
            )
        ):
            emitted.add("discovery_recognition")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Observer recognized staged qualification of a safety-relevant control path.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["rule-based", "optionb-mitre", "observer"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="discovery_recognition",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            self.role == "observer"
            and "manipulation_detection" not in emitted
            and (
                manipulation_started
                or rw_high_high(observation)
                or rw_low_or_spoof(observation)
                or unexpected_process2_selection(observation)
            )
        ):
            emitted.add("manipulation_detection")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Observer detected active plant-facing misuse before physical impact escalated further.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["rule-based", "optionb-mitre", "observer"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="manipulation_detection",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            self.role == "defender"
            and "access_presence_suspicion" not in emitted
            and attacker_recon_count(context) >= 1
        ):
            emitted.add("access_presence_suspicion")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Defender flagged early staged-path presence from reconnaissance against plant-facing surfaces.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["rule-based", "optionb-mitre", "defender"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="access_presence_suspicion",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            self.role == "defender"
            and "discovery_recognition" not in emitted
            and phase_reached(
                context,
                role="attacker",
                phase_family="attacker",
                phase="control_path_qualification",
            )
        ):
            emitted.add("discovery_recognition")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Defender recognized staged qualification of a process-relevant attack surface.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["rule-based", "optionb-mitre", "defender"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="discovery_recognition",
                        policy_mode="adaptive",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if attack_profile["family"] == "sensor_spoofing":
            if (
                attack_profile["runtime_required"]
                and state_name == "RUNNING"
                and "containment_mitigation" not in emitted
                and phase_reached(
                    context,
                    role="attacker",
                    phase_family="attacker",
                    phase="control_path_qualification",
                )
                and not manipulation_started
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Preemptively stop the plant once staged qualification identifies a critical sensor manipulation path.",
                        time_sec=now,
                        action={"type": "hmi_stop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="sensor_spoofing_preempted",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            evidence="qualified_sensor_surface_pre_manipulation",
                        ),
                    )
                ]
            if rw_high_high(observation) and "containment_mitigation" not in emitted:
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Emergency-stop the plant on high-high sensor misuse evidence.",
                        time_sec=now,
                        action={"type": "hmi_estop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="raw_water_alarm_condition",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            evidence="rw_high_high",
                        ),
                    )
                ]
            if (
                manipulation_started
                and "containment_mitigation" not in emitted
                and (rw_low_or_spoof(observation) or uf_high(observation))
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Stop the plant once adaptive detection confirms live sensor misuse.",
                        time_sec=now,
                        action={"type": "hmi_stop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="sensor_spoofing",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            evidence="sensor_runtime_threshold",
                        ),
                    )
                ]

        if attack_profile["family"] in {
            "process2_manual_control",
            "process2_selection_abuse",
        }:
            if (
                unexpected_process2_selection(observation)
                and "containment_mitigation" not in emitted
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Immediately deselect Process 2 once adaptive detection sees unauthorized selection.",
                        time_sec=now,
                        action={"process_id": "WT-P2", "type": "select_process", "active": False},
                        confidence=1.0,
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            evidence="unexpected_process2_selection",
                        ),
                    )
                ]
            if (
                process2_valve_open(observation)
                and "manipulation_detection" not in emitted
            ):
                emitted.add("manipulation_detection")
                return [
                    self.decision(
                        decision_type="alert",
                        summary="Defender confirmed active Process 2 manipulation from valve-state evidence.",
                        time_sec=now,
                        confidence=1.0,
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="manipulation_detection",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                        ),
                    )
                ]

        if attack_profile["family"] == "hmi_control_abuse":
            if (
                latest_attacker_action
                and latest_attacker_action.get("type") == "hmi_start"
                and state_name == "RUNNING"
                and "containment_mitigation" not in emitted
                and manipulation_started
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Stop the plant once adaptive logic confirms HMI chatter while the process is still running.",
                        time_sec=now,
                        action={"type": "hmi_stop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="hmi_control_abuse",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            evidence="hmi_chatter_running",
                        ),
                    )
                ]
            if (
                manipulation_started
                and "manipulation_detection" not in emitted
            ):
                emitted.add("manipulation_detection")
                return [
                    self.decision(
                        decision_type="alert",
                        summary="Defender detected HMI misuse after staged qualification reached an execution-capable state.",
                        time_sec=now,
                        confidence=1.0,
                        diagnosis="hmi_control_abuse",
                        labels=["rule-based", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="manipulation_detection",
                            policy_mode="adaptive",
                            attack_profile=attack_profile,
                            latest_action_type=(latest_attacker_action or {}).get("type"),
                        ),
                    )
                ]

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
