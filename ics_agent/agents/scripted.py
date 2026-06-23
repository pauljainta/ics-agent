from __future__ import annotations

from typing import Any

from .base import RoleAgent
from ..mitre import (
    attacker_manipulation_started,
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


class ScriptedRoleAgent(RoleAgent):
    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.mode == "offline":
            return []
        if optionb_mitre_enabled(self.profile):
            return self._step_optionb_mitre(context)
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
        phase_schedule = dict(attack_profile.get("phase_schedule") or {})
        emitted: set[str] = self._state.setdefault("mitre_phases_emitted", set())

        phase_steps = [
            (
                "staged_entry_context",
                "note",
                None,
                "Option B staged entry context established through access and engineering hops.",
            ),
            (
                "discovery",
                "action",
                {
                    "type": "inspect_surface",
                    "surface": attack_profile["discovery_surface"],
                },
                f"Inspect {attack_profile['discovery_surface']} to enumerate reachable process-facing surfaces.",
            ),
            (
                "process_understanding",
                "note",
                None,
                f"Identify {attack_profile['family']} as the likely process-facing tactic family for this scenario.",
            ),
            (
                "control_path_qualification",
                "action",
                {
                    "type": "inspect_surface",
                    "surface": attack_profile["qualification_surface"],
                },
                f"Qualify {attack_profile['qualification_surface']} as the plant-facing control surface for this staged path.",
            ),
        ]

        for phase, decision_type, action, summary in phase_steps:
            if phase in emitted:
                continue
            if now < float(phase_schedule.get(phase, 0.0)):
                continue
            emitted.add(phase)
            return [
                self.decision(
                    decision_type=decision_type,
                    summary=summary,
                    time_sec=now,
                    action=action,
                    confidence=1.0,
                    labels=["scripted", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase=phase,
                        policy_mode="scripted",
                        attack_profile=attack_profile,
                        qualification_surface=attack_profile["qualification_surface"],
                        target_tags=list(attack_profile.get("target_tags") or []),
                        chemicals=list(attack_profile.get("chemicals") or []),
                    ),
                )
            ]

        plan = self._resolve_plan(context)
        cursor = int(self._state.get("cursor", 0))
        decisions: list[RoleDecision] = []
        while cursor < len(plan) and now >= float(plan[cursor]["time_sec"]):
            item = plan[cursor]
            action = normalize_action(item.get("action"))
            if not action:
                cursor += 1
                continue
            summary = str(
                item.get("summary")
                or f"Execute staged manipulation via `{action.get('type', 'unknown')}`."
            )
            decisions.append(
                self.decision(
                    decision_type="action",
                    summary=summary,
                    time_sec=now,
                    action=action,
                    confidence=float(item.get("confidence", 1.0)),
                    labels=["scripted", "optionb-mitre", "attacker"],
                    metadata=phase_metadata(
                        phase_family="attacker",
                        phase="manipulation",
                        policy_mode="scripted",
                        attack_profile=attack_profile,
                        plan_index=cursor,
                        scheduled_time_sec=float(item.get("time_sec", now)),
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
        state_name = str(observation.get("STATE_NAME") or observation.get("state_name") or "")
        latest_attacker_event = context_phase_events(
            context,
            role="attacker",
            phase_family="attacker",
        )
        manipulation_started = attacker_manipulation_started(context)
        latest_attacker_action = None
        if context.get("executed_actions"):
            for item in context.get("executed_actions", []):
                if item.get("role") == "attacker":
                    latest_attacker_action = normalize_action(item.get("action"))

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
                    summary="Observer recognized staged qualification of a process-facing control path.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["scripted", "optionb-mitre", "observer"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="discovery_recognition",
                        policy_mode="scripted",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if (
            manipulation_started
            and self.role == "observer"
            and "manipulation_detection" not in emitted
        ):
            emitted.add("manipulation_detection")
            return [
                self.decision(
                    decision_type="alert",
                    summary="Observer detected active plant-facing misuse on the staged Option B path.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["scripted", "optionb-mitre", "observer"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="manipulation_detection",
                        policy_mode="scripted",
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
                    summary="Defender observed staged qualification activity against the process surface.",
                    time_sec=now,
                    confidence=1.0,
                    diagnosis=str(attack_profile["family"]),
                    labels=["scripted", "optionb-mitre", "defender"],
                    metadata=phase_metadata(
                        phase_family="defender",
                        phase="discovery_recognition",
                        policy_mode="scripted",
                        attack_profile=attack_profile,
                    ),
                )
            ]

        if attack_profile["family"] == "sensor_spoofing":
            if rw_high_high(observation) and "containment_mitigation" not in emitted:
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Emergency-stop the plant on fixed high-high evidence.",
                        time_sec=now,
                        action={"type": "hmi_estop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="raw_water_alarm_condition",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="scripted",
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
                        summary="Stop the plant after fixed manipulation evidence crosses a safety threshold.",
                        time_sec=now,
                        action={"type": "hmi_stop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="sensor_spoofing",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="scripted",
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
                process2_valve_open(observation)
                and "containment_mitigation" not in emitted
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Deselect Process 2 after fixed confirmation of unauthorized chemical actuation.",
                        time_sec=now,
                        action={"process_id": "WT-P2", "type": "select_process", "active": False},
                        confidence=1.0,
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="scripted",
                            attack_profile=attack_profile,
                            evidence="process2_valve_open",
                        ),
                    )
                ]
            if (
                manipulation_started
                and attack_profile["family"] == "process2_selection_abuse"
                and unexpected_process2_selection(observation)
                and "containment_mitigation" not in emitted
                and len(latest_attacker_event) >= 2
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Deselect Process 2 after repeated fixed-threshold selection abuse.",
                        time_sec=now,
                        action={"process_id": "WT-P2", "type": "select_process", "active": False},
                        confidence=1.0,
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="scripted",
                            attack_profile=attack_profile,
                            evidence="repeated_process2_selection",
                        ),
                    )
                ]

        if attack_profile["family"] == "hmi_control_abuse":
            if (
                manipulation_started
                and "manipulation_detection" not in emitted
            ):
                emitted.add("manipulation_detection")
                return [
                    self.decision(
                        decision_type="alert",
                        summary="Defender detected fixed-threshold HMI misuse on the staged path.",
                        time_sec=now,
                        confidence=1.0,
                        diagnosis="hmi_control_abuse",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="manipulation_detection",
                            policy_mode="scripted",
                            attack_profile=attack_profile,
                            latest_action_type=(latest_attacker_action or {}).get("type"),
                        ),
                    )
                ]
            if (
                latest_attacker_action
                and latest_attacker_action.get("type") == "hmi_start"
                and manipulation_started
                and state_name == "RUNNING"
                and "containment_mitigation" not in emitted
            ):
                emitted.add("containment_mitigation")
                return [
                    self.decision(
                        decision_type="action",
                        summary="Stop the plant after fixed-threshold HMI chatter persists.",
                        time_sec=now,
                        action={"type": "hmi_stop", "process_id": "WT-P1"},
                        confidence=1.0,
                        diagnosis="hmi_control_abuse",
                        labels=["scripted", "optionb-mitre", "defender"],
                        metadata=phase_metadata(
                            phase_family="defender",
                            phase="containment_mitigation",
                            policy_mode="scripted",
                            attack_profile=attack_profile,
                            evidence="hmi_chatter_running",
                        ),
                    )
                ]

        return []

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
