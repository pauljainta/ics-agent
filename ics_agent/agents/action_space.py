from __future__ import annotations

import json
from typing import Any


def normalize_action(action: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, str):
        return {"type": action}
    return dict(action)


def action_key(action: dict[str, Any] | str | None) -> str:
    normalized = normalize_action(action)
    if normalized is None:
        return "null"
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def build_candidate_actions(
    *,
    role: str,
    allowed_action_types: list[str],
    scenario: Any,
    observation: dict[str, Any],
    risk_flags: list[str] | None = None,
    active_alarms: list[str] | None = None,
    relationship_notes: list[str] | None = None,
    recent_alerts: list[dict[str, Any]] | None = None,
    incident: dict[str, Any] | None = None,
    remaining_playbook: list[dict[str, Any]],
    now: float,
    playbook_anticipation_sec: float,
) -> list[dict[str, Any]]:
    allowed = set(allowed_action_types)
    process_id = scenario.execution.get("process_id")
    safety_bounds = scenario.metadata.get("safety_bounds", {})
    risk_flags = list(risk_flags or [])
    active_alarms = list(active_alarms or [])
    relationship_notes = list(relationship_notes or [])
    recent_alerts = list(recent_alerts or [])
    incident = dict(incident or {})
    candidates: list[dict[str, Any]] = []

    def add(
        summary: str,
        action: dict[str, Any] | str | None,
        *,
        source: str,
        priority: int = 0,
        guardrail_reason: str | None = None,
    ) -> None:
        normalized = normalize_action(action)
        if normalized is None:
            return
        action_type = str(normalized.get("type", ""))
        if action_type not in allowed:
            return
        key = action_key(normalized)
        if any(item["id"] == key for item in candidates):
            return
        candidates.append(
            {
                "id": key,
                "summary": summary,
                "action": normalized,
                "source": source,
                "priority": priority,
                "guardrail_reason": guardrail_reason,
            }
        )

    for item in remaining_playbook[:3]:
        action = normalize_action(item.get("action"))
        if action is None:
            continue
        if now + playbook_anticipation_sec >= float(item.get("time_sec", 0.0)):
            add(
                f"Follow the scenario playbook with `{action.get('type', 'unknown')}`.",
                action,
                source="playbook",
            )

    if role != "defender":
        if "hmi_start" in allowed:
            add(
                "Start Process 1 from the HMI if the plant should begin operating.",
                {"type": "hmi_start", "process_id": "WT-P1"},
                source="action_space",
            )
        if "hmi_stop" in allowed:
            add(
                "Stop Process 1 from the HMI to halt plant operation in a controlled way.",
                {"type": "hmi_stop", "process_id": "WT-P1"},
                source="action_space",
            )
        if "hmi_reset" in allowed:
            add(
                "Reset Process 1 from the HMI after a stop or fault condition.",
                {"type": "hmi_reset", "process_id": "WT-P1"},
                source="action_space",
            )
        if "hmi_estop" in allowed:
            add(
                "Emergency stop Process 1 immediately to prioritize safety over continuity.",
                {"type": "hmi_estop", "process_id": "WT-P1"},
                source="action_space",
            )

    p2_selected = _coerce_number(observation.get("P2_Selected"))
    allow_process_selection = process_id == "WT-P2" or p2_selected == 1.0
    if role != "defender" and "select_process" in allowed and allow_process_selection:
        add(
            "Enable Process 2 manual controls for dosing actions.",
            {"process_id": "WT-P2", "type": "select_process", "active": True},
            source="action_space",
        )
        add(
            "Disable Process 2 manual controls to constrain further dosing.",
            {"process_id": "WT-P2", "type": "select_process", "active": False},
            source="action_space",
        )

    if "set_manual_valve" in allowed:
        for chemical in _available_chemicals(observation):
            add(
                f"Open the {chemical} manual valve on Process 2.",
                {
                    "process_id": "WT-P2",
                    "type": "set_manual_valve",
                    "chemical": chemical,
                    "open": True,
                },
                source="action_space",
            )
            add(
                f"Close the {chemical} manual valve on Process 2.",
                {
                    "process_id": "WT-P2",
                    "type": "set_manual_valve",
                    "chemical": chemical,
                    "open": False,
                },
                source="action_space",
            )

    if "set_sensor" in allowed:
        for field, values in _sensor_profiles(observation, safety_bounds, process_id):
            add(
                f"Bias `{field}` toward a low reading to perturb controller perception.",
                {"type": "set_sensor", "name": field, "value": values["low"]},
                source="action_space",
            )
            add(
                f"Bias `{field}` toward a high reading to perturb controller perception.",
                {"type": "set_sensor", "name": field, "value": values["high"]},
                source="action_space",
            )

    if role == "defender":
        _add_defender_candidates(
            candidates=candidates,
            add=add,
            observation=observation,
            process_id=process_id,
            allow_process_selection=allow_process_selection,
            risk_flags=risk_flags,
            active_alarms=active_alarms,
            relationship_notes=relationship_notes,
            recent_alerts=recent_alerts,
            incident=incident,
        )

    if "noop" in allowed:
        incident_status = str(incident.get("status", "none"))
        incident_type = str(incident.get("type", "none"))
        if role == "defender" and incident_status in {"suspected", "confirmed"}:
            summary = (
                "Hold only if one more sample is needed; the current evidence already "
                "supports containment and continued delay increases risk."
            )
        elif role == "defender" and incident_type != "none":
            summary = (
                "Hold position while the incident stabilizes and verify that no fresh "
                "alarms, attacker actions, or Process 2 changes appear."
            )
        else:
            summary = "Hold position and gather more evidence before acting."
        add(
            summary,
            {"type": "noop"},
            source="action_space",
        )

    return candidates


def _add_defender_candidates(
    *,
    candidates: list[dict[str, Any]],
    add,
    observation: dict[str, Any],
    process_id: str | None,
    allow_process_selection: bool,
    risk_flags: list[str],
    active_alarms: list[str],
    relationship_notes: list[str],
    recent_alerts: list[dict[str, Any]],
    incident: dict[str, Any],
) -> None:
    incident_type = str(incident.get("type", "none"))
    incident_status = str(incident.get("status", "none"))
    incident_severity = int(incident.get("severity", 0) or 0)
    recent_observer_alert = any(item.get("role") == "observer" for item in recent_alerts)
    pump_running = _is_active_flag(observation.get("RW_Pump_Sts"))
    planned_refill_response = "planned_refill_response" in relationship_notes

    process2_selected = "process2_selected" in relationship_notes
    process2_manual_dosing = "process2_manual_dosing_active" in relationship_notes
    process2_depleting = [flag for flag in risk_flags if flag.endswith("_depleting")]
    process2_containment_needed = (
        process2_manual_dosing
        or bool(process2_depleting)
        or (
            process2_selected
            and recent_observer_alert
            and incident_type == "process2_manual_dosing"
        )
        or (
            incident_type == "process2_manual_dosing"
            and incident_status in {"confirmed", "containing"}
        )
    )

    raw_water_low = (
        "Alarm_RW_Tank_LL" in active_alarms
        or (
            not planned_refill_response
            and (
                "raw_water_protection_risk" in risk_flags
                or "raw_water_low_band" in risk_flags
            )
        )
    )
    raw_water_high = (
        "Alarm_RW_Tank_HH" in active_alarms
        or "overflow_risk" in risk_flags
        or "raw_water_high_high_band" in risk_flags
    )
    command_divergence = "command_status_divergence" in risk_flags
    multi_asset_process2 = len(process2_depleting) >= 2

    if allow_process_selection and process2_containment_needed:
        summary = (
            "Deselect Process 2 now to isolate the manual dosing path because "
            "manual valve activity or tank depletion indicates active unauthorized "
            "chemical movement. This is the least disruptive containment."
        )
        add(
            summary,
            {"process_id": "WT-P2", "type": "select_process", "active": False},
            source="action_space",
            priority=95 if process2_manual_dosing or process2_depleting else 80,
            guardrail_reason="process2_containment",
        )

    if raw_water_low:
        summary = (
            "Issue a controlled stop because raw-water protection is degraded while "
            "WT-P1 is running. A stop halts flow cleanly before low-level faults or "
            "spoofed permissives can propagate."
        )
        add(
            summary,
            {"type": "hmi_stop", "process_id": "WT-P1"},
            source="action_space",
            priority=88 if pump_running else 76,
            guardrail_reason="raw_water_low_controlled_stop",
        )

    if raw_water_high:
        stop_summary = (
            "Issue a controlled stop because high-high raw-water evidence is present. "
            "If shutdown does not arrest the condition immediately, escalate further."
        )
        add(
            stop_summary,
            {"type": "hmi_stop", "process_id": "WT-P1"},
            source="action_space",
            priority=90 if pump_running else 78,
            guardrail_reason="raw_water_high_controlled_stop",
        )
        if incident_severity >= 3 or (pump_running and command_divergence):
            estop_summary = (
                "Escalate to estop because high-high raw-water risk is immediate and "
                "the plant is still driving the unsafe condition. This is justified "
                "only when delay is unacceptable."
            )
            add(
                estop_summary,
                {"type": "hmi_estop", "process_id": "WT-P1"},
                source="action_space",
                priority=98,
                guardrail_reason="raw_water_immediate_estop",
            )

    if process2_containment_needed and (multi_asset_process2 or incident_status == "confirmed"):
        summary = (
            "Use a controlled stop if Process 2 isolation alone is not enough because "
            "multiple chemical paths or sustained evidence suggest a wider process upset."
        )
        add(
            summary,
            {"type": "hmi_stop", "process_id": "WT-P1"},
            source="action_space",
            priority=70 if multi_asset_process2 else 60,
            guardrail_reason="process2_controlled_stop",
        )


def _is_active_flag(value: Any) -> bool:
    numeric = _coerce_number(value)
    if numeric is not None:
        return numeric >= 1.0
    return str(value).strip().lower() in {"true", "on", "active", "running"}


def _available_chemicals(observation: dict[str, Any]) -> list[str]:
    chemicals: list[str] = []
    for chemical in ("NaCl", "NaOCl", "HCl"):
        level_field = f"ChemTreat_{chemical}_Level"
        if level_field in observation:
            chemicals.append(chemical)
    return chemicals or ["NaCl", "NaOCl", "HCl"]


def _sensor_profiles(
    observation: dict[str, Any],
    safety_bounds: dict[str, Any],
    process_id: str | None,
) -> list[tuple[str, dict[str, int]]]:
    candidates: list[tuple[str, dict[str, int]]] = []
    for field, bounds in safety_bounds.items():
        if field not in observation:
            continue
        if process_id == "WT-P2" and not field.startswith("ChemTreat_"):
            continue
        if process_id == "WT-P1" and field.startswith("ChemTreat_"):
            continue

        current = _coerce_number(observation.get(field))
        minimum = _coerce_number(bounds.get("min"))
        maximum = _coerce_number(bounds.get("max"))
        if current is None or minimum is None or maximum is None or maximum <= minimum:
            continue

        span = maximum - minimum
        low_value = int(round(minimum + (span * 0.33)))
        high_value = int(round(minimum + (span * 0.99)))
        if low_value == high_value:
            continue
        candidates.append((field, {"low": low_value, "high": high_value}))
    return candidates


def _coerce_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
