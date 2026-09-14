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
    recent_actions: list[dict[str, Any]] | None = None,
    incident: dict[str, Any] | None = None,
    remaining_playbook: list[dict[str, Any]],
    now: float,
    playbook_anticipation_sec: float,
    defender_process_only: bool = False,
) -> list[dict[str, Any]]:
    allowed = set(allowed_action_types)
    testbed_id = _scenario_testbed_id(scenario)
    primary_process_id = _primary_process_id(testbed_id)
    process_id = scenario.execution.get("process_id") or primary_process_id
    safety_bounds = scenario.metadata.get("safety_bounds", {})
    risk_flags = list(risk_flags or [])
    active_alarms = list(active_alarms or [])
    relationship_notes = list(relationship_notes or [])
    recent_alerts = list(recent_alerts or [])
    recent_actions = list(recent_actions or [])
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
                "Start the primary process from the HMI if the plant should begin operating.",
                {"type": "hmi_start", "process_id": primary_process_id},
                source="action_space",
            )
        if "hmi_stop" in allowed:
            add(
                "Stop the primary process from the HMI to halt plant operation in a controlled way.",
                {"type": "hmi_stop", "process_id": primary_process_id},
                source="action_space",
            )
        if "hmi_reset" in allowed:
            add(
                "Reset the primary process from the HMI after a stop or fault condition.",
                {"type": "hmi_reset", "process_id": primary_process_id},
                source="action_space",
            )
        if "hmi_estop" in allowed:
            add(
                "Emergency stop the primary process immediately to prioritize safety over continuity.",
                {"type": "hmi_estop", "process_id": primary_process_id},
                source="action_space",
            )

    if testbed_id == "water_distribution":
        _add_water_distribution_candidates(
            add=add,
            role=role,
            allowed=allowed,
            process_id=process_id,
            primary_process_id=primary_process_id,
            observation=observation,
            safety_bounds=safety_bounds,
            risk_flags=risk_flags,
            active_alarms=active_alarms,
            relationship_notes=relationship_notes,
            recent_alerts=recent_alerts,
            recent_actions=recent_actions,
            incident=incident,
            defender_process_only=defender_process_only,
        )
        if "noop" in allowed:
            add(
                "Hold position and gather one more sample before changing the distribution process.",
                {"type": "noop"},
                source="action_space",
            )
        return candidates

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
            recent_actions=recent_actions,
            incident=incident,
            process_only=defender_process_only,
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
                "alarms or process-state changes appear."
                if defender_process_only
                else "Hold position while the incident stabilizes and verify that no fresh "
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
    recent_actions: list[dict[str, Any]],
    incident: dict[str, Any],
    process_only: bool = False,
) -> None:
    incident_type = str(incident.get("type", "none"))
    incident_status = str(incident.get("status", "none"))
    incident_severity = int(incident.get("severity", 0) or 0)
    recent_observer_alert = any(item.get("role") == "observer" for item in recent_alerts)
    pump_running = _is_active_flag(observation.get("RW_Pump_Sts"))
    planned_refill_response = "planned_refill_response" in relationship_notes
    recent_attacker_actions = [] if process_only else [
        normalize_action(item.get("action"))
        for item in recent_actions
        if item.get("role") == "attacker"
    ]
    recent_attacker_action_types = {
        str(action.get("type", ""))
        for action in recent_attacker_actions
        if action is not None
    }
    recent_alert_diagnoses = {
        str(item.get("diagnosis", "")).strip().lower()
        for item in recent_alerts
        if item.get("diagnosis")
    }
    recent_alert_text = " ".join(
        str(item.get("summary", "")).strip().lower() for item in recent_alerts
    )

    process2_selected = "process2_selected" in relationship_notes
    process2_manual_dosing = "process2_manual_dosing_active" in relationship_notes
    process2_depleting = [flag for flag in risk_flags if flag.endswith("_depleting")]
    process2_containment_needed = (
        process2_manual_dosing
        or bool(process2_depleting)
        or bool(
            recent_attacker_action_types.intersection({"select_process", "set_manual_valve"})
        )
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
    sensor_write_pressure = (
        (
            not process_only
            and (
                "set_sensor" in recent_attacker_action_types
                or "sensor_spoof" in recent_alert_diagnoses
                or "spoof" in recent_alert_text
            )
        )
        or (
            process_only
            and (
                "sensor_or_process_jump" in risk_flags
                or any(str(note).startswith("abrupt_change:") for note in relationship_notes)
            )
        )
    )
    hmi_surface_pressure = bool(
        recent_attacker_action_types.intersection({"hmi_start", "hmi_stop", "hmi_estop"})
    ) or (
        process_only
        and (
            "command_status_divergence" in risk_flags
            or "HMI_Stop_Active" in active_alarms
        )
    )
    command_divergence = "command_status_divergence" in risk_flags
    multi_asset_process2 = len(process2_depleting) >= 2

    if process2_containment_needed:
        summary = (
            "Block the Process 2 manual path now to cut off the suspicious dosing "
            "channel before escalating to a plant-wide stop."
        )
        add(
            summary,
            {"type": "block_process2_manual_path", "process_id": "WT-P2"},
            source="action_space",
            priority=99 if process2_manual_dosing or process2_depleting else 84,
            guardrail_reason="process2_symbolic_containment",
        )

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

    if sensor_write_pressure and incident_type in {
        "telemetry_anomaly",
        "raw_water_low_protection",
        "raw_water_overflow",
    }:
        add(
            "Block further sensor writes immediately to cut off the attacker write path "
            "before the spoof escalates into a plant-wide condition.",
            {"type": "block_sensor_write"},
            source="action_space",
            priority=102 if not active_alarms else 96,
            guardrail_reason="sensor_write_containment",
        )
        add(
            "Quarantine the suspicious signal source to isolate the compromised "
            "telemetry channel while preserving other plant functions.",
            {"type": "quarantine_signal_source"},
            source="action_space",
            priority=92 if not active_alarms else 88,
            guardrail_reason="signal_source_quarantine",
        )

    if hmi_surface_pressure and incident_type != "none":
        add(
            "Disable the suspicious HMI session first to cut off the current command "
            "channel before escalating to plant-wide shutdown.",
            {"type": "disable_hmi_session"},
            source="action_space",
            priority=100 if incident_severity >= 2 else 82,
            guardrail_reason="hmi_session_containment",
        )
        add(
            "Isolate the engineering host if the HMI misuse is broader than a single "
            "session or the attacker channel must be cut off completely.",
            {"type": "isolate_engineering_host"},
            source="action_space",
            priority=90 if incident_severity >= 3 else 74,
            guardrail_reason="engineering_host_isolation",
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


def _scenario_testbed_id(scenario: Any) -> str:
    return str(
        getattr(scenario, "testbed_id", None)
        or scenario.metadata.get("testbed_id", "water_treatment")
    ).strip()


def _primary_process_id(testbed_id: str) -> str:
    if testbed_id == "water_distribution":
        return "WD-UC0"
    return "WT-P1"


def _add_water_distribution_candidates(
    *,
    add,
    role: str,
    allowed: set[str],
    process_id: str | None,
    primary_process_id: str,
    observation: dict[str, Any],
    safety_bounds: dict[str, Any],
    risk_flags: list[str],
    active_alarms: list[str],
    relationship_notes: list[str],
    recent_alerts: list[dict[str, Any]],
    recent_actions: list[dict[str, Any]],
    incident: dict[str, Any],
    defender_process_only: bool = False,
) -> None:
    del relationship_notes, recent_alerts, incident
    process_id = process_id or primary_process_id
    if role != "defender" and "set_sensor" in allowed:
        for field, values in _water_distribution_sensor_profiles(
            observation=observation,
            safety_bounds=safety_bounds,
            process_id=process_id,
        ):
            add(
                f"Bias `{field}` low to mislead the controller about distribution state.",
                {"type": "set_sensor", "name": field, "value": values["low"]},
                source="action_space",
            )
            add(
                f"Bias `{field}` high to mislead the controller about distribution state.",
                {"type": "set_sensor", "name": field, "value": values["high"]},
                source="action_space",
            )

    if role != "defender" and "set_actuator" in allowed:
        for action in _water_distribution_actuator_actions(process_id):
            name = str(action.get("name", "actuator"))
            value = action.get("value")
            summary = f"Force `{name}` to `{value}` through the actuator write path."
            add(summary, action, source="action_space")

    if role == "defender":
        _add_water_distribution_defender_candidates(
            add=add,
            allowed=allowed,
            primary_process_id=primary_process_id,
            observation=observation,
            risk_flags=risk_flags,
            active_alarms=active_alarms,
            recent_actions=recent_actions,
            process_only=defender_process_only,
        )


def _water_distribution_sensor_profiles(
    *,
    observation: dict[str, Any],
    safety_bounds: dict[str, Any],
    process_id: str | None,
) -> list[tuple[str, dict[str, int]]]:
    candidates: list[tuple[str, dict[str, int]]] = []
    for field in _water_distribution_sensor_fields(process_id):
        bounds = safety_bounds.get(field)
        if field not in observation or not isinstance(bounds, dict):
            continue
        current = _coerce_number(observation.get(field))
        minimum = _coerce_number(bounds.get("min"))
        maximum = _coerce_number(bounds.get("max"))
        if current is None or minimum is None or maximum is None or maximum <= minimum:
            continue
        span = maximum - minimum
        low_value = int(round(minimum + (span * 0.18)))
        high_value = int(round(minimum + (span * 0.92)))
        if low_value == high_value:
            continue
        candidates.append((field, {"low": low_value, "high": high_value}))
    return candidates


def _water_distribution_sensor_fields(process_id: str | None) -> list[str]:
    mapping = {
        "WD-Supply": [
            "Supply_Tank_Level",
            "Supply_Pump_Flow",
        ],
        "WD-Grid": [
            "Grid_Elev_Tank_Level",
            "Grid_Consum_Tank_Level",
        ],
        "WD-RWS": [
            "RWS_Tank_Level",
        ],
    }
    if process_id in mapping:
        return mapping[process_id]
    return [
        "Supply_Tank_Level",
        "Grid_Elev_Tank_Level",
        "Grid_Consum_Tank_Level",
        "RWS_Tank_Level",
    ]


def _water_distribution_actuator_actions(process_id: str | None) -> list[dict[str, Any]]:
    process_actions = {
        "WD-Supply": [
            {"type": "set_actuator", "name": "Supply_Pump_Speed", "value": 0},
            {"type": "set_actuator", "name": "Supply_Pump_Speed", "value": 100},
            {"type": "set_actuator", "name": "Supply_Tank_Valve", "value": True},
            {"type": "set_actuator", "name": "Supply_Tank_Valve", "value": False},
        ],
        "WD-Grid": [
            {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
            {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": False},
        ],
        "WD-RWS": [
            {"type": "set_actuator", "name": "RWS_Pump_Speed", "value": 0},
            {"type": "set_actuator", "name": "RWS_Pump_Speed", "value": 100},
            {"type": "set_actuator", "name": "RWS_Tank_Valve", "value": True},
            {"type": "set_actuator", "name": "RWS_Tank_Valve", "value": False},
        ],
    }
    if process_id in process_actions:
        return process_actions[process_id]
    return [
        {"type": "set_actuator", "name": "Supply_Pump_Speed", "value": 100},
        {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
        {"type": "set_actuator", "name": "RWS_Pump_Speed", "value": 100},
        {"type": "set_actuator", "name": "RWS_Tank_Valve", "value": True},
    ]


def _add_water_distribution_defender_candidates(
    *,
    add,
    allowed: set[str],
    primary_process_id: str,
    observation: dict[str, Any],
    risk_flags: list[str],
    active_alarms: list[str],
    recent_actions: list[dict[str, Any]],
    process_only: bool = False,
) -> None:
    recent_attacker_action_types = set()
    if not process_only:
        recent_attacker_action_types = {
            str((item.get("action") or {}).get("type", ""))
            for item in recent_actions
            if item.get("role") == "attacker"
        }
    supply_level = _coerce_number(observation.get("Supply_Tank_Level"))
    grid_level = _coerce_number(observation.get("Grid_Elev_Tank_Level"))
    rws_level = _coerce_number(observation.get("RWS_Tank_Level"))
    pump_running = _is_active_flag(observation.get("Supply_Pump_Sts"))

    sensor_risk_flags = {"sensor_or_process_jump"}
    if not process_only:
        sensor_risk_flags.update({"supply_low_band", "supply_high_high_band"})
    sensor_pressure = "set_sensor" in recent_attacker_action_types or bool(
        set(risk_flags).intersection(sensor_risk_flags)
    )
    actuator_pressure = "set_actuator" in recent_attacker_action_types or (
        "command_status_divergence" in risk_flags
    )
    hmi_pressure = bool(
        recent_attacker_action_types.intersection({"hmi_start", "hmi_stop", "hmi_estop"})
    )
    distribution_extreme = bool(
        "Alarm_Supply_Tank_HH" in active_alarms
        or (supply_level is not None and (supply_level <= 320.0 or supply_level >= 1180.0))
        or (grid_level is not None and (grid_level <= 250.0 or grid_level >= 1850.0))
        or (rws_level is not None and (rws_level <= 120.0 or rws_level >= 1380.0))
    )

    if sensor_pressure and "block_sensor_write" in allowed:
        add(
            "Block further sensor writes now to preserve controller trust in distribution telemetry.",
            {"type": "block_sensor_write"},
            source="action_space",
            priority=98,
            guardrail_reason="distribution_sensor_containment",
        )
        add(
            "Quarantine the suspicious signal source before more deceptive telemetry reaches the controller.",
            {"type": "quarantine_signal_source"},
            source="action_space",
            priority=90,
            guardrail_reason="distribution_signal_quarantine",
        )

    if actuator_pressure and "block_actuator_write" in allowed:
        add(
            "Block actuator writes to cut off unauthorized pump or valve forcing on the distribution process.",
            {"type": "block_actuator_write"},
            source="action_space",
            priority=101,
            guardrail_reason="distribution_actuator_containment",
        )
        add(
            "Isolate the engineering host if actuator misuse suggests a broader control-path compromise.",
            {"type": "isolate_engineering_host"},
            source="action_space",
            priority=89,
            guardrail_reason="distribution_engineering_isolation",
        )

    if hmi_pressure and "disable_hmi_session" in allowed:
        add(
            "Disable the suspicious HMI session to stop further unauthorized start or stop commands.",
            {"type": "disable_hmi_session"},
            source="action_space",
            priority=94,
            guardrail_reason="distribution_hmi_session_containment",
        )

    if distribution_extreme and "hmi_stop" in allowed:
        add(
            "Issue a controlled stop because supply, grid, or return-water evidence indicates unsafe distribution behavior.",
            {"type": "hmi_stop", "process_id": primary_process_id},
            source="action_space",
            priority=88 if pump_running else 78,
            guardrail_reason="distribution_controlled_stop",
        )
    if (
        distribution_extreme
        and "hmi_estop" in allowed
        and ("Alarm_Supply_Tank_HH" in active_alarms or actuator_pressure)
    ):
        add(
            "Escalate to estop because the distribution process is still driving an unsafe condition.",
            {"type": "hmi_estop", "process_id": primary_process_id},
            source="action_space",
            priority=97,
            guardrail_reason="distribution_estop",
        )


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
