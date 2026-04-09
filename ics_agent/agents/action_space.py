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
    remaining_playbook: list[dict[str, Any]],
    now: float,
    playbook_anticipation_sec: float,
) -> list[dict[str, Any]]:
    allowed = set(allowed_action_types)
    process_id = scenario.execution.get("process_id")
    safety_bounds = scenario.metadata.get("safety_bounds", {})
    candidates: list[dict[str, Any]] = []

    def add(summary: str, action: dict[str, Any] | str | None, *, source: str) -> None:
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
    if "select_process" in allowed and allow_process_selection:
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

    if "noop" in allowed:
        add(
            "Hold position and gather more evidence before acting.",
            {"type": "noop"},
            source="action_space",
        )

    return candidates


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
