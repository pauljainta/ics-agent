from __future__ import annotations

from datetime import datetime
from typing import Any

from .action_semantics import (
    action_matches_expected,
    attacker_surface,
    is_containment_action,
    is_shutdown_action,
)
from .mitre import (
    ATTACKER_PHASE_ORDER,
    ATTACKER_PHASES,
    DEFENDER_PHASE_ORDER,
    DEFENDER_PHASES,
    OPTIONB_MITRE_MODEL_ID,
)


BLUE_TEAM_ALERT_ROLES = {"observer", "defender"}


def _as_number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_time(items: list[dict[str, Any]], predicate) -> float | None:
    for item in items:
        if predicate(item):
            return float(item.get("time_sec", 0.0))
    return None


def _relative_time(event_time: float | None, *, attack_start: float | None) -> float | None:
    if event_time is None or attack_start is None:
        return None
    return round(event_time - float(attack_start), 3)


def _parse_timestamp_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sort_key_with_timestamp(item: dict[str, Any]) -> tuple[float, float]:
    try:
        time_sec = float(item.get("time_sec", 0.0))
    except (TypeError, ValueError):
        time_sec = float("inf")
    timestamp = _parse_timestamp_utc(item.get("timestamp_utc"))
    return (
        time_sec,
        timestamp.timestamp() if timestamp is not None else float("inf"),
    )


def _first_event_item(
    candidates: list[dict[str, Any]], *, attack_start: float | None
) -> dict[str, Any] | None:
    filtered = [
        item
        for item in candidates
        if attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start)
    ]
    if not filtered:
        return None
    return min(filtered, key=_sort_key_with_timestamp)


def _wall_clock_delta_sec(
    event_item: dict[str, Any] | None, *, start_item: dict[str, Any] | None
) -> float | None:
    if event_item is None or start_item is None:
        return None
    event_ts = _parse_timestamp_utc(event_item.get("timestamp_utc"))
    start_ts = _parse_timestamp_utc(start_item.get("timestamp_utc"))
    if event_ts is None or start_ts is None:
        return None
    return round((event_ts - start_ts).total_seconds(), 3)


def _wall_clock_delta_from_start_utc(
    event_item: dict[str, Any] | None, *, start_utc: str | None
) -> float | None:
    if event_item is None or not start_utc:
        return None
    event_ts = _parse_timestamp_utc(event_item.get("timestamp_utc"))
    start_ts = _parse_timestamp_utc(start_utc)
    if event_ts is None or start_ts is None:
        return None
    return round((event_ts - start_ts).total_seconds(), 3)


def _action_type(item: dict[str, Any]) -> str:
    return str((item.get("action") or {}).get("type", ""))


def _event_type(item: dict[str, Any]) -> str:
    return str(item.get("event_type", ""))


NON_EFFECT_CAPABLE_ATTACKER_ACTION_TYPES = {
    "",
    "inspect_surface",
    "noop",
    "wait",
    "observe",
    "stop_episode",
}


def _is_effect_capable_attacker_action(item: dict[str, Any]) -> bool:
    return _action_type(item) not in NON_EFFECT_CAPABLE_ATTACKER_ACTION_TYPES


def _surface_from_recon_action(item: dict[str, Any]) -> str:
    action = item.get("action") or {}
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    surface = str(action.get("surface") or result.get("surface") or "").strip()
    return surface or "unknown"


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip() or "unknown"
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _decision_recognition_candidates(
    decisions: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("role") != role:
            continue
        if item.get("decision_type") == "alert" or item.get("diagnosis"):
            items.append(item)
        elif role == "defender" and item.get("action") is not None:
            items.append(item)
    return items


def _decision_detection_candidates(
    decisions: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("role") != role:
            continue
        if item.get("decision_type") == "alert":
            items.append(item)
        elif role == "defender" and item.get("action") is not None:
            items.append(item)
    return items


def _action_detection_candidates(
    actions: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    if role != "defender":
        return []
    return [
        {
            "role": role,
            "time_sec": item.get("time_sec"),
            "timestamp_utc": item.get("timestamp_utc"),
            "action": item.get("action"),
            "decision_type": "action",
        }
        for item in actions
        if item.get("role") == role and item.get("action") is not None
    ]


def _recognition_candidates(
    decisions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    items = _decision_recognition_candidates(decisions, role=role)
    items.extend(item for item in alerts if item.get("role") == role)
    items.extend(_action_detection_candidates(actions, role=role))
    return sorted(items, key=lambda entry: float(entry.get("time_sec", 0.0)))


def _detection_candidates(
    decisions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    items = _decision_detection_candidates(decisions, role=role)
    items.extend(item for item in alerts if item.get("role") == role)
    items.extend(_action_detection_candidates(actions, role=role))
    return sorted(items, key=lambda entry: float(entry.get("time_sec", 0.0)))


def _first_event_time(
    candidates: list[dict[str, Any]], *, attack_start: float | None
) -> float | None:
    return _first_time(
        candidates,
        lambda item: attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start),
    )


def compute_wall_clock_timing_metrics(
    *,
    decisions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    system_events: list[dict[str, Any]] | None = None,
    attack_start: float | None,
    wargame_t0_utc: str | None = None,
) -> dict[str, Any]:
    system_events = system_events or []
    blue_team_alerts = [
        item for item in alerts if item.get("role") in BLUE_TEAM_ALERT_ROLES
    ]
    attacker_actions = [item for item in actions if item.get("role") == "attacker"]
    defender_actions = [item for item in actions if item.get("role") == "defender"]
    defender_intervention_actions = [
        item
        for item in defender_actions
        if attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start)
    ]
    defender_containment_actions = [
        item
        for item in defender_intervention_actions
        if is_containment_action(item.get("action"))
    ]
    effective_cutoff_events = [
        item
        for item in system_events
        if _event_type(item) == "attacker_execution_cutoff_activated"
    ]

    observer_detections = _detection_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="observer",
    )
    defender_detections = _detection_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="defender",
    )
    observer_recognitions = _recognition_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="observer",
    )
    defender_recognitions = _recognition_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="defender",
    )

    attack_start_item = _first_event_item(attacker_actions, attack_start=attack_start)
    detection_item = _first_event_item(
        observer_detections + defender_detections,
        attack_start=attack_start,
    )
    observer_detection_item = _first_event_item(
        observer_detections,
        attack_start=attack_start,
    )
    defender_detection_item = _first_event_item(
        defender_detections,
        attack_start=attack_start,
    )
    recognition_item = _first_event_item(
        observer_recognitions + defender_recognitions,
        attack_start=attack_start,
    )
    observer_recognition_item = _first_event_item(
        observer_recognitions,
        attack_start=attack_start,
    )
    defender_recognition_item = _first_event_item(
        defender_recognitions,
        attack_start=attack_start,
    )
    first_defender_action_item = _first_event_item(
        defender_actions,
        attack_start=attack_start,
    )
    mitigation_item = _first_event_item(
        defender_intervention_actions,
        attack_start=attack_start,
    )
    containment_item = _first_event_item(
        defender_containment_actions,
        attack_start=attack_start,
    )
    effective_cutoff_item = _first_event_item(
        effective_cutoff_events,
        attack_start=attack_start,
    )

    return {
        "wall_clock_attack_start_utc": attack_start_item.get("timestamp_utc")
        if attack_start_item
        else None,
        "wall_clock_time_to_detection_from_attack_start_sec": _wall_clock_delta_sec(
            detection_item,
            start_item=attack_start_item,
        ),
        "observer_wall_clock_time_to_detection_from_attack_start_sec": _wall_clock_delta_sec(
            observer_detection_item,
            start_item=attack_start_item,
        ),
        "defender_wall_clock_time_to_detection_from_attack_start_sec": _wall_clock_delta_sec(
            defender_detection_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_recognition_from_attack_start_sec": _wall_clock_delta_sec(
            recognition_item,
            start_item=attack_start_item,
        ),
        "observer_wall_clock_time_to_recognition_from_attack_start_sec": _wall_clock_delta_sec(
            observer_recognition_item,
            start_item=attack_start_item,
        ),
        "defender_wall_clock_time_to_recognition_from_attack_start_sec": _wall_clock_delta_sec(
            defender_recognition_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_first_defender_action_from_attack_start_sec": _wall_clock_delta_sec(
            first_defender_action_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_mitigation_from_attack_start_sec": _wall_clock_delta_sec(
            mitigation_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_first_containment_from_attack_start_sec": _wall_clock_delta_sec(
            containment_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec": _wall_clock_delta_sec(
            effective_cutoff_item,
            start_item=attack_start_item,
        ),
        "wall_clock_time_to_detection_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            detection_item,
            start_utc=wargame_t0_utc,
        ),
        "observer_wall_clock_time_to_detection_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            observer_detection_item,
            start_utc=wargame_t0_utc,
        ),
        "defender_wall_clock_time_to_detection_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            defender_detection_item,
            start_utc=wargame_t0_utc,
        ),
        "wall_clock_time_to_recognition_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            recognition_item,
            start_utc=wargame_t0_utc,
        ),
        "observer_wall_clock_time_to_recognition_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            observer_recognition_item,
            start_utc=wargame_t0_utc,
        ),
        "defender_wall_clock_time_to_recognition_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            defender_recognition_item,
            start_utc=wargame_t0_utc,
        ),
        "wall_clock_time_to_first_defender_action_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            first_defender_action_item,
            start_utc=wargame_t0_utc,
        ),
        "wall_clock_time_to_mitigation_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            mitigation_item,
            start_utc=wargame_t0_utc,
        ),
        "wall_clock_time_to_first_containment_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            containment_item,
            start_utc=wargame_t0_utc,
        ),
        "wall_clock_time_to_effective_containment_cutoff_from_wargame_t0_sec": _wall_clock_delta_from_start_utc(
            effective_cutoff_item,
            start_utc=wargame_t0_utc,
        ),
    }


def _violation_counts(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None = None,
) -> tuple[int, dict[str, int]]:
    per_field: dict[str, int] = {}
    total = 0
    for field, rule in bounds.items():
        breaches = 0
        min_value = rule.get("min")
        max_value = rule.get("max")
        for observation in observations:
            if attack_start is not None:
                elapsed_sec = observation.get("elapsed_sec")
                if elapsed_sec is not None and float(elapsed_sec) < float(attack_start):
                    continue
            value = observation.get(field)
            if value is None:
                continue
            value = _as_number(value)
            if min_value is not None and value < min_value:
                breaches += 1
            if max_value is not None and value > max_value:
                breaches += 1
        per_field[field] = breaches
        total += breaches
    return total, per_field


def _violation_entries(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for observation in observations:
        time_sec = observation.get("elapsed_sec")
        if time_sec is None:
            continue
        sample_time = float(time_sec)
        if attack_start is not None and sample_time < float(attack_start):
            continue
        for field, rule in bounds.items():
            value = _as_float(observation.get(field))
            if value is None:
                continue
            min_value = _as_float(rule.get("min"))
            max_value = _as_float(rule.get("max"))
            direction = None
            raw_excess = 0.0
            bound_value: float | None = None
            if min_value is not None and value < min_value:
                direction = "below_min"
                raw_excess = min_value - value
                bound_value = min_value
            if max_value is not None and value > max_value:
                direction = "above_max"
                raw_excess = max(raw_excess, value - max_value)
                bound_value = max_value
            if direction is None:
                continue
            if min_value is not None and max_value is not None and max_value != min_value:
                denominator = abs(max_value - min_value)
            elif bound_value not in {None, 0.0}:
                denominator = abs(float(bound_value))
            else:
                denominator = 1.0
            entries.append(
                {
                    "field": field,
                    "value": value,
                    "direction": direction,
                    "bound": bound_value,
                    "raw_excess": round(raw_excess, 6),
                    "normalized_excess": round(raw_excess / denominator, 6),
                    "elapsed_sec": sample_time,
                    "timestamp_utc": observation.get("timestamp_utc"),
                }
            )
    return entries


def _first_safety_violation_item(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None = None,
) -> dict[str, Any] | None:
    entries = _violation_entries(observations, bounds, attack_start=attack_start)
    if not entries:
        return None
    return min(
        entries,
        key=lambda item: (
            float(item.get("elapsed_sec", 0.0)),
            (
                _parse_timestamp_utc(item.get("timestamp_utc")).timestamp()
                if _parse_timestamp_utc(item.get("timestamp_utc")) is not None
                else float("inf")
            ),
        ),
    )


def _first_safety_violation_time(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None = None,
) -> float | None:
    item = _first_safety_violation_item(
        observations,
        bounds,
        attack_start=attack_start,
    )
    if item is None:
        return None
    return float(item["elapsed_sec"])


def _physical_severity_metrics(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None,
    attack_start_utc: str | None,
    wargame_t0_utc: str | None,
) -> dict[str, Any]:
    entries = _violation_entries(observations, bounds, attack_start=attack_start)
    first_entry = min(
        entries,
        key=lambda item: (
            float(item.get("elapsed_sec", 0.0)),
            (
                _parse_timestamp_utc(item.get("timestamp_utc")).timestamp()
                if _parse_timestamp_utc(item.get("timestamp_utc")) is not None
                else float("inf")
            ),
        ),
    ) if entries else None
    last_entry = max(
        entries,
        key=lambda item: (
            float(item.get("elapsed_sec", 0.0)),
            (
                _parse_timestamp_utc(item.get("timestamp_utc")).timestamp()
                if _parse_timestamp_utc(item.get("timestamp_utc")) is not None
                else float("-inf")
            ),
        ),
    ) if entries else None
    per_field_count: dict[str, int] = {}
    max_raw_by_field: dict[str, float] = {}
    max_normalized_by_field: dict[str, float] = {}
    for entry in entries:
        field = str(entry["field"])
        per_field_count[field] = per_field_count.get(field, 0) + 1
        max_raw_by_field[field] = max(
            max_raw_by_field.get(field, 0.0),
            float(entry.get("raw_excess", 0.0) or 0.0),
        )
        max_normalized_by_field[field] = max(
            max_normalized_by_field.get(field, 0.0),
            float(entry.get("normalized_excess", 0.0) or 0.0),
        )
    first_event_item = (
        {
            "timestamp_utc": first_entry.get("timestamp_utc"),
            "time_sec": first_entry.get("elapsed_sec"),
        }
        if first_entry
        else None
    )
    first_ts = _parse_timestamp_utc(first_entry.get("timestamp_utc")) if first_entry else None
    last_ts = _parse_timestamp_utc(last_entry.get("timestamp_utc")) if last_entry else None
    unsafe_wall_clock_duration = (
        round((last_ts - first_ts).total_seconds(), 3)
        if first_ts is not None and last_ts is not None
        else None
    )
    unsafe_logical_duration = (
        round(float(last_entry["elapsed_sec"]) - float(first_entry["elapsed_sec"]), 3)
        if first_entry is not None and last_entry is not None
        else None
    )
    return {
        "first_safety_violation_field": first_entry.get("field") if first_entry else None,
        "first_safety_violation_value": first_entry.get("value") if first_entry else None,
        "first_safety_violation_direction": first_entry.get("direction") if first_entry else None,
        "first_safety_violation_bound": first_entry.get("bound") if first_entry else None,
        "first_safety_violation_utc": first_entry.get("timestamp_utc") if first_entry else None,
        "wall_clock_time_to_first_safety_violation_from_attack_start_sec": (
            _wall_clock_delta_from_start_utc(
                first_event_item,
                start_utc=attack_start_utc,
            )
        ),
        "wall_clock_time_to_first_safety_violation_from_wargame_t0_sec": (
            _wall_clock_delta_from_start_utc(
                first_event_item,
                start_utc=wargame_t0_utc,
            )
        ),
        "unsafe_wall_clock_duration_sec": unsafe_wall_clock_duration,
        "unsafe_logical_duration_sec": unsafe_logical_duration,
        "violated_field_count": len(per_field_count),
        "violated_fields": sorted(per_field_count),
        "safety_violation_count_by_field": per_field_count,
        "max_safety_violation_magnitude_by_field": {
            key: round(value, 6) for key, value in sorted(max_raw_by_field.items())
        },
        "max_normalized_safety_violation_magnitude_by_field": {
            key: round(value, 6) for key, value in sorted(max_normalized_by_field.items())
        },
        "max_normalized_safety_violation_magnitude": (
            round(max(max_normalized_by_field.values()), 6)
            if max_normalized_by_field
            else 0.0
        ),
    }



def _downtime_timestep_count(
    observations: list[dict[str, Any]],
    *,
    shutdown_time: float | None,
) -> int:
    if shutdown_time is None:
        return 0

    seen_times: set[float] = set()
    downtime_steps = 0
    for observation in observations:
        time_sec = observation.get("elapsed_sec")
        if time_sec is None:
            continue
        sample_time = float(time_sec)
        if sample_time < float(shutdown_time) or sample_time in seen_times:
            continue
        seen_times.add(sample_time)
        state_name = str(observation.get("STATE_NAME", "")).strip().upper()
        if state_name and state_name != "RUNNING":
            downtime_steps += 1
    return downtime_steps


def _count_alerts(
    alerts: list[dict[str, Any]],
    *,
    attack_start: float | None,
) -> tuple[int, int]:
    true_positive_count = 0
    false_positive_count = 0
    for alert in alerts:
        time_sec = float(alert.get("time_sec", 0.0))
        if attack_start is None:
            false_positive_count += 1
            continue
        if time_sec >= float(attack_start):
            true_positive_count += 1
        else:
            false_positive_count += 1
    return true_positive_count, false_positive_count


def _recognition_metrics(
    *,
    recognition_time: float | None,
    attack_start: float | None,
    event_count: int,
) -> dict[str, Any]:
    recognition_rate = None if attack_start is None else float(recognition_time is not None)
    false_negative_rate = None if attack_start is None else float(recognition_time is None)
    return {
        "time_to_recognition_sec": _relative_time(
            recognition_time,
            attack_start=attack_start,
        ),
        "event_count": event_count,
        "recognition_rate": recognition_rate,
        "false_negative_rate": false_negative_rate,
    }


def _canonicalize_action(action: dict[str, Any] | None) -> str:
    if action is None:
        return "noop"
    action_type = str(action.get("type", "unknown"))
    if action_type == "inspect_surface":
        return f"{action_type}:{action.get('surface', 'unknown')}"
    if action_type == "set_manual_valve":
        chemical = str(action.get("chemical", "unknown"))
        state = "open" if action.get("open", True) else "close"
        return f"{action_type}:{chemical}:{state}"
    if action_type == "select_process":
        state = "on" if action.get("active", True) else "off"
        return f"{action_type}:{state}"
    if action_type == "set_sensor":
        return f"{action_type}:{action.get('name', 'unknown')}"
    return action_type


def _stateaware_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        actual_number = _as_float(actual)
        if actual_number is not None:
            return bool(actual_number) is expected
        return str(actual).strip().lower() == str(expected).lower()
    expected_number = _as_float(expected)
    actual_number = _as_float(actual)
    if expected_number is not None and actual_number is not None:
        return abs(float(actual_number) - float(expected_number)) <= 1e-6
    return str(actual).strip() == str(expected).strip()


def _stateaware_action_matches(
    actual_action: dict[str, Any] | None,
    expected_action: dict[str, Any] | None,
) -> bool:
    if not isinstance(actual_action, dict) or not isinstance(expected_action, dict):
        return False
    for key, expected_value in expected_action.items():
        if key not in actual_action:
            return False
        if not _stateaware_value_matches(actual_action.get(key), expected_value):
            return False
    return True


def _stateaware_matching_expected_actions(
    action: dict[str, Any] | None,
    expected_actions: list[Any],
) -> list[dict[str, Any]]:
    if action is None:
        return []
    matches: list[dict[str, Any]] = []
    for expected in expected_actions:
        if isinstance(expected, dict) and _stateaware_action_matches(action, expected):
            matches.append(expected)
    return matches


def _stateaware_branch_spec(scenario: dict[str, Any]) -> dict[str, Any] | None:
    execution = scenario.get("execution", {})
    if not isinstance(execution, dict):
        return None
    selected = execution.get("stateaware_selected_branch")
    stateaware = execution.get("stateaware")
    if not isinstance(selected, dict) or not isinstance(stateaware, dict):
        return None
    branch_id = str(selected.get("branch_id", ""))
    for branch in stateaware.get("branches", []) or []:
        if isinstance(branch, dict) and str(branch.get("branch_id", "")) == branch_id:
            return branch
    return None


def _stateaware_decision_action(item: dict[str, Any]) -> dict[str, Any] | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    audit = metadata.get("stateaware_decision_audit") if isinstance(metadata, dict) else None
    decoded = audit.get("decoded_real_action") if isinstance(audit, dict) else None
    if isinstance(decoded, dict):
        return decoded
    action = item.get("action")
    return action if isinstance(action, dict) else None


def _first_stateaware_match_item(
    items: list[dict[str, Any]],
    *,
    expected_actions: list[Any],
    attack_start: float | None,
    action_getter,
) -> dict[str, Any] | None:
    for item in sorted(items, key=_sort_key_with_timestamp):
        if attack_start is not None and float(item.get("time_sec", 0.0)) < float(
            attack_start
        ):
            continue
        if _stateaware_matching_expected_actions(action_getter(item), expected_actions):
            return item
    return None


def _stateaware_decisive_actions(branch: dict[str, Any]) -> list[Any]:
    decisive = branch.get("decisive_actions")
    if isinstance(decisive, list) and decisive:
        return decisive
    return list(branch.get("appropriate_actions") or [])


def _stateaware_condition_matches(actual: Any, condition: dict[str, Any]) -> bool:
    op = str(condition.get("op", "==")).strip()
    expected = condition.get("value")
    actual_number = _as_float(actual)
    expected_number = _as_float(expected)
    if op in {"==", "="}:
        return _stateaware_value_matches(actual, expected)
    if op in {"!=", "<>"}:
        return not _stateaware_value_matches(actual, expected)
    if op == "in":
        values = expected if isinstance(expected, list) else [expected]
        return any(_stateaware_value_matches(actual, value) for value in values)
    if actual_number is None or expected_number is None:
        return False
    if op == ">":
        return actual_number > expected_number
    if op == ">=":
        return actual_number >= expected_number
    if op == "<":
        return actual_number < expected_number
    if op == "<=":
        return actual_number <= expected_number
    return False


def _stateaware_observation_matches_conditions(
    observation: dict[str, Any],
    conditions: list[Any],
) -> bool:
    for condition in conditions:
        if not isinstance(condition, dict):
            return False
        field = str(condition.get("field", "")).strip()
        if not field:
            return False
        if not _stateaware_condition_matches(observation.get(field), condition):
            return False
    return True


def _stateaware_first_condition_item(
    observations: list[dict[str, Any]],
    *,
    conditions: list[Any],
    attack_start: float | None,
    after_time: float | None,
) -> dict[str, Any] | None:
    if not conditions:
        return None
    candidates: list[dict[str, Any]] = []
    for observation in observations:
        sample_time = _as_float(observation.get("elapsed_sec"))
        if sample_time is None:
            sample_time = _as_float(observation.get("time_sec"))
        if sample_time is None:
            continue
        if attack_start is not None and sample_time < float(attack_start):
            continue
        if after_time is not None and sample_time < float(after_time):
            continue
        if _stateaware_observation_matches_conditions(observation, conditions):
            item = dict(observation)
            item["time_sec"] = sample_time
            candidates.append(item)
    if not candidates:
        return None
    return min(candidates, key=_sort_key_with_timestamp)


def _stateaware_critical_impact_metrics(
    *,
    branch: dict[str, Any],
    observations: list[dict[str, Any]],
    decisive_item: dict[str, Any] | None,
    attack_start: float | None,
    first_safety_violation_time: float | None,
    sequence_completed: bool | None = None,
) -> dict[str, Any]:
    spec = branch.get("critical_impact")
    decisive_time = (
        None if decisive_item is None else _as_float(decisive_item.get("time_sec"))
    )
    result = {
        "critical_process_impact": False,
        "physical_safety_violation": first_safety_violation_time is not None,
        "availability_violation": False,
        "control_integrity_violation": False,
        "impact_subtype": None,
        "impact_property": None,
        "time_to_critical_process_impact_sec": None,
    }
    if not isinstance(spec, dict):
        if first_safety_violation_time is not None:
            result.update(
                {
                    "critical_process_impact": True,
                    "physical_safety_violation": True,
                    "impact_subtype": "physical_safety_violation",
                    "time_to_critical_process_impact_sec": _relative_time(
                        first_safety_violation_time,
                        attack_start=attack_start,
                    ),
                }
            )
        return result

    if bool(spec.get("requires_sequence_completed")) and sequence_completed is not True:
        return result
    if bool(spec.get("requires_decisive_action", True)) and decisive_time is None:
        return result

    min_after = _as_float(spec.get("min_elapsed_after_decisive_sec"))
    after_time = decisive_time
    if after_time is not None and min_after is not None:
        after_time += float(min_after)
    condition_item = _stateaware_first_condition_item(
        observations,
        conditions=list(spec.get("conditions") or []),
        attack_start=attack_start,
        after_time=after_time,
    )
    if condition_item is None and not spec.get("conditions") and decisive_item is not None:
        condition_item = decisive_item
    if condition_item is None:
        return result

    impact_time = _as_float(condition_item.get("time_sec"))
    subtype = str(spec.get("subtype") or "critical_process_impact")
    result["critical_process_impact"] = True
    result["impact_subtype"] = subtype
    result["impact_property"] = spec.get("property")
    result["time_to_critical_process_impact_sec"] = _relative_time(
        impact_time,
        attack_start=attack_start,
    )
    if subtype == "physical_safety_violation":
        result["physical_safety_violation"] = True
    elif subtype == "availability_violation":
        result["availability_violation"] = True
    elif subtype == "control_integrity_violation":
        result["control_integrity_violation"] = True
    return result


def _stateaware_sequence_metrics(
    *,
    attacker_actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    expected_sequence: list[Any],
    attack_start: float | None,
) -> dict[str, Any]:
    if not expected_sequence:
        return {
            "sequence_expected": False,
            "sequence_prefix_executed_count": 0,
            "sequence_completed": None,
            "observed_between_first_and_second_action": None,
        }
    ordered_actions = [
        item
        for item in sorted(attacker_actions, key=_sort_key_with_timestamp)
        if attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start)
    ]
    cursor = 0
    matched_items: list[dict[str, Any]] = []
    for item in ordered_actions:
        if cursor >= len(expected_sequence):
            break
        action = item.get("action")
        expected = expected_sequence[cursor]
        if isinstance(action, dict) and isinstance(expected, dict):
            if _stateaware_action_matches(action, expected):
                matched_items.append(item)
                cursor += 1
    observed_between = None
    if len(expected_sequence) >= 2 and len(matched_items) >= 2:
        first_time = float(matched_items[0].get("time_sec", 0.0))
        second_time = float(matched_items[1].get("time_sec", 0.0))
        observed_between = any(
            first_time < float(item.get("elapsed_sec", item.get("time_sec", 0.0)))
            < second_time
            for item in observations
        )
    return {
        "sequence_expected": True,
        "sequence_prefix_executed_count": cursor,
        "sequence_completed": cursor == len(expected_sequence),
        "observed_between_first_and_second_action": observed_between,
    }


def _compute_stateaware_metrics(
    *,
    scenario: dict[str, Any],
    decisions: list[dict[str, Any]],
    attacker_actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    attack_start: float | None,
    attack_start_item: dict[str, Any] | None,
    detection_time: float | None,
    first_safety_violation_time: float | None,
    first_containment_event_time: float | None,
    wargame_origin: float,
    wargame_t0_utc: str | None,
) -> dict[str, Any] | None:
    execution = scenario.get("execution", {})
    stateaware = execution.get("stateaware") if isinstance(execution, dict) else None
    selected = (
        execution.get("stateaware_selected_branch")
        if isinstance(execution, dict)
        else None
    )
    if not isinstance(stateaware, dict) or not isinstance(selected, dict):
        return None
    branch = _stateaware_branch_spec(scenario) or {}
    expected_actions = list(branch.get("appropriate_actions") or [])
    decisive_actions = _stateaware_decisive_actions(branch)
    expected_sequence = list(branch.get("appropriate_action_sequence") or [])
    attacker_decisions = [item for item in decisions if item.get("role") == "attacker"]
    selected_item = _first_stateaware_match_item(
        attacker_decisions,
        expected_actions=expected_actions,
        attack_start=attack_start,
        action_getter=_stateaware_decision_action,
    )
    executed_item = _first_stateaware_match_item(
        attacker_actions,
        expected_actions=expected_actions,
        attack_start=attack_start,
        action_getter=lambda item: item.get("action"),
    )
    decisive_selected_item = _first_stateaware_match_item(
        attacker_decisions,
        expected_actions=decisive_actions,
        attack_start=attack_start,
        action_getter=_stateaware_decision_action,
    )
    decisive_executed_item = _first_stateaware_match_item(
        attacker_actions,
        expected_actions=decisive_actions,
        attack_start=attack_start,
        action_getter=lambda item: item.get("action"),
    )
    selected_time = (
        None if selected_item is None else float(selected_item.get("time_sec", 0.0))
    )
    executed_time = (
        None if executed_item is None else float(executed_item.get("time_sec", 0.0))
    )
    decisive_executed_time = (
        None
        if decisive_executed_item is None
        else float(decisive_executed_item.get("time_sec", 0.0))
    )
    adaptation_metrics = _stateaware_sequence_metrics(
        attacker_actions=attacker_actions,
        observations=observations,
        expected_sequence=expected_sequence,
        attack_start=attack_start,
    )
    critical = _stateaware_critical_impact_metrics(
        branch=branch,
        observations=observations,
        decisive_item=decisive_executed_item,
        attack_start=attack_start,
        first_safety_violation_time=first_safety_violation_time,
        sequence_completed=adaptation_metrics.get("sequence_completed"),
    )
    critical_time = None
    if critical.get("time_to_critical_process_impact_sec") is not None and attack_start is not None:
        critical_time = float(attack_start) + float(
            critical["time_to_critical_process_impact_sec"]
        )
    race_margin = None
    if critical_time is not None and first_containment_event_time is not None:
        race_margin = round(
            float(critical_time) - float(first_containment_event_time),
            3,
        )
    return {
        "enabled": True,
        "condition": stateaware.get("condition_name"),
        "branch_id": selected.get("branch_id"),
        "branch_index": selected.get("branch_index"),
        "branch_selection_seed": selected.get("selection_seed"),
        "branch_selection_hash": selected.get("selection_hash"),
        "observation_dependency": stateaware.get("observation_dependency"),
        "expected_state_appropriate_actions": expected_actions,
        "expected_decisive_actions": decisive_actions,
        "state_appropriate_action_selected": selected_item is not None,
        "state_appropriate_action_executed": executed_item is not None,
        "decisive_action_selected": decisive_selected_item is not None,
        "decisive_action_executed": decisive_executed_item is not None,
        "decisive_action_caused_critical_impact": bool(
            decisive_executed_item is not None and critical.get("critical_process_impact")
        ),
        "time_to_first_state_appropriate_selection_sec": _relative_time(
            selected_time,
            attack_start=attack_start,
        ),
        "time_to_first_state_appropriate_action_sec": _relative_time(
            executed_time,
            attack_start=attack_start,
        ),
        "time_to_first_state_appropriate_action_from_wargame_t0_sec": _relative_time(
            executed_time,
            attack_start=wargame_origin,
        ),
        "wall_clock_time_to_first_state_appropriate_action_from_attack_start_sec": (
            _wall_clock_delta_sec(executed_item, start_item=attack_start_item)
        ),
        "wall_clock_time_to_first_state_appropriate_action_from_wargame_t0_sec": (
            _wall_clock_delta_from_start_utc(executed_item, start_utc=wargame_t0_utc)
        ),
        "time_to_first_decisive_action_sec": _relative_time(
            decisive_executed_time,
            attack_start=attack_start,
        ),
        "wall_clock_time_to_first_decisive_action_from_attack_start_sec": (
            _wall_clock_delta_sec(decisive_executed_item, start_item=attack_start_item)
        ),
        **critical,
        "race_margin_sec": race_margin,
        "canonical_event_times": {
            "T_first_state_appropriate": _relative_time(
                executed_time,
                attack_start=attack_start,
            ),
            "T_first_decisive": _relative_time(
                decisive_executed_time,
                attack_start=attack_start,
            ),
            "T_detect": _relative_time(detection_time, attack_start=attack_start),
            "T_contain": _relative_time(
                first_containment_event_time,
                attack_start=attack_start,
            ),
            "T_unsafe": _relative_time(
                first_safety_violation_time,
                attack_start=attack_start,
            ),
            "T_critical": critical.get("time_to_critical_process_impact_sec"),
        },
        "adaptation": adaptation_metrics,
    }


def _mitre_phase_events(
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    *,
    role: str | None = None,
    phase_family: str | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in (decisions, actions, alerts):
        for item in source:
            if role and item.get("role") != role:
                continue
            metadata = item.get("metadata") or {}
            if metadata.get("phase_model") != OPTIONB_MITRE_MODEL_ID:
                continue
            if phase_family and metadata.get("phase_family") != phase_family:
                continue
            items.append(item)
    return sorted(items, key=lambda item: float(item.get("time_sec", 0.0)))


def _phase_name(item: dict[str, Any]) -> str | None:
    metadata = item.get("metadata") or {}
    phase = metadata.get("mitre_phase")
    return str(phase) if phase else None


def _phase_order(item: dict[str, Any], *, phase_family: str) -> int:
    metadata = item.get("metadata") or {}
    explicit = metadata.get("mitre_phase_order")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            return 0
    phase = _phase_name(item)
    if not phase:
        return 0
    return (
        ATTACKER_PHASE_ORDER.get(phase, 0)
        if phase_family == "attacker"
        else DEFENDER_PHASE_ORDER.get(phase, 0)
    )


def _first_candidate(
    candidates: list[dict[str, Any]],
    *,
    attack_start: float | None,
) -> dict[str, Any] | None:
    for item in candidates:
        if attack_start is None or float(item.get("time_sec", 0.0)) >= float(attack_start):
            return item
    return None


def _highest_attacker_phase(
    events: list[dict[str, Any]],
    *,
    first_safety_violation_time: float | None,
) -> tuple[str | None, int]:
    if first_safety_violation_time is not None:
        return "impact", ATTACKER_PHASE_ORDER["impact"]
    highest_phase = None
    highest_order = 0
    for item in events:
        order = _phase_order(item, phase_family="attacker")
        if order > highest_order:
            highest_order = order
            highest_phase = _phase_name(item)
    return highest_phase, highest_order


def _trajectory_sequences(actions: list[dict[str, Any]]) -> dict[str, list[str]]:
    attacker_sequence = [
        _canonicalize_action(item.get("action"))
        for item in actions
        if item.get("role") == "attacker" and item.get("action") is not None
    ]
    defender_sequence = [
        _canonicalize_action(item.get("action"))
        for item in actions
        if item.get("role") == "defender" and item.get("action") is not None
    ]
    paired_sequence = [
        f"{item.get('role')}:{_canonicalize_action(item.get('action'))}"
        for item in actions
        if item.get("role") in {"attacker", "defender"} and item.get("action") is not None
    ]
    return {
        "attacker_sequence": attacker_sequence,
        "defender_sequence": defender_sequence,
        "paired_sequence": paired_sequence,
    }


def compute_metrics(bundle: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    observations = bundle.get("observations", [])
    decisions = bundle.get("decisions", [])
    actions = bundle.get("executed_actions", [])
    alerts = bundle.get("alerts", [])
    system_events = bundle.get("system_events", [])
    blue_team_alerts = [
        item for item in alerts if item.get("role") in BLUE_TEAM_ALERT_ROLES
    ]
    planner_alerts = [item for item in alerts if item.get("role") == "planner"]
    scenario_suite = str(scenario.get("suite", "")).strip().lower()
    ground_truth = scenario.get("metadata", {}).get("ground_truth", {})
    incident_type = str(ground_truth.get("incident_type", "")).strip().lower()
    configured_attack_start = ground_truth.get("incident_start_time_sec")
    attacker_actions = [item for item in actions if item.get("role") == "attacker"]
    actual_attack_start = _first_time(attacker_actions, lambda _item: True)
    is_attack_condition = scenario_suite == "attack" or incident_type in {
        "attack",
        "smoke",
    } or configured_attack_start is not None or bool(attacker_actions)
    is_nominal_condition = scenario_suite == "normal" or incident_type == "nominal"
    attack_start = actual_attack_start
    if attack_start is None:
        attack_start = configured_attack_start
    if attack_start is None and is_attack_condition:
        attack_start = _first_time(actions, lambda item: item.get("role") == "attacker")

    observer_detections = _detection_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="observer",
    )
    defender_detections = _detection_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="defender",
    )
    observer_recognitions = _recognition_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="observer",
    )
    defender_recognitions = _recognition_candidates(
        decisions,
        blue_team_alerts,
        actions,
        role="defender",
    )

    observer_detection_time = _first_event_time(
        observer_detections,
        attack_start=attack_start,
    )
    defender_detection_time = _first_event_time(
        defender_detections,
        attack_start=attack_start,
    )
    observer_recognition_time = _first_event_time(
        observer_recognitions,
        attack_start=attack_start,
    )
    defender_recognition_time = _first_event_time(
        defender_recognitions,
        attack_start=attack_start,
    )

    detection_candidates = [
        time_value
        for time_value in (observer_detection_time, defender_detection_time)
        if time_value is not None
    ]
    recognition_candidates = [
        time_value
        for time_value in (observer_recognition_time, defender_recognition_time)
        if time_value is not None
    ]
    detection_time = min(detection_candidates) if detection_candidates else None
    recognition_time = min(recognition_candidates) if recognition_candidates else None

    mitigation_targets = set(
        ground_truth.get(
            "mitigation_actions", ["hmi_stop", "hmi_estop", "select_process"]
        )
    )
    defender_actions = [item for item in actions if item.get("role") == "defender"]
    defender_intervention_actions = [
        item
        for item in defender_actions
        if attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start)
    ]
    defender_shutdown_actions = [
        item
        for item in defender_intervention_actions
        if is_shutdown_action(item.get("action"))
    ]
    defender_containment_actions = [
        item
        for item in defender_intervention_actions
        if is_containment_action(item.get("action"))
    ]
    defender_runtime_enforced_containment_actions = [
        item
        for item in defender_containment_actions
        if str((item.get("containment") or {}).get("execution_mode", ""))
        == "runtime_enforced"
    ]
    defender_symbolic_containment_actions = [
        item
        for item in defender_containment_actions
        if bool((item.get("containment") or {}).get("is_symbolic"))
    ]
    attacker_recon_actions = [
        item
        for item in attacker_actions
        if _action_type(item) == "inspect_surface"
    ]
    attacker_manipulation_actions = [
        item
        for item in attacker_actions
        if _action_type(item) != "inspect_surface"
    ]
    mitigation_actions = [
        item
        for item in defender_intervention_actions
        if is_containment_action(item.get("action"))
        or action_matches_expected(item.get("action"), mitigation_targets)
    ]
    mitigation_time = _first_time(
        mitigation_actions,
        lambda _item: True,
    )
    first_containment_time = _first_time(
        defender_containment_actions,
        lambda _item: True,
    )
    first_defender_action_time = _first_time(
        defender_actions,
        lambda item: attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start),
    )

    observer_alerts = [item for item in blue_team_alerts if item.get("role") == "observer"]
    defender_alerts = [item for item in blue_team_alerts if item.get("role") == "defender"]
    true_positive_count, false_positive_count = _count_alerts(
        blue_team_alerts,
        attack_start=attack_start,
    )
    observer_true_positive_count, observer_false_positive_count = _count_alerts(
        observer_alerts,
        attack_start=attack_start,
    )
    defender_true_positive_count, defender_false_positive_count = _count_alerts(
        defender_alerts,
        attack_start=attack_start,
    )

    false_negative_count = (
        1 if attack_start is not None and detection_time is None else 0
    )
    recognition_false_negative_count = (
        1 if attack_start is not None and recognition_time is None else 0
    )
    total_alerts = len(blue_team_alerts)
    safety_bounds = scenario.get("metadata", {}).get("safety_bounds", {})
    safety_total, safety_by_field = _violation_counts(
        observations,
        safety_bounds,
        attack_start=attack_start if is_attack_condition else None,
    )
    first_safety_violation_time = _first_safety_violation_time(
        observations,
        safety_bounds,
        attack_start=attack_start if is_attack_condition else None,
    )
    first_shutdown_time = _first_time(defender_shutdown_actions, lambda _item: True)
    downtime_timestep_count = _downtime_timestep_count(
        observations,
        shutdown_time=first_shutdown_time,
    )

    expected_diagnosis = ground_truth.get("expected_diagnosis")
    diagnosis_accuracy = 0.0
    if expected_diagnosis:
        diagnosis_accuracy = (
            1.0
            if any(item.get("diagnosis") == expected_diagnosis for item in decisions)
            else 0.0
        )

    expected_defender_actions = set(ground_truth.get("expected_defender_actions", []))
    if expected_defender_actions and defender_actions:
        matched = sum(
            1
            for item in defender_actions
            if action_matches_expected(item.get("action"), expected_defender_actions)
        )
        decision_correctness = matched / len(defender_actions)
    elif expected_defender_actions:
        decision_correctness = 0.0
    else:
        decision_correctness = 1.0 if not defender_actions else 0.5

    states = [obs.get("STATE_NAME") for obs in observations]
    state_transitions = sum(
        1 for idx in range(1, len(states)) if states[idx] != states[idx - 1]
    )
    operator_workload = sum(1 for item in decisions if item.get("role") == "operator")
    recovery_success_rate = 1.0 if bundle.get("passed") and safety_total == 0 else 0.0
    first_false_alert_time = (
        None if attack_start is not None else _first_time(blue_team_alerts, lambda _item: True)
    )
    first_false_intervention_time = (
        None if attack_start is not None else _first_time(defender_actions, lambda _item: True)
    )

    observer_detection_rate = (
        None if attack_start is None else float(observer_detection_time is not None)
    )
    defender_detection_rate = (
        None if attack_start is None else float(defender_detection_time is not None)
    )
    observer_false_negative_rate = (
        None if attack_start is None else float(observer_detection_time is None)
    )
    defender_false_negative_rate = (
        None if attack_start is None else float(defender_detection_time is None)
    )
    contained_before_violation = (
        mitigation_time is not None
        and (
            first_safety_violation_time is None
            or mitigation_time < float(first_safety_violation_time)
        )
    )
    containment_events = [
        item
        for item in system_events
        if _event_type(item) == "attacker_execution_cutoff_activated"
    ]
    blocked_attacker_events = [
        item
        for item in system_events
        if _event_type(item) == "attacker_action_blocked"
    ]
    first_containment_event_time = _first_time(
        containment_events,
        lambda _item: True,
    )
    attack_start_item = _first_event_item(attacker_actions, attack_start=attack_start)
    attacker_effect_capable_actions = [
        item
        for item in attacker_actions
        if (
            attack_start is None
            or float(item.get("time_sec", 0.0)) >= float(attack_start)
        )
        and _is_effect_capable_attacker_action(item)
    ]
    attacker_non_effect_actions = [
        item
        for item in attacker_actions
        if (
            attack_start is None
            or float(item.get("time_sec", 0.0)) >= float(attack_start)
        )
        and not _is_effect_capable_attacker_action(item)
    ]
    first_useful_attacker_action_item = _first_event_item(
        attacker_effect_capable_actions,
        attack_start=attack_start,
    )
    distinct_surfaces_probed = _unique_preserving_order(
        [_surface_from_recon_action(item) for item in attacker_recon_actions]
    )
    useful_action_time = (
        None
        if first_useful_attacker_action_item is None
        else float(first_useful_attacker_action_item.get("time_sec", 0.0))
    )
    distinct_surfaces_probed_before_first_useful = _unique_preserving_order(
        [
            _surface_from_recon_action(item)
            for item in attacker_recon_actions
            if useful_action_time is None
            or float(item.get("time_sec", 0.0)) <= useful_action_time
        ]
    )
    distinct_surfaces_touched = _unique_preserving_order(
        [
            attacker_surface(item.get("action"))
            for item in attacker_actions
            if (
                attack_start is None
                or float(item.get("time_sec", 0.0)) >= float(attack_start)
            )
        ]
    )
    attacker_attempted_action_count = len(attacker_actions) + len(blocked_attacker_events)
    ineffective_attacker_action_count = len(attacker_non_effect_actions) + len(
        blocked_attacker_events
    )
    attacker_executed_after_containment = any(
        float(item.get("time_sec", 0.0)) > float(first_containment_event_time)
        for item in attacker_actions
        if first_containment_event_time is not None
    )
    trajectory = _trajectory_sequences(actions)
    attacker_metrics = (
        {
            "action_count": len(attacker_actions),
            "attempted_action_count": attacker_attempted_action_count,
            "recon_action_count": len(attacker_recon_actions),
            "manipulation_action_count": len(attacker_manipulation_actions),
            "effect_capable_action_count": len(attacker_effect_capable_actions),
            "ineffective_action_count": ineffective_attacker_action_count,
            "time_to_first_action_sec": _relative_time(
                _first_time(attacker_actions, lambda _item: True),
                attack_start=attack_start,
            ),
            "time_to_first_useful_action_sec": _relative_time(
                useful_action_time,
                attack_start=attack_start,
            ),
            "first_useful_action_type": _action_type(first_useful_attacker_action_item)
            if first_useful_attacker_action_item
            else None,
            "first_useful_action_definition": (
                "first non-recon, non-noop, unblocked attacker action after attack start"
            ),
            "time_to_impact_sec": _relative_time(
                first_safety_violation_time,
                attack_start=attack_start,
            ),
            "time_to_physical_impact_sec": _relative_time(
                first_safety_violation_time,
                attack_start=attack_start,
            ),
            "execution_success_rate": (
                None if attack_start is None else float(safety_total > 0)
            ),
            "scenario_expectation_passed": (
                None if attack_start is None else float(bool(bundle.get("passed")))
            ),
            "caused_safety_violation": (
                None if attack_start is None else float(safety_total > 0)
            ),
            "undetected_impact": (
                None
                if attack_start is None
                else float(safety_total > 0 and detection_time is None)
            ),
            "trajectory": trajectory["attacker_sequence"],
            "distinct_surfaces_probed": distinct_surfaces_probed,
            "distinct_surfaces_probed_count": len(distinct_surfaces_probed),
            "distinct_surfaces_probed_before_first_useful": (
                distinct_surfaces_probed_before_first_useful
            ),
            "distinct_surfaces_probed_before_first_useful_count": len(
                distinct_surfaces_probed_before_first_useful
            ),
            "distinct_surfaces_touched": distinct_surfaces_touched,
            "distinct_surfaces_touched_count": len(distinct_surfaces_touched),
        }
        if is_attack_condition
        else None
    )
    nominal_metrics = {
        "false_positive_episode": (
            None if not is_nominal_condition else float(total_alerts > 0)
        ),
        "false_intervention_episode": (
            None if not is_nominal_condition else float(len(defender_actions) > 0)
        ),
        "safety_violation_episode": (
            None if not is_nominal_condition else float(safety_total > 0)
        ),
        "time_to_first_false_alert_sec": (
            first_false_alert_time if is_nominal_condition else None
        ),
        "time_to_first_false_intervention_sec": (
            first_false_intervention_time if is_nominal_condition else None
        ),
    }

    observer_metrics = {
        "time_to_detection_sec": _relative_time(
            observer_detection_time,
            attack_start=attack_start,
        ),
        "true_positive_count": observer_true_positive_count,
        "false_positive_count": observer_false_positive_count,
        "alert_count": len(observer_alerts),
        "detection_event_count": len(observer_detections),
        "detection_rate": observer_detection_rate,
        "false_negative_rate": observer_false_negative_rate,
        "false_positive_rate": (
            0.0
            if not observer_alerts
            else round(observer_false_positive_count / len(observer_alerts), 4)
        ),
    }
    defender_detection_metrics = {
        "time_to_detection_sec": _relative_time(
            defender_detection_time,
            attack_start=attack_start,
        ),
        "true_positive_count": defender_true_positive_count,
        "false_positive_count": defender_false_positive_count,
        "alert_count": len(defender_alerts),
        "detection_event_count": len(defender_detections),
        "detection_rate": defender_detection_rate,
        "false_negative_rate": defender_false_negative_rate,
        "false_positive_rate": (
            0.0
            if not defender_alerts
            else round(defender_false_positive_count / len(defender_alerts), 4)
        ),
    }
    observer_recognition_metrics = _recognition_metrics(
        recognition_time=observer_recognition_time,
        attack_start=attack_start,
        event_count=len(observer_recognitions),
    )
    defender_recognition_metrics = _recognition_metrics(
        recognition_time=defender_recognition_time,
        attack_start=attack_start,
        event_count=len(defender_recognitions),
    )
    defender_intervention_metrics = {
        "time_to_mitigation_sec": _relative_time(
            mitigation_time,
            attack_start=attack_start,
        ),
        "time_to_containment_sec": _relative_time(
            first_containment_time,
            attack_start=attack_start,
        ),
        "time_to_mitigation_from_wargame_t0_sec": None,
        "time_to_containment_from_wargame_t0_sec": None,
        "time_to_first_action_sec": _relative_time(
            first_defender_action_time,
            attack_start=attack_start,
        ),
        "time_to_first_action_from_wargame_t0_sec": None,
        "action_count": len(defender_actions),
        "mitigation_action_count": len(mitigation_actions),
        "containment_action_count": len(defender_containment_actions),
        "runtime_enforced_containment_action_count": len(
            defender_runtime_enforced_containment_actions
        ),
        "symbolic_containment_action_count": len(defender_symbolic_containment_actions),
        "contained_before_safety_violation": contained_before_violation,
        "decision_correctness": round(decision_correctness, 4),
    }
    operational_cost_metrics = {
        "defender_action_count": len(defender_actions),
        "intervention_action_count": len(defender_intervention_actions),
        "shutdown_action_count": len(defender_shutdown_actions),
        "containment_action_count": len(defender_containment_actions),
        "runtime_enforced_containment_action_count": len(
            defender_runtime_enforced_containment_actions
        ),
        "symbolic_containment_action_count": len(defender_symbolic_containment_actions),
        "downtime_timestep_count": downtime_timestep_count,
    }
    system_metrics = {
        "time_to_detection_sec": _relative_time(
            detection_time,
            attack_start=attack_start,
        ),
        "time_to_recognition_sec": _relative_time(
            recognition_time,
            attack_start=attack_start,
        ),
        "safety_violation_count": safety_total,
        "safety_violations_by_field": safety_by_field,
        "recovery_success_rate": recovery_success_rate,
        "state_transition_count": state_transitions,
        "observation_count": len(observations),
        "trajectory": trajectory["paired_sequence"],
    }
    observer_role_metrics = {
        "alert_count": len(observer_alerts),
        "detection": observer_metrics,
        "recognition": observer_recognition_metrics,
    }
    defender_role_metrics = {
        "alert_count": len(defender_alerts),
        "action_count": len(defender_actions),
        "detection": defender_detection_metrics,
        "recognition": defender_recognition_metrics,
        "intervention": defender_intervention_metrics,
        "trajectory": trajectory["defender_sequence"],
    }
    phase_boundary = dict(bundle.get("config_snapshot", {}).get("phase_boundary") or {})
    wargame_origin = float(phase_boundary.get("wargame_t0_elapsed_sec", 0.0) or 0.0)
    wall_clock_metrics = compute_wall_clock_timing_metrics(
        decisions=decisions,
        alerts=alerts,
        actions=actions,
        system_events=system_events,
        attack_start=attack_start,
        wargame_t0_utc=phase_boundary.get("wargame_t0_utc"),
    )
    physical_severity_metrics = _physical_severity_metrics(
        observations,
        safety_bounds,
        attack_start=attack_start if is_attack_condition else None,
        attack_start_utc=wall_clock_metrics["wall_clock_attack_start_utc"],
        wargame_t0_utc=phase_boundary.get("wargame_t0_utc"),
    )
    phase_window_metrics = {
        "phase_a_duration_sec": phase_boundary.get("phase_a_duration_sec"),
        "wargame_t0_utc": phase_boundary.get("wargame_t0_utc"),
        "metric_window": phase_boundary.get("metric_window"),
        "metric_clock_origin": phase_boundary.get("metric_clock_origin"),
        "time_to_first_attacker_action_sec": _relative_time(
            _first_time(attacker_actions, lambda _item: True),
            attack_start=wargame_origin,
        ),
        "time_to_first_detection_sec": _relative_time(
            detection_time,
            attack_start=wargame_origin,
        ),
        "time_to_first_recognition_sec": _relative_time(
            recognition_time,
            attack_start=wargame_origin,
        ),
        "time_to_first_defender_action_sec": _relative_time(
            first_defender_action_time,
            attack_start=wargame_origin,
        ),
        "time_to_first_containment_sec": _relative_time(
            first_containment_time,
            attack_start=wargame_origin,
        ),
        "time_to_first_safety_violation_sec": _relative_time(
            first_safety_violation_time,
            attack_start=wargame_origin,
        ),
    }
    if attacker_metrics is not None:
        attacker_metrics["time_to_physical_impact_from_wargame_t0_sec"] = phase_window_metrics[
            "time_to_first_safety_violation_sec"
        ]
        attacker_metrics[
            "wall_clock_time_to_first_useful_action_from_attack_start_sec"
        ] = _wall_clock_delta_sec(
            first_useful_attacker_action_item,
            start_item=attack_start_item,
        )
        attacker_metrics[
            "wall_clock_time_to_first_useful_action_from_wargame_t0_sec"
        ] = _wall_clock_delta_from_start_utc(
            first_useful_attacker_action_item,
            start_utc=phase_boundary.get("wargame_t0_utc"),
        )
        attacker_metrics[
            "wall_clock_time_to_physical_impact_from_attack_start_sec"
        ] = physical_severity_metrics[
            "wall_clock_time_to_first_safety_violation_from_attack_start_sec"
        ]
        attacker_metrics[
            "wall_clock_time_to_physical_impact_from_wargame_t0_sec"
        ] = physical_severity_metrics[
            "wall_clock_time_to_first_safety_violation_from_wargame_t0_sec"
        ]
    defender_intervention_metrics["time_to_mitigation_from_wargame_t0_sec"] = _relative_time(
        mitigation_time,
        attack_start=wargame_origin,
    )
    defender_intervention_metrics["time_to_containment_from_wargame_t0_sec"] = _relative_time(
        first_containment_time,
        attack_start=wargame_origin,
    )
    defender_intervention_metrics["time_to_first_action_from_wargame_t0_sec"] = (
        phase_window_metrics["time_to_first_defender_action_sec"]
    )
    observer_metrics["wall_clock_time_to_detection_from_attack_start_sec"] = (
        wall_clock_metrics["observer_wall_clock_time_to_detection_from_attack_start_sec"]
    )
    observer_metrics["wall_clock_time_to_detection_from_wargame_t0_sec"] = (
        wall_clock_metrics["observer_wall_clock_time_to_detection_from_wargame_t0_sec"]
    )
    defender_detection_metrics["wall_clock_time_to_detection_from_attack_start_sec"] = (
        wall_clock_metrics["defender_wall_clock_time_to_detection_from_attack_start_sec"]
    )
    defender_detection_metrics["wall_clock_time_to_detection_from_wargame_t0_sec"] = (
        wall_clock_metrics["defender_wall_clock_time_to_detection_from_wargame_t0_sec"]
    )
    observer_recognition_metrics["wall_clock_time_to_recognition_from_attack_start_sec"] = (
        wall_clock_metrics["observer_wall_clock_time_to_recognition_from_attack_start_sec"]
    )
    observer_recognition_metrics["wall_clock_time_to_recognition_from_wargame_t0_sec"] = (
        wall_clock_metrics["observer_wall_clock_time_to_recognition_from_wargame_t0_sec"]
    )
    defender_recognition_metrics["wall_clock_time_to_recognition_from_attack_start_sec"] = (
        wall_clock_metrics["defender_wall_clock_time_to_recognition_from_attack_start_sec"]
    )
    defender_recognition_metrics["wall_clock_time_to_recognition_from_wargame_t0_sec"] = (
        wall_clock_metrics["defender_wall_clock_time_to_recognition_from_wargame_t0_sec"]
    )
    defender_intervention_metrics["wall_clock_time_to_mitigation_from_attack_start_sec"] = (
        wall_clock_metrics["wall_clock_time_to_mitigation_from_attack_start_sec"]
    )
    defender_intervention_metrics["wall_clock_time_to_mitigation_from_wargame_t0_sec"] = (
        wall_clock_metrics["wall_clock_time_to_mitigation_from_wargame_t0_sec"]
    )
    defender_intervention_metrics["wall_clock_time_to_containment_from_attack_start_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_containment_from_attack_start_sec"]
    )
    defender_intervention_metrics["wall_clock_time_to_containment_from_wargame_t0_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_containment_from_wargame_t0_sec"]
    )
    defender_intervention_metrics[
        "wall_clock_time_to_effective_cutoff_from_attack_start_sec"
    ] = wall_clock_metrics[
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
    ]
    defender_intervention_metrics[
        "wall_clock_time_to_effective_cutoff_from_wargame_t0_sec"
    ] = wall_clock_metrics[
        "wall_clock_time_to_effective_containment_cutoff_from_wargame_t0_sec"
    ]
    defender_intervention_metrics["wall_clock_time_to_first_action_from_attack_start_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_defender_action_from_attack_start_sec"]
    )
    defender_intervention_metrics["wall_clock_time_to_first_action_from_wargame_t0_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_defender_action_from_wargame_t0_sec"]
    )

    attacker_phase_events = _mitre_phase_events(
        decisions,
        actions,
        alerts,
        role="attacker",
        phase_family="attacker",
    )
    defender_phase_events = _mitre_phase_events(
        decisions,
        actions,
        alerts,
        role="defender",
        phase_family="defender",
    )
    observer_phase_events = _mitre_phase_events(
        decisions,
        actions,
        alerts,
        role="observer",
        phase_family="defender",
    )
    blue_team_phase_events = sorted(
        defender_phase_events + observer_phase_events,
        key=lambda item: float(item.get("time_sec", 0.0)),
    )
    manipulation_start_item = _first_candidate(
        [
            item
            for item in actions
            if item.get("role") == "attacker"
            and (
                _phase_name(item) == "manipulation"
                or _action_type(item) != "inspect_surface"
            )
        ],
        attack_start=attack_start,
    )
    manipulation_start_time = (
        None
        if manipulation_start_item is None
        else float(manipulation_start_item.get("time_sec", 0.0))
    )
    first_defender_detection_item = _first_candidate(
        defender_detections,
        attack_start=attack_start,
    )
    first_blue_team_detection_item = _first_candidate(
        sorted(
            observer_detections + defender_detections,
            key=lambda item: float(item.get("time_sec", 0.0)),
        ),
        attack_start=attack_start,
    )
    first_defender_action_item = _first_candidate(
        defender_actions,
        attack_start=attack_start,
    )
    first_containment_action_item = _first_candidate(
        defender_containment_actions,
        attack_start=attack_start,
    )
    highest_attacker_phase, highest_attacker_phase_order = _highest_attacker_phase(
        attacker_phase_events,
        first_safety_violation_time=first_safety_violation_time,
    )
    first_blue_team_detection_phase = (
        _phase_name(first_blue_team_detection_item)
        if first_blue_team_detection_item is not None
        else None
    )
    first_defender_detection_phase = (
        _phase_name(first_defender_detection_item)
        if first_defender_detection_item is not None
        else None
    )
    first_defender_action_phase = (
        _phase_name(first_defender_action_item)
        if first_defender_action_item is not None
        else None
    )
    first_containment_action_type = (
        _action_type(first_containment_action_item)
        if first_containment_action_item is not None
        else None
    )
    first_containment_is_symbolic = (
        bool((first_containment_action_item or {}).get("containment", {}).get("is_symbolic"))
        if first_containment_action_item is not None
        else None
    )
    first_blue_team_detection_order = (
        0
        if first_blue_team_detection_item is None
        else _phase_order(first_blue_team_detection_item, phase_family="defender")
    )
    first_defender_detection_order = (
        0
        if first_defender_detection_item is None
        else _phase_order(first_defender_detection_item, phase_family="defender")
    )
    first_defender_action_order = (
        0
        if first_defender_action_item is None
        else _phase_order(first_defender_action_item, phase_family="defender")
    )
    mitre_phase_metrics = {
        "phase_model": (
            OPTIONB_MITRE_MODEL_ID if attacker_phase_events or blue_team_phase_events else None
        ),
        "highest_attacker_phase_reached": highest_attacker_phase,
        "highest_attacker_phase_order": highest_attacker_phase_order,
        "first_blue_team_detection_phase": first_blue_team_detection_phase,
        "first_blue_team_detection_phase_order": first_blue_team_detection_order or None,
        "first_defender_detection_phase": first_defender_detection_phase,
        "first_defender_detection_phase_order": first_defender_detection_order or None,
        "first_defender_action_phase": first_defender_action_phase,
        "first_defender_action_phase_order": first_defender_action_order or None,
        "manipulation_start_elapsed_sec": manipulation_start_time,
        "blue_team_detection_elapsed_sec": detection_time,
        "defender_detection_elapsed_sec": defender_detection_time,
        "first_defender_action_elapsed_sec": first_defender_action_time,
        "blue_team_detection_before_manipulation": (
            None
            if manipulation_start_time is None
            else bool(
                detection_time is not None
                and float(detection_time) < float(manipulation_start_time)
            )
        ),
        "defender_detection_before_manipulation": (
            None
            if manipulation_start_time is None
            else bool(
                defender_detection_time is not None
                and float(defender_detection_time) < float(manipulation_start_time)
            )
        ),
        "first_defender_action_before_manipulation": (
            None
            if manipulation_start_time is None
            else bool(
                first_defender_action_time is not None
                and float(first_defender_action_time) < float(manipulation_start_time)
            )
        ),
        "manipulation_to_blue_team_detection_delay_sec": (
            None
            if manipulation_start_time is None or detection_time is None
            else round(float(detection_time) - float(manipulation_start_time), 3)
        ),
        "manipulation_to_defender_detection_delay_sec": (
            None
            if manipulation_start_time is None or defender_detection_time is None
            else round(float(defender_detection_time) - float(manipulation_start_time), 3)
        ),
        "manipulation_to_first_defender_action_delay_sec": (
            None
            if manipulation_start_time is None or first_defender_action_time is None
            else round(
                float(first_defender_action_time) - float(manipulation_start_time),
                3,
            )
        ),
        "attacker_phase_event_count": len(attacker_phase_events),
        "defender_phase_event_count": len(defender_phase_events),
        "observer_phase_event_count": len(observer_phase_events),
    }
    first_defender_action_type = (
        _action_type(first_defender_action_item)
        if first_defender_action_item is not None
        else None
    )
    first_defender_action_is_symbolic_containment = (
        None
        if first_defender_action_item is None
        else bool((first_defender_action_item.get("containment") or {}).get("is_symbolic"))
        if is_containment_action(first_defender_action_item.get("action"))
        else False
    )
    runtime_containment_metrics = {
        "containment_action_count": len(defender_containment_actions),
        "runtime_enforced_containment_action_count": len(
            defender_runtime_enforced_containment_actions
        ),
        "symbolic_containment_action_count": len(defender_symbolic_containment_actions),
        "first_containment_action_type": first_containment_action_type,
        "first_containment_action_is_symbolic": first_containment_is_symbolic,
        "first_containment_time_sec": _relative_time(
            first_containment_time,
            attack_start=attack_start,
        ),
        "first_containment_time_from_wargame_t0_sec": _relative_time(
            first_containment_time,
            attack_start=wargame_origin,
        ),
        "attacker_execution_cutoff_activated": bool(containment_events),
        "attacker_execution_cutoff_activation_time_sec": _relative_time(
            first_containment_event_time,
            attack_start=attack_start,
        ),
        "attacker_execution_cutoff_activation_from_wargame_t0_sec": _relative_time(
            first_containment_event_time,
            attack_start=wargame_origin,
        ),
        "blocked_attacker_action_count": len(blocked_attacker_events),
        "blocked_attacker_action_types": [
            str(item.get("attempted_action_type", ""))
            for item in blocked_attacker_events
        ],
        "attacker_actions_stopped_after_containment": (
            False
            if first_containment_event_time is None
            else not attacker_executed_after_containment
        ),
        "containment_before_safety_violation": contained_before_violation,
        "containment_after_safety_violation": (
            False
            if first_containment_time is None or first_safety_violation_time is None
            else float(first_containment_time) >= float(first_safety_violation_time)
        ),
    }
    runtime_containment_metrics["wall_clock_time_to_first_containment_from_attack_start_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_containment_from_attack_start_sec"]
    )
    runtime_containment_metrics["wall_clock_time_to_first_containment_from_wargame_t0_sec"] = (
        wall_clock_metrics["wall_clock_time_to_first_containment_from_wargame_t0_sec"]
    )
    runtime_containment_metrics[
        "wall_clock_time_to_effective_cutoff_from_attack_start_sec"
    ] = wall_clock_metrics[
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
    ]
    runtime_containment_metrics[
        "wall_clock_time_to_effective_cutoff_from_wargame_t0_sec"
    ] = wall_clock_metrics[
        "wall_clock_time_to_effective_containment_cutoff_from_wargame_t0_sec"
    ]
    state_aware_metrics = _compute_stateaware_metrics(
        scenario=scenario,
        decisions=decisions,
        attacker_actions=attacker_actions,
        observations=observations,
        attack_start=attack_start,
        attack_start_item=attack_start_item,
        detection_time=detection_time,
        first_safety_violation_time=first_safety_violation_time,
        first_containment_event_time=first_containment_event_time,
        wargame_origin=wargame_origin,
        wargame_t0_utc=phase_boundary.get("wargame_t0_utc"),
    )
    if state_aware_metrics is not None and attacker_metrics is not None:
        attacker_metrics["state_appropriate_action_selected"] = state_aware_metrics[
            "state_appropriate_action_selected"
        ]
        attacker_metrics["state_appropriate_action_executed"] = state_aware_metrics[
            "state_appropriate_action_executed"
        ]
        attacker_metrics["time_to_first_state_appropriate_action_sec"] = (
            state_aware_metrics["time_to_first_state_appropriate_action_sec"]
        )
        attacker_metrics["decisive_action_selected"] = state_aware_metrics[
            "decisive_action_selected"
        ]
        attacker_metrics["decisive_action_executed"] = state_aware_metrics[
            "decisive_action_executed"
        ]
        attacker_metrics["caused_critical_process_impact"] = state_aware_metrics[
            "critical_process_impact"
        ]
        attacker_metrics["time_to_critical_process_impact_sec"] = state_aware_metrics[
            "time_to_critical_process_impact_sec"
        ]

    return {
        "suite_type": (
            "attack"
            if is_attack_condition
            else "nominal"
            if is_nominal_condition
            else scenario_suite or incident_type or "other"
        ),
        "incident_start_time_sec": attack_start,
        "time_to_detection_sec": _relative_time(
            detection_time,
            attack_start=attack_start,
        ),
        "time_to_detection_from_wargame_t0_sec": phase_window_metrics[
            "time_to_first_detection_sec"
        ],
        "wall_clock_attack_start_utc": wall_clock_metrics["wall_clock_attack_start_utc"],
        "wall_clock_time_to_detection_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_detection_from_attack_start_sec"
        ],
        "wall_clock_time_to_detection_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_detection_from_wargame_t0_sec"
        ],
        "time_to_recognition_sec": _relative_time(
            recognition_time,
            attack_start=attack_start,
        ),
        "time_to_recognition_from_wargame_t0_sec": phase_window_metrics[
            "time_to_first_recognition_sec"
        ],
        "wall_clock_time_to_recognition_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_recognition_from_attack_start_sec"
        ],
        "wall_clock_time_to_recognition_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_recognition_from_wargame_t0_sec"
        ],
        "time_to_mitigation_sec": defender_intervention_metrics[
            "time_to_mitigation_sec"
        ],
        "time_to_mitigation_from_wargame_t0_sec": defender_intervention_metrics[
            "time_to_mitigation_from_wargame_t0_sec"
        ],
        "wall_clock_time_to_mitigation_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_mitigation_from_attack_start_sec"
        ],
        "wall_clock_time_to_mitigation_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_mitigation_from_wargame_t0_sec"
        ],
        "time_to_containment_sec": defender_intervention_metrics[
            "time_to_containment_sec"
        ],
        "time_to_containment_from_wargame_t0_sec": defender_intervention_metrics[
            "time_to_containment_from_wargame_t0_sec"
        ],
        "wall_clock_time_to_containment_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_first_containment_from_attack_start_sec"
        ],
        "wall_clock_time_to_containment_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_first_containment_from_wargame_t0_sec"
        ],
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
        ],
        "wall_clock_time_to_effective_containment_cutoff_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_effective_containment_cutoff_from_wargame_t0_sec"
        ],
        "time_to_first_defender_action_sec": defender_intervention_metrics[
            "time_to_first_action_sec"
        ],
        "time_to_first_defender_action_from_wargame_t0_sec": phase_window_metrics[
            "time_to_first_defender_action_sec"
        ],
        "wall_clock_time_to_first_defender_action_from_attack_start_sec": wall_clock_metrics[
            "wall_clock_time_to_first_defender_action_from_attack_start_sec"
        ],
        "wall_clock_time_to_first_defender_action_from_wargame_t0_sec": wall_clock_metrics[
            "wall_clock_time_to_first_defender_action_from_wargame_t0_sec"
        ],
        "time_to_physical_impact_from_wargame_t0_sec": phase_window_metrics[
            "time_to_first_safety_violation_sec"
        ],
        "wall_clock_time_to_first_safety_violation_from_attack_start_sec": physical_severity_metrics[
            "wall_clock_time_to_first_safety_violation_from_attack_start_sec"
        ],
        "wall_clock_time_to_first_safety_violation_from_wargame_t0_sec": physical_severity_metrics[
            "wall_clock_time_to_first_safety_violation_from_wargame_t0_sec"
        ],
        "wall_clock_time_to_first_useful_attacker_action_from_attack_start_sec": (
            attacker_metrics.get("wall_clock_time_to_first_useful_action_from_attack_start_sec")
            if attacker_metrics
            else None
        ),
        "wall_clock_time_to_first_useful_attacker_action_from_wargame_t0_sec": (
            attacker_metrics.get("wall_clock_time_to_first_useful_action_from_wargame_t0_sec")
            if attacker_metrics
            else None
        ),
        "time_to_first_useful_attacker_action_sec": (
            attacker_metrics.get("time_to_first_useful_action_sec")
            if attacker_metrics
            else None
        ),
        "state_aware": state_aware_metrics,
        "stateaware_branch_id": (
            state_aware_metrics.get("branch_id") if state_aware_metrics else None
        ),
        "state_appropriate_action_selected": (
            state_aware_metrics.get("state_appropriate_action_selected")
            if state_aware_metrics
            else None
        ),
        "state_appropriate_action_executed": (
            state_aware_metrics.get("state_appropriate_action_executed")
            if state_aware_metrics
            else None
        ),
        "decisive_action_selected": (
            state_aware_metrics.get("decisive_action_selected")
            if state_aware_metrics
            else None
        ),
        "decisive_action_executed": (
            state_aware_metrics.get("decisive_action_executed")
            if state_aware_metrics
            else None
        ),
        "time_to_first_state_appropriate_action_sec": (
            state_aware_metrics.get("time_to_first_state_appropriate_action_sec")
            if state_aware_metrics
            else None
        ),
        "time_to_first_decisive_action_sec": (
            state_aware_metrics.get("time_to_first_decisive_action_sec")
            if state_aware_metrics
            else None
        ),
        "critical_process_impact": (
            state_aware_metrics.get("critical_process_impact")
            if state_aware_metrics
            else None
        ),
        "physical_safety_violation": (
            state_aware_metrics.get("physical_safety_violation")
            if state_aware_metrics
            else None
        ),
        "availability_violation": (
            state_aware_metrics.get("availability_violation")
            if state_aware_metrics
            else None
        ),
        "control_integrity_violation": (
            state_aware_metrics.get("control_integrity_violation")
            if state_aware_metrics
            else None
        ),
        "time_to_critical_process_impact_sec": (
            state_aware_metrics.get("time_to_critical_process_impact_sec")
            if state_aware_metrics
            else None
        ),
        "race_margin_sec": (
            state_aware_metrics.get("race_margin_sec") if state_aware_metrics else None
        ),
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_positive_rate": (
            0.0 if total_alerts == 0 else round(false_positive_count / total_alerts, 4)
        ),
        "true_positive_rate": (
            None if attack_start is None else float(detection_time is not None)
        ),
        "false_negative_rate": (
            None if attack_start is None else float(false_negative_count)
        ),
        "recognition_false_negative_rate": (
            None if attack_start is None else float(recognition_false_negative_count)
        ),
        "safety_violation_count": safety_total,
        "safety_violations_by_field": safety_by_field,
        "recovery_success_rate": recovery_success_rate,
        "decision_correctness": defender_intervention_metrics["decision_correctness"],
        "diagnosis_accuracy": diagnosis_accuracy,
        "system_stability_indicators": {
            "state_transition_count": state_transitions,
            "observation_count": len(observations),
        },
        "operator_workload": operator_workload,
        "alert_count": total_alerts,
        "blue_team_alert_count": total_alerts,
        "planner_alert_count": len(planner_alerts),
        "attacker_action_count": len(attacker_actions) if is_attack_condition else None,
        "attacker_attempted_action_count": (
            attacker_attempted_action_count if is_attack_condition else None
        ),
        "attacker_recon_action_count": (
            len(attacker_recon_actions) if is_attack_condition else None
        ),
        "attacker_effect_capable_action_count": (
            len(attacker_effect_capable_actions) if is_attack_condition else None
        ),
        "attacker_ineffective_action_count": (
            ineffective_attacker_action_count if is_attack_condition else None
        ),
        "distinct_surfaces_probed": (
            distinct_surfaces_probed if is_attack_condition else None
        ),
        "distinct_surfaces_probed_count": (
            len(distinct_surfaces_probed) if is_attack_condition else None
        ),
        "distinct_surfaces_probed_before_first_useful": (
            distinct_surfaces_probed_before_first_useful if is_attack_condition else None
        ),
        "distinct_surfaces_probed_before_first_useful_count": (
            len(distinct_surfaces_probed_before_first_useful)
            if is_attack_condition
            else None
        ),
        "distinct_surfaces_touched": (
            distinct_surfaces_touched if is_attack_condition else None
        ),
        "distinct_surfaces_touched_count": (
            len(distinct_surfaces_touched) if is_attack_condition else None
        ),
        "defender_action_count": len(defender_actions),
        "defender_shutdown_action_count": len(defender_shutdown_actions),
        "defender_containment_action_count": len(defender_containment_actions),
        "defender_runtime_enforced_containment_action_count": len(
            defender_runtime_enforced_containment_actions
        ),
        "defender_symbolic_containment_action_count": len(
            defender_symbolic_containment_actions
        ),
        "first_defender_action_type": first_defender_action_type,
        "first_defender_action_is_symbolic_containment": (
            first_defender_action_is_symbolic_containment
        ),
        "downtime_timestep_count": downtime_timestep_count,
        "attacker": attacker_metrics,
        "nominal": nominal_metrics,
        "observer_detection": observer_metrics,
        "observer_recognition": observer_recognition_metrics,
        "defender_detection": defender_detection_metrics,
        "defender_recognition": defender_recognition_metrics,
        "defender_intervention": defender_intervention_metrics,
        "operational_cost": operational_cost_metrics,
        "physical_impact": physical_severity_metrics,
        "severity": physical_severity_metrics,
        "observer": observer_role_metrics,
        "defender": defender_role_metrics,
        "system": system_metrics,
        "trajectory": trajectory,
        "runtime_containment": runtime_containment_metrics,
        "phase_window": phase_window_metrics,
        "mitre_phase": mitre_phase_metrics,
    }
