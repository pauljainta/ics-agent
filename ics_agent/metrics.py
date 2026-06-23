from __future__ import annotations

from typing import Any

from .mitre import (
    ATTACKER_PHASE_ORDER,
    ATTACKER_PHASES,
    DEFENDER_PHASE_ORDER,
    DEFENDER_PHASES,
    OPTIONB_MITRE_MODEL_ID,
)


BLUE_TEAM_ALERT_ROLES = {"observer", "defender"}
SHUTDOWN_ACTION_TYPES = {"hmi_stop", "hmi_estop"}


def _as_number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _first_time(items: list[dict[str, Any]], predicate) -> float | None:
    for item in items:
        if predicate(item):
            return float(item.get("time_sec", 0.0))
    return None


def _relative_time(event_time: float | None, *, attack_start: float | None) -> float | None:
    if event_time is None or attack_start is None:
        return None
    return round(event_time - float(attack_start), 3)


def _action_type(item: dict[str, Any]) -> str:
    return str((item.get("action") or {}).get("type", ""))


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


def _first_safety_violation_time(
    observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    *,
    attack_start: float | None = None,
) -> float | None:
    for observation in observations:
        time_sec = observation.get("elapsed_sec")
        if time_sec is None:
            continue
        if attack_start is not None and float(time_sec) < float(attack_start):
            continue
        for field, rule in bounds.items():
            value = observation.get(field)
            if value is None:
                continue
            value = _as_number(value)
            min_value = rule.get("min")
            max_value = rule.get("max")
            if min_value is not None and value < min_value:
                return float(time_sec)
            if max_value is not None and value > max_value:
                return float(time_sec)
    return None


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
        if _action_type(item) in SHUTDOWN_ACTION_TYPES
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
        if _action_type(item) in mitigation_targets
    ]
    mitigation_time = _first_time(
        mitigation_actions,
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
            if str(item.get("action", {}).get("type")) in expected_defender_actions
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
    trajectory = _trajectory_sequences(actions)
    attacker_metrics = (
        {
            "action_count": len(attacker_actions),
            "recon_action_count": len(attacker_recon_actions),
            "manipulation_action_count": len(attacker_manipulation_actions),
            "time_to_first_action_sec": _relative_time(
                _first_time(attacker_actions, lambda _item: True),
                attack_start=attack_start,
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
        "time_to_first_action_sec": _relative_time(
            first_defender_action_time,
            attack_start=attack_start,
        ),
        "action_count": len(defender_actions),
        "mitigation_action_count": len(mitigation_actions),
        "contained_before_safety_violation": contained_before_violation,
        "decision_correctness": round(decision_correctness, 4),
    }
    operational_cost_metrics = {
        "defender_action_count": len(defender_actions),
        "intervention_action_count": len(defender_intervention_actions),
        "shutdown_action_count": len(defender_shutdown_actions),
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
        "time_to_first_safety_violation_sec": _relative_time(
            first_safety_violation_time,
            attack_start=wargame_origin,
        ),
    }

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
        "time_to_recognition_sec": _relative_time(
            recognition_time,
            attack_start=attack_start,
        ),
        "time_to_mitigation_sec": defender_intervention_metrics[
            "time_to_mitigation_sec"
        ],
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
        "attacker_recon_action_count": (
            len(attacker_recon_actions) if is_attack_condition else None
        ),
        "defender_action_count": len(defender_actions),
        "defender_shutdown_action_count": len(defender_shutdown_actions),
        "downtime_timestep_count": downtime_timestep_count,
        "attacker": attacker_metrics,
        "nominal": nominal_metrics,
        "observer_detection": observer_metrics,
        "observer_recognition": observer_recognition_metrics,
        "defender_detection": defender_detection_metrics,
        "defender_recognition": defender_recognition_metrics,
        "defender_intervention": defender_intervention_metrics,
        "operational_cost": operational_cost_metrics,
        "observer": observer_role_metrics,
        "defender": defender_role_metrics,
        "system": system_metrics,
        "trajectory": trajectory,
        "phase_window": phase_window_metrics,
        "mitre_phase": mitre_phase_metrics,
    }
