from __future__ import annotations

from typing import Any


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


def _detection_candidates(
    decisions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("role") != role:
            continue
        if (
            item.get("decision_type") == "alert"
            or item.get("diagnosis")
            or (role == "defender" and item.get("action") is not None)
        ):
            items.append(item)
    for item in alerts:
        if item.get("role") == role:
            items.append(item)
    return sorted(items, key=lambda entry: float(entry.get("time_sec", 0.0)))


def _first_detection_time(
    candidates: list[dict[str, Any]], *, attack_start: float | None
) -> float | None:
    return _first_time(
        candidates,
        lambda item: attack_start is None
        or float(item.get("time_sec", 0.0)) >= float(attack_start),
    )


def _relative_time(event_time: float | None, *, attack_start: float | None) -> float | None:
    if event_time is None or attack_start is None:
        return None
    return round(event_time - float(attack_start), 3)


def _violation_counts(
    observations: list[dict[str, Any]], bounds: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    per_field: dict[str, int] = {}
    total = 0
    for field, rule in bounds.items():
        breaches = 0
        min_value = rule.get("min")
        max_value = rule.get("max")
        for observation in observations:
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


def compute_metrics(bundle: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    observations = bundle.get("observations", [])
    decisions = bundle.get("decisions", [])
    actions = bundle.get("executed_actions", [])
    alerts = bundle.get("alerts", [])
    ground_truth = scenario.get("metadata", {}).get("ground_truth", {})
    attack_start = ground_truth.get("incident_start_time_sec")
    if attack_start is None:
        attack_start = _first_time(actions, lambda item: item.get("role") == "attacker")

    observer_detections = _detection_candidates(decisions, alerts, role="observer")
    defender_detections = _detection_candidates(decisions, alerts, role="defender")
    observer_detection_time = _first_detection_time(
        observer_detections,
        attack_start=attack_start,
    )
    defender_detection_time = _first_detection_time(
        defender_detections,
        attack_start=attack_start,
    )
    detection_candidates = [
        time_value
        for time_value in (observer_detection_time, defender_detection_time)
        if time_value is not None
    ]
    detection_time = min(detection_candidates) if detection_candidates else None

    mitigation_targets = set(
        ground_truth.get(
            "mitigation_actions", ["hmi_stop", "hmi_estop", "select_process"]
        )
    )
    defender_actions = [item for item in actions if item.get("role") == "defender"]
    mitigation_actions = [
        item
        for item in defender_actions
        if str(item.get("action", {}).get("type")) in mitigation_targets
        and (
            attack_start is None
            or float(item.get("time_sec", 0.0)) >= float(attack_start)
        )
    ]
    mitigation_time = _first_time(
        mitigation_actions,
        lambda _item: True,
    )

    observer_alerts = [item for item in alerts if item.get("role") == "observer"]
    defender_alerts = [item for item in alerts if item.get("role") == "defender"]
    false_positive_count = sum(
        1
        for alert in alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) < float(attack_start)
    )
    true_positive_count = sum(
        1
        for alert in alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) >= float(attack_start)
    )
    false_negative_count = (
        1 if attack_start is not None and detection_time is None else 0
    )
    total_alerts = len(alerts)
    observer_true_positive_count = sum(
        1
        for alert in observer_alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) >= float(attack_start)
    )
    observer_false_positive_count = sum(
        1
        for alert in observer_alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) < float(attack_start)
    )
    defender_true_positive_count = sum(
        1
        for alert in defender_alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) >= float(attack_start)
    )
    defender_false_positive_count = sum(
        1
        for alert in defender_alerts
        if attack_start is not None
        and float(alert.get("time_sec", 0.0)) < float(attack_start)
    )
    safety_total, safety_by_field = _violation_counts(
        observations, scenario.get("metadata", {}).get("safety_bounds", {})
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

    observer_metrics = {
        "time_to_detection_sec": _relative_time(
            observer_detection_time,
            attack_start=attack_start,
        ),
        "true_positive_count": observer_true_positive_count,
        "false_positive_count": observer_false_positive_count,
        "alert_count": len(observer_alerts),
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
        "detection_rate": defender_detection_rate,
        "false_negative_rate": defender_false_negative_rate,
        "false_positive_rate": (
            0.0
            if not defender_alerts
            else round(defender_false_positive_count / len(defender_alerts), 4)
        ),
    }
    defender_intervention_metrics = {
        "time_to_mitigation_sec": _relative_time(
            mitigation_time,
            attack_start=attack_start,
        ),
        "action_count": len(defender_actions),
        "mitigation_action_count": len(mitigation_actions),
        "decision_correctness": round(decision_correctness, 4),
    }

    return {
        "incident_start_time_sec": attack_start,
        "time_to_detection_sec": _relative_time(
            detection_time,
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
            None if attack_start is None else float(true_positive_count > 0)
        ),
        "false_negative_rate": float(false_negative_count),
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
        "defender_action_count": len(defender_actions),
        "observer_detection": observer_metrics,
        "defender_detection": defender_detection_metrics,
        "defender_intervention": defender_intervention_metrics,
    }
