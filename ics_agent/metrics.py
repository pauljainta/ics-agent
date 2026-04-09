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


def _violation_counts(observations: list[dict[str, Any]], bounds: dict[str, Any]) -> tuple[int, dict[str, int]]:
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

    detection_time = _first_time(
        alerts,
        lambda item: item.get("role") in {"observer", "defender"}
        and (attack_start is None or float(item.get("time_sec", 0.0)) >= float(attack_start)),
    )
    mitigation_targets = set(ground_truth.get("mitigation_actions", ["hmi_stop", "hmi_estop", "select_process"]))
    mitigation_time = _first_time(
        actions,
        lambda item: item.get("role") == "defender"
        and str(item.get("action", {}).get("type")) in mitigation_targets
        and (attack_start is None or float(item.get("time_sec", 0.0)) >= float(attack_start)),
    )

    false_positive_count = sum(
        1
        for alert in alerts
        if attack_start is not None and float(alert.get("time_sec", 0.0)) < float(attack_start)
    )
    false_negative_count = 1 if attack_start is not None and detection_time is None else 0
    total_alerts = len(alerts)
    safety_total, safety_by_field = _violation_counts(observations, scenario.get("metadata", {}).get("safety_bounds", {}))

    expected_diagnosis = ground_truth.get("expected_diagnosis")
    diagnosis_accuracy = 0.0
    if expected_diagnosis:
        diagnosis_accuracy = 1.0 if any(item.get("diagnosis") == expected_diagnosis for item in decisions) else 0.0

    expected_defender_actions = set(ground_truth.get("expected_defender_actions", []))
    defender_actions = [item for item in actions if item.get("role") == "defender"]
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
    state_transitions = sum(1 for idx in range(1, len(states)) if states[idx] != states[idx - 1])
    operator_workload = sum(1 for item in decisions if item.get("role") == "operator")
    recovery_success_rate = 1.0 if bundle.get("passed") and safety_total == 0 else 0.0

    return {
        "incident_start_time_sec": attack_start,
        "time_to_detection_sec": None if detection_time is None or attack_start is None else round(detection_time - float(attack_start), 3),
        "time_to_mitigation_sec": None if mitigation_time is None or attack_start is None else round(mitigation_time - float(attack_start), 3),
        "false_positive_rate": 0.0 if total_alerts == 0 else round(false_positive_count / total_alerts, 4),
        "false_negative_rate": float(false_negative_count),
        "safety_violation_count": safety_total,
        "safety_violations_by_field": safety_by_field,
        "recovery_success_rate": recovery_success_rate,
        "decision_correctness": round(decision_correctness, 4),
        "diagnosis_accuracy": diagnosis_accuracy,
        "system_stability_indicators": {
            "state_transition_count": state_transitions,
            "observation_count": len(observations),
        },
        "operator_workload": operator_workload,
        "alert_count": total_alerts,
        "defender_action_count": len(defender_actions),
    }
