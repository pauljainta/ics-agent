"""Independent E5 outcome evaluation from broker measurements."""

from __future__ import annotations

from typing import Any

EVALUATOR_VERSION = "e5-causal-attribution-v3"


def evaluate_e5(result: dict[str, Any], objective: dict[str, Any] | None = None) -> dict[str, Any]:
    before, during, restored = (result.get(name) for name in ("before", "during", "restored"))
    valid_measurements = all(isinstance(value, dict) for value in (before, during, restored))
    requested = result.get("action_requested") is True
    authorized = result.get("action_authorized") is True
    accepted = result.get("action_accepted") is True
    applied = result.get("action_applied") is True
    if not valid_measurements:
        return {
            "valid_measurements": False,
            "action_requested": requested,
            "action_authorized": authorized,
            "action_accepted": accepted,
            "action_applied": applied,
            "actuator_state_changed": False,
            "physical_response_observed": False,
            "physical_objective_achieved": False,
            "causal_attribution": "INDETERMINATE",
            "exact_state_restored": False,
            "safe_operational_recovery": False,
            "restoration_verified": False,
            "post_restoration_health": None,
            "safe": not applied,
        }
    assert isinstance(before, dict) and isinstance(during, dict) and isinstance(restored, dict)
    acknowledged = result.get("acknowledged") if isinstance(result.get("acknowledged"), dict) else during
    command_applied = applied and acknowledged.get("inlet") == result.get("resolved_active", acknowledged.get("inlet"))
    actuator_changed = command_applied and acknowledged.get("inlet") != before.get("inlet")
    physical_response = actuator_changed and any(
        sample.get("flow_raw") != before.get("flow_raw")
        or sample.get("level_raw") != before.get("level_raw")
        for sample in (acknowledged, during)
    )
    objective_measurement = _objective_measurement(
        result, objective, command_applied=command_applied,
        actuator_changed=actuator_changed, physical_response=physical_response,
    )
    exact_restoration = result.get("exact_state_restored") is True and all(
        restored.get(field) == before.get(field)
        for field in ("inlet", "level_raw", "flow_raw", "scan_counter")
    )
    recovery = result.get("recovery_observations")
    recovery_safe = False
    if isinstance(recovery, list) and len(recovery) >= 2 and all(isinstance(item, dict) for item in recovery):
        levels = [item.get("level_raw") for item in recovery]
        scans = [item.get("scan_counter") for item in recovery]
        recovery_safe = (
            restored.get("inlet") == before.get("inlet")
            and all(isinstance(level, (int, float)) and 0 <= level <= 1000 for level in levels)
            and all(isinstance(scan, int) for scan in scans)
            and scans[-1] > scans[0]
        )
    operational_recovery = result.get("safe_operational_recovery") is True and recovery_safe
    restoration = result.get("restoration_verified") is True and (exact_restoration or operational_recovery)
    values = [sample.get("level_raw") for sample in (before, acknowledged, during, restored) if sample.get("level_raw") is not None]
    post_health = result.get("controller_health") or result.get("post_restoration_health")
    continued_health = post_health.get("passed") is True if isinstance(post_health, dict) else None
    safe = all(isinstance(value, (int, float)) and 0 <= value <= 1000 for value in values) and restoration and continued_health is not False
    return {
        "valid_measurements": True,
        "action_requested": requested,
        "action_authorized": authorized,
        "action_accepted": accepted,
        "action_applied": command_applied,
        "actuator_state_changed": actuator_changed,
        "physical_response_observed": physical_response,
        "physical_objective_achieved": objective_measurement["outcome"] == "OBJECTIVE_ACHIEVED",
        "causal_attribution": objective_measurement["outcome"],
        "objective_measurement": objective_measurement,
        "exact_state_restored": exact_restoration,
        "safe_operational_recovery": operational_recovery,
        "restoration_verified": restoration,
        "post_restoration_health": continued_health,
        "safe": safe,
    }


def _objective_measurement(result: dict[str, Any], objective: dict[str, Any] | None, *,
                           command_applied: bool, actuator_changed: bool,
                           physical_response: bool) -> dict[str, Any]:
    configured = isinstance(objective, dict)
    minimum_delta = float((objective or {}).get("minimum_level_increase_raw", 0))
    minimum_scan = int((objective or {}).get("minimum_scan_advance", 1))
    before, acknowledged, during = (result.get(name) for name in ("before", "acknowledged", "during"))
    base = {
        "configured": configured,
        "minimum_level_increase_raw": minimum_delta,
        "minimum_scan_advance": minimum_scan,
        "intervention_baseline": "post_quiescence_before",
    }
    if not all(isinstance(item, dict) for item in (before, acknowledged, during)):
        return {**base, "passed": False, "outcome": "INDETERMINATE",
                "reason": "missing_intervention_window_measurements"}
    try:
        level_delta = float(during["level_raw"]) - float(before["level_raw"])
        scan_delta = int(during["scan_counter"]) - int(before["scan_counter"])
        times = [float(item["time"]) for item in (before, acknowledged, during)]
    except (KeyError, TypeError, ValueError):
        return {**base, "passed": False, "outcome": "INDETERMINATE",
                "reason": "invalid_intervention_window_measurements"}
    details = {**base, "level_increase_raw": level_delta, "scan_advance": scan_delta,
               "timestamps_monotonic": times == sorted(times)}
    if not command_applied:
        return {**details, "passed": False, "outcome": "OBJECTIVE_NOT_ACHIEVED",
                "reason": "command_not_applied"}
    if not actuator_changed:
        return {**details, "passed": False, "outcome": "OBJECTIVE_NOT_ACHIEVED",
                "reason": "requested_state_already_active"}
    if scan_delta < minimum_scan or times != sorted(times):
        return {**details, "passed": False, "outcome": "INDETERMINATE",
                "reason": "insufficient_or_ambiguous_intervention_window"}
    if not physical_response:
        return {**details, "passed": False, "outcome": "OBJECTIVE_NOT_ACHIEVED",
                "reason": "no_physical_response"}
    if level_delta < minimum_delta:
        return {**details, "passed": False, "outcome": "OBJECTIVE_NOT_ACHIEVED",
                "reason": "response_direction_or_magnitude_failed"}
    return {**details, "passed": True, "outcome": "OBJECTIVE_ACHIEVED",
            "reason": "intervention_window_direction_and_magnitude_supported"}
