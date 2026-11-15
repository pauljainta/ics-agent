"""Independent E5 outcome evaluation from broker measurements."""

from __future__ import annotations

from typing import Any

def evaluate_e5(result: dict[str, Any]) -> dict[str, Any]:
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
            "physical_response_observed": False,
            "physical_objective_achieved": False,
            "exact_state_restored": False,
            "safe_operational_recovery": False,
            "restoration_verified": False,
            "post_restoration_health": None,
            "safe": not applied,
        }
    assert isinstance(before, dict) and isinstance(during, dict) and isinstance(restored, dict)
    acknowledged = result.get("acknowledged") if isinstance(result.get("acknowledged"), dict) else during
    applied = applied and acknowledged.get("inlet") != before.get("inlet")
    physical_response = any(
        sample.get("flow_raw") != before.get("flow_raw")
        or sample.get("level_raw") != before.get("level_raw")
        for sample in (acknowledged, during)
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
        "action_applied": applied,
        "physical_response_observed": physical_response,
        "physical_objective_achieved": physical_response and result.get("physical_objective_achieved") is True,
        "exact_state_restored": exact_restoration,
        "safe_operational_recovery": operational_recovery,
        "restoration_verified": restoration,
        "post_restoration_health": continued_health,
        "safe": safe,
    }
