from __future__ import annotations

from typing import Any


RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES = {
    "isolate_engineering_host",
    "disable_hmi_session",
    "block_sensor_write",
    "block_actuator_write",
    "block_process2_manual_path",
    "quarantine_signal_source",
}
SHUTDOWN_ACTION_TYPES = {"hmi_stop", "hmi_estop"}


def normalize_action(action: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, str):
        return {"type": action}
    return dict(action)


def action_type(action: dict[str, Any] | str | None) -> str:
    normalized = normalize_action(action)
    if normalized is None:
        return ""
    return str(normalized.get("type", ""))


def attacker_surface(action: dict[str, Any] | str | None) -> str:
    normalized = normalize_action(action)
    if normalized is None:
        return "unknown"
    current_type = str(normalized.get("type", ""))
    if current_type == "set_sensor":
        return "sensor_write_path"
    if current_type == "set_actuator":
        return "actuator_write_path"
    if current_type in {"select_process", "set_manual_valve"}:
        return "process2_manual_path"
    if current_type in {"hmi_start", "hmi_stop", "hmi_estop"}:
        return "hmi_session"
    if current_type == "inspect_surface":
        return "reconnaissance"
    if current_type in {
        "discover_hosts",
        "enumerate_services",
        "probe_protocol",
        "enumerate_protocol_objects",
    }:
        return "reconnaissance"
    if current_type == "write_protocol_object":
        return "raw_protocol_write_path"
    return "unknown"


def is_symbolic_containment_action(action: dict[str, Any] | str | None) -> bool:
    return False


def is_runtime_enforced_containment_action(
    action: dict[str, Any] | str | None,
) -> bool:
    return action_type(action) in RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES


def is_shutdown_action(action: dict[str, Any] | str | None) -> bool:
    return action_type(action) in SHUTDOWN_ACTION_TYPES


def is_process2_deselect_action(action: dict[str, Any] | str | None) -> bool:
    normalized = normalize_action(action)
    if normalized is None:
        return False
    return str(normalized.get("type", "")) == "select_process" and not bool(
        normalized.get("active", True)
    )


def is_containment_action(action: dict[str, Any] | str | None) -> bool:
    return (
        is_runtime_enforced_containment_action(action)
        or is_shutdown_action(action)
        or is_process2_deselect_action(action)
    )


def containment_surfaces(action: dict[str, Any] | str | None) -> list[str]:
    normalized = normalize_action(action)
    if normalized is None:
        return []
    current_type = str(normalized.get("type", ""))
    if current_type in {"isolate_engineering_host", "hmi_stop", "hmi_estop"}:
        return ["all"]
    if current_type == "disable_hmi_session":
        return ["hmi_session"]
    if current_type in {"block_sensor_write", "quarantine_signal_source"}:
        return ["sensor_write_path"]
    if current_type == "block_actuator_write":
        return ["actuator_write_path"]
    if current_type == "block_process2_manual_path" or is_process2_deselect_action(
        normalized
    ):
        return ["process2_manual_path"]
    return []


def containment_metadata(action: dict[str, Any] | str | None) -> dict[str, Any]:
    surfaces = containment_surfaces(action)
    current_type = action_type(action)
    execution_mode = None
    if current_type in RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES:
        execution_mode = "runtime_enforced"
    elif is_shutdown_action(action) or is_process2_deselect_action(action):
        execution_mode = "plant_action"
    return {
        "is_containment_action": is_containment_action(action),
        "is_symbolic": False,
        "target_surfaces": surfaces,
        "cuts_off_attacker_execution": is_containment_action(action),
        "surface_scope": "all" if "all" in surfaces else (surfaces[0] if surfaces else None),
        "execution_mode": execution_mode,
        "requires_runtime_enforcement": current_type
        in RUNTIME_ENFORCED_CONTAINMENT_ACTION_TYPES,
    }


def action_matches_expected(
    action: dict[str, Any] | str | None,
    expected_action_types: set[str] | list[str] | tuple[str, ...],
) -> bool:
    current_type = action_type(action)
    expected = {str(item) for item in expected_action_types}
    if current_type in expected:
        return True
    if current_type in {"block_sensor_write", "quarantine_signal_source"}:
        return bool(expected.intersection({"hmi_stop", "hmi_estop"}))
    if current_type == "block_actuator_write":
        return bool(expected.intersection({"hmi_stop", "hmi_estop"}))
    if current_type == "block_process2_manual_path":
        return bool(expected.intersection({"select_process", "hmi_stop"}))
    if current_type == "disable_hmi_session":
        return bool(expected.intersection({"hmi_stop", "hmi_estop"}))
    if current_type == "isolate_engineering_host":
        return bool(
            expected.intersection(
                {
                    "disable_hmi_session",
                    "block_sensor_write",
                    "block_actuator_write",
                    "block_process2_manual_path",
                    "hmi_stop",
                    "hmi_estop",
                    "select_process",
                }
            )
        )
    return False
