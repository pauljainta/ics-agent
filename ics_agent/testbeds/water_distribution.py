from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TestbedError(RuntimeError):
    """Raised when the local water-distribution testbed cannot satisfy a request."""


@dataclass
class ExpectationCheck:
    check_id: str
    passed: bool
    detail: str


FIELD_GROUPS = {
    "levels": [
        ("Supply_Tank_Level", 300),
        ("Supply_Pump_Flow", 301),
        ("Supply_NaOCl_Level", 302),
        ("Supply_NH4Cl_Level", 303),
        ("Grid_Elev_Tank_Level", 304),
        ("Grid_Consum_Tank_Level", 305),
        ("RWS_Tank_Level", 306),
    ],
    "status": [
        ("Supply_Pump_Sts", 320),
        ("Supply_Mixer_Sts", 321),
        ("Supply_Tank_Valve_Sts", 322),
        ("Grid_Elev_Valve_Sts", 323),
        ("RWS_Pump_Sts", 324),
        ("RWS_Tank_Valve_Sts", 325),
    ],
    "commands": [
        ("Supply_Tank_Valve_Cmd", 40),
        ("Grid_Elev_Valve_Cmd", 41),
        ("RWS_Tank_Valve_Cmd", 42),
    ],
    "states": [
        ("SYS_IDLE", 56),
        ("SYS_START", 57),
        ("SYS_RUNNING", 58),
        ("SYS_SHUTDOWN", 59),
    ],
    "alarms": [
        ("Alarm_Supply_Tank_HH", 64),
    ],
    "hmi": [
        ("HMI_Start_PB", 0),
        ("HMI_Stop_PB", 1),
    ],
    "process_controls": [
        ("Supply_Pump_Speed_Cmd", 100),
        ("RWS_Pump_Speed_Cmd", 101),
    ],
}

LEVEL_REGISTER_MAP = {name: address for name, address in FIELD_GROUPS["levels"]}
STATUS_COIL_MAP = {name: address for name, address in FIELD_GROUPS["status"]}
COMMAND_COIL_MAP = {name: address for name, address in FIELD_GROUPS["commands"]}
STATE_COIL_MAP = {name: address for name, address in FIELD_GROUPS["states"]}
ALARM_COIL_MAP = {name: address for name, address in FIELD_GROUPS["alarms"]}
HMI_COIL_MAP = {name: address for name, address in FIELD_GROUPS["hmi"]}
PROCESS_CONTROL_REGISTER_MAP = {
    name: address for name, address in FIELD_GROUPS["process_controls"]
}
ACTUATOR_FIELD_MAP = {
    "Supply_Pump_Speed": ("register", PROCESS_CONTROL_REGISTER_MAP["Supply_Pump_Speed_Cmd"]),
    "RWS_Pump_Speed": ("register", PROCESS_CONTROL_REGISTER_MAP["RWS_Pump_Speed_Cmd"]),
    "Supply_Tank_Valve": ("coil", COMMAND_COIL_MAP["Supply_Tank_Valve_Cmd"]),
    "Grid_Elev_Valve": ("coil", COMMAND_COIL_MAP["Grid_Elev_Valve_Cmd"]),
    "RWS_Tank_Valve": ("coil", COMMAND_COIL_MAP["RWS_Tank_Valve_Cmd"]),
}

SENSOR_FIELD_BOUNDS = {
    "Supply_Tank_Level": (0.0, 1500.0),
    "Supply_Pump_Flow": (0.0, 200.0),
    "Supply_NaOCl_Level": (0.0, 1000.0),
    "Supply_NH4Cl_Level": (0.0, 1000.0),
    "Grid_Elev_Tank_Level": (0.0, 2000.0),
    "Grid_Consum_Tank_Level": (0.0, 2000.0),
    "RWS_Tank_Level": (0.0, 1500.0),
}

NUMERIC_ACTUATOR_BOUNDS = {
    "Supply_Pump_Speed": (0.0, 100.0),
    "RWS_Pump_Speed": (0.0, 100.0),
}


DEFAULT_LEVELS = {
    "Supply_Tank_Level": 1000.0,
    "Supply_NaOCl_Level": 800.0,
    "Supply_NH4Cl_Level": 800.0,
    "Grid_Elev_Tank_Level": 1200.0,
    "Grid_Consum_Tank_Level": 1000.0,
    "RWS_Tank_Level": 500.0,
}

VALID_SENSOR_FIELDS = set(DEFAULT_LEVELS) | {"Supply_Pump_Flow"}
RUNTIME_RESET_COIL = 2
SIMULATOR_LEVEL_SEED_FIELDS = tuple(DEFAULT_LEVELS)
SIMULATOR_LEVEL_SEED_REGISTER_MAP = {
    name: 1200 + index for index, name in enumerate(SIMULATOR_LEVEL_SEED_FIELDS)
}
SIMULATOR_LEVEL_APPLY_REGISTER_MAP = {
    name: 1210 + index for index, name in enumerate(SIMULATOR_LEVEL_SEED_FIELDS)
}
SENSOR_OVERRIDE_VALUE_REGISTER_MAP = {
    name: 1300 + index for index, (name, _) in enumerate(FIELD_GROUPS["levels"])
}
SENSOR_OVERRIDE_ENABLE_REGISTER_MAP = {
    name: 1310 + index for index, (name, _) in enumerate(FIELD_GROUPS["levels"])
}
ACTUATOR_OVERRIDE_FIELDS = (
    "Supply_Pump_Speed",
    "RWS_Pump_Speed",
    "Supply_Tank_Valve",
    "Grid_Elev_Valve",
    "RWS_Tank_Valve",
)
ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP = {
    name: 1400 + index for index, name in enumerate(ACTUATOR_OVERRIDE_FIELDS)
}
ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP = {
    name: 1410 + index for index, name in enumerate(ACTUATOR_OVERRIDE_FIELDS)
}


def normalize_sensor_override_value(field: str, value: Any) -> int:
    # White-box sensor overrides can intentionally exceed nominal process bounds;
    # only clamp them to the 16-bit register space used by the runtime.
    del field
    return int(round(_clamp(float(value), 0.0, 65535.0)))


def normalize_numeric_actuator_override_value(field: str, value: Any) -> int:
    minimum, maximum = NUMERIC_ACTUATOR_BOUNDS.get(field, (0.0, 65535.0))
    return int(round(_clamp(float(value), minimum, maximum)))


def state_name(sample: dict[str, Any]) -> str:
    if int(sample.get("SYS_RUNNING", 0) or 0) == 1:
        return "RUNNING"
    if int(sample.get("SYS_START", 0) or 0) == 1:
        return "START"
    if int(sample.get("SYS_SHUTDOWN", 0) or 0) == 1:
        return "SHUTDOWN"
    return "IDLE"


class OpenPLCWaterDistributionTestbed:
    def __init__(
        self,
        *,
        base_dir: Path,
        controller_host: str,
        controller_port: int | None,
        simulator_host: str,
        simulator_port: int | None,
        unit_id: int,
        use_remote_modbus: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.controller_host = str(controller_host)
        self.controller_port = int(controller_port) if controller_port is not None else None
        self.simulator_host = str(simulator_host)
        self.simulator_port = int(simulator_port) if simulator_port is not None else None
        self.unit_id = int(unit_id)
        self._remote_mode = bool(use_remote_modbus)
        if self._remote_mode and (
            self.controller_port is None or self.simulator_port is None
        ):
            raise TestbedError(
                "remote Modbus mode requires both controller and simulator ports"
            )
        self.started = False
        self._dt_sec = 0.1
        self._last_update = time.monotonic()
        self._shutdown_until = 0.0
        self._pulse_until: dict[str, float] = {}
        self._sensor_overrides: dict[str, float] = {}
        self._actuator_overrides: dict[str, Any] = {}
        self._controller_outputs: dict[str, Any] = {}
        self._plant_status: dict[str, Any] = {}
        self._levels: dict[str, float] = {}
        self._state_name = "IDLE"
        self._supply_pump_latched = False
        self._grid_valve_latched = False
        self._rws_pump_latched = False
        if not self._remote_mode:
            self.reset(DEFAULT_LEVELS)

    def ensure_stack(
        self,
        *,
        build_if_missing: bool,
        timeout_sec: float,
        start_bridge: bool,
    ) -> dict[str, Any]:
        if self._remote_mode:
            del build_if_missing, timeout_sec, start_bridge
            with self.clients() as (ctrl, sim):
                self._read_remote_coils(ctrl, HMI_COIL_MAP["HMI_Start_PB"], 2)
                self._read_remote_registers(
                    sim,
                    LEVEL_REGISTER_MAP["Supply_Tank_Level"],
                    len(FIELD_GROUPS["levels"]),
                )
            self.started = True
            return {
                "started": True,
                "mode": "modbus_remote",
                "testbed": "water_distribution",
                "support_root": str(self.base_dir),
                "controller_endpoint": f"{self.controller_host}:{self.controller_port}",
                "simulator_endpoint": f"{self.simulator_host}:{self.simulator_port}",
            }
        del build_if_missing, timeout_sec, start_bridge
        self.started = True
        return {
            "started": True,
            "mode": "python_sim",
            "testbed": "water_distribution",
            "support_root": str(self.base_dir),
        }

    def reset(
        self,
        initial_conditions: dict[str, int] | None = None,
        *,
        wait_sec: float = 0.0,
    ) -> dict[str, Any]:
        if self._remote_mode:
            conditions = {
                key: int(value)
                for key, value in {**DEFAULT_LEVELS, **dict(initial_conditions or {})}.items()
                if key in SIMULATOR_LEVEL_SEED_REGISTER_MAP
            }
            with self.clients() as (ctrl, sim):
                self._write_remote_coil(ctrl, RUNTIME_RESET_COIL, True)
                time.sleep(0.1)
                self._write_remote_coil(ctrl, RUNTIME_RESET_COIL, False)
                for field in FIELD_GROUPS["levels"]:
                    enable_address = SENSOR_OVERRIDE_ENABLE_REGISTER_MAP[field[0]]
                    self._write_remote_register(sim, enable_address, 0)
                for key, value in conditions.items():
                    value_address = SIMULATOR_LEVEL_SEED_REGISTER_MAP[key]
                    apply_address = SIMULATOR_LEVEL_APPLY_REGISTER_MAP[key]
                    self._write_remote_register(
                        sim, value_address, int(round(float(value)))
                    )
                    self._write_remote_register(sim, apply_address, 1)
            time.sleep(wait_sec)
            snapshot = self.observe_flat()
            snapshot["reset"] = True
            snapshot["reset_conditions"] = dict(conditions)
            return snapshot
        del wait_sec
        self._sensor_overrides.clear()
        self._actuator_overrides.clear()
        self._pulse_until.clear()
        self._shutdown_until = 0.0
        self._levels = {
            key: float(value)
            for key, value in {**DEFAULT_LEVELS, **dict(initial_conditions or {})}.items()
        }
        self._state_name = "IDLE"
        self._supply_pump_latched = False
        self._grid_valve_latched = False
        self._rws_pump_latched = False
        self._controller_outputs = {
            "Supply_Pump_Speed_Cmd": 0.0,
            "RWS_Pump_Speed_Cmd": 0.0,
            "Supply_Tank_Valve_Cmd": 0,
            "Grid_Elev_Valve_Cmd": 0,
            "RWS_Tank_Valve_Cmd": 0,
            "Alarm_Supply_Tank_HH": 0,
        }
        self._plant_status = {
            "Supply_Pump_Sts": 0,
            "Supply_Mixer_Sts": 1,
            "Supply_Tank_Valve_Sts": 0,
            "Grid_Elev_Valve_Sts": 0,
            "RWS_Pump_Sts": 0,
            "RWS_Tank_Valve_Sts": 0,
            "Supply_Pump_Flow": 0.0,
        }
        self._last_update = time.monotonic()
        return {
            "reset": True,
            "state_name": self._state_name,
            "initial_conditions": dict(self._levels),
        }

    def down(self) -> dict[str, Any]:
        self.started = False
        return {"stopped": True}

    def _clear_external_controls(self) -> None:
        if self._remote_mode:
            return
        self._sensor_overrides.clear()
        self._actuator_overrides.clear()

    def set_levels(
        self,
        updates: dict[str, int | float],
        *,
        wait_sec: float = 0.0,
    ) -> None:
        if self._remote_mode:
            with self.clients() as (_, sim):
                for key, value in updates.items():
                    value_address = SIMULATOR_LEVEL_SEED_REGISTER_MAP.get(key)
                    apply_address = SIMULATOR_LEVEL_APPLY_REGISTER_MAP.get(key)
                    if value_address is None or apply_address is None:
                        continue
                    self._write_remote_register(
                        sim, value_address, int(round(float(value)))
                    )
                    self._write_remote_register(sim, apply_address, 1)
            time.sleep(wait_sec)
            return
        del wait_sec
        for key, value in updates.items():
            if key in self._levels:
                self._levels[key] = float(value)

    def observe_flat(self) -> dict[str, Any]:
        if self._remote_mode:
            with self.clients() as (ctrl, sim):
                levels = self._read_remote_registers(
                    sim,
                    LEVEL_REGISTER_MAP["Supply_Tank_Level"],
                    len(FIELD_GROUPS["levels"]),
                )
                status = self._read_remote_coils(
                    sim,
                    STATUS_COIL_MAP["Supply_Pump_Sts"],
                    len(FIELD_GROUPS["status"]),
                )
                commands = self._read_remote_coils(
                    ctrl,
                    COMMAND_COIL_MAP["Supply_Tank_Valve_Cmd"],
                    len(FIELD_GROUPS["commands"]),
                )
                process_controls = self._read_remote_registers(
                    ctrl,
                    PROCESS_CONTROL_REGISTER_MAP["Supply_Pump_Speed_Cmd"],
                    len(FIELD_GROUPS["process_controls"]),
                )
                states = self._read_remote_coils(
                    ctrl,
                    STATE_COIL_MAP["SYS_IDLE"],
                    len(FIELD_GROUPS["states"]),
                )
                alarms = self._read_remote_coils(
                    ctrl,
                    ALARM_COIL_MAP["Alarm_Supply_Tank_HH"],
                    len(FIELD_GROUPS["alarms"]),
                )
                hmi = self._read_remote_coils(
                    ctrl,
                    HMI_COIL_MAP["HMI_Start_PB"],
                    len(FIELD_GROUPS["hmi"]),
                )

            snapshot = {
                name: levels[index]
                for index, (name, _) in enumerate(FIELD_GROUPS["levels"])
            }
            snapshot.update(
                {
                    name: status[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["status"])
                }
            )
            snapshot.update(
                {
                    name: commands[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["commands"])
                }
            )
            snapshot.update(
                {
                    name: process_controls[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["process_controls"])
                }
            )
            snapshot.update(
                {
                    name: states[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["states"])
                }
            )
            snapshot.update(
                {
                    name: alarms[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["alarms"])
                }
            )
            snapshot.update(
                {
                    name: hmi[index]
                    for index, (name, _) in enumerate(FIELD_GROUPS["hmi"])
                }
            )
            snapshot["STATE_NAME"] = state_name(snapshot)
            return snapshot
        self._advance()
        return self._snapshot()

    def observe(self) -> dict[str, Any]:
        flat = self.observe_flat()
        return {
            "levels": {key: flat[key] for key, _ in FIELD_GROUPS["levels"]},
            "status": {key: flat[key] for key, _ in FIELD_GROUPS["status"]},
            "commands": {key: flat[key] for key, _ in FIELD_GROUPS["commands"]},
            "states": {key: flat[key] for key, _ in FIELD_GROUPS["states"]},
            "alarms": {key: flat[key] for key, _ in FIELD_GROUPS["alarms"]},
            "hmi": {key: flat[key] for key, _ in FIELD_GROUPS["hmi"]},
            "process_controls": {
                key: flat[key] for key, _ in FIELD_GROUPS["process_controls"]
            },
            "state_name": flat["STATE_NAME"],
        }

    def apply_action(self, action: dict[str, Any] | str) -> dict[str, Any]:
        descriptor = {"type": action} if isinstance(action, str) else dict(action)
        action_type = str(descriptor.get("type", "")).strip()
        if self._remote_mode:
            if action_type == "noop":
                return {"state_name": self.observe_flat()["STATE_NAME"]}
            if action_type == "hmi_start":
                with self.clients() as (ctrl, _):
                    self._set_remote_hmi(ctrl, start=True, stop=False)
                    self._pulse_remote_coil(ctrl, HMI_COIL_MAP["HMI_Start_PB"])
            elif action_type in {"hmi_stop", "hmi_estop"}:
                with self.clients() as (ctrl, _):
                    self._set_remote_hmi(ctrl, start=False, stop=False)
                    self._pulse_remote_coil(ctrl, HMI_COIL_MAP["HMI_Stop_PB"])
            elif action_type == "hmi_reset":
                with self.clients() as (ctrl, _):
                    self._write_remote_coil(ctrl, RUNTIME_RESET_COIL, True)
                    time.sleep(0.1)
                    self._write_remote_coil(ctrl, RUNTIME_RESET_COIL, False)
            elif action_type == "set_sensor":
                field = str(descriptor.get("name", "")).strip()
                if field not in VALID_SENSOR_FIELDS:
                    raise TestbedError(f"unknown sensor field `{field}`")
                with self.clients() as (_, sim):
                    enable_address = SENSOR_OVERRIDE_ENABLE_REGISTER_MAP[field]
                    if descriptor.get("clear") or descriptor.get("value") is None:
                        self._write_remote_register(sim, enable_address, 0)
                    else:
                        value_address = SENSOR_OVERRIDE_VALUE_REGISTER_MAP[field]
                        self._write_remote_register(
                            sim,
                            value_address,
                            normalize_sensor_override_value(field, descriptor["value"]),
                        )
                        self._write_remote_register(sim, enable_address, 1)
            elif action_type == "set_actuator":
                field = str(descriptor.get("name", "")).strip()
                if field not in ACTUATOR_FIELD_MAP:
                    raise TestbedError(f"unknown actuator field `{field}`")
                with self.clients() as (ctrl, _):
                    enable_address = ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP[field]
                    if descriptor.get("clear") or descriptor.get("value") is None:
                        self._write_remote_register(ctrl, enable_address, 0)
                    else:
                        value_address = ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP[field]
                        if field.endswith("_Valve"):
                            value = int(bool(descriptor["value"]))
                        else:
                            value = normalize_numeric_actuator_override_value(
                                field,
                                descriptor["value"],
                            )
                        self._write_remote_register(ctrl, value_address, value)
                        self._write_remote_register(ctrl, enable_address, 1)
            else:
                raise TestbedError(f"unsupported action `{action_type}`")
            time.sleep(0.2)
            return {"state_name": self.observe_flat()["STATE_NAME"]}
        self._advance()
        if action_type == "noop":
            return {"state_name": self._state_name}
        if action_type == "hmi_start":
            self._state_name = "START"
            self._pulse("HMI_Start_PB")
            return {"state_name": self._state_name}
        if action_type in {"hmi_stop", "hmi_estop"}:
            self._state_name = "SHUTDOWN"
            self._shutdown_until = time.monotonic() + 1.0
            self._pulse("HMI_Stop_PB")
            return {"state_name": self._state_name}
        if action_type == "hmi_reset":
            self._state_name = "IDLE"
            self._shutdown_until = 0.0
            return {"state_name": self._state_name}
        if action_type == "set_sensor":
            field = str(descriptor.get("name", "")).strip()
            if field not in VALID_SENSOR_FIELDS:
                raise TestbedError(f"unknown sensor field `{field}`")
            if descriptor.get("clear") or descriptor.get("value") is None:
                self._sensor_overrides.pop(field, None)
            else:
                self._sensor_overrides[field] = float(
                    normalize_sensor_override_value(field, descriptor["value"])
                )
            return {"state_name": self._state_name}
        if action_type == "set_actuator":
            field = str(descriptor.get("name", "")).strip()
            if field not in {
                "Supply_Pump_Speed",
                "Supply_Tank_Valve",
                "Grid_Elev_Valve",
                "RWS_Pump_Speed",
                "RWS_Tank_Valve",
            }:
                raise TestbedError(f"unknown actuator field `{field}`")
            if descriptor.get("clear") or descriptor.get("value") is None:
                self._actuator_overrides.pop(field, None)
            else:
                value = descriptor["value"]
                if field.endswith("_Valve"):
                    self._actuator_overrides[field] = bool(value)
                else:
                    self._actuator_overrides[field] = float(
                        normalize_numeric_actuator_override_value(field, value)
                    )
            return {"state_name": self._state_name}
        raise TestbedError(f"unsupported action `{action_type}`")

    def inspect_surface(self, surface: str) -> dict[str, Any] | None:
        flat = self.observe_flat()
        groups = self._surface_groups(flat, surface)
        if groups is None:
            return None
        findings = [f"state={flat['STATE_NAME']}"]
        for field in ("Supply_Tank_Level", "Grid_Elev_Tank_Level", "RWS_Tank_Level"):
            if field in flat:
                findings.append(f"{field}={flat[field]}")
        if int(flat.get("Alarm_Supply_Tank_HH", 0) or 0) == 1:
            findings.append("active_alarms=Alarm_Supply_Tank_HH")
        if self._actuator_overrides:
            findings.append(
                "actuator_overrides=" + ",".join(sorted(self._actuator_overrides))
            )
        if self._sensor_overrides:
            findings.append(
                "sensor_overrides=" + ",".join(sorted(self._sensor_overrides))
            )
        return {
            "state_name": flat["STATE_NAME"],
            "reconnaissance": {
                "category": "reconnaissance",
                "surface": surface,
                "summary": f"Observed {surface} on the water-distribution control path.",
                "findings": findings[:8],
                "groups": groups,
            },
        }

    def _surface_groups(self, flat: dict[str, Any], surface: str) -> dict[str, Any] | None:
        if surface == "tags":
            return {
                "levels": {
                    key: flat[key]
                    for key in (
                        "Supply_Tank_Level",
                        "Supply_Pump_Flow",
                        "Supply_NaOCl_Level",
                        "Supply_NH4Cl_Level",
                        "Grid_Elev_Tank_Level",
                        "Grid_Consum_Tank_Level",
                        "RWS_Tank_Level",
                    )
                },
                "status": {
                    key: flat[key]
                    for key in (
                        "Supply_Pump_Sts",
                        "Supply_Mixer_Sts",
                        "Supply_Tank_Valve_Sts",
                        "Grid_Elev_Valve_Sts",
                        "RWS_Pump_Sts",
                        "RWS_Tank_Valve_Sts",
                    )
                },
                "commands": {
                    key: flat[key]
                    for key in (
                        "Supply_Pump_Speed_Cmd",
                        "RWS_Pump_Speed_Cmd",
                        "Supply_Tank_Valve_Cmd",
                        "Grid_Elev_Valve_Cmd",
                        "RWS_Tank_Valve_Cmd",
                    )
                },
            }
        if surface == "registers":
            return {
                "levels": {
                    "plane": "simulator",
                    "kind": "holding_registers",
                    "values": {
                        key: flat[key] for key in (
                            "Supply_Tank_Level",
                            "Supply_Pump_Flow",
                            "Supply_NaOCl_Level",
                            "Supply_NH4Cl_Level",
                            "Grid_Elev_Tank_Level",
                            "Grid_Consum_Tank_Level",
                            "RWS_Tank_Level",
                        )
                    },
                },
                "commands": {
                    "plane": "controller",
                    "kind": "holding_registers",
                    "values": {
                        "Supply_Pump_Speed_Cmd": flat["Supply_Pump_Speed_Cmd"],
                        "RWS_Pump_Speed_Cmd": flat["RWS_Pump_Speed_Cmd"],
                    },
                },
            }
        if surface == "io_image":
            return {
                "commands": {
                    "plane": "controller",
                    "kind": "coils",
                    "values": {
                        "Supply_Tank_Valve_Cmd": flat["Supply_Tank_Valve_Cmd"],
                        "Grid_Elev_Valve_Cmd": flat["Grid_Elev_Valve_Cmd"],
                        "RWS_Tank_Valve_Cmd": flat["RWS_Tank_Valve_Cmd"],
                    },
                },
                "states": {
                    "plane": "controller",
                    "kind": "coils",
                    "values": {
                        "SYS_IDLE": flat["SYS_IDLE"],
                        "SYS_START": flat["SYS_START"],
                        "SYS_RUNNING": flat["SYS_RUNNING"],
                        "SYS_SHUTDOWN": flat["SYS_SHUTDOWN"],
                    },
                },
                "hmi": {
                    "plane": "controller",
                    "kind": "coils",
                    "values": {
                        "HMI_Start_PB": flat["HMI_Start_PB"],
                        "HMI_Stop_PB": flat["HMI_Stop_PB"],
                    },
                },
            }
        if surface == "alarm_bits":
            return {
                "alarms": {
                    "plane": "controller",
                    "kind": "coils",
                    "values": {"Alarm_Supply_Tank_HH": flat["Alarm_Supply_Tank_HH"]},
                }
            }
        if surface == "actuator_state_bits":
            return {
                "commands": {
                    "plane": "controller",
                    "kind": "coils",
                    "values": {
                        "Supply_Tank_Valve_Cmd": flat["Supply_Tank_Valve_Cmd"],
                        "Grid_Elev_Valve_Cmd": flat["Grid_Elev_Valve_Cmd"],
                        "RWS_Tank_Valve_Cmd": flat["RWS_Tank_Valve_Cmd"],
                    },
                },
                "status": {
                    "plane": "simulator",
                    "kind": "coils",
                    "values": {
                        "Supply_Pump_Sts": flat["Supply_Pump_Sts"],
                        "Supply_Tank_Valve_Sts": flat["Supply_Tank_Valve_Sts"],
                        "Grid_Elev_Valve_Sts": flat["Grid_Elev_Valve_Sts"],
                        "RWS_Pump_Sts": flat["RWS_Pump_Sts"],
                        "RWS_Tank_Valve_Sts": flat["RWS_Tank_Valve_Sts"],
                    },
                },
            }
        if surface == "supply_control_plane":
            return {
                "levels": {
                    "values": {
                        "Supply_Tank_Level": flat["Supply_Tank_Level"],
                        "Supply_Pump_Flow": flat["Supply_Pump_Flow"],
                    }
                },
                "commands": {
                    "values": {
                        "Supply_Pump_Speed_Cmd": flat["Supply_Pump_Speed_Cmd"],
                        "Supply_Tank_Valve_Cmd": flat["Supply_Tank_Valve_Cmd"],
                    }
                },
                "status": {
                    "values": {
                        "Supply_Pump_Sts": flat["Supply_Pump_Sts"],
                        "Supply_Tank_Valve_Sts": flat["Supply_Tank_Valve_Sts"],
                    }
                },
            }
        if surface == "grid_control_plane":
            return {
                "levels": {
                    "values": {
                        "Grid_Elev_Tank_Level": flat["Grid_Elev_Tank_Level"],
                        "Grid_Consum_Tank_Level": flat["Grid_Consum_Tank_Level"],
                    }
                },
                "commands": {"values": {"Grid_Elev_Valve_Cmd": flat["Grid_Elev_Valve_Cmd"]}},
                "status": {"values": {"Grid_Elev_Valve_Sts": flat["Grid_Elev_Valve_Sts"]}},
            }
        if surface == "return_control_plane":
            return {
                "levels": {"values": {"RWS_Tank_Level": flat["RWS_Tank_Level"]}},
                "commands": {
                    "values": {
                        "RWS_Pump_Speed_Cmd": flat["RWS_Pump_Speed_Cmd"],
                        "RWS_Tank_Valve_Cmd": flat["RWS_Tank_Valve_Cmd"],
                    }
                },
                "status": {
                    "values": {
                        "RWS_Pump_Sts": flat["RWS_Pump_Sts"],
                        "RWS_Tank_Valve_Sts": flat["RWS_Tank_Valve_Sts"],
                    }
                },
            }
        if surface == "sensor_write_path":
            return {
                "sensor_write_path": {
                    "values": {
                        field: flat[field]
                        for field in (
                            "Supply_Tank_Level",
                            "Grid_Elev_Tank_Level",
                            "Grid_Consum_Tank_Level",
                            "RWS_Tank_Level",
                        )
                    }
                }
            }
        if surface == "actuator_write_path":
            return {
                "actuator_write_path": {
                    "values": {
                        "Supply_Pump_Speed_Cmd": flat["Supply_Pump_Speed_Cmd"],
                        "Grid_Elev_Valve_Cmd": flat["Grid_Elev_Valve_Cmd"],
                        "RWS_Pump_Speed_Cmd": flat["RWS_Pump_Speed_Cmd"],
                        "RWS_Tank_Valve_Cmd": flat["RWS_Tank_Valve_Cmd"],
                    }
                }
            }
        return None

    def evaluate_expectations(
        self,
        observations: list[dict[str, Any]],
        expectation_events: list[dict[str, Any]],
        expectations: dict[str, Any] | None,
    ) -> tuple[bool, list[ExpectationCheck]]:
        expectations = dict(expectations or {})
        checks: list[ExpectationCheck] = []
        last_observation = observations[-1] if observations else {}

        final_state = expectations.get("final_state")
        if final_state is not None:
            actual = str(last_observation.get("STATE_NAME", "UNKNOWN"))
            checks.append(
                ExpectationCheck(
                    check_id="final_state",
                    passed=actual == str(final_state),
                    detail=f"expected={final_state} actual={actual}",
                )
            )

        must_reach = [str(item) for item in expectations.get("must_reach_states", [])]
        for state in must_reach:
            reached = any(str(item.get("STATE_NAME")) == state for item in observations)
            checks.append(
                ExpectationCheck(
                    check_id=f"must_reach:{state}",
                    passed=reached,
                    detail=f"state={state} reached={reached}",
                )
            )

        for field in expectations.get("ever_nonzero", []):
            seen = any(float(item.get(field, 0) or 0) > 0 for item in observations)
            checks.append(
                ExpectationCheck(
                    check_id=f"ever_nonzero:{field}",
                    passed=seen,
                    detail=f"field={field} seen_nonzero={seen}",
                )
            )

        min_samples = expectations.get("min_samples")
        if min_samples is not None:
            checks.append(
                ExpectationCheck(
                    check_id="min_samples",
                    passed=len(observations) >= int(min_samples),
                    detail=f"required={min_samples} actual={len(observations)}",
                )
            )

        final_equals = dict(expectations.get("final_equals") or {})
        for field, expected in final_equals.items():
            actual = last_observation.get(field)
            checks.append(
                ExpectationCheck(
                    check_id=f"final_equals:{field}",
                    passed=actual == expected,
                    detail=f"expected={expected} actual={actual}",
                )
            )

        event_count = dict(expectations.get("event_count") or {})
        for action_name, expected_count in event_count.items():
            actual_count = sum(
                1 for item in expectation_events if str(item.get("action")) == action_name
            )
            checks.append(
                ExpectationCheck(
                    check_id=f"event_count:{action_name}",
                    passed=actual_count == int(expected_count),
                    detail=f"expected={expected_count} actual={actual_count}",
                )
            )

        events_lead_to_state = dict(expectations.get("events_lead_to_state") or {})
        for action_name, expected_state in events_lead_to_state.items():
            matched = any(
                str(item.get("action")) == action_name
                and str(item.get("state_after")) == str(expected_state)
                for item in expectation_events
            )
            checks.append(
                ExpectationCheck(
                    check_id=f"events_lead_to_state:{action_name}",
                    passed=matched,
                    detail=f"action={action_name} expected_state={expected_state}",
                )
            )

        passed = all(item.passed for item in checks)
        return passed, checks

    def _pulse(self, field: str) -> None:
        self._pulse_until[field] = time.monotonic() + 0.75

    def _advance(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_update)
        steps = max(1, int(round(elapsed / self._dt_sec))) if elapsed > 0 else 0
        for _ in range(steps):
            self._step_once(self._dt_sec)
        self._last_update = now

    def _sensor_value(self, field: str) -> float:
        if field in self._sensor_overrides:
            return float(self._sensor_overrides[field])
        if field == "Supply_Pump_Flow":
            return float(self._plant_status.get("Supply_Pump_Flow", 0.0))
        return float(self._levels.get(field, 0.0))

    def _actual_actuator(self, field: str, controller_value: Any) -> Any:
        if field in self._actuator_overrides:
            return self._actuator_overrides[field]
        return controller_value

    def _step_once(self, dt_sec: float) -> None:
        sensed_supply = self._sensor_value("Supply_Tank_Level")
        sensed_grid = self._sensor_value("Grid_Elev_Tank_Level")
        sensed_rws = self._sensor_value("RWS_Tank_Level")

        if self._state_name == "START":
            if sensed_supply > 300.0 and sensed_grid < 1800.0:
                self._state_name = "RUNNING"
            elif time.monotonic() >= self._shutdown_until and self._shutdown_until > 0:
                self._state_name = "IDLE"
        elif self._state_name == "SHUTDOWN":
            if time.monotonic() >= self._shutdown_until:
                self._state_name = "IDLE"

        if self._state_name != "RUNNING":
            self._controller_outputs.update(
                {
                    "Supply_Pump_Speed_Cmd": 0.0,
                    "RWS_Pump_Speed_Cmd": 0.0,
                    "Supply_Tank_Valve_Cmd": 0,
                    "Grid_Elev_Valve_Cmd": 0,
                    "RWS_Tank_Valve_Cmd": 0,
                }
            )
            self._supply_pump_latched = False
            self._rws_pump_latched = False
        else:
            if sensed_supply <= 500.0:
                self._supply_pump_latched = True
            elif sensed_supply >= 1200.0:
                self._supply_pump_latched = False
            self._controller_outputs["Supply_Pump_Speed_Cmd"] = (
                75.0 if self._supply_pump_latched else 0.0
            )
            self._controller_outputs["Supply_Tank_Valve_Cmd"] = 1 if self._supply_pump_latched else 0

            if sensed_grid <= 600.0:
                self._grid_valve_latched = True
            elif sensed_grid >= 1600.0:
                self._grid_valve_latched = False
            self._controller_outputs["Grid_Elev_Valve_Cmd"] = 1 if self._grid_valve_latched else 0

            if sensed_rws >= 1200.0:
                self._rws_pump_latched = True
            elif sensed_rws <= 300.0:
                self._rws_pump_latched = False
            self._controller_outputs["RWS_Pump_Speed_Cmd"] = (
                50.0 if self._rws_pump_latched else 0.0
            )
            self._controller_outputs["RWS_Tank_Valve_Cmd"] = 1 if self._rws_pump_latched else 0

        self._controller_outputs["Alarm_Supply_Tank_HH"] = 1 if sensed_supply >= 1200.0 else 0

        supply_speed = float(
            self._actual_actuator(
                "Supply_Pump_Speed", self._controller_outputs["Supply_Pump_Speed_Cmd"]
            )
        )
        supply_valve_open = bool(
            self._actual_actuator(
                "Supply_Tank_Valve", bool(self._controller_outputs["Supply_Tank_Valve_Cmd"])
            )
        )
        grid_valve_open = bool(
            self._actual_actuator(
                "Grid_Elev_Valve", bool(self._controller_outputs["Grid_Elev_Valve_Cmd"])
            )
        )
        rws_speed = float(
            self._actual_actuator(
                "RWS_Pump_Speed", self._controller_outputs["RWS_Pump_Speed_Cmd"]
            )
        )
        rws_valve_open = bool(
            self._actual_actuator(
                "RWS_Tank_Valve", bool(self._controller_outputs["RWS_Tank_Valve_Cmd"])
            )
        )

        supply_flow = (supply_speed / 100.0) * 200.0 if supply_valve_open else 0.0
        rws_flow = (rws_speed / 100.0) * 38.0 if rws_valve_open else 0.0
        grid_demand = 100.0
        return_flow = grid_demand * 0.9
        elevated_drain = 120.0

        self._levels["Supply_Tank_Level"] = _clamp(
            self._levels["Supply_Tank_Level"]
            + (((150.0 + rws_flow) - supply_flow) / 40.0) * dt_sec / 60.0 * 1000.0,
            0.0,
            1500.0,
        )
        if supply_flow > 0.0:
            self._levels["Supply_NaOCl_Level"] = _clamp(
                self._levels["Supply_NaOCl_Level"]
                - (0.038 / 0.1) * dt_sec / 60.0 * 1000.0,
                0.0,
                1000.0,
            )
            self._levels["Supply_NH4Cl_Level"] = _clamp(
                self._levels["Supply_NH4Cl_Level"]
                - (0.019 / 0.1) * dt_sec / 60.0 * 1000.0,
                0.0,
                1000.0,
            )
        self._levels["Grid_Elev_Tank_Level"] = _clamp(
            self._levels["Grid_Elev_Tank_Level"]
            + (((supply_flow if grid_valve_open else 0.0) - elevated_drain) / 60.0)
            * dt_sec
            / 60.0
            * 1000.0,
            0.0,
            2000.0,
        )
        self._levels["Grid_Consum_Tank_Level"] = _clamp(
            self._levels["Grid_Consum_Tank_Level"]
            + ((elevated_drain - grid_demand) / 10.0) * dt_sec / 60.0 * 1000.0,
            0.0,
            2000.0,
        )
        self._levels["RWS_Tank_Level"] = _clamp(
            self._levels["RWS_Tank_Level"]
            + ((return_flow - rws_flow) / 0.3) * dt_sec / 60.0 * 1000.0,
            0.0,
            1500.0,
        )

        self._plant_status.update(
            {
                "Supply_Pump_Sts": 1 if supply_flow > 0.1 else 0,
                "Supply_Mixer_Sts": 1,
                "Supply_Tank_Valve_Sts": 1 if supply_valve_open else 0,
                "Grid_Elev_Valve_Sts": 1 if grid_valve_open else 0,
                "RWS_Pump_Sts": 1 if rws_flow > 0.1 else 0,
                "RWS_Tank_Valve_Sts": 1 if rws_valve_open else 0,
                "Supply_Pump_Flow": round(supply_flow, 3),
            }
        )

    def _snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        snapshot = {
            "STATE_NAME": self._state_name,
            "Supply_Tank_Level": round(self._sensor_value("Supply_Tank_Level"), 3),
            "Supply_Pump_Flow": round(self._sensor_value("Supply_Pump_Flow"), 3),
            "Supply_NaOCl_Level": round(self._sensor_value("Supply_NaOCl_Level"), 3),
            "Supply_NH4Cl_Level": round(self._sensor_value("Supply_NH4Cl_Level"), 3),
            "Grid_Elev_Tank_Level": round(self._sensor_value("Grid_Elev_Tank_Level"), 3),
            "Grid_Consum_Tank_Level": round(self._sensor_value("Grid_Consum_Tank_Level"), 3),
            "RWS_Tank_Level": round(self._sensor_value("RWS_Tank_Level"), 3),
            "Supply_Pump_Speed_Cmd": round(
                float(self._controller_outputs["Supply_Pump_Speed_Cmd"]), 3
            ),
            "RWS_Pump_Speed_Cmd": round(
                float(self._controller_outputs["RWS_Pump_Speed_Cmd"]), 3
            ),
            "Supply_Tank_Valve_Cmd": int(self._controller_outputs["Supply_Tank_Valve_Cmd"]),
            "Grid_Elev_Valve_Cmd": int(self._controller_outputs["Grid_Elev_Valve_Cmd"]),
            "RWS_Tank_Valve_Cmd": int(self._controller_outputs["RWS_Tank_Valve_Cmd"]),
            "Alarm_Supply_Tank_HH": int(self._controller_outputs["Alarm_Supply_Tank_HH"]),
            "SYS_IDLE": 1 if self._state_name == "IDLE" else 0,
            "SYS_START": 1 if self._state_name == "START" else 0,
            "SYS_RUNNING": 1 if self._state_name == "RUNNING" else 0,
            "SYS_SHUTDOWN": 1 if self._state_name == "SHUTDOWN" else 0,
            "HMI_Start_PB": 1 if self._pulse_until.get("HMI_Start_PB", 0.0) > now else 0,
            "HMI_Stop_PB": 1 if self._pulse_until.get("HMI_Stop_PB", 0.0) > now else 0,
        }
        snapshot.update(self._plant_status)
        return snapshot

    @contextmanager
    def clients(self):
        if self._remote_mode:
            from pymodbus.client import ModbusTcpClient

            assert self.controller_port is not None
            assert self.simulator_port is not None
            ctrl = ModbusTcpClient(
                self.controller_host, port=self.controller_port, timeout=3
            )
            sim = ModbusTcpClient(
                self.simulator_host, port=self.simulator_port, timeout=3
            )
            if not ctrl.connect():
                raise TestbedError(
                    f"controller Modbus endpoint is unreachable at {self.controller_host}:{self.controller_port}"
                )
            if not sim.connect():
                ctrl.close()
                raise TestbedError(
                    f"simulator Modbus endpoint is unreachable at {self.simulator_host}:{self.simulator_port}"
                )
            try:
                yield ctrl, sim
            finally:
                ctrl.close()
                sim.close()
            return
        yield _FakeClient(self, plane="controller"), _FakeClient(self, plane="simulator")

    @staticmethod
    def _response_failed(response: Any) -> bool:
        return response is None or bool(getattr(response, "isError", lambda: True)())

    def _read_remote_coils(self, client: Any, address: int, count: int) -> list[int]:
        response = client.read_coils(address=address, count=count)
        if self._response_failed(response):
            raise TestbedError(f"failed to read coils {address}-{address + count - 1}")
        return [1 if bool(bit) else 0 for bit in response.bits[:count]]

    def _read_remote_registers(self, client: Any, address: int, count: int) -> list[int]:
        response = client.read_holding_registers(address=address, count=count)
        if self._response_failed(response):
            raise TestbedError(
                f"failed to read holding registers {address}-{address + count - 1}"
            )
        return [int(value) for value in response.registers[:count]]

    def _write_remote_coil(self, client: Any, address: int, value: bool) -> None:
        response = client.write_coil(address=address, value=bool(value))
        if self._response_failed(response):
            raise TestbedError(f"failed to write coil {address}")

    def _write_remote_register(self, client: Any, address: int, value: int) -> None:
        response = client.write_register(address=address, value=int(value))
        if self._response_failed(response):
            raise TestbedError(f"failed to write holding register {address}")

    def _set_remote_hmi(self, client: Any, *, start: bool, stop: bool) -> None:
        self._write_remote_coil(client, HMI_COIL_MAP["HMI_Start_PB"], start)
        self._write_remote_coil(client, HMI_COIL_MAP["HMI_Stop_PB"], stop)

    def _pulse_remote_coil(self, client: Any, address: int) -> None:
        self._write_remote_coil(client, address, True)
        time.sleep(0.1)
        self._write_remote_coil(client, address, False)


class _FakeReadResult:
    def __init__(self, values: list[Any], *, bits: bool = False) -> None:
        self.bits = [bool(value) for value in values] if bits else []
        self.registers = [int(round(float(value))) for value in values] if not bits else []

    def isError(self) -> bool:
        return False


class _FakeClient:
    def __init__(self, testbed: OpenPLCWaterDistributionTestbed, *, plane: str) -> None:
        self.testbed = testbed
        self.plane = plane

    def read_coils(self, address: int, count: int) -> _FakeReadResult:
        snapshot = self.testbed.observe_flat()
        values: list[int] = []
        for offset in range(count):
            current_address = address + offset
            values.append(
                int(
                    next(
                        (
                            snapshot[name]
                            for group in ("commands", "states", "alarms", "hmi", "status")
                            for name, item_address in FIELD_GROUPS[group]
                            if item_address == current_address
                        ),
                        0,
                    )
                )
            )
        return _FakeReadResult(values, bits=True)

    def read_holding_registers(self, address: int, count: int) -> _FakeReadResult:
        snapshot = self.testbed.observe_flat()
        values: list[float] = []
        for offset in range(count):
            current_address = address + offset
            values.append(
                float(
                    next(
                        (
                            snapshot[name]
                            for group in ("levels", "process_controls")
                            for name, item_address in FIELD_GROUPS[group]
                            if item_address == current_address
                        ),
                        0.0,
                    )
                )
            )
        return _FakeReadResult(values)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
