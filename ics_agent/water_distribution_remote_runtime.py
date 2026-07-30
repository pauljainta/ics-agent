from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from typing import Any

try:
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
        ModbusSlaveContext,
    )
    from pymodbus.server.sync import ModbusTcpServer
except ImportError:
    from pymodbus.datastore import (
        ModbusDeviceContext as ModbusSlaveContext,
        ModbusSequentialDataBlock,
        ModbusServerContext,
    )
    from pymodbus.server import ModbusTcpServer

from .testbeds.water_distribution import (
    ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP,
    ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP,
    ALARM_COIL_MAP,
    COMMAND_COIL_MAP,
    DEFAULT_LEVELS,
    FIELD_GROUPS,
    HMI_COIL_MAP,
    LEVEL_REGISTER_MAP,
    PROCESS_CONTROL_REGISTER_MAP,
    RUNTIME_RESET_COIL,
    SENSOR_OVERRIDE_ENABLE_REGISTER_MAP,
    SENSOR_OVERRIDE_VALUE_REGISTER_MAP,
    SIMULATOR_LEVEL_APPLY_REGISTER_MAP,
    SIMULATOR_LEVEL_SEED_REGISTER_MAP,
    STATE_COIL_MAP,
    STATUS_COIL_MAP,
)


LOG = logging.getLogger("water_distribution_remote_runtime")
REGISTER_CAPACITY = 2048
COIL_CAPACITY = 2048


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _register_capacity(maximum_address: int) -> int:
    return max(REGISTER_CAPACITY, maximum_address + 8)


class LockedSequentialDataBlock(ModbusSequentialDataBlock):
    def __init__(self, address: int, values: list[int]) -> None:
        super().__init__(address, values)
        self._lock = threading.RLock()

    def getValues(self, address: int, count: int = 1):
        with self._lock:
            return list(super().getValues(address, count))

    def setValues(self, address: int, values):
        if not isinstance(values, list):
            values = [values]
        with self._lock:
            super().setValues(address, list(values))

    def runtime_get_values(self, address: int, count: int = 1) -> list[int]:
        return [int(value) for value in self.getValues(address, count)]

    def runtime_set_values(self, address: int, values: list[int] | tuple[int, ...] | int) -> None:
        self.setValues(address, list(values) if isinstance(values, (list, tuple)) else [values])


class _BaseRuntime:
    def __init__(self, *, host: str, port: int, scan_ms: int) -> None:
        self.host = host
        self.port = int(port)
        self.scan_sec = max(0.01, float(scan_ms) / 1000.0)
        self._stop = threading.Event()
        self._scan_thread: threading.Thread | None = None

        self.coils = LockedSequentialDataBlock(0, [0] * COIL_CAPACITY)
        self.holding = LockedSequentialDataBlock(0, [0] * REGISTER_CAPACITY)
        self.discrete = LockedSequentialDataBlock(0, [0] * COIL_CAPACITY)
        self.inputs = LockedSequentialDataBlock(0, [0] * REGISTER_CAPACITY)

        context_kwargs = {
            "co": self.coils,
            "hr": self.holding,
            "di": self.discrete,
            "ir": self.inputs,
        }
        try:
            self.slave = ModbusSlaveContext(zero_mode=True, **context_kwargs)
        except TypeError:
            self.slave = ModbusSlaveContext(**context_kwargs)
        try:
            self.context = ModbusServerContext(self.slave, single=True)
        except TypeError:
            self.context = ModbusServerContext(slaves=self.slave, single=True)

        try:
            self.server = ModbusTcpServer(
                self.context,
                address=(self.host, self.port),
                allow_reuse_address=True,
            )
        except TypeError:
            self.server = ModbusTcpServer(
                self.context,
                address=(self.host, self.port),
            )

    def _read_coil(self, address: int) -> int:
        return int(self.coils.runtime_get_values(address, 1)[0] or 0)

    def _read_holding(self, address: int) -> int:
        return int(self.holding.runtime_get_values(address, 1)[0] or 0)

    def _write_coils(self, address: int, values: list[int] | tuple[int, ...] | int) -> None:
        self.coils.runtime_set_values(address, values)

    def _write_holding(self, address: int, values: list[int] | tuple[int, ...] | int) -> None:
        self.holding.runtime_set_values(address, values)

    def start(self) -> None:
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        LOG.info("Listening on %s:%d", self.host, self.port)
        self.server.serve_forever()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=2.0)

    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            started_at = time.monotonic()
            try:
                self._scan_once()
            except Exception:
                LOG.exception("Scan failed")
            elapsed = time.monotonic() - started_at
            sleep_sec = self.scan_sec - elapsed
            if sleep_sec > 0:
                self._stop.wait(sleep_sec)

    def _scan_once(self) -> None:
        raise NotImplementedError


class ControllerRuntime(_BaseRuntime):
    def __init__(self, *, host: str, port: int, scan_ms: int) -> None:
        super().__init__(host=host, port=port, scan_ms=scan_ms)
        self.state_name = "IDLE"
        self.shutdown_until = 0.0
        self.supply_pump_latched = False
        self.grid_valve_latched = False
        self.rws_pump_latched = False
        self._publish_state(
            supply_speed=0,
            rws_speed=0,
            supply_valve=0,
            grid_valve=0,
            rws_valve=0,
            alarm=0,
        )

    def _publish_state(
        self,
        *,
        supply_speed: int,
        rws_speed: int,
        supply_valve: int,
        grid_valve: int,
        rws_valve: int,
        alarm: int,
    ) -> None:
        self._write_holding(
            PROCESS_CONTROL_REGISTER_MAP["Supply_Pump_Speed_Cmd"],
            [int(supply_speed), int(rws_speed)],
        )
        self._write_coils(
            COMMAND_COIL_MAP["Supply_Tank_Valve_Cmd"],
            [int(supply_valve), int(grid_valve), int(rws_valve)],
        )
        self._write_coils(
            STATE_COIL_MAP["SYS_IDLE"],
            [
                1 if self.state_name == "IDLE" else 0,
                1 if self.state_name == "START" else 0,
                1 if self.state_name == "RUNNING" else 0,
                1 if self.state_name == "SHUTDOWN" else 0,
            ],
        )
        self._write_coils(ALARM_COIL_MAP["Alarm_Supply_Tank_HH"], [int(alarm)])

    def _clear_reset_and_overrides(self) -> None:
        self.state_name = "IDLE"
        self.shutdown_until = 0.0
        self.supply_pump_latched = False
        self.grid_valve_latched = False
        self.rws_pump_latched = False
        self._write_coils(HMI_COIL_MAP["HMI_Start_PB"], [0, 0, 0])
        for field in ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP:
            self._write_holding(ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP[field], 0)
            self._write_holding(ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP[field], 0)

    def _override_value(self, field: str, default: int) -> int:
        enabled = self._read_holding(ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP[field]) == 1
        if not enabled:
            return int(default)
        value = self._read_holding(ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP[field])
        if field.endswith("_Valve"):
            return 1 if value else 0
        return max(0, value)

    def _scan_once(self) -> None:
        now = time.monotonic()
        if self._read_coil(RUNTIME_RESET_COIL) == 1:
            self._clear_reset_and_overrides()

        hmi_start = self._read_coil(HMI_COIL_MAP["HMI_Start_PB"]) == 1
        hmi_stop = self._read_coil(HMI_COIL_MAP["HMI_Stop_PB"]) == 1
        sensed_supply = float(self._read_holding(LEVEL_REGISTER_MAP["Supply_Tank_Level"]))
        sensed_grid = float(self._read_holding(LEVEL_REGISTER_MAP["Grid_Elev_Tank_Level"]))
        sensed_rws = float(self._read_holding(LEVEL_REGISTER_MAP["RWS_Tank_Level"]))

        if self.state_name == "IDLE" and hmi_start and not hmi_stop:
            self.state_name = "START"
        elif hmi_stop:
            self.state_name = "SHUTDOWN"
            self.shutdown_until = now + 1.0

        if self.state_name == "START":
            if sensed_supply > 300.0 and sensed_grid < 1800.0:
                self.state_name = "RUNNING"
            elif self.shutdown_until > 0.0 and now >= self.shutdown_until:
                self.state_name = "IDLE"
        elif self.state_name == "SHUTDOWN" and now >= self.shutdown_until:
            self.state_name = "IDLE"

        supply_speed = 0
        rws_speed = 0
        supply_valve = 0
        grid_valve = 0
        rws_valve = 0
        if self.state_name == "RUNNING":
            if sensed_supply <= 500.0:
                self.supply_pump_latched = True
            elif sensed_supply >= 1200.0:
                self.supply_pump_latched = False
            if sensed_grid <= 600.0:
                self.grid_valve_latched = True
            elif sensed_grid >= 1600.0:
                self.grid_valve_latched = False
            if sensed_rws >= 1200.0:
                self.rws_pump_latched = True
            elif sensed_rws <= 300.0:
                self.rws_pump_latched = False

            supply_speed = 75 if self.supply_pump_latched else 0
            rws_speed = 50 if self.rws_pump_latched else 0
            supply_valve = 1 if self.supply_pump_latched else 0
            grid_valve = 1 if self.grid_valve_latched else 0
            rws_valve = 1 if self.rws_pump_latched else 0
        else:
            self.supply_pump_latched = False
            self.rws_pump_latched = False

        alarm = 1 if sensed_supply >= 1200.0 else 0
        self._publish_state(
            supply_speed=self._override_value("Supply_Pump_Speed", supply_speed),
            rws_speed=self._override_value("RWS_Pump_Speed", rws_speed),
            supply_valve=self._override_value("Supply_Tank_Valve", supply_valve),
            grid_valve=self._override_value("Grid_Elev_Valve", grid_valve),
            rws_valve=self._override_value("RWS_Tank_Valve", rws_valve),
            alarm=alarm,
        )


class SimulatorRuntime(_BaseRuntime):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        scan_ms: int,
        initial_conditions: dict[str, int] | None,
    ) -> None:
        super().__init__(host=host, port=port, scan_ms=scan_ms)
        self.levels = {
            key: float(value)
            for key, value in {**DEFAULT_LEVELS, **dict(initial_conditions or {})}.items()
        }
        self.supply_pump_flow = 0.0
        self._publish_outputs()

    def _apply_level_seed_updates(self) -> None:
        for field, apply_address in SIMULATOR_LEVEL_APPLY_REGISTER_MAP.items():
            if self._read_holding(apply_address) != 1:
                continue
            value = float(self._read_holding(SIMULATOR_LEVEL_SEED_REGISTER_MAP[field]))
            if field == "Supply_Tank_Level":
                self.levels[field] = _clamp(value, 0.0, 1500.0)
            elif field in {"Supply_NaOCl_Level", "Supply_NH4Cl_Level"}:
                self.levels[field] = _clamp(value, 0.0, 1000.0)
            elif field in {"Grid_Elev_Tank_Level", "Grid_Consum_Tank_Level"}:
                self.levels[field] = _clamp(value, 0.0, 2000.0)
            elif field == "RWS_Tank_Level":
                self.levels[field] = _clamp(value, 0.0, 1500.0)
            self._write_holding(apply_address, 0)

    def _override_level(self, field: str, actual: float) -> int:
        enabled = self._read_holding(SENSOR_OVERRIDE_ENABLE_REGISTER_MAP[field]) == 1
        if not enabled:
            return int(round(actual))
        return int(self._read_holding(SENSOR_OVERRIDE_VALUE_REGISTER_MAP[field]))

    def _publish_outputs(self) -> None:
        level_values = []
        for field, _ in FIELD_GROUPS["levels"]:
            if field == "Supply_Pump_Flow":
                actual = self.supply_pump_flow
            else:
                actual = self.levels[field]
            level_values.append(self._override_level(field, actual))
        self._write_holding(LEVEL_REGISTER_MAP["Supply_Tank_Level"], level_values)
        self._write_coils(
            STATUS_COIL_MAP["Supply_Pump_Sts"],
            [
                1 if self.supply_pump_flow > 0.1 else 0,
                1,
                1 if self._read_coil(24) == 1 and self.supply_pump_flow > 0.1 else 0,
                1 if self._read_coil(25) == 1 else 0,
                1 if self._read_holding(201) > 0 and self._read_coil(26) == 1 else 0,
                1 if self._read_coil(26) == 1 else 0,
            ],
        )

    def _scan_once(self) -> None:
        self._apply_level_seed_updates()

        supply_speed = float(max(0, self._read_holding(200)))
        rws_speed = float(max(0, self._read_holding(201)))
        supply_valve_open = self._read_coil(24) == 1
        grid_valve_open = self._read_coil(25) == 1
        rws_valve_open = self._read_coil(26) == 1

        supply_flow = (supply_speed / 100.0) * 200.0 if supply_valve_open else 0.0
        rws_flow = (rws_speed / 100.0) * 38.0 if rws_valve_open else 0.0
        grid_demand = 100.0
        return_flow = grid_demand * 0.9
        elevated_drain = 120.0

        self.levels["Supply_Tank_Level"] = _clamp(
            self.levels["Supply_Tank_Level"]
            + (((150.0 + rws_flow) - supply_flow) / 40.0) * self.scan_sec / 60.0 * 1000.0,
            0.0,
            1500.0,
        )
        if supply_flow > 0.0:
            self.levels["Supply_NaOCl_Level"] = _clamp(
                self.levels["Supply_NaOCl_Level"]
                - (0.038 / 0.1) * self.scan_sec / 60.0 * 1000.0,
                0.0,
                1000.0,
            )
            self.levels["Supply_NH4Cl_Level"] = _clamp(
                self.levels["Supply_NH4Cl_Level"]
                - (0.019 / 0.1) * self.scan_sec / 60.0 * 1000.0,
                0.0,
                1000.0,
            )
        self.levels["Grid_Elev_Tank_Level"] = _clamp(
            self.levels["Grid_Elev_Tank_Level"]
            + (((supply_flow if grid_valve_open else 0.0) - elevated_drain) / 60.0)
            * self.scan_sec
            / 60.0
            * 1000.0,
            0.0,
            2000.0,
        )
        self.levels["Grid_Consum_Tank_Level"] = _clamp(
            self.levels["Grid_Consum_Tank_Level"]
            + ((elevated_drain - grid_demand) / 10.0) * self.scan_sec / 60.0 * 1000.0,
            0.0,
            2000.0,
        )
        self.levels["RWS_Tank_Level"] = _clamp(
            self.levels["RWS_Tank_Level"]
            + ((return_flow - rws_flow) / 0.3) * self.scan_sec / 60.0 * 1000.0,
            0.0,
            1500.0,
        )
        self.supply_pump_flow = round(supply_flow, 3)
        self._publish_outputs()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Water-distribution remote Modbus runtime")
    parser.add_argument("--role", choices=("controller", "simulator"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scan-ms", type=int, default=100)
    parser.add_argument("--initial-conditions", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    initial_conditions = None
    if args.initial_conditions:
        initial_conditions = json.loads(args.initial_conditions)

    runtime: _BaseRuntime
    if args.role == "controller":
        runtime = ControllerRuntime(host=args.host, port=args.port, scan_ms=args.scan_ms)
    else:
        runtime = SimulatorRuntime(
            host=args.host,
            port=args.port,
            scan_ms=args.scan_ms,
            initial_conditions=initial_conditions,
        )

    def _handle_signal(signum, _frame) -> None:
        LOG.info("Signal %d received; shutting down", signum)
        runtime.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        runtime.start()
    finally:
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
