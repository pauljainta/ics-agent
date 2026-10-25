"""Fixture-only reset for the experimental WD storage v2 OpenPLC pair."""
from __future__ import annotations

import struct
import time
from typing import Any, Mapping


def _request(endpoint: tuple[str, int], operation: str, address: int,
             value: bool | None = None) -> Any:
    """One Modbus operation per connection; retries cannot reuse a stale frame."""
    from pymodbus.client import ModbusTcpClient

    last_error: Exception | None = None
    for _ in range(3):
        client = ModbusTcpClient(endpoint[0], port=endpoint[1], timeout=1, retries=0)
        try:
            if not client.connect():
                raise RuntimeError("connect failed")
            if operation == "real":
                response = client.read_holding_registers(address, count=2)
                if response.isError() or len(response.registers) != 2:
                    raise RuntimeError("register read rejected")
                return struct.unpack(">f", struct.pack(">HH", *response.registers))[0]
            if operation == "coil":
                response = client.read_coils(address, count=1)
                if response.isError() or not response.bits:
                    raise RuntimeError("coil read rejected")
                return int(response.bits[0])
            if operation == "write":
                response = client.write_coil(address, bool(value))
                if response.isError():
                    raise RuntimeError("coil write rejected")
                return None
            raise ValueError(operation)
        except Exception as exc:
            last_error = exc
            time.sleep(0.15)
        finally:
            client.close()
    raise RuntimeError(f"{operation} failed at {endpoint[0]}:{endpoint[1]}:{address}: {last_error}")


def reset_wd_storage_v2(controller: tuple[str, int], simulator: tuple[str, int],
                        mapping: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Clear controller memory, pulse simulator-only reset, start, and verify."""
    address = lambda tag: int(mapping[tag]["address"])
    cr = lambda tag: _request(controller, "coil", address(tag))
    cw = lambda tag, value: _request(controller, "write", address(tag), value)
    sr = lambda tag: _request(simulator, "real", address(tag))
    sw = lambda tag, value: _request(simulator, "write", address(tag), value)

    cw("HMI_Start_PB", False)
    cw("HMI_Stop_PB", True)
    time.sleep(0.5)
    cw("HMI_Stop_PB", False)
    deadline = time.monotonic() + 3.0
    while cr("Sys_Idle") != 1:
        if time.monotonic() >= deadline:
            raise RuntimeError("controller did not return to idle")
        time.sleep(0.2)

    generation_before = sr("Fixture_Reset_Generation")
    sw("Fixture_Reset_PB", False)
    time.sleep(0.2)
    sw("Fixture_Reset_PB", True)
    time.sleep(0.3)
    sw("Fixture_Reset_PB", False)
    time.sleep(0.2)
    generation_after = sr("Fixture_Reset_Generation")
    if generation_after != generation_before + 1.0:
        raise RuntimeError(f"fixture reset edge not proven: {generation_before} -> {generation_after}")

    cw("HMI_Start_PB", True)
    time.sleep(0.5)
    cw("HMI_Start_PB", False)
    deadline = time.monotonic() + 5.0
    while cr("Sys_Running") != 1:
        if time.monotonic() >= deadline:
            raise RuntimeError("controller did not return to running")
        time.sleep(0.2)
    values = {tag: sr(tag) for tag in (
        "Supply_Tank_Level", "Grid_Elev_Tank_Level", "Grid_Consum_Tank_Level",
        "Fixture_Demand_Flow", "Grid_To_Consumer_Flow", "Consumer_Served_Flow",
        "Total_Spill_Flow")}
    if not (450.0 <= values["Supply_Tank_Level"] <= 465.0
            and 540.0 <= values["Grid_Elev_Tank_Level"] <= 600.0
            and 499.0 <= values["Grid_Consum_Tank_Level"] <= 501.0
            and values["Fixture_Demand_Flow"] == 100.0
            and values["Grid_To_Consumer_Flow"] == 100.0
            and values["Consumer_Served_Flow"] == 100.0
            and values["Total_Spill_Flow"] == 0.0):
        raise RuntimeError(f"WD v2 reset state outside fixture band: {values}")
    return {"generation_before": generation_before,
            "generation_after": generation_after, "Sys_Running": 1,
            "readback": values}
