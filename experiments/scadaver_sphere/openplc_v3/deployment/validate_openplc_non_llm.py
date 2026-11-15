#!/usr/bin/env python3
"""Non-LLM health check for the isolated OpenPLC realization.

This is operator-side validation, not an agent tool.  It performs bounded reads
of the controller and simulator and never writes either endpoint.
"""

from __future__ import annotations

import argparse
import json
import time

from pymodbus.client import ModbusTcpClient


def read(client: ModbusTcpClient, *, holding: bool, address: int, count: int):
    result = client.read_holding_registers(address, count, slave=1) if holding else client.read_coils(address, count, slave=1)
    if result.isError():
        return {"ok": False, "error": str(result)}
    return {"ok": True, "values": list(result.registers if holding else result.bits[:count])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="198.18.4.10")
    parser.add_argument("--controller-port", type=int, default=1502)
    parser.add_argument("--simulator", default="198.18.4.20")
    parser.add_argument("--simulator-port", type=int, default=1503)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if not 2 <= args.samples <= 10 or args.interval < 0.5:
        parser.error("samples must be 2..10 and interval must be at least 0.5 seconds")
    controller = ModbusTcpClient(args.controller, port=args.controller_port, timeout=2)
    simulator = ModbusTcpClient(args.simulator, port=args.simulator_port, timeout=2)
    try:
        if not controller.connect() or not simulator.connect():
            raise SystemExit("controller or simulator Modbus service is unreachable")
        snapshots = []
        for index in range(args.samples):
            snapshots.append({"timestamp": time.time(), "controller_holding": read(controller, holding=True, address=0, count=2), "controller_coil": read(controller, holding=False, address=0, count=1), "simulator_holding": read(simulator, holding=True, address=0, count=3), "simulator_coil": read(simulator, holding=False, address=0, count=1)})
            if index + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        controller.close(); simulator.close()
    if not all(item["controller_holding"]["ok"] and item["simulator_holding"]["ok"] and item["simulator_coil"]["ok"] for item in snapshots):
        raise SystemExit("one or more bounded Modbus reads failed")
    print(json.dumps({"status": "read_only_validation_complete", "snapshots": snapshots}, indent=2))


if __name__ == "__main__":
    main()
