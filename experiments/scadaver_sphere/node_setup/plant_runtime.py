#!/usr/bin/env python3
"""Small runtime-backed controller/simulator pair for the SCADAver lab.

This is experiment infrastructure, not an agent tool.  The controller polls the
simulator and commands it over the private process interface; the agent can only
read each endpoint via the restricted access worker.
"""

from __future__ import annotations

import argparse
import threading
import time

from pymodbus.client import ModbusTcpClient
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
from pymodbus.server import StartTcpServer


def context() -> ModbusServerContext:
    store = ModbusSlaveContext(
        co=ModbusSequentialDataBlock(0, [0] * 64),
        hr=ModbusSequentialDataBlock(0, [0] * 256),
    )
    return ModbusServerContext(slaves=store, single=True)


def simulator_loop(store: ModbusServerContext) -> None:
    level = 500
    while True:
        inlet = bool(store[0].getValues(1, 0, count=1)[0])
        level = max(0, min(1000, level + (12 if inlet else -7)))
        store[0].setValues(3, 0, [level, 120 if inlet else 0, int(time.time()) % 65535])
        time.sleep(1)


def controller_loop(store: ModbusServerContext, simulator: str) -> None:
    client = ModbusTcpClient(simulator, port=1503, timeout=1)
    while True:
        try:
            if client.connect():
                response = client.read_holding_registers(0, count=1)
                if not response.isError():
                    level = response.registers[0]
                    inlet = level < 450
                    client.write_coil(0, inlet)
                    store[0].setValues(3, 0, [level, 1 if inlet else 0])
        except Exception:
            pass
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("controller", "simulator"))
    parser.add_argument("--controller-simulator", default="198.18.4.20")
    args = parser.parse_args()
    store = context()
    if args.role == "simulator":
        threading.Thread(target=simulator_loop, args=(store,), daemon=True).start()
        port = 1503
    else:
        threading.Thread(target=controller_loop, args=(store, args.controller_simulator), daemon=True).start()
        port = 1502
    StartTcpServer(context=store, address=("0.0.0.0", port))


if __name__ == "__main__":
    main()
