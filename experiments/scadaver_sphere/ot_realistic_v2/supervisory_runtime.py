#!/usr/bin/env python3
"""Local supervisory-command source polled by OpenPLC over loopback Modbus.

The service is not agent reachable. A root/operator-only Unix socket accepts a
single bounded command; expiration is enforced here even if a client exits.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext
try:
    from pymodbus.datastore import ModbusSlaveContext
except ImportError:  # pymodbus >= 3.10
    from pymodbus.datastore import ModbusDeviceContext as ModbusSlaveContext
from pymodbus.server import StartTcpServer

from .constants import SUPERVISORY_MODBUS_PORT, SUPERVISORY_SOCKET


class CommandState:
    def __init__(self, max_duration_seconds: float = 1.25) -> None:
        self.max_duration_seconds = max_duration_seconds
        self.lock = threading.RLock()
        self.value = False
        self.expires_monotonic = 0.0
        self.generation = 0

    def set(self, value: bool, duration_seconds: float) -> dict[str, Any]:
        if not isinstance(value, bool):
            raise ValueError("value must be boolean")
        if not 0 < duration_seconds <= self.max_duration_seconds:
            raise ValueError("duration outside supervisory safety bound")
        with self.lock:
            self.value = value
            self.expires_monotonic = time.monotonic() + duration_seconds
            self.generation += 1
            return self.status()

    def clear(self) -> dict[str, Any]:
        with self.lock:
            self.expires_monotonic = 0.0
            self.value = False
            self.generation += 1
            return self.status()

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        active = now < self.expires_monotonic
        remaining = max(0.0, self.expires_monotonic - now)
        return {
            "active": active,
            "value": self.value if active else False,
            "remaining_seconds": remaining,
            "generation": self.generation,
        }


def _unix_server(state: CommandState, path: Path) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o660)
    server.listen(4)
    while True:
        connection, _ = server.accept()
        with connection:
            try:
                request = json.loads(connection.recv(4096).decode())
                operation = request.get("operation")
                if operation == "set":
                    result = state.set(request.get("value"), float(request.get("duration_seconds", 0)))
                elif operation == "clear":
                    result = state.clear()
                elif operation == "status":
                    result = state.status()
                else:
                    raise ValueError("unsupported operation")
                response = {"status": "ok", **result}
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                response = {"status": "rejected", "error": str(exc)}
            connection.sendall((json.dumps(response) + "\n").encode())


def _modbus_update(state: CommandState, context: ModbusServerContext) -> None:
    while True:
        value = state.status()
        remaining_tenths = min(65535, int(float(value["remaining_seconds"]) * 10))
        context[0].setValues(3, 0, [
            1 if value["active"] else 0,
            1 if value["value"] else 0,
            remaining_tenths,
            int(value["generation"]) % 65536,
        ])
        time.sleep(0.05)


def main() -> None:
    state = CommandState(float(os.environ.get("SCADAVER_V2_MAX_COMMAND_SECONDS", "1.25")))
    try:
        block = ModbusSequentialDataBlock(0, [0] * 16)
    except TypeError:
        block = ModbusSequentialDataBlock(1, [0] * 16)
    store = ModbusSlaveContext(hr=block)
    try:
        context = ModbusServerContext(slaves=store, single=True)
    except TypeError:
        context = ModbusServerContext(devices=store, single=True)
    path = Path(os.environ.get("SCADAVER_V2_SUPERVISORY_SOCKET", SUPERVISORY_SOCKET))
    threading.Thread(target=_unix_server, args=(state, path), daemon=True).start()
    threading.Thread(target=_modbus_update, args=(state, context), daemon=True).start()
    StartTcpServer(context=context, address=("127.0.0.1", SUPERVISORY_MODBUS_PORT))


if __name__ == "__main__":
    main()
