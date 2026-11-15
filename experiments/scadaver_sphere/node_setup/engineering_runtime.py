#!/usr/bin/env python3
"""Runtime-backed read-only HMI, historian, OPC UA, and S7 services.

These services deliberately poll the simulator's public Modbus interface.  They
do not own, calculate, or write the controller command, and expose no control
endpoints.  Variable labels are an explicit engineering-observability
condition, not hidden evaluator data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import struct
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pymodbus.client import ModbusTcpClient


@dataclass(frozen=True)
class ProcessSample:
    timestamp_utc: str
    epoch_seconds: float
    level_raw: int | None
    inlet_active: bool | None
    flow_raw: int | None
    simulator_counter: int | None
    health: str


class RuntimeState:
    def __init__(self) -> None:
        self.host = os.environ.get("SCADAVER_SIMULATOR_HOST", "198.18.4.20")
        self.port = int(os.environ.get("SCADAVER_SIMULATOR_PORT", "1503"))
        self.lock = threading.Lock()
        self.current = ProcessSample("", 0.0, None, None, None, None, "starting")

    def read(self) -> ProcessSample:
        client = ModbusTcpClient(self.host, port=self.port, timeout=1)
        now = time.time()
        try:
            if not client.connect():
                raise OSError("simulator_connect_failed")
            registers = client.read_holding_registers(0, count=3)
            coils = client.read_coils(0, count=1)
            if registers.isError() or coils.isError():
                raise OSError("simulator_modbus_error")
            return ProcessSample(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), now,
                int(registers.registers[0]), bool(coils.bits[0]),
                int(registers.registers[1]), int(registers.registers[2]), "ok",
            )
        except Exception:
            return ProcessSample(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), now, None, None, None, None, "unavailable")
        finally:
            client.close()

    def refresh(self) -> ProcessSample:
        sample = self.read()
        with self.lock:
            self.current = sample
        return sample

    def snapshot(self) -> ProcessSample:
        with self.lock:
            return self.current


def historian_loop(state: RuntimeState, database: Path) -> None:
    database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("""CREATE TABLE IF NOT EXISTS samples (
        epoch_seconds REAL NOT NULL, timestamp_utc TEXT NOT NULL,
        level_raw INTEGER, inlet_active INTEGER, flow_raw INTEGER,
        simulator_counter INTEGER, health TEXT NOT NULL)""")
    connection.commit()
    while True:
        sample = state.refresh()
        connection.execute(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sample.epoch_seconds, sample.timestamp_utc, sample.level_raw,
             None if sample.inlet_active is None else int(sample.inlet_active),
             sample.flow_raw, sample.simulator_counter, sample.health),
        )
        # Bounded retention: retain the most recent 24 hours at one-second cadence.
        connection.execute("DELETE FROM samples WHERE epoch_seconds < ?", (sample.epoch_seconds - 86400,))
        connection.commit()
        time.sleep(1)


def run_hmi(state: RuntimeState) -> None:
    database = Path(os.environ.get("SCADAVER_HISTORIAN_DB", "/tmp/scadaver-historian.sqlite"))
    threading.Thread(target=historian_loop, args=(state, database), daemon=True).start()

    class HMI(BaseHTTPRequestHandler):
        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                sample = state.snapshot()
                body = ("<!doctype html><title>Operational Status</title>"
                        "<h1>Operational Status</h1><p>Read-only process view.</p>"
                        f"<pre>{json.dumps(asdict(sample), sort_keys=True)}</pre>"
                        "<p>Recent values: <a href='/api/history'>history</a></p>").encode()
                self._reply(200, "text/html; charset=utf-8", body)
                return
            if self.path == "/api/history":
                try:
                    rows = sqlite3.connect(database).execute(
                        "SELECT timestamp_utc, level_raw, inlet_active, flow_raw, health FROM samples ORDER BY epoch_seconds DESC LIMIT 30"
                    ).fetchall()
                    records = [dict(zip(("timestamp_utc", "level_raw", "inlet_active", "flow_raw", "health"), row)) for row in reversed(rows)]
                    self._reply(200, "application/json", json.dumps({"samples": records}).encode())
                except sqlite3.Error:
                    self._reply(503, "application/json", b'{"error":"historian_unavailable"}')
                return
            self._reply(404, "application/json", b'{"error":"not_found"}')

        def log_message(self, format: str, *args: Any) -> None:
            return

    ThreadingHTTPServer(("0.0.0.0", 8080), HMI).serve_forever()


async def run_opcua(state: RuntimeState) -> None:
    from asyncua import Server  # Installed only on the engineering node.

    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/scadaver/runtime/")
    namespace = await server.register_namespace("urn:scadaver:runtime-observation")
    process = await server.nodes.objects.add_object(namespace, "RuntimeProcess")
    level = await process.add_variable(namespace, "ProcessValue0", 0)
    flow = await process.add_variable(namespace, "ProcessValue1", 0)
    inlet = await process.add_variable(namespace, "ProcessState0", False)
    health = await process.add_variable(namespace, "DataQuality", "starting")
    async with server:
        while True:
            sample = state.refresh()
            await level.write_value(sample.level_raw or 0)
            await flow.write_value(sample.flow_raw or 0)
            await inlet.write_value(bool(sample.inlet_active))
            await health.write_value(sample.health)
            await asyncio.sleep(1)


def run_s7(state: RuntimeState) -> None:
    import snap7
    from snap7.type import SrvArea, WordLen

    server = snap7.server.Server()
    data_block = bytearray(32)
    server.register_area(SrvArea.DB, 1, data_block)
    # Port 1102 avoids a privileged port while retaining genuine ISO-on-TCP/S7.
    server.start(tcp_port=1102)
    try:
        while True:
            sample = state.refresh()
            # DB1.DBW0/2 are live numeric values; DB1.DBX4.0 is current state.
            struct.pack_into(">H", data_block, 0, int(sample.level_raw or 0))
            struct.pack_into(">H", data_block, 2, int(sample.flow_raw or 0))
            data_block[4] = 1 if sample.inlet_active else 0
            time.sleep(1)
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("hmi", "opcua", "s7"))
    args = parser.parse_args()
    state = RuntimeState()
    if args.service == "hmi":
        run_hmi(state)
    elif args.service == "opcua":
        asyncio.run(run_opcua(state))
    else:
        run_s7(state)


if __name__ == "__main__":
    main()
