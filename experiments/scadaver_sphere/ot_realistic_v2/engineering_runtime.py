#!/usr/bin/env python3
"""Read-only HMI/OPC UA/S7 views sourced from OpenPLC OT telemetry.

Unlike V1, this service has no simulator address or process-network access.
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
class ControllerSample:
    timestamp_utc: str
    epoch_seconds: float
    process_value_raw: int | None
    output_active: bool | None
    output_flow_raw: int | None
    source_sequence: int
    health: str


class ControllerTelemetry:
    def __init__(self) -> None:
        self.host = os.environ.get("SCADAVER_CONTROLLER_HOST", "198.18.4.10")
        self.port = int(os.environ.get("SCADAVER_CONTROLLER_PORT", "1502"))
        self.lock = threading.Lock()
        self.sequence = 0
        self.current = ControllerSample("", 0.0, None, None, None, 0, "starting")

    def read(self) -> ControllerSample:
        client = ModbusTcpClient(self.host, port=self.port, timeout=1)
        now = time.time()
        self.sequence += 1
        try:
            if not client.connect():
                raise OSError("controller_connect_failed")
            inputs = client.read_input_registers(100, count=1)
            outputs = client.read_holding_registers(1, count=1)
            if inputs.isError() or outputs.isError():
                raise OSError("controller_telemetry_error")
            active = bool(outputs.registers[0])
            return ControllerSample(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), now,
                int(inputs.registers[0]), active, 120 if active else 0,
                self.sequence, "ok",
            )
        except Exception:
            return ControllerSample(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), now,
                None, None, None, self.sequence, "unavailable",
            )
        finally:
            client.close()

    def refresh(self) -> ControllerSample:
        sample = self.read()
        with self.lock:
            self.current = sample
        return sample

    def snapshot(self) -> ControllerSample:
        with self.lock:
            return self.current


def historian_loop(state: ControllerTelemetry, database: Path) -> None:
    database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("""CREATE TABLE IF NOT EXISTS samples (
        epoch_seconds REAL NOT NULL, timestamp_utc TEXT NOT NULL,
        process_value_raw INTEGER, output_active INTEGER, output_flow_raw INTEGER,
        source_sequence INTEGER, health TEXT NOT NULL)""")
    connection.commit()
    while True:
        sample = state.refresh()
        connection.execute(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sample.epoch_seconds, sample.timestamp_utc, sample.process_value_raw,
             None if sample.output_active is None else int(sample.output_active),
             sample.output_flow_raw, sample.source_sequence, sample.health),
        )
        connection.execute("DELETE FROM samples WHERE epoch_seconds < ?", (sample.epoch_seconds - 86400,))
        connection.commit()
        time.sleep(1)


def run_hmi(state: ControllerTelemetry) -> None:
    database = Path(os.environ.get("SCADAVER_HISTORIAN_DB", "/var/lib/scadaver-v2/historian.sqlite"))
    threading.Thread(target=historian_loop, args=(state, database), daemon=True).start()

    class HMI(BaseHTTPRequestHandler):
        def reply(self, status: int, content_type: str, value: Any) -> None:
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self.reply(200, "application/json", {
                    "service": "Operational Status", "mode": "read-only",
                    "current": asdict(state.snapshot()), "history": "/api/history",
                })
                return
            if self.path == "/api/history":
                try:
                    rows = sqlite3.connect(database).execute(
                        "SELECT timestamp_utc, process_value_raw, output_active, "
                        "output_flow_raw, health FROM samples ORDER BY epoch_seconds DESC LIMIT 30"
                    ).fetchall()
                    keys = ("timestamp_utc", "process_value_raw", "output_active", "output_flow_raw", "health")
                    self.reply(200, "application/json", {"samples": [dict(zip(keys, row)) for row in reversed(rows)]})
                except sqlite3.Error:
                    self.reply(503, "application/json", {"error": "historian_unavailable"})
                return
            self.reply(404, "application/json", {"error": "not_found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    ThreadingHTTPServer(("0.0.0.0", 8080), HMI).serve_forever()


async def run_opcua(state: ControllerTelemetry) -> None:
    from asyncua import Server

    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/scadaver/runtime/")
    namespace = await server.register_namespace("urn:scadaver:runtime")
    process = await server.nodes.objects.add_object(namespace, "RuntimeProcess")
    value = await process.add_variable(namespace, "ProcessValue0", 0)
    flow = await process.add_variable(namespace, "OutputValue0", 0)
    output = await process.add_variable(namespace, "OutputState0", False)
    quality = await process.add_variable(namespace, "DataQuality", "starting")
    async with server:
        while True:
            sample = state.refresh()
            await value.write_value(sample.process_value_raw or 0)
            await flow.write_value(sample.output_flow_raw or 0)
            await output.write_value(bool(sample.output_active))
            await quality.write_value(sample.health)
            await asyncio.sleep(1)


def run_s7(state: ControllerTelemetry) -> None:
    import snap7
    from snap7.type import SrvArea

    server = snap7.server.Server()
    data_block = bytearray(32)
    server.register_area(SrvArea.DB, 1, data_block)
    server.start(tcp_port=1102)
    try:
        while True:
            sample = state.refresh()
            struct.pack_into(">H", data_block, 0, int(sample.process_value_raw or 0))
            struct.pack_into(">H", data_block, 2, int(sample.output_flow_raw or 0))
            data_block[4] = 1 if sample.output_active else 0
            time.sleep(1)
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("hmi", "opcua", "s7"))
    service = parser.parse_args().service
    state = ControllerTelemetry()
    if service == "hmi":
        run_hmi(state)
    elif service == "opcua":
        asyncio.run(run_opcua(state))
    else:
        run_s7(state)


if __name__ == "__main__":
    main()
