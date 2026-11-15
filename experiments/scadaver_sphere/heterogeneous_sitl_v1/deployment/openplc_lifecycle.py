#!/usr/bin/env python3
"""Supervise the deployed OpenPLC web process and compiled PLC runtime core."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


RUNTIME_ROOT = Path(os.environ.get("SCADAVER_OPENPLC_ROOT", "/opt/openplc-v3"))
WEB_ROOT = RUNTIME_ROOT / "webserver"
CORE = WEB_ROOT / "core/openplc"
WEBSERVER = WEB_ROOT / "webserver.py"
DATABASE = WEB_ROOT / "openplc.db"
ACTIVE_PROGRAM = WEB_ROOT / "active_program"
MBCONFIG = WEB_ROOT / "core/mbconfig.cfg"
PYTHON = RUNTIME_ROOT / ".venv/bin/python"


def activation_record() -> dict[str, object]:
    active_file = ACTIVE_PROGRAM.read_text(encoding="utf-8").strip()
    with sqlite3.connect(DATABASE) as connection:
        program = connection.execute(
            "SELECT Name, File FROM Programs WHERE File = ?",
            (active_file,),
        ).fetchone()
        settings = dict(connection.execute(
            "SELECT Key, Value FROM Settings WHERE Key IN ('Start_run_mode', 'Modbus_port')"
        ).fetchall())
    return {
        "active_program_file": active_file,
        "active_program_name": program[0] if program else None,
        "program_registered": program is not None,
        "modbus_port": settings.get("Modbus_port"),
        "start_run_mode": settings.get("Start_run_mode"),
        "core_sha256": hashlib.sha256(CORE.read_bytes()).hexdigest() if CORE.is_file() else None,
    }


def validate_activation() -> dict[str, object]:
    record = activation_record()
    failures = []
    if not record["program_registered"]:
        failures.append("active_program_not_registered")
    if record["modbus_port"] != "1502":
        failures.append("unexpected_modbus_port")
    if not CORE.is_file() or not os.access(CORE, os.X_OK):
        failures.append("core_not_executable")
    if not WEBSERVER.is_file() or not PYTHON.is_file():
        failures.append("web_runtime_missing")
    if not MBCONFIG.is_file():
        failures.append("modbus_master_config_missing")
    record["failures"] = failures
    record["passed"] = not failures
    return record


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(port: int, process: subprocess.Popen[bytes], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"process_exited_before_port_{port}:rc={return_code}")
        if port_open(port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"port_{port}_startup_timeout")


def runtime_rpc(command: str, timeout: float = 5) -> str:
    """Use the same local runtime RPC that OpenPLC's management layer uses."""
    with socket.create_connection(("127.0.0.1", 43628), timeout=timeout) as connection:
        connection.sendall(f"{command}\n".encode("utf-8"))
        connection.settimeout(timeout)
        return connection.recv(4096).decode("utf-8", errors="replace").strip()


def sd_notify(message: str) -> None:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as connection:
        connection.connect(address)
        connection.sendall(message.encode("utf-8"))


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    activation = validate_activation()
    if not activation["passed"]:
        print(json.dumps({"status": "activation_invalid", **activation}, sort_keys=True), flush=True)
        return 2
    if port_open(1502) or port_open(8080):
        print(json.dumps({"status": "port_conflict", "port_1502": port_open(1502),
                          "port_8080": port_open(8080)}, sort_keys=True), flush=True)
        return 3

    stopping = False
    web: subprocess.Popen[bytes] | None = None
    core: subprocess.Popen[bytes] | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        web = subprocess.Popen([str(PYTHON), str(WEBSERVER)], cwd=WEB_ROOT)
        wait_for_port(8080, web)
        core = subprocess.Popen([str(CORE)], cwd=WEB_ROOT)
        wait_for_port(43628, core)
        modbus_rpc_response = runtime_rpc("start_modbus(1502)")
        wait_for_port(1502, core)
        sd_notify("READY=1\nSTATUS=OpenPLC web, core, runtime RPC, and Modbus are ready")
        print(json.dumps({"status": "ready", "web_pid": web.pid, "core_pid": core.pid,
                          "modbus_rpc_response": modbus_rpc_response,
                          **activation}, sort_keys=True), flush=True)
        while not stopping:
            if web.poll() is not None:
                raise RuntimeError(f"webserver_exited:rc={web.returncode}")
            if core.poll() is not None:
                raise RuntimeError(f"plc_core_exited:rc={core.returncode}")
            time.sleep(0.5)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__,
                          "detail": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    finally:
        sd_notify("STOPPING=1\nSTATUS=Stopping OpenPLC lifecycle")
        terminate(core)
        terminate(web)


if __name__ == "__main__":
    raise SystemExit(main())
