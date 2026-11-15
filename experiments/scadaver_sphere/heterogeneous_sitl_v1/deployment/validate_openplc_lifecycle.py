#!/usr/bin/env python3
"""Fail closed unless the complete OpenPLC lifecycle is genuinely healthy."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

try:
    from .openplc_lifecycle import activation_record, port_open
except ImportError:
    from openplc_lifecycle import activation_record, port_open


def core_processes() -> list[dict[str, object]]:
    result = subprocess.run(
        ["pgrep", "-af", "/opt/openplc-v3/webserver/core/openplc"],
        capture_output=True,
        text=True,
        check=False,
    )
    values = []
    for line in result.stdout.splitlines():
        pid, _, command = line.partition(" ")
        if pid.isdigit() and command.strip().endswith("/opt/openplc-v3/webserver/core/openplc"):
            values.append({"pid": int(pid), "command": command.strip()})
    return values


def service_state(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "is-active", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def health_snapshot() -> dict[str, object]:
    activation = activation_record()
    cores = core_processes()
    snapshot = {
        "activation": activation,
        "core_processes": cores,
        "core_process_count": len(cores),
        "http_8080_open": port_open(8080),
        "runtime_rpc_43628_open": port_open(43628),
        "modbus_1502_open": port_open(1502),
        "openplc_controller_state": service_state("scadaver-openplc-controller.service"),
        "legacy_controller_state": service_state("scadaver-legacy-controller.service"),
        "time": time.time(),
    }
    snapshot["passed"] = bool(
        activation.get("program_registered") is True
        and activation.get("modbus_port") == "1502"
        and len(cores) == 1
        and snapshot["http_8080_open"] is True
        and snapshot["runtime_rpc_43628_open"] is True
        and snapshot["modbus_1502_open"] is True
        and snapshot["openplc_controller_state"] in {"active", "activating"}
        and snapshot["legacy_controller_state"] != "active"
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        snapshot = health_snapshot()
        if snapshot["passed"] or time.monotonic() >= deadline:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return 0 if snapshot["passed"] else 1
        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
