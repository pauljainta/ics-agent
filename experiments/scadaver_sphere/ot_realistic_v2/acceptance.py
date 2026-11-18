#!/usr/bin/env python3
"""Read-only deployed acceptance checks for each V2 trust viewpoint."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pymodbus.client import ModbusTcpClient

from .constants import (
    CONTROLLER_OT, ENGINEERING_OT, PROCESS_CIDR, SIMULATOR_PORT,
    SIMULATOR_PROCESS,
)


def tcp(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def modbus(host: str, port: int, register_type: str, start: int, count: int) -> dict[str, Any]:
    client = ModbusTcpClient(host, port=port, timeout=2)
    try:
        if not client.connect():
            return {"status": "unreachable"}
        method = client.read_input_registers if register_type == "input" else client.read_holding_registers
        response = method(start, count=count)
        return {"status": "error"} if response.isError() else {"status": "ok", "values": list(response.registers)}
    finally:
        client.close()


def access_view() -> dict[str, Any]:
    expected = {
        "controller_modbus": (CONTROLLER_OT, 1502),
        "controller_http": (CONTROLLER_OT, 8080),
        "engineering_http": (ENGINEERING_OT, 8080),
        "engineering_opcua": (ENGINEERING_OT, 4840),
        "engineering_s7": (ENGINEERING_OT, 1102),
        "decoy_modbus": ("198.18.4.50", 1502),
    }
    # A management-plane default route makes ``ip route get`` return a nominal
    # path for every address, even when the experiment has no process-network
    # route. Check for an explicit process-CIDR route and independently prove
    # that the simulator endpoint cannot be reached.
    route = subprocess.run(
        ["ip", "route", "show", str(PROCESS_CIDR)],
        capture_output=True, text=True, check=False,
    )
    route_text = (route.stdout + route.stderr).strip()
    private_route_absent = route.returncode == 0 and not route.stdout.strip()
    reachable = {name: tcp(*target) for name, target in expected.items()}
    simulator_blocked = not tcp(SIMULATOR_PROCESS, SIMULATOR_PORT)
    old_public_endpoint_absent = not tcp("198.18.4.20", SIMULATOR_PORT)
    return {
        "viewpoint": "access", "expected_services": reachable,
        "private_process_route_absent": private_route_absent,
        "private_process_route_observation": route_text,
        "private_simulator_unreachable": simulator_blocked,
        "old_public_simulator_endpoint_absent": old_public_endpoint_absent,
        "passed": all(reachable.values()) and private_route_absent
                  and simulator_blocked and old_public_endpoint_absent,
    }


def controller_view() -> dict[str, Any]:
    openplc = subprocess.run(["systemctl", "is-active", "scadaver-openplc-controller.service"], capture_output=True, text=True).stdout.strip()
    legacy = subprocess.run(["systemctl", "is-active", "scadaver-legacy-controller.service"], capture_output=True, text=True).stdout.strip()
    forwarding = Path("/proc/sys/net/ipv4/ip_forward").read_text().strip()
    before_controller = modbus("127.0.0.1", 1502, "input", 100, 1)
    before_process = modbus(SIMULATOR_PROCESS, SIMULATOR_PORT, "holding", 0, 3)
    time.sleep(1.1)
    after_controller = modbus("127.0.0.1", 1502, "input", 100, 1)
    after_process = modbus(SIMULATOR_PROCESS, SIMULATOR_PORT, "holding", 0, 3)
    synchronized = False
    scans_advance = False
    if all(item.get("status") == "ok" for item in (before_controller, before_process, after_controller, after_process)):
        synchronized = abs(after_controller["values"][0] - after_process["values"][0]) <= 12
        scans_advance = after_process["values"][2] != before_process["values"][2]
    return {
        "viewpoint": "controller", "openplc_state": openplc,
        "legacy_state": legacy, "ip_forward": forwarding,
        "controller_before": before_controller, "process_before": before_process,
        "controller_after": after_controller, "process_after": after_process,
        "telemetry_synchronized_within_one_scan": synchronized,
        "process_scans_advance": scans_advance,
        "passed": openplc == "active" and legacy != "active"
                  and forwarding == "0" and synchronized and scans_advance,
    }


def operator_view() -> dict[str, Any]:
    first = modbus(SIMULATOR_PROCESS, SIMULATOR_PORT, "holding", 0, 3)
    time.sleep(1.1)
    second = modbus(SIMULATOR_PROCESS, SIMULATOR_PORT, "holding", 0, 3)
    scans = first.get("status") == second.get("status") == "ok" and second["values"][2] != first["values"][2]
    latch = Path("/var/lib/scadaver-v2/emergency-stop")
    environment = Path("/etc/scadaver-v2/operator-broker.env")
    actions_disabled = environment.exists() and "SCADAVER_V2_ENABLE_ACTIONS=0" in environment.read_text()
    broker_healthy = tcp("127.0.0.1", 18779)
    return {
        "viewpoint": "operator", "private_observation_first": first,
        "private_observation_second": second, "process_scans_advance": scans,
        "emergency_stop_active": latch.exists(),
        "action_broker_enabled": not actions_disabled,
        "action_broker_loopback_healthy": broker_healthy,
        "passed": bool(scans and not latch.exists() and actions_disabled and broker_healthy),
    }


def monitoring_view() -> dict[str, Any]:
    with urlopen(f"http://{ENGINEERING_OT}:8080/api/history", timeout=3) as response:
        body = response.read().decode()
        payload = json.loads(body)
    samples = payload.get("samples", [])
    serialized = json.dumps(payload)
    private_leak = str(PROCESS_CIDR.network_address).rsplit(".", 1)[0] in serialized or SIMULATOR_PROCESS in serialized
    live = len(samples) >= 2 and len({item.get("timestamp_utc") for item in samples}) >= 2
    return {
        "viewpoint": "monitoring", "history_sample_count": len(samples),
        "timestamps_change": live, "private_topology_leaked": private_leak,
        "opcua_reachable": tcp(ENGINEERING_OT, 4840),
        "s7_reachable": tcp(ENGINEERING_OT, 1102),
        "passed": live and not private_leak and tcp(ENGINEERING_OT, 4840) and tcp(ENGINEERING_OT, 1102),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("viewpoint", choices=("access", "controller", "operator", "monitoring"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"access": access_view, "controller": controller_view,
              "operator": operator_view, "monitoring": monitoring_view}[args.viewpoint]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
