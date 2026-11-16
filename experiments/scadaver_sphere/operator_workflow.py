#!/usr/bin/env python3
"""Idempotent Access-VM lifecycle and read-only readiness workflow.

This module never clears the emergency latch, creates an action token, enables
the broker, or launches a model. It is intended to run on the existing Access
VM from the deployed repository bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
VALIDATOR = ROOT / "heterogeneous_sitl_v1/deployment/validate_heterogeneous_sitl.py"
LIFECYCLE = "/home/researcher/scadaver-lab/experiments/scadaver_sphere/heterogeneous_sitl_v1/deployment/validate_openplc_lifecycle.py"
WORKER = ROOT / "node_setup/access_worker_service.py"
WORKER_URL = "http://127.0.0.1:18777"
LATCH = Path(os.environ.get("SCADAVER_ACTION_EMERGENCY_STOP", "/tmp/scadaver-action-emergency-stop"))

SERVICES = {
    "simulator": ("198.18.4.20", "scadaver-simulator.service"),
    "engineering_http": ("198.18.4.40", "scadaver-engineering.service"),
    "engineering_opcua": ("198.18.4.40", "scadaver-opcua.service"),
    "engineering_s7": ("198.18.4.40", "scadaver-s7.service"),
    "openplc": ("198.18.4.10", "scadaver-openplc-controller.service"),
    "legacy_controller": ("198.18.4.10", "scadaver-legacy-controller.service"),
}
PORTS = {
    "controller_modbus": ("198.18.4.10", 1502),
    "controller_http": ("198.18.4.10", 8080),
    "simulator_modbus": ("198.18.4.20", 1503),
    "engineering_http": ("198.18.4.40", 8080),
    "engineering_opcua": ("198.18.4.40", 4840),
    "engineering_s7": ("198.18.4.40", 1102),
    "independent_modbus": ("198.18.4.50", 1502),
}


def run(command: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def remote(host: str, command: str, *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=accept-new", host, command], timeout=timeout)


def service_state(host: str, service: str) -> str:
    result = remote(host, f"systemctl is-active {service}")
    return result.stdout.strip() or "unknown"


def tcp_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def worker_session() -> dict[str, Any]:
    request = Request(WORKER_URL + "/session", method="POST",
                      data=b'{"harness_profile":"heterogeneous_e4"}',
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=4) as response:
            value = json.loads(response.read())
        session_id = value.get("session_id")
        if session_id:
            close = Request(WORKER_URL + "/session/close", method="POST",
                            data=json.dumps({"session_id": session_id}).encode(),
                            headers={"Content-Type": "application/json"})
            with urlopen(close, timeout=4):
                pass
        return {"healthy": True, "profile": value.get("harness_profile"),
                "tool_count": len(value.get("tool_names", [])),
                "tool_names": value.get("tool_names", [])}
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return {"healthy": False, "error": type(exc).__name__}


def lifecycle() -> dict[str, Any]:
    result = remote("198.18.4.10", f"python3 {LIFECYCLE}", timeout=20)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        value = {"passed": False, "error": "invalid_lifecycle_response"}
    value["returncode"] = result.returncode
    return value


def status() -> dict[str, Any]:
    services = {name: service_state(*target) for name, target in SERVICES.items()}
    ports = {name: tcp_open(*target) for name, target in PORTS.items()}
    worker = worker_session()
    action_environment = {
        "enabled": os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") == "1",
        "token_present": bool(os.environ.get("SCADAVER_SITL_ACTION_TOKEN")),
        "emergency_latch_present": LATCH.exists(),
    }
    core = lifecycle()
    mandatory = (all(ports.values()) and worker.get("healthy") is True
                 and core.get("passed") is True
                 and services["legacy_controller"] != "active"
                 and not action_environment["enabled"]
                 and not action_environment["token_present"]
                 and not action_environment["emergency_latch_present"])
    return {"timestamp": time.time(), "services": services, "ports": ports,
            "openplc_lifecycle": core, "worker": worker,
            "action_environment": action_environment,
            "TESTBED_READY": mandatory}


def start() -> dict[str, Any]:
    """Start only missing managed components, then perform strict verification."""
    changes: list[dict[str, Any]] = []
    # Dependencies first. The independent endpoint is bootstrap-managed and is
    # deliberately not recreated here if missing.
    for name in ("simulator", "engineering_http", "engineering_opcua", "engineering_s7"):
        host, service = SERVICES[name]
        if service_state(host, service) != "active":
            result = remote(host, f"sudo systemctl start {service}")
            changes.append({"component": name, "operation": "start", "returncode": result.returncode})
            if result.returncode:
                return {"status": "failed", "changes": changes, "failure": name,
                        "TESTBED_READY": False}
    core = lifecycle()
    if not core.get("passed"):
        # Enforce single-controller order and use the validated service lifecycle.
        stop = remote("198.18.4.10", "sudo systemctl stop scadaver-legacy-controller.service")
        changes.append({"component": "legacy_controller", "operation": "stop", "returncode": stop.returncode})
        result = remote("198.18.4.10", "sudo systemctl restart scadaver-openplc-controller.service")
        changes.append({"component": "openplc", "operation": "validated_service_restart", "returncode": result.returncode})
        if result.returncode:
            return {"status": "failed", "changes": changes, "failure": "openplc_restart",
                    "TESTBED_READY": False}
        time.sleep(1)
    if not worker_session().get("healthy"):
        command = (f"nohup {sys.executable} {WORKER} "
                   ">/tmp/scadaver-access-worker.log 2>&1 </dev/null &")
        result = run(["sh", "-c", command])
        changes.append({"component": "access_worker", "operation": "start", "returncode": result.returncode})
        time.sleep(0.5)
    final = status()
    return {"status": "ready" if final["TESTBED_READY"] else "failed",
            "changes": changes, "verification": final,
            "TESTBED_READY": final["TESTBED_READY"]}


def verify() -> dict[str, Any]:
    snapshot = status()
    result = run([sys.executable, str(VALIDATOR)], timeout=45)
    try:
        acceptance = json.loads(result.stdout)
    except json.JSONDecodeError:
        acceptance = {"error": "invalid_acceptance_output", "stderr": result.stderr[-1000:]}
    passed = snapshot["TESTBED_READY"] and result.returncode == 0
    return {"TESTBED_READY": passed, "status": snapshot,
            "deployed_acceptance": acceptance,
            "acceptance_returncode": result.returncode}


def baseline() -> dict[str, Any]:
    result = verify()
    closed_loop = result.get("deployed_acceptance", {}).get("closed_loop", {})
    samples = closed_loop.get("samples", [])
    result["initial_process_state"] = samples[-1] if samples else None
    result["initial_condition_record"] = {
        "captured_at_unix": time.time(),
        "source": "deployed_read_only_closed_loop_validation",
        "process_sample": samples[-1] if samples else None,
        "openplc_core": result.get("status", {}).get("openplc_lifecycle"),
        "service_health": result.get("status", {}).get("ports"),
        "controller_mode": "automatic_running" if result.get("TESTBED_READY") else "unconfirmed",
        "process_phase": "continuously_evolving_uncontrolled_phase",
        "reset_performed": False,
        "limitations": [
            "No deployed process reset was authorized for this engineering validation.",
            "Comparable future runs must select and record an approved initial-state band.",
        ],
    }
    result["initial_condition_policy"] = (
        "continuous_evolving_state_recorded_no_deterministic_reset_claimed")
    return result


def monitor(path: Path, interval: float, iterations: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with path.open("a", encoding="utf-8") as handle:
        for _ in range(iterations):
            value = status()
            failed = failed or not value["TESTBED_READY"]
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            time.sleep(interval)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "start", "verify", "baseline", "recover"):
        sub.add_parser(name)
    watch = sub.add_parser("monitor")
    watch.add_argument("--out", type=Path, required=True)
    watch.add_argument("--interval-seconds", type=float, default=10)
    watch.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    if args.command == "monitor":
        return monitor(args.out, args.interval_seconds, args.iterations)
    operation: Callable[[], dict[str, Any]] = {
        "status": status, "start": start, "verify": verify,
        "baseline": baseline, "recover": start,
    }[args.command]
    value = operation()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("TESTBED_READY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
