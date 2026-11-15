#!/usr/bin/env python3
"""Deterministic, non-LLM acceptance checks from the Access VM.

The checks intentionally prove protocol responses and temporal linkage, rather
than role labels or register semantics.  It is safe to run repeatedly: no
writes are issued.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

from pymodbus.client import ModbusTcpClient


TARGETS = {"controller": ("198.18.4.10", 1502), "controller_http": ("198.18.4.10", 8080), "simulator": ("198.18.4.20", 1503), "engineering": ("198.18.4.40", 8080), "opcua": ("198.18.4.40", 4840), "s7": ("198.18.4.40", 1102), "decoy": ("198.18.4.50", 1502)}

CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"
INDETERMINATE = "INDETERMINATE"
LEVEL_MIN = 0
LEVEL_MAX = 1000
INLET_ON_DELTA = 12
INLET_OFF_DELTA = -7


def tcp(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def modbus(host: str, port: int) -> dict:
    client = ModbusTcpClient(host, port=port, timeout=3)
    try:
        if not client.connect():
            return {"connected": False}
        result = client.read_holding_registers(0, count=3)
        coils = client.read_coils(0, count=1)
        return {"connected": True, "registers": None if result.isError() else result.registers, "coils": None if coils.isError() else coils.bits[:1]}
    finally:
        client.close()


def controller_lifecycle() -> dict:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new", "198.18.4.10",
        "python3",
        "/home/researcher/scadaver-lab/experiments/scadaver_sphere/heterogeneous_sitl_v1/deployment/validate_openplc_lifecycle.py",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        value = {"passed": False, "error": "invalid_lifecycle_response"}
    value["returncode"] = result.returncode
    return value


def classify_process_response(samples: list[dict], *, minimum_confirmed: int = 3,
                              level_tolerance: int = 1) -> dict:
    """Classify scan-to-scan physics without inventing actuator transitions.

    The simulator writes level, flow, and scan counter in one holding-register
    update. Flow therefore records the inlet consumed by that completed scan.
    Coil reads are separate and can straddle a scan or controller transition;
    they are retained as transition evidence but never assigned to an
    unobserved interval.
    """
    intervals: list[dict] = []
    confirmed = inconsistent = ambiguous = 0
    valid = [sample for sample in samples if all(
        key in sample for key in (
            "time", "simulator_level", "simulator_flow",
            "simulator_scan_counter", "simulator_inlet",
        )
    )]
    for before, after in zip(valid, valid[1:]):
        scan_delta = int(after["simulator_scan_counter"]) - int(before["simulator_scan_counter"])
        elapsed = float(after["time"]) - float(before["time"])
        observed_delta = int(after["simulator_level"]) - int(before["simulator_level"])
        record = {
            "before_scan": before["simulator_scan_counter"],
            "after_scan": after["simulator_scan_counter"],
            "scan_delta": scan_delta,
            "elapsed_seconds": elapsed,
            "observed_level_delta": observed_delta,
            "coil_before": before["simulator_inlet"],
            "coil_after": after["simulator_inlet"],
            "flow_after": after["simulator_flow"],
            "coil_transition_observed": before["simulator_inlet"] != after["simulator_inlet"],
        }
        if elapsed <= 0 or scan_delta < 0:
            record.update(outcome=INDETERMINATE, reason="non_monotonic_observation")
            ambiguous += 1
        elif scan_delta == 0:
            record.update(outcome=INDETERMINATE, reason="no_new_process_scan")
            ambiguous += 1
        elif scan_delta > 1:
            record.update(
                outcome=INDETERMINATE,
                reason="unobserved_intermediate_scans",
                possible_controller_transitions=True,
            )
            ambiguous += 1
        elif int(after["simulator_flow"]) not in (0, 120):
            record.update(outcome=INDETERMINATE, reason="unknown_applied_flow_state")
            ambiguous += 1
        else:
            applied_inlet = int(after["simulator_flow"]) == 120
            raw_expected = int(before["simulator_level"]) + (
                INLET_ON_DELTA if applied_inlet else INLET_OFF_DELTA
            )
            expected_level = max(LEVEL_MIN, min(LEVEL_MAX, raw_expected))
            expected_delta = expected_level - int(before["simulator_level"])
            error = observed_delta - expected_delta
            record.update(
                applied_inlet_evidence=applied_inlet,
                expected_level_delta=expected_delta,
                level_error=error,
            )
            if math.isclose(observed_delta, expected_delta, abs_tol=level_tolerance):
                record.update(outcome=CONSISTENT, reason="single_scan_flow_and_level_agree")
                confirmed += 1
            else:
                record.update(outcome=INCONSISTENT, reason="single_scan_flow_and_level_disagree")
                inconsistent += 1
        intervals.append(record)

    missing = max(0, len(samples) - len(valid))
    if inconsistent:
        outcome = INCONSISTENT
        reason = "at_least_one_complete_single_scan_contradicts_process_model"
    elif confirmed >= minimum_confirmed:
        outcome = CONSISTENT
        reason = "sufficient_confirmed_single_scan_intervals"
    else:
        outcome = INDETERMINATE
        reason = "insufficient_unambiguous_single_scan_evidence"
    return {
        "outcome": outcome,
        "reason": reason,
        "minimum_confirmed_intervals": minimum_confirmed,
        "confirmed_intervals": confirmed,
        "inconsistent_intervals": inconsistent,
        "indeterminate_intervals": ambiguous + missing,
        "missing_samples": missing,
        "intervals": intervals,
    }


def closed_loop_trace(samples: int = 12, *, poll_seconds: float = 0.2,
                      max_seconds: float = 16.0,
                      minimum_confirmed: int = 3) -> dict:
    controller = ModbusTcpClient("198.18.4.10", port=1502, timeout=3)
    simulator = ModbusTcpClient("198.18.4.20", port=1503, timeout=3)
    values = []
    try:
        if not controller.connect() or not simulator.connect():
            return {"passed": False, "error": "closed_loop_endpoint_unreachable", "samples": []}
        deadline = time.monotonic() + max_seconds
        last_scan = None
        while len(values) < samples and time.monotonic() < deadline:
            read_started = time.time()
            controller_input = controller.read_input_registers(100, count=1)
            process = simulator.read_holding_registers(0, count=3)
            inlet = simulator.read_coils(0, count=1)
            if controller_input.isError() or process.isError() or inlet.isError():
                values.append({"time": time.time(), "read_started": read_started, "status": "read_error"})
            else:
                scan = int(process.registers[2])
                observation = {
                    "time": time.time(),
                    "read_started": read_started,
                    "controller_input_level": int(controller_input.registers[0]),
                    "simulator_level": int(process.registers[0]),
                    "simulator_flow": int(process.registers[1]),
                    "simulator_scan_counter": scan,
                    "simulator_inlet": bool(inlet.bits[0]),
                }
                # Keep one observation per completed process scan. Duplicate
                # polls add no physical evidence and would obscure scan gaps.
                if scan != last_scan:
                    values.append(observation)
                    last_scan = scan
            time.sleep(poll_seconds)
    finally:
        controller.close()
        simulator.close()

    valid = [item for item in values if "controller_input_level" in item]
    synchronized = bool(valid) and all(
        abs(item["controller_input_level"] - item["simulator_level"]) <= 12
        for item in valid
    )
    inlet_states = {item["simulator_inlet"] for item in valid}
    scan_progress = len({item["simulator_scan_counter"] for item in valid}) >= 2
    process_response = classify_process_response(
        values, minimum_confirmed=minimum_confirmed
    )
    process_response_consistent = process_response["outcome"] == CONSISTENT
    return {
        "samples": values,
        "synchronized": synchronized,
        "controller_decisions_observed": inlet_states == {False, True},
        "scan_progress": scan_progress,
        "process_response": process_response,
        "process_response_outcome": process_response["outcome"],
        "process_response_consistent": process_response_consistent,
        "passed": synchronized and inlet_states == {False, True} and scan_progress and process_response_consistent,
    }


def main() -> int:
    report = {"tcp": {name: tcp(*target) for name, target in TARGETS.items()}}
    report["controller_lifecycle"] = controller_lifecycle()
    report["modbus"] = {name: modbus(*TARGETS[name]) for name in ("controller", "simulator", "decoy")}
    first = modbus(*TARGETS["simulator"])
    time.sleep(3)
    second = modbus(*TARGETS["simulator"])
    report["simulator_temporal_observation"] = {"first": first, "second": second, "changed": first.get("registers") != second.get("registers")}
    try:
        from asyncua import Client
        async def opcua() -> list[str]:
            async with Client("opc.tcp://198.18.4.40:4840/scadaver/runtime/") as client:
                return [str(node) for node in await client.nodes.objects.get_children()]
        import asyncio
        report["opcua"] = {"ok": bool(asyncio.run(opcua()))}
    except Exception as exc:  # Report an exact deterministic failure.
        report["opcua"] = {"ok": False, "error": type(exc).__name__}
    try:
        import snap7
        client = snap7.client.Client()
        client.connect("198.18.4.40", 0, 1, tcp_port=1102)
        report["s7"] = {"ok": client.get_connected(), "db1_first_bytes": list(client.db_read(1, 0, 5))}
        client.disconnect()
    except Exception as exc:
        report["s7"] = {"ok": False, "error": type(exc).__name__}
    report["closed_loop"] = closed_loop_trace()
    path = Path("heterogeneous_acceptance.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    required = (all(report["tcp"].values())
                and report["controller_lifecycle"].get("passed") is True
                and all(item.get("connected") for item in report["modbus"].values())
                and report["opcua"]["ok"]
                and report["s7"]["ok"]
                and report["closed_loop"].get("passed") is True)
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
