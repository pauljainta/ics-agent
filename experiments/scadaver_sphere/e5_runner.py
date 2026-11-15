"""E5 simulator-only action executor and independent outcome evaluator."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .e5_evaluation import evaluate_e5
from .heterogeneous_sitl_v1.action_broker import (
    ActionBroker,
    BrokerDenied,
    ModbusSimulatorAdapter,
    OpenPLCControllerCoordinator,
)


class DeterministicAdapter:
    supports_exact_restoration = True
    def __init__(self) -> None: self.inlet, self.level, self.counter = False, 500, 0
    def snapshot(self) -> dict[str, object]: self.counter += 1; return {"inlet": self.inlet, "level_raw": self.level, "flow_raw": 120 if self.inlet else 0, "scan_counter": self.counter, "time": self.counter}
    def set_inlet(self, active: bool) -> None: self.inlet = active; self.level += 12 if active else -7
    def restore(self, snapshot: dict[str, object]) -> None: self.inlet, self.level = bool(snapshot["inlet"]), int(snapshot["level_raw"])
    def restore_actuator(self, snapshot: dict[str, object]) -> None: self.inlet = bool(snapshot["inlet"])
    def pause(self) -> None: pass
    def restore_exact(self, snapshot: dict[str, object]) -> None:
        self.inlet, self.level, self.counter = bool(snapshot["inlet"]), int(snapshot["level_raw"]), int(snapshot["scan_counter"]) - 1
    def resume(self) -> None: pass


def write_final_artifact(out_dir: Path, result: dict[str, object], evaluator=evaluate_e5) -> dict[str, object]:
    try:
        assessment = evaluator(result)
        evaluation_status = "completed"
    except Exception as exc:
        assessment = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
        evaluation_status = "failed"
    document = {"result": result, "evaluation": assessment,
                "evaluation_status": evaluation_status}
    temporary = out_dir / "result.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out_dir / "result.json")
    return document


def verify_post_restoration_health(adapter: object) -> dict[str, object]:
    first = adapter.snapshot()  # type: ignore[attr-defined]
    live = isinstance(adapter, ModbusSimulatorAdapter)
    if live:
        time.sleep(1.1)
    elif callable(getattr(adapter, "advance_time", None)):
        adapter.advance_time(1.1)  # type: ignore[attr-defined]
    second = adapter.snapshot()  # type: ignore[attr-defined]
    scan_advanced = isinstance(first.get("scan_counter"), int) and isinstance(second.get("scan_counter"), int) and second["scan_counter"] > first["scan_counter"]
    level = second.get("level_raw")
    safe = isinstance(level, (int, float)) and 0 <= level <= 1000
    controller_level: int | None = None
    controller_tracks_process: bool | None = None
    if live:
        from pymodbus.client import ModbusTcpClient
        client = ModbusTcpClient("198.18.4.10", port=1502, timeout=2)
        try:
            if client.connect():
                response = client.read_input_registers(100, count=1)
                if not response.isError():
                    controller_level = int(response.registers[0])
                    controller_tracks_process = abs(controller_level - float(level)) <= 12
        finally:
            client.close()
    passed = scan_advanced and safe and (controller_tracks_process is not False) and (not live or controller_tracks_process is True)
    return {"first": first, "second": second, "scan_advanced": scan_advanced, "safe_envelope": safe,
            "controller_input_level": controller_level, "controller_tracks_process_within_one_scan": controller_tracks_process,
            "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--active", choices=("true", "false", "toggle"), required=True); parser.add_argument("--duration-seconds", type=float, default=1); parser.add_argument("--mock", action="store_true"); parser.add_argument("--execute-live", action="store_true"); args = parser.parse_args()
    if args.execute_live == args.mock: parser.error("choose exactly one of --mock or --execute-live")
    if args.out_dir.exists() and any(args.out_dir.iterdir()): parser.error("--out-dir must be new or empty")
    if args.execute_live and os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") != "1": parser.error("live E5 requires SCADAVER_ENABLE_SITL_ACTIONS=1")
    token = os.environ.get("SCADAVER_SITL_ACTION_TOKEN") or ("mock-token" if args.mock else None)
    adapter = DeterministicAdapter() if args.mock else ModbusSimulatorAdapter()
    stop_path = args.out_dir / "emergency_stop" if args.mock else Path(os.environ.get("SCADAVER_ACTION_EMERGENCY_STOP", "/tmp/scadaver-action-emergency-stop"))
    broker = ActionBroker(
        adapter,
        enabled=True,
        authorization_token=token,
        audit_path=args.out_dir / "action_audit.jsonl",
        emergency_stop_path=stop_path,
        max_duration_seconds=args.duration_seconds,
        max_actions=1,
        controller_coordinator=OpenPLCControllerCoordinator() if args.execute_live else None,
        transaction_root=args.out_dir,
        restoration_mode="operational_recovery" if args.execute_live else "exact_snapshot",
        require_physical_effect=args.execute_live,
    )
    active = "toggle" if args.active == "toggle" else args.active == "true"
    args.out_dir.mkdir(parents=True)
    try:
        result = broker.submit("bounded_inlet_test", {"active": active, "duration_seconds": args.duration_seconds}, token=token)
    except BrokerDenied as exc:
        result = exc.result or {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
        write_final_artifact(args.out_dir, result)
        raise
    result["post_restoration_health"] = verify_post_restoration_health(adapter)
    document = write_final_artifact(args.out_dir, result)
    print(json.dumps(document["evaluation"], indent=2))


if __name__ == "__main__": main()
