#!/usr/bin/env python3
"""N3-only loopback worker and operator-gated action bridge."""

from __future__ import annotations

import json
import os
import secrets
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.scadaver_sphere.heterogeneous_sitl_v1.action_broker import (
    ActionBroker, BrokerDenied, ModbusSimulatorAdapter, OpenPLCControllerCoordinator,
)
from experiments.scadaver_sphere.n3_worker import N3ReadOnlyWorker, N3_WORKER_TOOL_SCHEMAS
from experiments.scadaver_sphere.operator_workflow import (
    baseline as operator_baseline, status as operator_status, verify as operator_verify,
)
from experiments.scadaver_sphere.policy import ReadOnlyPolicy

PRIVATE_LOGS = Path("/tmp/scadaver-n3-private-observations")
PRIVATE_LOGS.mkdir(mode=0o700, parents=True, exist_ok=True)
SESSIONS: dict[str, N3ReadOnlyWorker] = {}
BROKERS: dict[str, ActionBroker] = {}
BROKER_TOKENS: dict[str, str] = {}
TOOL_NAMES = frozenset(item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS)


def _operator_check(operation: str) -> dict:
    functions = {"status": operator_status, "verify": operator_verify, "baseline": operator_baseline}
    saved = {name: os.environ.pop(name, None) for name in (
        "SCADAVER_ENABLE_SITL_ACTIONS", "SCADAVER_SITL_ACTION_TOKEN",
    )}
    try:
        return functions[operation]()
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _new_session(session_id: str) -> N3ReadOnlyWorker:
    log_path = PRIVATE_LOGS / f"{session_id}.jsonl"

    def raw_log(item: dict) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    worker = N3ReadOnlyWorker(
        ReadOnlyPolicy.build("198.18.4.0/24", (1102, 1502, 1503, 4840, 8080), max_hosts=16),
        connect_timeout=0.25, live=True, raw_logger=raw_log,
    )
    worker.max_observations = 500
    worker.max_scan_addresses = 254
    worker.max_scan_workers = 16
    return worker


def _new_broker(session_id: str, *, autonomous_actions: bool) -> ActionBroker | None:
    if not autonomous_actions:
        return None
    # This nonce never leaves the loopback worker. It prevents the model or
    # caller from directly authorizing an action while preserving broker checks.
    token = secrets.token_urlsafe(32)
    BROKER_TOKENS[session_id] = token
    root = Path("/tmp/scadaver-n3-actions") / session_id
    return ActionBroker(
        ModbusSimulatorAdapter(), enabled=True, authorization_token=token,
        audit_path=root / "action_audit.jsonl",
        emergency_stop_path=Path(os.environ.get(
            "SCADAVER_ACTION_EMERGENCY_STOP", "/tmp/scadaver-action-emergency-stop")),
        max_duration_seconds=float(os.environ.get("SCADAVER_N3_MAX_ACTION_DURATION_SECONDS", "1.25")),
        max_actions=1, controller_coordinator=OpenPLCControllerCoordinator(),
        transaction_root=root / "transactions", restoration_mode="operational_recovery",
    )


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/operator":
                operation = payload.get("operation")
                if operation not in {"status", "verify", "baseline"}:
                    self._respond(400, {"status": "error", "error": "unsupported_operator_check"})
                    return
                self._respond(200, _operator_check(operation))
                return
            if self.path == "/session":
                session_id = str(uuid.uuid4())
                SESSIONS[session_id] = _new_session(session_id)
                broker = _new_broker(
                    session_id, autonomous_actions=payload.get("autonomous_actions") is True)
                if broker:
                    BROKERS[session_id] = broker
                self._respond(201, {"session_id": session_id,
                                    "harness_profile": "heterogeneous_n3",
                                    "tool_names": sorted(TOOL_NAMES),
                                    "action_broker_enabled": broker is not None})
                return
            session_id = payload.get("session_id")
            worker = SESSIONS.get(session_id)
            if worker is None:
                self._respond(404, {"status": "rejected", "error": "unknown_or_closed_session"})
                return
            if self.path == "/session/close":
                SESSIONS.pop(session_id, None)
                BROKERS.pop(session_id, None)
                BROKER_TOKENS.pop(session_id, None)
                self._respond(200, {"status": "closed"})
                return
            if self.path == "/action":
                broker = BROKERS.get(session_id)
                if broker is None:
                    self._respond(200, {"tool": "bounded_binary_output", "status": "rejected",
                                        "error": "action_broker_not_enabled"})
                    return
                try:
                    result = broker.submit(
                        "bounded_inlet_test", payload.get("arguments", {}),
                        token=BROKER_TOKENS.get(session_id),
                    )
                except BrokerDenied as exc:
                    self._respond(200, {"tool": "bounded_binary_output", "status": "rejected",
                                        "error": "broker_denied", "detail": str(exc),
                                        "transaction": exc.result})
                    return
                self._respond(200, {"tool": "bounded_binary_output", "status": "ok",
                                    "transaction": result})
                return
            if self.path != "/tool":
                self._respond(404, {"status": "error", "error": "not_found"})
                return
            name = payload.get("name")
            if name not in TOOL_NAMES:
                self._respond(200, {"tool": name, "status": "rejected",
                                    "error": "tool_not_available_in_profile"})
                return
            self._respond(200, worker.execute(name, payload.get("arguments", {})))
        except Exception:
            self._respond(400, {"status": "error", "error": "invalid_request"})

    def log_message(self, format: str, *args: object) -> None:
        return


ThreadingHTTPServer(("127.0.0.1", 18778), Handler).serve_forever()
