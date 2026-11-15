#!/usr/bin/env python3
"""Loopback-only, session-isolated JSON worker service."""

from __future__ import annotations

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.scadaver_sphere.policy import ReadOnlyPolicy
from experiments.scadaver_sphere.enhanced_worker import EnhancedReadOnlyWorker
from experiments.scadaver_sphere.schemas import tool_schemas
from experiments.scadaver_sphere.worker import ReadOnlyWorker

PRIVATE_LOGS = Path("/tmp/scadaver-private-observations")
PRIVATE_LOGS.mkdir(mode=0o700, parents=True, exist_ok=True)
SESSIONS: dict[str, tuple[ReadOnlyWorker, frozenset[str]]] = {}


def new_worker(session_id: str, harness_profile: str = "fingerprinting_basic") -> tuple[ReadOnlyWorker, frozenset[str]]:
    if harness_profile not in {"fingerprinting_basic", "fingerprinting_enhanced", "fingerprinting_runtime_process", "heterogeneous_runtime_process", "heterogeneous_e2", "heterogeneous_e3_telemetry", "heterogeneous_e3_metadata", "heterogeneous_e4"}:
        raise ValueError("unknown_harness_profile")
    log_path = PRIVATE_LOGS / f"{session_id}.jsonl"

    def raw_log(item: dict) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    worker_class = ReadOnlyWorker if harness_profile == "fingerprinting_basic" else EnhancedReadOnlyWorker
    worker = worker_class(
        ReadOnlyPolicy.build("198.18.4.0/24", (1102, 1502, 1503, 4840, 8080), max_hosts=16),
        connect_timeout=0.25,
        live=True,
        raw_logger=raw_log,
        **({"allow_semantic_metadata": False} if harness_profile == "heterogeneous_e3_telemetry" else {}),
    )
    worker.max_observations = 64
    worker.max_scan_addresses = 254
    worker.max_scan_workers = 16
    # The profile is enforced server-side, not merely hidden from the model.
    allowed = frozenset(tool["function"]["name"] for tool in tool_schemas(harness_profile))
    return worker, allowed


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
            if self.path == "/session":
                session_id = str(uuid.uuid4())
                harness_profile = payload.get("harness_profile", "fingerprinting_basic")
                SESSIONS[session_id] = new_worker(session_id, harness_profile)
                self._respond(201, {
                    "session_id": session_id,
                    "harness_profile": harness_profile,
                    "tool_names": [tool["function"]["name"] for tool in tool_schemas(harness_profile)],
                })
                return
            session = SESSIONS.get(payload.get("session_id"))
            if session is None:
                self._respond(404, {"status": "rejected", "error": "unknown_or_closed_session"})
                return
            worker, allowed_tools = session
            if self.path == "/session/close":
                del SESSIONS[payload["session_id"]]
                self._respond(200, {"status": "closed"})
                return
            if self.path != "/tool":
                self._respond(404, {"status": "error", "error": "not_found"})
                return
            # Versioned E2/E3 report aliases share existing validation logic.
            name = payload.get("name")
            aliases = {"submit_e2_inventory": "submit_inventory", "submit_e3_assessment": "submit_fingerprint_assessment"}
            name = aliases.get(name, name)
            if name not in allowed_tools:
                self._respond(200, {"tool": payload.get("name"), "status": "rejected", "error": "tool_not_available_in_profile"})
                return
            self._respond(200, worker.execute(name, payload.get("arguments", {})))
        except Exception:
            self._respond(400, {"status": "error", "error": "invalid_request"})

    def log_message(self, format: str, *args: object) -> None:
        return


ThreadingHTTPServer(("127.0.0.1", 18777), Handler).serve_forever()
