#!/usr/bin/env python3
"""Loopback-only V2 Access-VM worker with no process-network action bridge."""

from __future__ import annotations

import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..n3_worker import N3_WORKER_TOOL_SCHEMAS
from .worker import build_worker

TOOLS = frozenset(item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS)
SESSIONS: dict[str, object] = {}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/session":
                session = str(uuid.uuid4())
                worker = build_worker(connect_timeout=0.25, live=True)
                worker.max_observations = 500
                worker.max_scan_addresses = 254
                worker.max_scan_workers = 16
                SESSIONS[session] = worker
                self.respond(201, {"session_id": session,
                                   "harness_profile": "ot_realistic_v2",
                                   "tool_names": sorted(TOOLS),
                                   "action_broker_enabled": False})
                return
            session = payload.get("session_id")
            worker = SESSIONS.get(session)
            if worker is None:
                self.respond(404, {"status": "rejected", "error": "unknown_or_closed_session"})
                return
            if self.path == "/session/close":
                SESSIONS.pop(session, None)
                self.respond(200, {"status": "closed"})
                return
            if self.path != "/tool":
                self.respond(404, {"status": "error", "error": "not_found"})
                return
            name = payload.get("name")
            if name not in TOOLS:
                self.respond(200, {"tool": name, "status": "rejected",
                                   "error": "tool_not_available_in_profile"})
                return
            self.respond(200, worker.execute(name, payload.get("arguments", {})))
        except Exception:
            self.respond(400, {"status": "error", "error": "invalid_request"})

    def log_message(self, format: str, *args: object) -> None:
        return


ThreadingHTTPServer(("127.0.0.1", 18780), Handler).serve_forever()
