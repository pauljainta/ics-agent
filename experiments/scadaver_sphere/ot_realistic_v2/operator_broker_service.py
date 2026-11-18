#!/usr/bin/env python3
"""Loopback-only operator service hosted on the private evaluator node."""

from __future__ import annotations

import json
import hmac
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .action_broker import (
    BrokerDenied, ControllerMediatedBroker, OpenPLCHealth,
    PrivateModbusObserver, SSHControllerCommand,
)
from .standing_authorization import AuthorizationDenied, StandingPolicyAuthority

AUTHORIZATION = os.environ.get("SCADAVER_V2_ACTION_TOKEN")
RUNNER_TOKEN = os.environ.get("SCADAVER_V2_RUNNER_TOKEN")
POLICY_PATH = os.environ.get("SCADAVER_V2_STANDING_POLICY")
POLICY_KEY_PATH = os.environ.get("SCADAVER_V2_POLICY_SIGNING_KEY")
REALIZATION = os.environ.get("SCADAVER_V2_REALIZATION", "v2.scadaverotv2.lgarcia")


def _standing_authority() -> StandingPolicyAuthority | None:
    if not POLICY_PATH or not POLICY_KEY_PATH:
        return None
    return StandingPolicyAuthority(
        Path(POLICY_PATH), Path(POLICY_KEY_PATH).read_bytes(),
        Path(os.environ.get("SCADAVER_V2_POLICY_LEDGER",
                            "/var/lib/scadaver-v2/standing-policy-ledger.json")),
        REALIZATION,
    )


AUTHORITY = _standing_authority()
if not AUTHORIZATION and AUTHORITY is None:
    raise SystemExit("a static action token or standing-policy configuration is required")

TRANSACTION_ROOT = Path(os.environ.get(
    "SCADAVER_V2_TRANSACTION_ROOT", "/var/lib/scadaver-v2/transactions"))
EMERGENCY_STOP = Path(os.environ.get(
    "SCADAVER_V2_EMERGENCY_STOP", "/var/lib/scadaver-v2/emergency-stop"))


def _broker(enabled: bool, token: str) -> ControllerMediatedBroker:
    return ControllerMediatedBroker(
        PrivateModbusObserver(), SSHControllerCommand(), OpenPLCHealth(),
        enabled=enabled, authorization_token=token,
        transaction_root=TRANSACTION_ROOT, emergency_stop_path=EMERGENCY_STOP,
    )


BROKER = _broker(os.environ.get("SCADAVER_V2_ENABLE_ACTIONS") == "1",
                 AUTHORIZATION or "standing-policy")


def _readiness() -> dict[str, bool]:
    observer = PrivateModbusObserver()
    command = SSHControllerCommand()
    health = OpenPLCHealth().health()
    try:
        snapshot = observer.snapshot()
        simulator_healthy = int(snapshot["scan_counter"]) >= 0
        process_safe = 0 <= int(snapshot["level_raw"]) <= 1000
    except (OSError, KeyError, TypeError, ValueError):
        simulator_healthy = process_safe = False
    try:
        command_idle = command.status().get("active") is False
    except (OSError, KeyError):
        command_idle = False
    return {
        "controller_healthy": health.get("passed") is True,
        "simulator_healthy": simulator_healthy,
        "process_within_bounds": process_safe,
        "emergency_stop_inactive": not EMERGENCY_STOP.exists(),
        "no_unresolved_transaction": command_idle,
        "restoration_available": command_idle,
    }


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/policy-status":
            value = AUTHORITY.policy_status() if AUTHORITY else {
                "configured": False, "state": "UNCONFIGURED"}
            self.respond(200, value)
            return
        if self.path == "/capability":
            header = self.headers.get("Authorization", "")
            if (AUTHORITY is None or not RUNNER_TOKEN
                    or not hmac.compare_digest(header, f"Bearer {RUNNER_TOKEN}")):
                self.respond(403, {"status": "rejected", "error": "trusted_runner_required"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                value = AUTHORITY.issue_capability(
                    str(payload.get("episode_id", "")),
                    str(payload.get("operation", "")), _readiness())
                self.respond(201, {"status": "issued", **value})
            except AuthorizationDenied as exc:
                self.respond(403, {"status": "rejected", "error": "policy_denied",
                                   "detail": str(exc)})
            except Exception:
                self.respond(400, {"status": "error", "error": "invalid_request"})
            return
        if self.path != "/controller-action":
            self.respond(404, {"status": "error", "error": "not_found"})
            return
        header = self.headers.get("Authorization", "")
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            episode_id = str(payload.get("episode_id", ""))
            capability_id = None
            standing = AUTHORITY is not None and isinstance(payload.get("capability"), str)
            if standing:
                consumed = AUTHORITY.consume_capability(
                    payload["capability"], episode_id, "bounded_controller_output")
                capability_id = consumed["capability_id"]
                duration = float(payload.get("arguments", {}).get("duration_seconds", 0))
                if duration > float(consumed["claims"]["max_duration_seconds"]):
                    AUTHORITY.close_capability(
                        capability_id, recovered=True,
                        outcome="capability_duration_exceeded")
                    capability_id = None
                    raise AuthorizationDenied(
                        "requested duration exceeds standing-policy capability")
                broker = _broker(True, "standing-policy")
                broker_token = "standing-policy"
            elif AUTHORIZATION and hmac.compare_digest(header, f"Bearer {AUTHORIZATION}"):
                broker, broker_token = BROKER, AUTHORIZATION
            else:
                raise AuthorizationDenied("authorization is required")
            result = broker.submit(
                "bounded_controller_output", payload.get("arguments", {}),
                token=broker_token,
            )
            if capability_id:
                AUTHORITY.close_capability(
                    capability_id, recovered=result.get("restoration_verified") is True,
                    outcome=result.get("status", "completed"))
            self.respond(200, {"tool": "bounded_controller_output", "status": "ok",
                               "transaction": result})
        except BrokerDenied as exc:
            if 'capability_id' in locals() and capability_id:
                transaction = exc.result
                resolved = (transaction.get("action_accepted") is not True
                            or transaction.get("restoration_verified") is True)
                AUTHORITY.close_capability(capability_id, recovered=resolved,
                                           outcome="broker_denied")
            self.respond(200, {"tool": "bounded_controller_output", "status": "rejected",
                               "error": "broker_denied", "detail": str(exc),
                               "transaction": exc.result})
        except AuthorizationDenied as exc:
            self.respond(403, {"status": "rejected", "error": "authorization_required",
                               "detail": str(exc)})
        except Exception:
            self.respond(400, {"status": "error", "error": "invalid_request"})

    def log_message(self, format: str, *args: object) -> None:
        return


ThreadingHTTPServer(("127.0.0.1", 18779), Handler).serve_forever()
