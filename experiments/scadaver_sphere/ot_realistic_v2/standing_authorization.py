"""Signed standing-policy enforcement for bounded V2 laboratory actions.

Policy activation is intentionally separate from policy enforcement.  The
operator service can validate an already approved policy, but cannot approve
or activate one itself.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


POLICY_STATES = frozenset({
    "UNCONFIGURED", "PENDING_APPROVAL", "APPROVED_ACTIVE", "EXPIRED",
    "REVOKED", "SUSPENDED",
})
REQUIRED_READINESS_CHECKS = frozenset({
    "controller_healthy", "simulator_healthy", "process_within_bounds",
    "emergency_stop_inactive", "no_unresolved_transaction",
    "restoration_available",
})


class AuthorizationDenied(PermissionError):
    """A standing-policy or episode-capability gate failed closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _parse_time(value: Any) -> float:
    if not isinstance(value, str):
        raise AuthorizationDenied("policy timestamp is missing")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise AuthorizationDenied("policy timestamp is invalid") from exc


def sign_policy(policy: dict[str, Any], key: bytes) -> dict[str, Any]:
    """Return a signed policy envelope; callers remain responsible for approval."""
    signature = hmac.new(key, _canonical(policy), hashlib.sha256).hexdigest()
    return {"policy": policy, "signature": signature, "algorithm": "HMAC-SHA256"}


@dataclass
class StandingPolicyAuthority:
    policy_path: Path
    signing_key: bytes
    ledger_path: Path
    expected_realization: str
    now: Callable[[], float] = time.time
    capability_lifetime_seconds: float = 120.0

    def policy_status(self) -> dict[str, Any]:
        try:
            policy, digest = self._load_policy()
            state = self._effective_state(policy)
            return {
                "configured": True,
                "policy_id": policy.get("policy_id"),
                "policy_version": policy.get("policy_version"),
                "state": state,
                "realization": policy.get("realization"),
                "operation": policy.get("operation"),
                "max_duration_seconds": policy.get("max_duration_seconds"),
                "max_actions_per_episode": policy.get("max_actions_per_episode"),
                "expires_at": policy.get("expires_at"),
                "policy_digest": digest,
            }
        except AuthorizationDenied as exc:
            return {"configured": self.policy_path.exists(), "state": "UNCONFIGURED",
                    "error": str(exc)}

    def issue_capability(
        self, episode_id: str, operation: str, readiness: dict[str, bool],
    ) -> dict[str, Any]:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise AuthorizationDenied("episode identifier is required")
        policy, digest = self._active_policy(operation)
        missing = sorted(REQUIRED_READINESS_CHECKS - {
            name for name, passed in readiness.items() if passed is True
        })
        if missing:
            raise AuthorizationDenied("readiness checks failed: " + ",".join(missing))
        now = self.now()
        with self._locked_ledger() as ledger:
            self._expire_issued(ledger, now)
            unresolved = [item for item in ledger["capabilities"]
                          if item["status"] in {"issued", "consumed"}]
            if unresolved:
                raise AuthorizationDenied("previous transaction is unresolved")
            per_episode = sum(item.get("episode_id") == episode_id
                              for item in ledger["capabilities"])
            if per_episode >= int(policy["max_actions_per_episode"]):
                raise AuthorizationDenied("per-episode action limit exceeded")
            period = float(policy["aggregate_period_seconds"])
            used = sum(
                item.get("consumed_at", 0) >= now - period
                for item in ledger["capabilities"]
                if item.get("status") in {"consumed", "closed", "unresolved"}
            )
            if used >= int(policy["max_aggregate_actions"]):
                raise AuthorizationDenied("aggregate action limit exceeded")
            capability_id = str(uuid.uuid4())
            expires = min(now + self.capability_lifetime_seconds,
                          _parse_time(policy["expires_at"]))
            claims = {
                "capability_id": capability_id,
                "episode_id": episode_id,
                "policy_id": policy["policy_id"],
                "policy_version": policy["policy_version"],
                "policy_digest": digest,
                "realization": policy["realization"],
                "operation": operation,
                "max_duration_seconds": policy["max_duration_seconds"],
                "issued_at": now,
                "expires_at": expires,
                "nonce": str(uuid.uuid4()),
            }
            token = self._sign_capability(claims)
            ledger["capabilities"].append({
                "capability_id": capability_id, "episode_id": episode_id,
                "operation": operation, "issued_at": now, "expires_at": expires,
                "status": "issued", "policy_digest": digest,
            })
        return {"capability": token, "capability_id": capability_id,
                "expires_at_unix": expires, "policy_id": policy["policy_id"]}

    def consume_capability(
        self, token: str, episode_id: str, operation: str,
    ) -> dict[str, Any]:
        claims = self._verify_capability(token)
        policy, digest = self._active_policy(operation)
        now = self.now()
        if claims.get("episode_id") != episode_id or claims.get("operation") != operation:
            raise AuthorizationDenied("capability scope mismatch")
        if claims.get("realization") != self.expected_realization:
            raise AuthorizationDenied("capability realization mismatch")
        if claims.get("policy_digest") != digest:
            raise AuthorizationDenied("capability policy is no longer current")
        if now >= float(claims.get("expires_at", 0)):
            raise AuthorizationDenied("capability expired")
        with self._locked_ledger() as ledger:
            record = next((item for item in ledger["capabilities"]
                           if item["capability_id"] == claims.get("capability_id")), None)
            if not record or record.get("status") != "issued":
                raise AuthorizationDenied("capability is unknown or already used")
            record.update(status="consumed", consumed_at=now)
        return {"capability_id": claims["capability_id"], "policy": policy,
                "claims": claims}

    def close_capability(self, capability_id: str, *, recovered: bool,
                         outcome: str) -> None:
        with self._locked_ledger() as ledger:
            record = next((item for item in ledger["capabilities"]
                           if item["capability_id"] == capability_id), None)
            if not record or record.get("status") != "consumed":
                raise AuthorizationDenied("consumed capability record is missing")
            record.update(status="closed" if recovered else "unresolved",
                          closed_at=self.now(), recovered=bool(recovered),
                          outcome=str(outcome)[:128])

    def sanitized_audit(self) -> dict[str, Any]:
        with self._locked_ledger() as ledger:
            return json.loads(json.dumps(ledger))

    def _load_policy(self) -> tuple[dict[str, Any], str]:
        if not self.policy_path.is_file():
            raise AuthorizationDenied("standing policy is not configured")
        try:
            envelope = json.loads(self.policy_path.read_text(encoding="utf-8"))
            policy, signature = envelope["policy"], envelope["signature"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AuthorizationDenied("standing policy envelope is invalid") from exc
        if not isinstance(policy, dict) or not isinstance(signature, str):
            raise AuthorizationDenied("standing policy envelope is invalid")
        expected = hmac.new(self.signing_key, _canonical(policy), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise AuthorizationDenied("standing policy signature is invalid")
        required = {
            "policy_id", "policy_version", "state", "realization", "environment",
            "operation", "max_duration_seconds", "max_actions_per_episode",
            "max_aggregate_actions", "aggregate_period_seconds", "safety_envelope",
            "controller_health_required", "simulator_health_required",
            "restoration_required", "emergency_stop_behavior", "created_at",
            "approved_at", "activated_at", "expires_at", "approver_identity",
            "revoked", "suspended", "audit",
        }
        if required - set(policy):
            raise AuthorizationDenied("standing policy is missing required fields")
        return policy, hashlib.sha256(_canonical(policy)).hexdigest()

    def _effective_state(self, policy: dict[str, Any]) -> str:
        state = policy.get("state")
        if state not in POLICY_STATES:
            raise AuthorizationDenied("standing policy state is invalid")
        if policy.get("revoked") is True:
            return "REVOKED"
        if policy.get("suspended") is True:
            return "SUSPENDED"
        if self.now() >= _parse_time(policy.get("expires_at")):
            return "EXPIRED"
        return str(state)

    def _active_policy(self, operation: str) -> tuple[dict[str, Any], str]:
        policy, digest = self._load_policy()
        if self._effective_state(policy) != "APPROVED_ACTIVE":
            raise AuthorizationDenied("standing policy is not approved and active")
        if policy.get("realization") != self.expected_realization:
            raise AuthorizationDenied("standing policy realization mismatch")
        if policy.get("operation") != operation:
            raise AuthorizationDenied("operation is outside standing policy")
        if not all(isinstance(policy.get(name), str) and policy[name].strip()
                   for name in ("approved_at", "activated_at", "approver_identity")):
            raise AuthorizationDenied("standing policy lacks authenticated approval provenance")
        if float(policy.get("max_duration_seconds", 0)) > 1.25:
            raise AuthorizationDenied("standing policy exceeds implementation safety bound")
        return policy, digest

    def _sign_capability(self, claims: dict[str, Any]) -> str:
        payload = _b64(_canonical(claims))
        signature = _b64(hmac.new(self.signing_key, payload.encode(), hashlib.sha256).digest())
        return payload + "." + signature

    def _verify_capability(self, token: str) -> dict[str, Any]:
        try:
            payload, signature = token.split(".", 1)
            expected = _b64(hmac.new(self.signing_key, payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                raise AuthorizationDenied("capability signature is invalid")
            claims = json.loads(_unb64(payload))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuthorizationDenied("capability is malformed") from exc
        if not isinstance(claims, dict):
            raise AuthorizationDenied("capability is malformed")
        return claims

    def _expire_issued(self, ledger: dict[str, Any], now: float) -> None:
        for record in ledger["capabilities"]:
            if record.get("status") == "issued" and now >= float(record["expires_at"]):
                record.update(status="expired", closed_at=now)

    class _LedgerContext:
        def __init__(self, authority: "StandingPolicyAuthority") -> None:
            self.authority = authority
            self.handle: Any = None
            self.ledger: dict[str, Any] = {}

        def __enter__(self) -> dict[str, Any]:
            path = self.authority.ledger_path
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.with_suffix(path.suffix + ".lock")
            self.handle = lock.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            if path.is_file():
                self.ledger = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.ledger = {"version": 1, "capabilities": []}
            return self.ledger

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            if exc_type is None:
                path = self.authority.ledger_path
                fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(self.ledger, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _locked_ledger(self) -> "StandingPolicyAuthority._LedgerContext":
        return self._LedgerContext(self)
