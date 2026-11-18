#!/usr/bin/env python3
"""Root/operator-only creation and activation of V2 standing policies."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .standing_authorization import sign_policy


def _secure_key(path: Path) -> bytes:
    stat = path.stat()
    if os.geteuid() != 0 or stat.st_uid != 0 or stat.st_mode & 0o077:
        raise PermissionError("activation requires root and a root-owned mode-0600 signing key")
    value = path.read_bytes()
    if len(value) < 32:
        raise ValueError("signing key must contain at least 32 bytes")
    return value


def activate(pending: Path, approval: Path, output: Path, key_path: Path) -> dict:
    key = _secure_key(key_path)
    policy = json.loads(pending.read_text(encoding="utf-8"))
    approved = json.loads(approval.read_text(encoding="utf-8"))
    if approved.get("approval_status") != "APPROVED" or not approved.get("approval_id"):
        raise ValueError("trusted approval record is absent")
    if approved.get("policy_id") != policy.get("policy_id"):
        raise ValueError("approval does not match policy")
    now = datetime.now(timezone.utc).isoformat()
    policy.update(
        state="APPROVED_ACTIVE", approved_at=approved["approved_at"],
        activated_at=now, approver_identity=approved["approver_identity"],
        approval_id=approved["approval_id"], revoked=False, suspended=False,
    )
    envelope = sign_policy(policy, key)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    return {"policy_id": policy["policy_id"], "state": policy["state"],
            "activated_at": now, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("activate", nargs="?")
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(activate(args.pending, args.approval, args.output, args.signing_key),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
