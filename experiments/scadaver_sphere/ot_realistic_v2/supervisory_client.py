#!/usr/bin/env python3
"""Operator CLI for the controller-local supervisory Unix socket."""

from __future__ import annotations

import argparse
import json
import os
import socket

from .constants import SUPERVISORY_SOCKET


def request(payload: dict[str, object], path: str = SUPERVISORY_SOCKET) -> dict[str, object]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    try:
        client.connect(path)
        client.sendall(json.dumps(payload).encode())
        response = json.loads(client.recv(4096).decode())
    finally:
        client.close()
    if response.get("status") != "ok":
        raise RuntimeError(str(response.get("error", "supervisory request rejected")))
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("set", "clear", "status"))
    parser.add_argument("--value", choices=("on", "off"))
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--socket", default=os.environ.get("SCADAVER_V2_SUPERVISORY_SOCKET", SUPERVISORY_SOCKET))
    args = parser.parse_args()
    payload: dict[str, object] = {"operation": args.operation}
    if args.operation == "set":
        if args.value is None or args.duration_seconds is None:
            parser.error("set requires --value and --duration-seconds")
        payload.update(value=args.value == "on", duration_seconds=args.duration_seconds)
    print(json.dumps(request(payload, args.socket), sort_keys=True))


if __name__ == "__main__":
    main()
