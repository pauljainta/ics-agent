"""Entry point inside a --network none Linux container.

Relays only internal 127.0.0.1 TCP to the mounted broker Unix socket, then
executes one bounded actor command. The host invokes this file, never the model.
"""

from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import threading


def _pump(left: socket.socket, right: socket.socket) -> None:
    peers = (left, right)
    while True:
        readable, _, _ = select.select(peers, [], [], 1)
        for source in readable:
            data = source.recv(65536)
            if not data:
                return
            (right if source is left else left).sendall(data)


def relay(unix_path: str, port: int, ready: threading.Event, stop: threading.Event) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", port))
    server.listen(5)
    server.settimeout(0.2)
    ready.set()
    try:
        while not stop.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            with client, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                upstream.connect(unix_path)
                _pump(client, upstream)
    finally:
        server.close()


def main() -> int:
    command = json.load(sys.stdin)
    port = int(command["port"])
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(target=relay, args=("/bridge/broker.sock", port, ready, stop), daemon=True)
    thread.start()
    if not ready.wait(3):
        raise RuntimeError("internal loopback relay did not start")
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/work", "TMPDIR": "/tmp",
           "PLCBENCH_BROKER_HOST": "127.0.0.1", "PLCBENCH_BROKER_PORT": str(port)}
    argv = ([sys.executable, "-I", "-c", command["payload"]] if command["kind"] == "python"
            else ["/bin/sh", "-c", command["payload"]] if command["kind"] == "shell" else None)
    if argv is None:
        raise ValueError("unsupported command kind")
    try:
        return subprocess.run(argv, cwd="/work", env=env, check=False).returncode
    finally:
        stop.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
