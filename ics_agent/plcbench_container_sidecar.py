"""Trusted Docker Desktop relay: shared-volume Unix socket to host broker only."""

from __future__ import annotations

import os
import select
import socket
import threading


def handle(client: socket.socket, host: str, port: int) -> None:
    with client, socket.create_connection((host, port), timeout=3) as upstream:
        peers = (client, upstream)
        while True:
            readable, _, _ = select.select(peers, [], [], 1)
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                (upstream if source is client else client).sendall(data)


def main() -> None:
    path = "/bridge/broker.sock"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o777)
    server.listen(16)
    host = os.environ["BROKER_HOST"]
    port = int(os.environ["BROKER_PORT"])
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, host, port), daemon=True).start()


if __name__ == "__main__":
    main()
