"""Fixed-target TCP relay for a disposable internal OpenPLC pair."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class Relay(socketserver.BaseRequestHandler):
    target: tuple[str, int]

    def handle(self) -> None:
        with socket.create_connection(self.target, timeout=5) as upstream:
            peers = (self.request, upstream)
            while True:
                readable, _, _ = select.select(peers, [], [], 900)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("controller:502", "simulator:502"), required=True)
    args = parser.parse_args()
    host, port = args.target.split(":")
    Relay.target = (host, int(port))
    with Server(("0.0.0.0", 502), Relay) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
