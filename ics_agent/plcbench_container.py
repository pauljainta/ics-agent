"""Linux container egress confinement for the PLCBench native child."""

from __future__ import annotations

import json
import os
import platform
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .plcbench_sandbox import SandboxResult, _cap

RELAY_PORT = 37665
SIDECAR_IMAGE = "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce"


def _forward(client: socket.socket, upstream_addr: tuple[str, int]) -> None:
    with client, socket.create_connection(upstream_addr, timeout=3) as upstream:
        peers = (client, upstream)
        while True:
            readable, _, _ = select.select(peers, [], [], 1)
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                (upstream if source is client else client).sendall(data)


class UnixBrokerRelay:
    def __init__(self, socket_path: Path, upstream: tuple[str, int]) -> None:
        self.path, self.upstream = Path(socket_path), upstream
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(5)
        self._server.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=_forward, args=(client, self.upstream), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


class LinuxContainerSandbox:
    """One command per networkless container; only mounted Unix broker path."""

    def __init__(self, workdir: Path, broker_addr: tuple[str, int], *, image: str,
                 wall_timeout_sec: float = 15.0, output_cap_bytes: int = 65536) -> None:
        if not image or image == "latest":
            raise ValueError("container image must be explicit")
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.bridge_dir: Path | None = None
        self.relay: UnixBrokerRelay | None = None
        self.volume: str | None = None
        self.sidecar: str | None = None
        self.bridge_mount: str
        if platform.system() == "Darwin":
            # Docker Desktop cannot bind-mount a macOS AF_UNIX socket into its
            # Linux VM. A trusted relay in a separate container owns the socket
            # on a named volume. The model child still runs with --network none.
            self.volume = f"plcbench-bridge-{uuid.uuid4().hex[:12]}"
            self.sidecar = f"{self.volume}-relay"
            subprocess.run(["docker", "volume", "create", self.volume], check=True,
                           capture_output=True, timeout=10)
            sidecar_script = Path(__file__).with_name("plcbench_container_sidecar.py").resolve()
            subprocess.run(["docker", "run", "-d", "--rm", "--name", self.sidecar,
                "--mount", f"type=volume,src={self.volume},dst=/bridge",
                "--mount", f"type=bind,src={sidecar_script},dst=/sidecar.py,readonly",
                "-e", "BROKER_HOST=host.docker.internal", "-e", f"BROKER_PORT={broker_addr[1]}",
                SIDECAR_IMAGE, "python", "/sidecar.py"], check=True,
                capture_output=True, timeout=15)
            for _ in range(30):
                ready = subprocess.run(["docker", "exec", self.sidecar, "test", "-S",
                                        "/bridge/broker.sock"], capture_output=True, timeout=3)
                if ready.returncode == 0:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Docker Desktop broker sidecar did not start")
            self.bridge_mount = f"type=volume,src={self.volume},dst=/bridge,readonly"
        else:
            self.bridge_dir = Path(tempfile.mkdtemp(prefix="plcbench_bridge_"))
            self.relay = UnixBrokerRelay(self.bridge_dir / "broker.sock", broker_addr)
            self.relay.start()
            self.bridge_mount = f"type=bind,src={self.bridge_dir},dst=/bridge,readonly"
        self.image = image
        self.timeout = wall_timeout_sec
        self.cap = output_cap_bytes
        self.name = f"plcbench-native-{uuid.uuid4().hex[:12]}"

    def run(self, kind: str, payload: str) -> SandboxResult:
        if kind not in ("python", "shell"):
            raise ValueError("unsupported command kind")
        child = Path(__file__).with_name("plcbench_container_child.py").resolve()
        argv = ["docker", "run", "--rm", "--name", self.name,
                "-i", "--network", "none", "--read-only",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--mount", self.bridge_mount,
                "--mount", f"type=bind,src={self.workdir},dst=/work",
                "--mount", f"type=bind,src={child},dst=/runner.py,readonly",
                self.image, "python", "/runner.py"]
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(argv, input=json.dumps({"kind": kind, "payload": payload,
                "port": RELAY_PORT}).encode(), capture_output=True, timeout=self.timeout, check=False)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            code, out, err = None, exc.stdout or b"", exc.stderr or b""
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True,
                           timeout=5, check=False)
        stdout, t1 = _cap(out, self.cap)
        stderr, t2 = _cap(err, self.cap)
        return SandboxResult(kind, not timed_out and code == 0, code, stdout, stderr,
                             timed_out, time.monotonic() - start, t1 or t2)

    def cleanup(self) -> None:
        if self.relay and self.bridge_dir:
            self.relay.stop()
            self.bridge_dir.rmdir()
        if self.sidecar:
            subprocess.run(["docker", "stop", self.sidecar], capture_output=True,
                           timeout=10, check=False)
        if self.volume:
            subprocess.run(["docker", "volume", "rm", self.volume], capture_output=True,
                           timeout=10, check=False)
        shutil.rmtree(self.workdir, ignore_errors=True)
