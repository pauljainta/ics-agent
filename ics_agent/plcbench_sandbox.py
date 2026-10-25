"""Locked-down subprocess sandbox for the PLCBench native actor (dev/OpenPLC).

Runs one actor-issued shell or Python command at a time in a constrained child:

- **Allowlisted environment**: the child's env starts empty and is populated only
  with a minimal PATH, HOME/TMPDIR pointed at the scratch workdir, and the broker
  address. Control-plane and credential variables (MRG_SERVER, MERGE_SERVER,
  ICS_AGENT_*, SSH_*, tokens) are therefore absent by construction, not by
  denylist.
- **Workdir confinement**: cwd is a per-episode scratch directory; HOME/TMPDIR
  resolve there too.
- **Resource limits**: CPU seconds and max file size via setrlimit (POSIX).
- **Process group + timeout**: the child runs in its own session; on wall-clock
  timeout the whole group is killed, so a spawned `sleep`/server cannot linger.
- **Output cap**: stdout/stderr are truncated to a byte ceiling.

Scope note: this is the legacy loopback-only subprocess mode. It does not
isolate network egress. Versioned batches use plcbench_container instead;
non-loopback execution remains closed pending independent review. Stdlib-only.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore


@dataclass
class SandboxConfig:
    workdir: Path
    broker_host: str | None = None
    broker_port: int | None = None
    cpu_seconds: int = 10
    wall_timeout_sec: float = 15.0
    output_cap_bytes: int = 64 * 1024
    max_file_bytes: int = 16 * 1024 * 1024
    python_executable: str = field(default_factory=lambda: sys.executable)
    path: str = "/usr/bin:/bin"


@dataclass
class SandboxResult:
    kind: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_sec: float
    truncated: bool


class Sandbox:
    """A reusable per-episode sandbox. Create, run() repeatedly, then cleanup()."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.workdir = Path(config.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._env = self._build_env()

    def _build_env(self) -> dict[str, str]:
        env = {
            "PATH": self.config.path,
            "HOME": str(self.workdir),
            "TMPDIR": str(self.workdir),
            "LANG": "C.UTF-8",
        }
        if self.config.broker_host is not None and self.config.broker_port is not None:
            # The only network endpoint the actor is told about.
            env["PLCBENCH_BROKER_HOST"] = str(self.config.broker_host)
            env["PLCBENCH_BROKER_PORT"] = str(self.config.broker_port)
        return env

    def _preexec(self) -> None:  # pragma: no cover - runs in child
        os.setsid()
        if resource is not None:
            cpu = self.config.cpu_seconds
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            except (ValueError, OSError):
                pass
            try:
                fs = self.config.max_file_bytes
                resource.setrlimit(resource.RLIMIT_FSIZE, (fs, fs))
            except (ValueError, OSError):
                pass

    def _argv(self, kind: str, payload: str) -> list[str]:
        if kind == "shell":
            return ["/bin/sh", "-c", payload]
        if kind == "python":
            return [self.config.python_executable, "-I", "-c", payload]
        raise ValueError(f"unknown sandbox command kind: {kind!r}")

    def run(self, kind: str, payload: str) -> SandboxResult:
        argv = self._argv(kind, payload)
        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=str(self.workdir),
            env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=self._preexec if os.name == "posix" else None,
            start_new_session=False,  # _preexec already calls setsid on POSIX
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=self.config.wall_timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_group(proc)
            try:
                out, err = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                out, err = b"", b""
        duration = time.monotonic() - start

        stdout, t1 = _cap(out, self.config.output_cap_bytes)
        stderr, t2 = _cap(err, self.config.output_cap_bytes)
        return SandboxResult(
            kind=kind,
            ok=(not timed_out and proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_sec=duration,
            truncated=t1 or t2,
        )

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        if os.name != "posix":
            proc.kill()
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def _cap(raw: bytes, cap: int) -> tuple[str, bool]:
    truncated = len(raw) > cap
    text = raw[:cap].decode("utf-8", errors="replace")
    if truncated:
        text += "\n[output truncated]"
    return text, truncated
