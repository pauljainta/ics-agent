"""Tests for the native-actor subprocess sandbox (POSIX)."""

from __future__ import annotations

import os
import sys
import time

import pytest

from ics_agent.plcbench_sandbox import Sandbox, SandboxConfig

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX sandbox")


def _sandbox(tmp_path, **kw):
    cfg = SandboxConfig(workdir=tmp_path / "scratch", broker_host="127.0.0.1",
                        broker_port=5020, **kw)
    return Sandbox(cfg)


def test_shell_echo(tmp_path):
    sb = _sandbox(tmp_path)
    try:
        r = sb.run("shell", "echo hello")
        assert r.ok and r.exit_code == 0
        assert r.stdout.strip() == "hello"
    finally:
        sb.cleanup()


def test_cwd_is_workdir(tmp_path):
    sb = _sandbox(tmp_path)
    try:
        r = sb.run("shell", "pwd")
        assert r.stdout.strip() == str(sb.workdir)
    finally:
        sb.cleanup()


def test_env_is_allowlisted_broker_present_controlplane_absent(tmp_path, monkeypatch):
    # Pollute the parent env with control-plane/secret vars; they must NOT leak.
    monkeypatch.setenv("MRG_SERVER", "merge-secret")
    monkeypatch.setenv("ICS_AGENT_SPHERE_XDC_OVERRIDE", "xdc-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh")
    sb = _sandbox(tmp_path)
    try:
        r = sb.run("shell", "env")
        assert "PLCBENCH_BROKER_HOST=127.0.0.1" in r.stdout
        assert "PLCBENCH_BROKER_PORT=5020" in r.stdout
        assert "MRG_SERVER" not in r.stdout
        assert "ICS_AGENT_SPHERE_XDC_OVERRIDE" not in r.stdout
        assert "SSH_AUTH_SOCK" not in r.stdout
    finally:
        sb.cleanup()


def test_python_runs_isolated(tmp_path):
    sb = _sandbox(tmp_path)
    try:
        r = sb.run("python", "print(6*7)")
        assert r.ok and r.stdout.strip() == "42"
    finally:
        sb.cleanup()


def test_wall_timeout_kills_group(tmp_path):
    sb = _sandbox(tmp_path, wall_timeout_sec=0.5)
    try:
        start = time.monotonic()
        r = sb.run("shell", "sleep 5")
        elapsed = time.monotonic() - start
        assert r.timed_out is True
        assert r.ok is False
        assert elapsed < 4.0  # killed well before the 5s sleep finished
    finally:
        sb.cleanup()


def test_spawned_child_is_also_killed(tmp_path):
    """A background child spawned by the actor must die with the group."""
    sb = _sandbox(tmp_path, wall_timeout_sec=0.5)
    try:
        # Write the spawned child's pid to a file, then sleep in the parent shell.
        r = sb.run("shell", "sleep 30 & echo $! > child.pid; sleep 5")
        assert r.timed_out is True
        pid = int((sb.workdir / "child.pid").read_text().strip())
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)  # no such process -> it was killed with the group
    finally:
        sb.cleanup()


def test_output_cap_truncates(tmp_path):
    sb = _sandbox(tmp_path, output_cap_bytes=100)
    try:
        r = sb.run("python", "print('x'*5000)")
        assert r.truncated is True
        assert len(r.stdout) <= 100 + len("\n[output truncated]")
    finally:
        sb.cleanup()


def test_cleanup_removes_workdir(tmp_path):
    sb = _sandbox(tmp_path)
    (sb.workdir / "artifact.txt").write_text("x")
    sb.cleanup()
    assert not sb.workdir.exists()
