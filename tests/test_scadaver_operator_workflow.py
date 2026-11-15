from __future__ import annotations

from pathlib import Path

from experiments.scadaver_sphere import operator_workflow as workflow


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def healthy_status(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(workflow, "LATCH", tmp_path / "latch")
    monkeypatch.delenv("SCADAVER_ENABLE_SITL_ACTIONS", raising=False)
    monkeypatch.delenv("SCADAVER_SITL_ACTION_TOKEN", raising=False)
    monkeypatch.setattr(workflow, "service_state", lambda host, service: "inactive" if "legacy" in service else "active")
    monkeypatch.setattr(workflow, "tcp_open", lambda host, port: True)
    monkeypatch.setattr(workflow, "worker_session", lambda: {"healthy": True, "profile": "heterogeneous_e4", "tool_count": 16})
    monkeypatch.setattr(workflow, "lifecycle", lambda: {"passed": True, "core_process_count": 1})


def test_status_requires_core_not_only_systemd(monkeypatch, tmp_path):
    healthy_status(monkeypatch, tmp_path)
    monkeypatch.setattr(workflow, "lifecycle", lambda: {"passed": False, "openplc_controller_state": "active", "core_process_count": 0})
    assert workflow.status()["TESTBED_READY"] is False


def test_status_fails_closed_on_action_state(monkeypatch, tmp_path):
    healthy_status(monkeypatch, tmp_path)
    monkeypatch.setenv("SCADAVER_ENABLE_SITL_ACTIONS", "1")
    assert workflow.status()["TESTBED_READY"] is False


def test_idempotent_start_does_not_restart_healthy_services(monkeypatch, tmp_path):
    healthy_status(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(workflow, "remote", lambda *args, **kwargs: calls.append(args) or Result())
    result = workflow.start()
    assert result["TESTBED_READY"] is True
    assert result["changes"] == []
    assert calls == []


def test_start_orders_dependencies_before_controller_recovery(monkeypatch, tmp_path):
    healthy_status(monkeypatch, tmp_path)
    state = {name: "inactive" for name in workflow.SERVICES}
    state["legacy_controller"] = "active"
    monkeypatch.setattr(workflow, "service_state", lambda host, service: state[next(name for name, value in workflow.SERVICES.items() if value == (host, service))])
    lifecycle_calls = iter([{"passed": False}, {"passed": True}])
    monkeypatch.setattr(workflow, "lifecycle", lambda: next(lifecycle_calls, {"passed": True}))
    calls = []
    def remote(host, command, **kwargs):
        calls.append((host, command))
        for name, (target_host, service) in workflow.SERVICES.items():
            if host == target_host and command == f"sudo systemctl start {service}": state[name] = "active"
        if "stop scadaver-legacy" in command: state["legacy_controller"] = "inactive"
        if "restart scadaver-openplc" in command: state["openplc"] = "active"
        return Result()
    monkeypatch.setattr(workflow, "remote", remote)
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)
    result = workflow.start()
    assert result["TESTBED_READY"] is True
    commands = [command for _, command in calls]
    assert commands.index("sudo systemctl start scadaver-simulator.service") < commands.index("sudo systemctl restart scadaver-openplc-controller.service")


def test_emergency_latch_is_never_cleared(monkeypatch, tmp_path):
    healthy_status(monkeypatch, tmp_path)
    workflow.LATCH.write_text("operator review required\n")
    result = workflow.start()
    assert result["TESTBED_READY"] is False
    assert workflow.LATCH.exists()
