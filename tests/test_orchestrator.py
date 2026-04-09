from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from ics_agent.orchestrator import ExperimentOrchestrator


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class FakeEnvironment:
    def __init__(self, config=None) -> None:
        self.config = config or {}
        self.state = {}

    def start(self):
        return {"backend": "fake"}

    def reset(self, initial_conditions=None):
        conditions = {
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
            "ChemTreat_NaCl_Level": 800,
            "ChemTreat_NaOCl_Level": 800,
            "ChemTreat_HCl_Level": 800,
        }
        if initial_conditions:
            conditions.update(initial_conditions)
        self.state = {
            **conditions,
            "STATE_NAME": "IDLE",
            "Alarm_RW_Tank_LL": 0,
            "Alarm_RW_Tank_HH": 0,
            "P2_Selected": 0,
            "RW_Tank_PR_Valve_Sts": 0,
            "RW_Pump_Sts": 0,
        }
        return self.observe_flat()

    def observe_flat(self):
        level = int(self.state["RW_Tank_Level"])
        self.state["Alarm_RW_Tank_LL"] = 1 if level < 450 else 0
        self.state["Alarm_RW_Tank_HH"] = 1 if level > 1180 else 0
        self.state["RW_Tank_PR_Valve_Sts"] = 1 if level < 450 and self.state["STATE_NAME"] == "RUNNING" else 0
        return dict(self.state)

    def act(self, action):
        action_type = action["type"]
        if action_type == "hmi_start":
            self.state["STATE_NAME"] = "RUNNING"
            self.state["RW_Pump_Sts"] = 1
        elif action_type in {"hmi_stop", "hmi_estop"}:
            self.state["STATE_NAME"] = "SHUTDOWN"
            self.state["RW_Pump_Sts"] = 0
        elif action_type == "set_sensor":
            self.state[action["name"]] = int(action["value"])
        elif action_type == "select_process":
            self.state["P2_Selected"] = 1 if action.get("active", True) else 0
        return {"state_name": self.state["STATE_NAME"]}

    def stop(self):
        return None


def test_orchestrator_runs_live_smoke_with_fake_environment(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "live.yaml"
    config = {
        "experiment_name": "pytest_live",
        "scenario_selectors": ["smoke/wt-smoke-rw-refill-spoof"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "role_matrix": {
            "attacker": ["scripted_attacker"],
            "defender": ["scripted_defender"],
            "observer": ["scripted_observer"],
            "planner": ["scripted_planner"],
            "operator": ["scripted_operator"],
            "offline_analysis": ["scripted_offline_analyst"],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    clock = FakeClock()
    monkeypatch.setattr("ics_agent.orchestrator.OpenPLCEnvironment", FakeEnvironment)
    monkeypatch.setattr("ics_agent.orchestrator.time.monotonic", clock.monotonic)
    monkeypatch.setattr("ics_agent.orchestrator.time.sleep", clock.sleep)

    orchestrator = ExperimentOrchestrator()
    orchestrator.agent_profiles["agentic_offline_analyst"]["llm"]["strict_backend"] = False
    summary = orchestrator.run(config_path)
    assert summary["conditions"]

    metric_files = list((tmp_path / "Results").rglob("metrics.json"))
    assert metric_files, "expected metrics.json to be written"
    metrics = json.loads(metric_files[0].read_text())
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["alert_count"] >= 1


def test_orchestrator_runs_replay_mode_from_saved_results(tmp_path) -> None:
    replay_dir = tmp_path / "replay_seed"
    replay_dir.mkdir(parents=True)
    with (replay_dir / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STATE_NAME", "RW_Tank_Level"])
        writer.writeheader()
        writer.writerow({"STATE_NAME": "RUNNING", "RW_Tank_Level": 390})
        writer.writerow({"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 390})
    (replay_dir / "actions.json").write_text(json.dumps([{"role": "defender", "time_sec": 1.0, "action": {"type": "hmi_stop"}}]))
    (replay_dir / "decisions.json").write_text(json.dumps([{"role": "observer", "time_sec": 0.9, "diagnosis": "raw_water_low_or_spoof"}]))
    (replay_dir / "alerts.json").write_text(json.dumps([{"role": "observer", "time_sec": 0.9}]))
    (replay_dir / "summary.json").write_text(json.dumps({"passed": True}))

    config_path = tmp_path / "replay.yaml"
    config = {
        "experiment_name": "pytest_replay",
        "execution_mode": "replay",
        "replay_source": str(replay_dir),
        "scenario_selectors": ["smoke/wt-smoke-rw-refill-spoof"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "role_matrix": {
            "attacker": ["scripted_attacker"],
            "defender": ["scripted_defender"],
            "observer": ["scripted_observer"],
            "planner": ["scripted_planner"],
            "operator": ["scripted_operator"],
            "offline_analysis": ["agentic_offline_analyst"],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    orchestrator = ExperimentOrchestrator()
    orchestrator.agent_profiles["agentic_offline_analyst"]["llm"]["strict_backend"] = False
    summary = orchestrator.run(config_path)
    assert summary["conditions"]
    analysis_files = list((tmp_path / "Results").rglob("offline_analysis.json"))
    assert analysis_files
    payload = json.loads(analysis_files[0].read_text())
    assert payload


def test_orchestrator_marks_failed_run_when_offline_agent_finalize_errors(tmp_path, monkeypatch) -> None:
    replay_dir = tmp_path / "replay_seed"
    replay_dir.mkdir(parents=True)
    with (replay_dir / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STATE_NAME", "RW_Tank_Level"])
        writer.writeheader()
        writer.writerow({"STATE_NAME": "RUNNING", "RW_Tank_Level": 390})
        writer.writerow({"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 390})
    (replay_dir / "actions.json").write_text(json.dumps([{"role": "defender", "time_sec": 1.0, "action": {"type": "hmi_stop"}}]))
    (replay_dir / "decisions.json").write_text(json.dumps([{"role": "observer", "time_sec": 0.9, "diagnosis": "raw_water_low_or_spoof"}]))
    (replay_dir / "alerts.json").write_text(json.dumps([{"role": "observer", "time_sec": 0.9}]))
    (replay_dir / "summary.json").write_text(json.dumps({"passed": True}))

    config_path = tmp_path / "replay_fail.yaml"
    config = {
        "experiment_name": "pytest_replay_fail",
        "execution_mode": "replay",
        "replay_source": str(replay_dir),
        "scenario_selectors": ["smoke/wt-smoke-rw-refill-spoof"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "role_matrix": {
            "attacker": ["scripted_attacker"],
            "defender": ["scripted_defender"],
            "observer": ["scripted_observer"],
            "planner": ["scripted_planner"],
            "operator": ["scripted_operator"],
            "offline_analysis": ["agentic_offline_analyst"],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def broken_finalize(self, context):
        raise RuntimeError("offline backend unavailable")

    monkeypatch.setattr("ics_agent.orchestrator.AgenticRoleAgent.finalize", broken_finalize)

    summary = ExperimentOrchestrator().run(config_path)

    assert summary["status"] == "failed"
    assert summary["conditions"][0]["status"] == "failed"
    run_summary = summary["conditions"][0]["runs"][0]
    assert run_summary["status"] == "failed"
    assert "offline backend unavailable" in run_summary["errors"][0]["error"]
