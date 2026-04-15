from __future__ import annotations

import csv
import json

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
        self.state["RW_Tank_PR_Valve_Sts"] = (
            1 if level < 450 and self.state["STATE_NAME"] == "RUNNING" else 0
        )
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


def test_orchestrator_runs_live_smoke_with_fake_environment(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "live.yaml"
    config = {
        "experiment_name": "pytest_live",
        "scenario_selectors": ["smoke/wt-smoke-rw-refill-spoof"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "conditions": [
            {
                "name": "scripted_baseline",
                "profiles": {
                    "attacker": "scripted_attacker",
                    "defender": "scripted_defender",
                    "observer": "scripted_observer",
                    "planner": "scripted_planner",
                    "operator": "scripted_operator",
                    "offline_analysis": "scripted_offline_analyst",
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    clock = FakeClock()
    monkeypatch.setattr("ics_agent.orchestrator.OpenPLCEnvironment", FakeEnvironment)
    monkeypatch.setattr("ics_agent.orchestrator.time.monotonic", clock.monotonic)
    monkeypatch.setattr("ics_agent.orchestrator.time.sleep", clock.sleep)

    orchestrator = ExperimentOrchestrator()
    summary = orchestrator.run(config_path)
    assert summary["conditions"]
    assert summary["conditions"][0]["condition_name"] == "scripted_baseline"

    metric_files = list((tmp_path / "Results").rglob("metrics.json"))
    assert metric_files, "expected metrics.json to be written"
    metrics = json.loads(metric_files[0].read_text())
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["alert_count"] >= 1


def test_orchestrator_runs_replay_mode_from_saved_results(
    tmp_path, monkeypatch
) -> None:
    replay_dir = tmp_path / "replay_seed"
    replay_dir.mkdir(parents=True)
    with (replay_dir / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STATE_NAME", "RW_Tank_Level"])
        writer.writeheader()
        writer.writerow({"STATE_NAME": "RUNNING", "RW_Tank_Level": 390})
        writer.writerow({"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 390})
    (replay_dir / "actions.json").write_text(
        json.dumps(
            [{"role": "defender", "time_sec": 1.0, "action": {"type": "hmi_stop"}}]
        )
    )
    (replay_dir / "decisions.json").write_text(
        json.dumps(
            [
                {
                    "role": "observer",
                    "time_sec": 0.9,
                    "diagnosis": "raw_water_low_or_spoof",
                }
            ]
        )
    )
    (replay_dir / "alerts.json").write_text(
        json.dumps([{"role": "observer", "time_sec": 0.9}])
    )
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

    monkeypatch.setattr(
        "ics_agent.agents.agentic.CachedQwenDecisionEngine.decide",
        lambda self, context: {
            "summary": "Offline agentic review completed.",
            "decision_type": "analysis",
            "confidence": 0.9,
            "diagnosis": "raw_water_low_or_spoof",
            "labels": ["offline", "agentic"],
        },
    )

    orchestrator = ExperimentOrchestrator()
    summary = orchestrator.run(config_path)
    assert summary["conditions"]
    analysis_files = list((tmp_path / "Results").rglob("offline_analysis.json"))
    assert analysis_files
    payload = json.loads(analysis_files[0].read_text())
    assert payload


def test_orchestrator_marks_failed_run_when_offline_agent_finalize_errors(
    tmp_path, monkeypatch
) -> None:
    replay_dir = tmp_path / "replay_seed"
    replay_dir.mkdir(parents=True)
    with (replay_dir / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STATE_NAME", "RW_Tank_Level"])
        writer.writeheader()
        writer.writerow({"STATE_NAME": "RUNNING", "RW_Tank_Level": 390})
        writer.writerow({"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 390})
    (replay_dir / "actions.json").write_text(
        json.dumps(
            [{"role": "defender", "time_sec": 1.0, "action": {"type": "hmi_stop"}}]
        )
    )
    (replay_dir / "decisions.json").write_text(
        json.dumps(
            [
                {
                    "role": "observer",
                    "time_sec": 0.9,
                    "diagnosis": "raw_water_low_or_spoof",
                }
            ]
        )
    )
    (replay_dir / "alerts.json").write_text(
        json.dumps([{"role": "observer", "time_sec": 0.9}])
    )
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

    monkeypatch.setattr(
        "ics_agent.orchestrator.AgenticRoleAgent.finalize", broken_finalize
    )

    summary = ExperimentOrchestrator().run(config_path)

    assert summary["status"] == "failed"
    assert summary["conditions"][0]["status"] == "failed"
    run_summary = summary["conditions"][0]["runs"][0]
    assert run_summary["status"] == "failed"
    assert "offline backend unavailable" in run_summary["errors"][0]["error"]


def test_remaining_role_events_preserves_pending_attacker_plan() -> None:
    orchestrator = ExperimentOrchestrator()
    planned_events = {
        "attacker": [
            {"time_sec": 8.0, "action": {"type": "select_process", "active": True}},
            {
                "time_sec": 10.0,
                "action": {
                    "type": "set_manual_valve",
                    "chemical": "NaCl",
                    "open": True,
                },
            },
        ]
    }
    executed_actions = [
        {
            "role": "attacker",
            "action": {"type": "select_process", "active": True},
        }
    ]

    remaining = orchestrator._remaining_role_events(
        role="attacker",
        planned_events=planned_events,
        executed_actions=executed_actions,
    )

    assert remaining == [planned_events["attacker"][1]]


class CoordinatedFakeAgent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.mode = "online"
        self.profile_name = f"fake_{role}"
        self._emitted = False

    def reset(self, scenario) -> None:
        self._emitted = False

    def step(self, context):
        from ics_agent.types import RoleDecision

        if self.role == "operator":
            if not self._emitted and float(context["time_sec"]) >= 5.0:
                self._emitted = True
                return [
                    RoleDecision(
                        role="operator",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Start the plant.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={"type": "hmi_start", "process_id": "WT-P1"},
                        labels=["operator"],
                    )
                ]
            return []

        if self.role == "attacker":
            role_actions = [
                item
                for item in context.get("executed_actions", [])
                if item.get("role") == "attacker"
            ]
            if float(context["time_sec"]) >= 8.0 and not role_actions:
                return [
                    RoleDecision(
                        role="attacker",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Enable process 2.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": True,
                        },
                        labels=["attacker"],
                    )
                ]
            return []

        if self.role == "observer":
            if context["observation"].get("P2_Selected") == 1 and not self._emitted:
                self._emitted = True
                return [
                    RoleDecision(
                        role="observer",
                        agent_name=self.profile_name,
                        decision_type="alert",
                        summary="Unexpected process 2 selection.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["observer"],
                    )
                ]
            return []

        if self.role == "defender":
            if context["observation"].get("P2_Selected") == 1 and not self._emitted:
                self._emitted = True
                return [
                    RoleDecision(
                        role="defender",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Deselect process 2 immediately.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": False,
                        },
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["defender"],
                    )
                ]
            return []

        return []

    def finalize(self, context):
        return []


class HandoffFakeAgent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.mode = "online"
        self.profile_name = f"handoff_{role}"
        self._emitted = False

    def reset(self, scenario) -> None:
        self._emitted = False

    def step(self, context):
        from ics_agent.types import RoleDecision

        if self.role == "operator":
            if not self._emitted and float(context["time_sec"]) >= 5.0:
                self._emitted = True
                return [
                    RoleDecision(
                        role="operator",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Start the plant.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={"type": "hmi_start", "process_id": "WT-P1"},
                        labels=["operator"],
                    )
                ]
            return []

        if self.role == "attacker":
            if not self._emitted and float(context["time_sec"]) >= 8.0:
                self._emitted = True
                return [
                    RoleDecision(
                        role="attacker",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Enable process 2.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": True,
                        },
                        labels=["attacker"],
                    )
                ]
            return []

        if self.role == "observer":
            if context["observation"].get("P2_Selected") == 1 and not self._emitted:
                self._emitted = True
                return [
                    RoleDecision(
                        role="observer",
                        agent_name=self.profile_name,
                        decision_type="alert",
                        summary="Observer saw unexpected Process 2 selection.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["observer"],
                    )
                ]
            return []

        if self.role == "defender":
            if self._emitted:
                return []
            observer_alert_present = any(
                item.get("role") == "observer" for item in context.get("alerts", [])
            )
            if context["observation"].get("P2_Selected") == 1 and observer_alert_present:
                self._emitted = True
                return [
                    RoleDecision(
                        role="defender",
                        agent_name=self.profile_name,
                        decision_type="action",
                        summary="Deselect Process 2 from observer evidence.",
                        time_sec=float(context["time_sec"]),
                        timestamp_utc="2026-04-09T00:00:00Z",
                        action={
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": False,
                        },
                        diagnosis="unauthorized_chemical_dosing",
                        labels=["defender"],
                    )
                ]
            return []

        return []

    def finalize(self, context):
        return []


def test_orchestrator_allows_defender_reaction_before_attacker_plan_finishes(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "attack.yaml"
    config = {
        "experiment_name": "pytest_attack",
        "scenario_selectors": ["attack/wt-attack-p2-multi-chemical-drain"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "conditions": [
            {
                "name": "agentic_agents",
                "profiles": {
                    "attacker": "agentic_attacker",
                    "defender": "agentic_defender",
                    "observer": "agentic_observer",
                    "planner": "agentic_planner",
                    "operator": "agentic_operator",
                    "offline_analysis": "agentic_offline_analyst",
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    clock = FakeClock()
    monkeypatch.setattr("ics_agent.orchestrator.OpenPLCEnvironment", FakeEnvironment)
    monkeypatch.setattr("ics_agent.orchestrator.time.monotonic", clock.monotonic)
    monkeypatch.setattr("ics_agent.orchestrator.time.sleep", clock.sleep)

    orchestrator = ExperimentOrchestrator()
    monkeypatch.setattr(
        orchestrator,
        "_create_agent",
        lambda profile_id: CoordinatedFakeAgent(
            orchestrator.agent_profiles[profile_id]["role"]
        ),
    )

    summary = orchestrator.run(config_path)

    assert summary["status"] == "completed"
    metric_files = list((tmp_path / "Results").rglob("metrics.json"))
    assert metric_files
    metrics = json.loads(metric_files[0].read_text())
    assert metrics["defender_intervention"]["mitigation_action_count"] == 1
    assert metrics["time_to_mitigation_sec"] == 0.0

    action_files = list((tmp_path / "Results").rglob("actions.json"))
    actions = json.loads(action_files[0].read_text())
    defender_action = next(
        item for item in actions if item["role"] == "defender"
    )
    attacker_actions = [item for item in actions if item["role"] == "attacker"]

    assert defender_action["action"]["type"] == "select_process"
    assert len(attacker_actions) == 1
    assert float(defender_action["time_sec"]) == float(attacker_actions[0]["time_sec"])


def test_orchestrator_rebuilds_context_between_observer_and_defender(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "attack_handoff.yaml"
    config = {
        "experiment_name": "pytest_attack_handoff",
        "scenario_selectors": ["attack/wt-attack-p2-hcl-drain"],
        "repeats": 1,
        "resume": False,
        "results_root": str(tmp_path / "Results"),
        "conditions": [
            {
                "name": "agentic_agents",
                "profiles": {
                    "attacker": "agentic_attacker",
                    "defender": "agentic_defender",
                    "observer": "agentic_observer",
                    "planner": "agentic_planner",
                    "operator": "agentic_operator",
                    "offline_analysis": "agentic_offline_analyst",
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    clock = FakeClock()
    monkeypatch.setattr("ics_agent.orchestrator.OpenPLCEnvironment", FakeEnvironment)
    monkeypatch.setattr("ics_agent.orchestrator.time.monotonic", clock.monotonic)
    monkeypatch.setattr("ics_agent.orchestrator.time.sleep", clock.sleep)

    orchestrator = ExperimentOrchestrator()
    monkeypatch.setattr(
        orchestrator,
        "_create_agent",
        lambda profile_id: HandoffFakeAgent(
            orchestrator.agent_profiles[profile_id]["role"]
        ),
    )

    summary = orchestrator.run(config_path)

    assert summary["status"] == "completed"
    action_files = list((tmp_path / "Results").rglob("actions.json"))
    actions = json.loads(action_files[0].read_text())
    defender_action = next(item for item in actions if item["role"] == "defender")
    attacker_action = next(item for item in actions if item["role"] == "attacker")
    alert_files = list((tmp_path / "Results").rglob("alerts.json"))
    alerts = json.loads(alert_files[0].read_text())

    assert any(item["role"] == "observer" for item in alerts)
    assert defender_action["action"]["type"] == "select_process"
    assert float(defender_action["time_sec"]) == float(attacker_action["time_sec"])
