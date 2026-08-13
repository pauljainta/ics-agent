from __future__ import annotations

from pathlib import Path

from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.types import RoleDecision, RunBundle, ScenarioDefinition
from ics_agent.utils import utc_now


class StubAgent:
    def __init__(self, decision: RoleDecision) -> None:
        self.mode = "online"
        self._decision = decision

    def step(self, context):
        return [self._decision]


class StubEnvironment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def act(self, action, *, role=None):
        descriptor = dict(action)
        self.calls.append((str(role), descriptor))
        if str(descriptor.get("type")) == "block_sensor_write":
            return {
                "state_name": "RUNNING",
                "containment": {
                    "is_containment_action": True,
                    "is_symbolic": False,
                    "target_surfaces": ["sensor_write_path"],
                    "surface_scope": "sensor_write_path",
                    "execution_mode": "runtime_enforced",
                },
                "runtime_containment": {
                    "active": True,
                    "surface_scope": "targeted",
                    "target_surfaces": ["sensor_write_path"],
                },
            }
        if str(descriptor.get("type")) == "select_process":
            return {"state_name": "RUNNING"}
        return {
            "state_name": "RUNNING",
            "blocked_by_runtime_containment": True,
            "blocked_surface": "sensor_write_path",
            "blocked_by_action": {"type": "block_sensor_write"},
            "blocked_by_action_type": "block_sensor_write",
            "reason": "contained telemetry-write surface blocks later attacker sensor writes",
            "runtime_containment": {
                "active": True,
                "surface_scope": "targeted",
                "target_surfaces": ["sensor_write_path"],
            },
        }

    def record_role_runtime(self, role, runtime_info) -> None:
        return None


def _bundle() -> RunBundle:
    scenario = ScenarioDefinition(
        catalog_id="attack/test",
        suite="attack",
        scenario_id="test",
        execution_path=Path("attack/test.yaml"),
        execution={},
        metadata={},
    )
    return RunBundle(
        experiment_id="exp",
        condition_name="cond",
        run_id="run-001",
        scenario=scenario,
        run_dir=Path("/tmp/run-001"),
        config_snapshot={},
    )


def test_orchestrator_records_runtime_activation_and_blocked_events() -> None:
    orchestrator = ExperimentOrchestrator()
    env = StubEnvironment()
    bundle = _bundle()
    runtime_containment = orchestrator._initial_runtime_containment()
    context = {"observation": {"STATE_NAME": "RUNNING"}}

    defender_decision = RoleDecision(
        role="defender",
        agent_name="defender",
        decision_type="action",
        summary="apply containment",
        time_sec=1.0,
        timestamp_utc=utc_now(),
        action={"type": "block_sensor_write"},
    )
    attacker_decision = RoleDecision(
        role="attacker",
        agent_name="attacker",
        decision_type="action",
        summary="spoof sensor",
        time_sec=2.0,
        timestamp_utc=utc_now(),
        action={"type": "set_sensor", "name": "RW_Tank_Level", "value": 111},
    )

    orchestrator._execute_online_roles(
        roles=["defender"],
        agents={"defender": StubAgent(defender_decision)},
        context=context,
        bundle=bundle,
        scenario=bundle.scenario,
        condition={"defender": "agentic_defender"},
        env=env,
        elapsed=1.0,
        effective_profiles={
            "defender": {
                "role": "defender",
                "allowed_action_types": ["block_sensor_write", "noop"],
            }
        },
        runtime_containment=runtime_containment,
    )
    orchestrator._execute_online_roles(
        roles=["attacker"],
        agents={"attacker": StubAgent(attacker_decision)},
        context=context,
        bundle=bundle,
        scenario=bundle.scenario,
        condition={"attacker": "gray_box_attacker"},
        env=env,
        elapsed=2.0,
        runtime_containment=runtime_containment,
    )

    assert runtime_containment["active"] is True
    assert runtime_containment["target_surfaces"] == ["sensor_write_path"]
    assert any(
        item["event_type"] == "attacker_execution_cutoff_activated"
        for item in bundle.system_events
    )
    assert any(
        item["event_type"] == "attacker_action_blocked" for item in bundle.system_events
    )
    assert [item["action"]["type"] for item in bundle.executed_actions] == [
        "block_sensor_write"
    ]
    assert env.calls == [
        ("defender", {"type": "block_sensor_write"}),
        ("attacker", {"type": "set_sensor", "name": "RW_Tank_Level", "value": 111}),
    ]


def test_process2_deselect_no_longer_triggers_global_attacker_cutoff() -> None:
    orchestrator = ExperimentOrchestrator()
    env = StubEnvironment()
    bundle = _bundle()
    runtime_containment = orchestrator._initial_runtime_containment()
    context = {"observation": {"STATE_NAME": "RUNNING"}}

    defender_decision = RoleDecision(
        role="defender",
        agent_name="defender",
        decision_type="action",
        summary="disable process2",
        time_sec=1.0,
        timestamp_utc=utc_now(),
        action={"type": "select_process", "process_id": "WT-P2", "active": False},
    )

    orchestrator._execute_online_roles(
        roles=["defender"],
        agents={"defender": StubAgent(defender_decision)},
        context=context,
        bundle=bundle,
        scenario=bundle.scenario,
        condition={"defender": "agentic_defender"},
        env=env,
        elapsed=1.0,
        runtime_containment=runtime_containment,
    )

    assert runtime_containment["active"] is False
    assert not any(
        item["event_type"] == "attacker_execution_cutoff_activated"
        for item in bundle.system_events
    )


def test_observe_alert_only_defender_emits_alert_without_cutoff_activation() -> None:
    orchestrator = ExperimentOrchestrator()
    env = StubEnvironment()
    bundle = _bundle()
    runtime_containment = orchestrator._initial_runtime_containment()
    context = {"observation": {"STATE_NAME": "RUNNING"}}

    defender_alert = RoleDecision(
        role="defender",
        agent_name="defender",
        decision_type="alert",
        summary="Suspicious telemetry-write behavior detected.",
        time_sec=1.0,
        timestamp_utc=utc_now(),
        action=None,
        diagnosis="sensor_write_abuse",
    )

    orchestrator._execute_online_roles(
        roles=["defender"],
        agents={"defender": StubAgent(defender_alert)},
        context=context,
        bundle=bundle,
        scenario=bundle.scenario,
        condition={"defender": "observe_alert_only_defender"},
        env=env,
        elapsed=1.0,
        effective_profiles={
            "defender": {
                "role": "defender",
                "allowed_action_types": ["noop"],
            }
        },
        runtime_containment=runtime_containment,
    )

    assert len(bundle.alerts) == 1
    assert runtime_containment["active"] is False
    assert not any(
        item["event_type"] == "attacker_execution_cutoff_activated"
        for item in bundle.system_events
    )
