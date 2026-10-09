from __future__ import annotations

from ics_agent.agents.rule_based import RuleBasedRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


def test_ctf_operator_reacts_to_observation_without_security_defender() -> None:
    config = load_config("experiments/ctf_water_treatment_operator_pilot.yaml")
    condition = config["conditions"][0]
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profiles = orchestrator._validate_condition_profiles(condition["profiles"])
    assert profiles["defender"] == "silent_defender"
    assert profiles["operator"] == "rule_based_operator"

    operator_profile = orchestrator._effective_condition_profiles(
        profiles=profiles,
        profile_overrides=condition["profile_overrides"],
    )["operator"]
    agent = RuleBasedRoleAgent("rule_based_operator", operator_profile)
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "ctf/wt-operator-verification-pilot"
    )
    assert scenario.execution["timeline"] == []
    agent.reset(scenario)

    def step(time_sec: float, level: int, state: str = "RUNNING"):
        return agent.step(
            {
                "time_sec": time_sec,
                "observation": {"STATE_NAME": state, "RW_Tank_Level": level},
            }
        )

    assert step(4.5, 600) == []
    start = step(5.0, 600)
    assert len(start) == 1
    assert start[0].action == {"type": "hmi_start"}
    assert start[0].metadata["actor_kind"] == "simulated_human_operator"
    assert step(6.0, 600) == []

    omission = step(15.0, 400)
    assert len(omission) == 1
    assert omission[0].decision_type == "note"
    assert omission[0].action is None
    assert omission[0].metadata["verification_state"] == "not_requested"
    assert step(16.0, 400) == []

    agent.reset(scenario)
    assert len(step(5.0, 600)) == 1


def test_ctf_operator_cannot_use_attacker_sensor_write() -> None:
    config = load_config("experiments/ctf_water_treatment_operator_pilot.yaml")
    condition = config["conditions"][0]
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profiles = orchestrator._validate_condition_profiles(condition["profiles"])
    effective = orchestrator._effective_condition_profiles(
        profiles=profiles,
        profile_overrides=condition["profile_overrides"],
    )
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "ctf/wt-operator-verification-pilot"
    )
    action, reason = orchestrator._sanitize_action(
        role="operator",
        action={"type": "set_sensor", "name": "RW_Tank_Level", "value": 400},
        scenario=scenario,
        profile_id=profiles["operator"],
        effective_profile=effective["operator"],
    )
    assert action is None
    assert "rejected by guardrails" in reason
