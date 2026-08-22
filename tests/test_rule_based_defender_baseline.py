from __future__ import annotations

from pathlib import Path

import pytest

from ics_agent.agents.rule_based import RuleBasedRoleAgent
from ics_agent.rule_based_defender_baseline import (
    DEFENDER_MODE_SPECS,
    DistributionRuleBasedDefenderRunner,
    TreatmentRuleBasedDefenderRunner,
)


@pytest.mark.parametrize("testbed,scenario_count", [("water_treatment", 18), ("water_distribution", 11)])
@pytest.mark.parametrize("attacker", ["black_box", "gray_box", "white_box"])
@pytest.mark.parametrize("defender_mode", sorted(DEFENDER_MODE_SPECS))
def test_rule_based_defender_runner_resolves_expected_profiles(
    tmp_path: Path,
    testbed: str,
    scenario_count: int,
    attacker: str,
    defender_mode: str,
) -> None:
    result_root = (
        tmp_path
        / f"{attacker}_agentic_attacker_vs_rule_based_defender_{defender_mode}"
    )
    if testbed == "water_treatment":
        runner = TreatmentRuleBasedDefenderRunner(result_root)
    else:
        runner = DistributionRuleBasedDefenderRunner(result_root)
    payload = runner.smoke()

    assert payload["status"] == "passed"
    assert payload["scenario_count"] == scenario_count
    assert payload["condition_name"] == (
        f"{attacker}_attacker__rule_based_{defender_mode}_defender"
    )


def test_advisory_rule_based_defender_alerts_without_action() -> None:
    agent = RuleBasedRoleAgent(
        "rule_based_advisory_response_defender",
        profile={
            "role": "defender",
            "family": "rule_based",
            "mode": "online",
            "allowed_action_types": ["noop"],
            "behavior_model": {
                "id": "optionb_mitre_staged",
                "policy": "rule_based_defender",
                "response_mode": "advisory_response",
            },
        },
    )
    decisions = agent.step(
        {
            "time_sec": 1.0,
            "observation": {"STATE_NAME": "RUNNING"},
            "scenario": {"metadata": {}},
            "executed_actions": [
                {
                    "role": "attacker",
                    "action": {"type": "set_sensor", "tag": "RW_Tank_Level", "value": 0.0},
                }
            ],
        }
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "alert"
    assert decisions[0].action is None
    assert "block_sensor_write" in decisions[0].summary


def test_active_rule_based_defender_blocks_matching_surface() -> None:
    agent = RuleBasedRoleAgent(
        "rule_based_active_intervention_defender",
        profile={
            "role": "defender",
            "family": "rule_based",
            "mode": "online",
            "allowed_action_types": ["block_actuator_write"],
            "behavior_model": {
                "id": "optionb_mitre_staged",
                "policy": "rule_based_defender",
                "response_mode": "active_intervention",
            },
        },
    )
    decisions = agent.step(
        {
            "time_sec": 2.0,
            "observation": {"STATE_NAME": "RUNNING"},
            "scenario": {"metadata": {}},
            "executed_actions": [
                {
                    "role": "attacker",
                    "action": {"type": "set_actuator", "tag": "Supply_Pump", "value": 1},
                }
            ],
        }
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "action"
    assert decisions[0].action == {"type": "block_actuator_write"}
