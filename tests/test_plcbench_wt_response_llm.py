"""WT LLM response policy preserves the fixture's observations and authority."""

import json
from pathlib import Path

import pytest

from ics_agent.plcbench_wt_response import ChemistryState
from ics_agent.plcbench_wt_response_llm import LLMResponsePolicy, parse_action


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'configs/plcbench/wt_response_llm_policy_v1.json').read_text())
FIXTURE = json.loads((ROOT / SPEC['fixture_ref']).read_text())


def test_one_step_policy_has_same_observation_and_authority_contract():
    observation = ChemistryState.fresh(FIXTURE['scenarios']['high_ph']).observe()
    policy = LLMResponsePolicy(SPEC, FIXTURE, lambda prompt: '{"action_id":"select_p2"}')
    assert policy(observation) == 'select_p2'
    assert set(SPEC['observation_fields']) == set(observation)
    assert tuple(policy.actions) == tuple(a['id'] for a in FIXTURE['permitted_interventions'])
    assert 'high_ph' not in policy.decisions[0]['prompt']


def test_invalid_or_missing_model_decision_is_not_replaced_with_an_action():
    observation = ChemistryState.fresh(FIXTURE['scenarios']['low_orp']).observe()
    policy = LLMResponsePolicy(SPEC, FIXTURE, lambda prompt: '{"action_id":"outside"}')
    with pytest.raises(ValueError, match='outside declared authority'):
        policy(observation)
    assert 'action_id' not in policy.decisions[0]
    with pytest.raises(ValueError, match='observation fields'):
        policy({key: value for key, value in observation.items() if key != 'P2_Selected'})
    assert parse_action('```json\n{"action_id":"observe"}\n```', set(policy.actions)) == 'observe'
