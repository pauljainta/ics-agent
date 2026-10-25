"""The decision adapter preserves the fixed observation and action boundary."""

import json
from pathlib import Path

import pytest

from scripts import plcbench_wt_response_systemone as runner
from ics_agent.plcbench_systemone import SystemOneChoicePolicy, validate_choice
from ics_agent.plcbench_wt_response import ChemistryState


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'configs/plcbench/wt_response_systemone_policy_v1.json').read_text())
FIXTURE = json.loads((ROOT / SPEC['fixture_ref']).read_text())


def observation():
    return ChemistryState.fresh(FIXTURE['scenarios']['high_ph']).observe()


def response(action='select_p2'):
    probs = {a: 0.02 for a in SPEC['permitted_actions']}
    probs[action] = 0.9
    return {'answers': {'next_action': {'type': 'choice', 'choice': action,
                                         'probabilities': probs}}}


def test_request_hides_case_and_matches_fixed_authority():
    seen = []
    policy = SystemOneChoicePolicy(SPEC, FIXTURE, lambda request: seen.append(request) or response())
    assert policy(observation()) == 'select_p2'
    assert tuple(seen[0]['questions']['next_action']['criteria']) == tuple(SPEC['permitted_actions'])
    assert set(seen[0]['state']['observation']) == set(SPEC['observation_fields'])
    assert 'high_ph' not in json.dumps(seen[0])
    assert policy.records[0]['choice'] == 'select_p2'


def test_invalid_model_output_cannot_become_an_intervention():
    policy = SystemOneChoicePolicy(SPEC, FIXTURE, lambda _: {'answers': {'next_action': {
        'type': 'choice', 'choice': 'outside', 'probabilities': {'outside': 1.0}}}})
    with pytest.raises(ValueError, match='authority'):
        policy(observation())
    assert 'choice' not in policy.records[0]
    with pytest.raises(ValueError, match='observation fields'):
        policy({key: value for key, value in observation().items() if key != 'SYS_RUNNING'})
    with pytest.raises(ValueError, match='probabilities'):
        validate_choice({'type': 'choice', 'choice': 'select_p2',
                         'probabilities': {a: float('nan') for a in SPEC['permitted_actions']}},
                        tuple(SPEC['permitted_actions']))


def test_preflight_keeps_safety_and_service_unassessable(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'source_identity', lambda *args, **kwargs: {'verified_for_test': True})
    calls = []

    def fake_http(url, *, body, key, timeout):
        calls.append((url, body))
        return {'data': [{'id': 'd3'}]} if body is None else response()

    monkeypatch.setattr(runner, 'http_json', fake_http)
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'schema_version': 'plcbench.d3-deployment/v1',
                                    'model_id': 'd3', 'model_source': 'vllm-sr/d3',
                                    'weight_revision': runner.D3_CHECKPOINT_REVISION,
                                    'server_sha256': '0' * 64}))
    summary = runner.run(ROOT / 'configs/plcbench/wt_response_systemone_policy_v1.json',
                         tmp_path, tmp_path / 'out', mode='preflight',
                         endpoint='http://127.0.0.1:9999', deployment_json=identity)
    assert summary['status'] == 'complete'
    assert len(summary['results']) == 2
    assert len(calls) == 3  # inventory plus one decision per hidden case
    assert all('unassessable' in row['assessment_status'] for row in summary['results'])
    assert all('safety_score' not in row and 'service_score' not in row
               for row in summary['results'])
    assert (tmp_path / 'out/deployment.json').read_bytes() == identity.read_bytes()


def test_endpoint_admission_failure_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'source_identity', lambda *args, **kwargs: {'verified_for_test': True})

    def unavailable(*args, **kwargs):
        raise ValueError('no model inventory at this port')

    monkeypatch.setattr(runner, 'http_json', unavailable)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='no model inventory'):
        runner.run(ROOT / 'configs/plcbench/wt_response_systemone_policy_v1.json',
                   tmp_path, out, mode='preflight', endpoint='http://127.0.0.1:8899')
    report = json.loads((out / 'summary.json').read_text())
    assert report['status'] == 'failed'
    assert report['assessment_status'] == 'unassessable_model_not_admitted'
    assert report['results'] == []


def test_mismatched_chpc_deployment_is_not_admitted(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'source_identity', lambda *args, **kwargs: {'verified_for_test': True})
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'schema_version': 'plcbench.d3-deployment/v1',
                                    'model_id': 'd3', 'model_source': 'vllm-sr/d3',
                                    'weight_revision': 'wrong', 'server_sha256': '0' * 64}))
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='deployment identity'):
        runner.run(ROOT / 'configs/plcbench/wt_response_systemone_policy_v1.json',
                   tmp_path, out, mode='preflight', endpoint='http://127.0.0.1:18080',
                   deployment_json=identity)
    report = json.loads((out / 'summary.json').read_text())
    assert report['assessment_status'] == 'unassessable_model_not_admitted'
    assert report['results'] == []
