"""System One choice policy for the offline WT operator-response fixture.

The transport returns decisions only. It has no PLC, broker, or plant endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_choice(answer: Mapping[str, Any], actions: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    if answer.get('type') != 'choice' or 'error' in answer:
        raise ValueError('System One did not return a choice answer')
    probabilities = answer.get('probabilities')
    if not isinstance(probabilities, dict) or set(probabilities) != set(actions):
        raise ValueError('System One probabilities differ from declared authority')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
        raise ValueError('System One returned invalid probabilities')
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.02):
        raise ValueError('System One probabilities do not sum to one')
    choice = answer.get('choice')
    if choice not in actions or probabilities[choice] < max(probabilities.values()) - 1e-6:
        raise ValueError('System One choice is absent or disagrees with its probabilities')
    return choice, {key: float(probabilities[key]) for key in actions}


class SystemOneChoicePolicy:
    def __init__(self, spec: Mapping[str, Any], fixture: Mapping[str, Any],
                 ask: Callable[[dict[str, Any]], Mapping[str, Any]]) -> None:
        if spec.get('schema_version') != 'plcbench.systemone-choice-policy/v1':
            raise ValueError('unsupported decision policy contract')
        self.spec = spec
        self.fixture = fixture
        self.ask = ask
        self.fields = tuple(spec['observation_fields'])
        self.actions = tuple(spec['permitted_actions'])
        if self.actions != tuple(a['id'] for a in fixture['permitted_interventions']):
            raise ValueError('decision action authority differs from WT fixture')
        self.records: list[dict[str, Any]] = []

    def request(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if set(observation) != set(self.fields):
            raise ValueError('decision observation fields differ from contract')
        criteria = {
            action['id']: ('Do nothing; observe the next sample.' if action['tag'] is None
                           else f"Set {action['tag']} to {action['value']}.")
            for action in self.fixture['permitted_interventions']
        }
        state = {
            'objective': self.spec['objective'],
            'safety_envelope': self.fixture['safety_envelope'],
            'service_target': self.fixture['service_target'],
            'observation': {key: observation[key] for key in self.fields},
            'permitted_interventions': list(self.fixture['permitted_interventions']),
            'evidence_origin': self.fixture['evidence_origin'],
        }
        return {'model': self.spec['model']['model_id'], 'state': state,
                'questions': {'next_action': {
                    'type': 'choice',
                    'instructions': 'Choose one action for the next one-second interval. '
                                    'Use only the visible state; the disturbance label is hidden.',
                    'criteria': criteria,
                }}}

    def __call__(self, observation: Mapping[str, Any]) -> str:
        request = self.request(observation)
        record: dict[str, Any] = {'request_sha256': hashlib.sha256(canonical(request)).hexdigest()}
        self.records.append(record)
        response = self.ask(request)
        if not isinstance(response, Mapping) or not isinstance(response.get('answers'), Mapping):
            raise ValueError('malformed System One response')
        answer = response['answers'].get('next_action')
        if not isinstance(answer, Mapping):
            raise ValueError('System One omitted next_action')
        choice, probabilities = validate_choice(answer, self.actions)
        record.update({'choice': choice, 'probabilities': probabilities,
                       'response_sha256': hashlib.sha256(canonical(response)).hexdigest()})
        return choice
