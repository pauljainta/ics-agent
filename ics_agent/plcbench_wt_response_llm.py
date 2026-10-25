"""Bounded LLM decision contract for the experimental WT response fixture."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping


def parse_action(raw: str, allowed: set[str]) -> str:
    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('model response is not a complete JSON block')
        text = '\n'.join(lines[1:-1])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError('model response is not JSON') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('action_id'), str):
        raise ValueError('model response lacks a string action_id')
    action = payload['action_id']
    if action not in allowed:
        raise ValueError(f'model action {action!r} is outside declared authority')
    return action


class LLMResponsePolicy:
    def __init__(self, spec: Mapping[str, Any], fixture: Mapping[str, Any],
                 generate: Callable[[str], str]) -> None:
        if spec.get('schema_version') != 'plcbench.wt-response-llm-policy/v1':
            raise ValueError('unsupported LLM policy contract')
        self.fields = tuple(spec['observation_fields'])
        self.actions = tuple(spec['permitted_actions'])
        fixture_actions = tuple(action['id'] for action in fixture['permitted_interventions'])
        if self.actions != fixture_actions:
            raise ValueError('LLM action authority differs from the fixed fixture')
        self.action_details = list(fixture['permitted_interventions'])
        self.objective = str(spec['objective'])
        self.safety = fixture['safety_envelope']
        self.service = fixture['service_target']
        self.generate = generate
        self.decisions: list[dict[str, Any]] = []

    def __call__(self, observation: Mapping[str, Any]) -> str:
        if set(observation) != set(self.fields):
            raise ValueError('LLM observation fields differ from the policy contract')
        prompt = (
            'You are a WT operator-response policy in an isolated experimental fixture. '
            'Choose exactly one permitted action for this decision. The disturbance '
            'label is hidden. Reply only as JSON: {"action_id":"<id>"}.\n'
            + json.dumps({'objective': self.objective,
                          'safety_envelope': self.safety,
                          'service_target': self.service,
                          'observation': {key: observation[key] for key in self.fields},
                          'permitted_actions': self.action_details}, sort_keys=True)
        )
        raw = self.generate(prompt)
        record = {'prompt': prompt, 'response': raw,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'response_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        self.decisions.append(record)
        action = parse_action(raw, set(self.actions))
        record['action_id'] = action
        return action
