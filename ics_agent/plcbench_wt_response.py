"""Experimental, source-derived WT operator-response fixture.

The reduced-order update copies the pinned ST P2 mixing and membrane equations.
It is an offline engineering fixture, not a validated or live plant simulation.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


MIX_RATE = 0.001667
DAMAGE_RATE = 0.000006
RO_REJECTION = 0.995
RO_CL_MAX = 0.1
NAOCL_GAIN = 1.0
HCL_GAIN = 0.8


def f32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', value))[0]


def source_identity(config: Mapping[str, Any], *, repo_root: Path, sphere_root: Path) -> dict:
    ref = config['realization']
    source = sphere_root / ref['path']
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != ref['sha256']:
        raise ValueError('WT source differs from pinned fixture realization')
    body = source.read_text()
    required = (
        'WQ_MIX_RATE : REAL := 0.001667',
        'WQ_DAMAGE_RATE : REAL := 0.000006',
        'WQ_NAOCL_GAIN : REAL := 1.0',
        'WQ_HCL_GAIN : REAL := 0.8',
        'WQ_RO_REJECTION : REAL := 0.995',
        'WQ_RO_CL_MAX : REAL := 0.1',
        'wq_cl := wq_cl + (wq_target - wq_cl) * WQ_MIX_RATE',
        'wq_ph := wq_ph + (wq_target - wq_ph) * WQ_MIX_RATE',
        'membrane_damage := membrane_damage',
    )
    if any(fragment not in body for fragment in required):
        raise ValueError('pinned WT equations missing from source')
    import yaml
    map_path = repo_root / ref['map_path']
    mapping = yaml.safe_load(map_path.read_text())['mappings']
    expected_tags = {'P2_Selected': 8, 'P2_NaOCl_Cmd': 10, 'P2_HCl_Cmd': 11}
    for tag, address in expected_tags.items():
        entry = mapping[tag]
        if (entry['owner'], entry['register_type'], entry['address']) != ('controller', 'coil', address):
            raise ValueError(f'WT command provenance mismatch for {tag}')
    actions = config['permitted_interventions']
    if {a['id'] for a in actions} != {'select_p2', 'naocl_on', 'naocl_off',
                                     'hcl_on', 'hcl_off', 'observe'}:
        raise ValueError('WT authority differs from declared P2 surface')
    if any(a['tag'] is not None and a['tag'] not in expected_tags for a in actions):
        raise ValueError('fixture action escapes P2 command surface')
    return {'simulator_st_sha256': actual,
            'map_yaml_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest()}


@dataclass
class ChemistryState:
    source_ph: float
    source_cl: float
    source_cond: float
    ph: float
    cl: float
    cond: float
    roft_cl: float = 0.0
    membrane_damage: float = 0.0
    selected: bool = False
    naocl_cmd: bool = False
    hcl_cmd: bool = False
    naocl_valve: bool = False
    hcl_valve: bool = False
    naocl_timer: int = 0
    hcl_timer: int = 0

    @classmethod
    def fresh(cls, scenario: Mapping[str, float]) -> 'ChemistryState':
        return cls(source_ph=float(scenario['source_ph']),
                   source_cl=float(scenario['source_cl']),
                   source_cond=float(scenario['source_cond']),
                   ph=float(scenario['source_ph']), cl=float(scenario['source_cl']),
                   cond=float(scenario['source_cond']))

    def apply(self, action: Mapping[str, Any]) -> None:
        tag, value = action['tag'], action['value']
        if tag == 'P2_Selected':
            self.selected = bool(value)
        elif tag == 'P2_NaOCl_Cmd':
            self.naocl_cmd = bool(value)
        elif tag == 'P2_HCl_Cmd':
            self.hcl_cmd = bool(value)
        elif tag is not None:
            raise ValueError(f'undeclared WT fixture action: {tag}')

    @staticmethod
    def delayed(requested: bool, actual: bool, timer: int, delay: int) -> tuple[bool, int]:
        if requested == actual:
            return actual, 0
        timer += 1
        if timer >= delay:
            return requested, 0
        return actual, timer

    def scan(self, delay: int) -> None:
        self.naocl_valve, self.naocl_timer = self.delayed(
            self.selected and self.naocl_cmd, self.naocl_valve, self.naocl_timer, delay)
        self.hcl_valve, self.hcl_timer = self.delayed(
            self.selected and self.hcl_cmd, self.hcl_valve, self.hcl_timer, delay)
        self.cl = f32(self.cl + ((self.source_cl + (NAOCL_GAIN if self.naocl_valve else 0))
                                  - self.cl) * MIX_RATE)
        self.ph = f32(self.ph + ((self.source_ph - (HCL_GAIN if self.hcl_valve else 0))
                                  - self.ph) * MIX_RATE)
        self.cond = f32(self.cond + (self.source_cond - self.cond) * MIX_RATE)
        # The full-horizon local admission found the P4 valve inactive; this
        # fixture fixes that observed state rather than inventing P4 authority.
        self.roft_cl = f32(self.roft_cl + (self.cl - self.roft_cl) * MIX_RATE)
        if self.roft_cl > RO_CL_MAX:
            self.membrane_damage = f32(self.membrane_damage +
                                       DAMAGE_RATE * (self.roft_cl - RO_CL_MAX))
        self.membrane_damage = min(1.0, self.membrane_damage)

    def observe(self) -> dict[str, float | int]:
        permeate = self.cond * (1.0 - RO_REJECTION * (1.0 - self.membrane_damage))
        return {'ChemTreat_pH': self.ph,
                'ChemTreat_ORP': f32(200.0 + 400.0 * self.cl),
                'ChemTreat_Conductivity': self.cond,
                'RO_Permeate_Conductivity': f32(min(20.0, max(0.02, permeate))),
                'ChemTreat_NaOCl_Valve_Sts': int(self.naocl_valve),
                'ChemTreat_HCl_Valve_Sts': int(self.hcl_valve),
                'P2_Selected': int(self.selected),
                'P2_NaOCl_Cmd': int(self.naocl_cmd),
                'P2_HCl_Cmd': int(self.hcl_cmd),
                'SYS_RUNNING': 1}


Policy = Callable[[Mapping[str, Any]], str]


def feedback_policy(observation: Mapping[str, float | int], *,
                    orp_low: float = 260.0, orp_high: float = 320.0) -> str:
    if not observation['P2_Selected']:
        return 'select_p2'
    if observation['ChemTreat_pH'] > 7.8 and not observation['P2_HCl_Cmd']:
        return 'hcl_on'
    if observation['ChemTreat_ORP'] < orp_low and not observation['P2_NaOCl_Cmd']:
        return 'naocl_on'
    if observation['ChemTreat_ORP'] > orp_high and observation['P2_NaOCl_Cmd']:
        return 'naocl_off'
    return 'observe'


def fixed_policy(*, hcl: bool, naocl: bool) -> Policy:
    def decide(observation: Mapping[str, float | int]) -> str:
        if not observation['P2_Selected']:
            return 'select_p2'
        if hcl and not observation['P2_HCl_Cmd']:
            return 'hcl_on'
        if naocl and not observation['P2_NaOCl_Cmd']:
            return 'naocl_on'
        return 'observe'
    return decide


def _within(observation: Mapping[str, float | int],
            bands: Mapping[str, list[float]]) -> bool:
    return (observation['SYS_RUNNING'] == 1 and
            all(math.isfinite(float(observation[tag])) and lo <= float(observation[tag]) <= hi
                for tag, (lo, hi) in bands.items()))


def run_case(config: Mapping[str, Any], case_id: str, policy_name: str,
             policy: Policy) -> dict[str, Any]:
    scenario = config['scenarios'][case_id]
    timing = config['timing']
    scans_per_decision = round(timing['decision_interval_sec'] / timing['scan_sec'])
    if abs(scans_per_decision * timing['scan_sec'] - timing['decision_interval_sec']) > 1e-9:
        raise ValueError('decision interval must contain whole scans')
    state = ChemistryState.fresh(scenario)
    actions = {item['id']: item for item in config['permitted_interventions']}
    events: list[dict[str, Any]] = []
    observations: list[dict[str, float | int]] = []
    steps = round(timing['horizon_sec'] / timing['decision_interval_sec'])
    for step in range(steps):
        t = step * timing['decision_interval_sec']
        before = state.observe()
        decision = policy(dict(before))
        if decision not in actions:
            raise ValueError(f'policy {policy_name} chose unauthorized {decision}')
        action = actions[decision]
        state.apply(action)
        for _ in range(scans_per_decision):
            state.scan(timing['valve_delay_scans'])
        after = state.observe()
        observations.append(after)
        events.extend((
            {'type': 'observation', 't_sec': t, 'values': before},
            {'type': 'decision', 't_sec': t, 'policy': policy_name, 'action_id': decision},
            {'type': 'intervention', 't_sec': t, 'action_id': decision,
             'tag': action['tag'], 'value': action['value'], 'authorized': True},
            {'type': 'plant_response', 't_sec': t + timing['decision_interval_sec'],
             'values': after, 'origin': config['evidence_origin']},
        ))
    service_count = round(timing['service_window_sec'] / timing['decision_interval_sec'])
    safety_fraction = sum(_within(row, config['safety_envelope']) for row in observations) / len(observations)
    service_fraction = sum(_within(row, config['service_target'])
                           for row in observations[-service_count:]) / service_count
    return {'cell_id': config['cell_id'], 'case_id': case_id, 'policy': policy_name,
            'authority': list(actions), 'evidence_origin': config['evidence_origin'],
            'safety_score': safety_fraction, 'service_score': service_fraction,
            'safety_pass': safety_fraction == 1.0,
            'service_pass': service_fraction >= config['service_required_fraction'],
            'observation_decision_intervention_response_verified': verify_chain(events),
            'event_sha256': hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
            'events': events, 'final_membrane_damage': state.membrane_damage}


def verify_chain(events: list[Mapping[str, Any]]) -> bool:
    if not events or len(events) % 4:
        return False
    for i in range(0, len(events), 4):
        observation, decision, intervention, response = events[i:i + 4]
        if ([x['type'] for x in (observation, decision, intervention, response)] !=
            ['observation', 'decision', 'intervention', 'plant_response']):
            return False
        if (decision['action_id'] != intervention['action_id'] or
            not intervention['authorized'] or
            response['t_sec'] <= observation['t_sec'] or
            decision['t_sec'] != observation['t_sec'] or
            intervention['t_sec'] != observation['t_sec']):
            return False
        if i + 4 < len(events) and response['values'] != events[i + 4]['values']:
            return False
    return True
