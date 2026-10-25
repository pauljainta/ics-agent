#!/usr/bin/env python3
"""Independently assess one isolated OpenPLC WT response admission from saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(row, bands):
    return row['SYS_RUNNING'] == 1 and all(
        math.isfinite(float(row[tag])) and lo <= float(row[tag]) <= hi
        for tag, (lo, hi) in bands.items())


def assess(attempt: Path) -> dict:
    admission = json.loads((attempt / 'admission.json').read_text())
    config_path = attempt / 'experiment.json'
    config = json.loads(config_path.read_text())
    if admission['status'] != 'complete':
        raise ValueError('admission did not complete')
    base_hash = config['realization'].get('derived_from', config['realization'])['sha256']
    if admission['source_hashes']['base_simulator_st'] != base_hash:
        raise ValueError('base simulator differs from frozen objective')
    if ('derived_from' in config['realization'] and
        admission['source_hashes']['fixture_simulator_st'] != config['realization']['sha256']):
        raise ValueError('fixture simulator differs from live descriptor')
    policy = admission.get('policy', 'deterministic')
    if policy not in ('deterministic', 'reference'):
        raise ValueError('unsupported admission policy')
    cases = admission['cases']
    if [item['case_id'] for item in cases] != ['high_ph', 'low_orp']:
        raise ValueError('both predeclared cases are required in order')
    if len({item['generation'] for item in cases}) != len(cases):
        raise ValueError('simulator generation reused')
    actions = {item['id']: item for item in config['permitted_interventions']}
    observed_tags = set(config['observations']) | {
        'P2_Selected', 'P2_NaOCl_Cmd', 'P2_HCl_Cmd'}
    results = []
    for case in cases:
        case_id = case['case_id']
        stem = f'{case_id}__{policy}'
        meta_path = attempt / f'{stem}.json'
        trace_path = attempt / f'{stem}.jsonl'
        meta = json.loads(meta_path.read_text())
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if (meta['status'] != 'complete' or not meta['qualified'] or meta['smoke'] or
            meta['case_id'] != case_id or meta['policy'] != policy or
            case.get('policy', policy) != policy or
            meta['authority'] != list(actions) or
            meta['evidence_origin'] != admission['evidence_origin'] or
            meta['generation'] != case['generation'] or
            meta['config_sha256'] != sha(config_path) or
            meta['simulator_st_sha256'] != admission['source_hashes']['fixture_simulator_st'] or
            meta['map_yaml_sha256'] != admission['source_hashes']['map_copy']):
            raise ValueError(f'{case_id}: identity or qualification mismatch')
        clean = meta['clean_readback']
        if (not 7.49 <= clean['ChemTreat_pH'] <= 7.51 or
            not 199 <= clean['ChemTreat_ORP'] <= 201 or
            not 1.48 <= clean['RO_Permeate_Conductivity'] <= 1.52 or
            clean['P2_Selected'] or clean['P2_NaOCl_Cmd'] or
            clean['ChemTreat_NaOCl_Valve_Sts']):
            raise ValueError(f'{case_id}: reset readback invalid')
        target = config['scenarios'][case_id]
        fixture = meta['fixture_readback']
        if (abs(fixture['ChemTreat_pH'] - target['source_ph']) >= 0.04 or
            abs(fixture['ChemTreat_ORP'] - (200 + 400*target['source_cl'])) >= 3):
            raise ValueError(f'{case_id}: fixture readback invalid')
        if len(events) != 4 * 180:
            raise ValueError(f'{case_id}: incomplete event chain')
        rows = []
        response_times = []
        for i in range(0, len(events), 4):
            observation, decision, intervention, response = events[i:i+4]
            if [x['type'] for x in events[i:i+4]] != [
                    'observation', 'decision', 'intervention', 'plant_response']:
                raise ValueError(f'{case_id}: event sequence invalid at {i}')
            action = actions.get(decision['action_id'])
            if decision['policy'] != policy:
                raise ValueError(f'{case_id}: policy mismatch at {i}')
            if (action is None or intervention['action_id'] != decision['action_id'] or
                intervention['tag'] != action['tag'] or intervention['value'] != action['value'] or
                not intervention['authorized'] or
                not observation['t_sec'] <= decision['t_sec'] <= intervention['t_sec'] < response['t_sec']):
                raise ValueError(f'{case_id}: decision/action chain invalid at {i}')
            if i and (observation['values'] != events[i-1]['values'] or
                      observation['t_sec'] < events[i-1]['t_sec']):
                raise ValueError(f'{case_id}: response/next-observation mismatch at {i}')
            if (set(observation['values']) != observed_tags or
                set(response['values']) != observed_tags):
                raise ValueError(f'{case_id}: observation surface mismatch')
            rows.append(response['values'])
            response_times.append(float(response['t_sec']))
        gaps = [end - start for start, end in zip(response_times, response_times[1:])]
        max_gap = max(gaps)
        if (response_times[0] > 3 or response_times[-1] < 175 or
            response_times[-1] > 190 or max_gap > 2.5 or min(gaps) <= 0):
            raise ValueError(f'{case_id}: timing or horizon gate failed')
        safety = sum(within(row, config['safety_envelope']) for row in rows) / len(rows)
        service_rows = rows[-round(config['timing']['service_window_sec']):]
        service = sum(within(row, config['service_target']) for row in service_rows) / len(service_rows)
        safety_pass = safety == 1
        service_pass = service >= config['service_required_fraction']
        if (abs(meta['safety_score'] - safety) > 1e-12 or
            abs(meta['service_score'] - service) > 1e-12 or
            abs(case['safety_score'] - safety) > 1e-12 or
            abs(case['service_score'] - service) > 1e-12 or
            meta['safety_pass'] != safety_pass or
            meta['service_pass'] != service_pass or
            case['safety_pass'] != safety_pass or
            case['service_pass'] != service_pass or
            not case['qualified']):
            raise ValueError(f'{case_id}: score mismatch')
        results.append({'case_id': case_id, 'generation': case['generation'],
                        'trace_sha256': sha(trace_path), 'meta_sha256': sha(meta_path),
                        'safety_score': safety, 'service_score': service,
                        'safety_pass': safety_pass,
                        'service_pass': service_pass,
                        'chain_verified': True,
                        'max_sample_gap_sec': max_gap,
                        'last_response_sec': response_times[-1]})
    return {'schema_version': 'plcbench.wt-response-assessment/v1',
            'status': 'complete', 'evidence_origin': admission['evidence_origin'],
            'policy': policy,
            'experiment_sha256': sha(config_path),
            'admission_sha256': sha(attempt / 'admission.json'),
            'all_pass': all(r['safety_pass'] and r['service_pass'] for r in results),
            'results': results, 'llm_status': 'unassessable_no_model_decisions'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt', type=Path)
    args = p.parse_args()
    out = args.attempt / 'assessment.json'
    if out.exists():
        p.error('assessment already exists; refusing overwrite')
    result = assess(args.attempt)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'all_pass': result['all_pass'],
                      'llm_status': result['llm_status']}))

if __name__ == '__main__':
    main()
