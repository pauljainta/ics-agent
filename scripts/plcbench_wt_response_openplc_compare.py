#!/usr/bin/env python3
"""Compare saved reference and deterministic WT fixture gates without pooling cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plcbench_wt_response_openplc_assess import assess


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference: Path, deterministic: Path) -> dict:
    assessments = {'reference': assess(reference),
                   'deterministic': assess(deterministic)}
    attempts = {'reference': reference, 'deterministic': deterministic}
    admissions = {name: json.loads((path / 'admission.json').read_text())
                  for name, path in attempts.items()}
    if any(value['policy'] != name or value['status'] != 'complete'
           for name, value in assessments.items()):
        raise ValueError('policy identity or assessment status mismatch')
    if len({value['experiment_sha256'] for value in assessments.values()}) != 1:
        raise ValueError('reference and deterministic objective snapshots differ')
    shared_sources = ('controller_st', 'base_simulator_st', 'fixture_simulator_st',
                      'canonical_map', 'map_copy', 'map_json', 'bridge', 'compose')
    for key in shared_sources:
        if len({admission['source_hashes'][key] for admission in admissions.values()}) != 1:
            raise ValueError(f'realization differs: {key}')
    if admissions['reference']['image_ids'] != admissions['deterministic']['image_ids']:
        raise ValueError('OpenPLC/bridge images differ')
    cases = []
    for case_id in ('high_ph', 'low_orp'):
        rows = {}
        for policy, attempt in attempts.items():
            item = next((r for r in assessments[policy]['results']
                         if r['case_id'] == case_id), None)
            if item is None or not item['chain_verified']:
                raise ValueError(f'missing verified {policy}/{case_id} chain')
            meta = json.loads((attempt / f'{case_id}__{policy}.json').read_text())
            rows[policy] = {'safety_score': item['safety_score'],
                            'service_score': item['service_score'],
                            'max_sample_gap_sec': item['max_sample_gap_sec'],
                            'last_response_sec': item['last_response_sec'],
                            'safety_pass': item['safety_pass'],
                            'service_pass': item['service_pass'],
                            'generation': item['generation'],
                            'trace_sha256': item['trace_sha256'],
                            'authority': meta['authority']}
        if rows['reference']['authority'] != rows['deterministic']['authority']:
            raise ValueError(f'{case_id}: policy authorities differ')
        if rows['reference']['generation'] == rows['deterministic']['generation']:
            raise ValueError(f'{case_id}: simulator generation reused across policies')
        cases.append({'case_id': case_id, 'reference': rows['reference'],
                      'deterministic': rows['deterministic'], 'authority_equal': True})
    return {'schema_version': 'plcbench.wt-response-comparison/v1',
            'status': 'complete', 'cell_id': 'wt-observation-response-openplc-v1',
            'experiment_sha256': assessments['reference']['experiment_sha256'],
            'shared_source_sha256': {
                key: admissions['reference']['source_hashes'][key]
                for key in shared_sources},
            'image_ids': admissions['reference']['image_ids'],
            'policy_probe_sha256': {
                name: admissions[name]['source_hashes']['probe']
                for name in attempts},
            'assessment_sha256': {
                name: sha(path / 'assessment.json')
                for name, path in attempts.items()},
            'cases': cases,
            'replications_per_case_per_policy': 1,
            'policy_all_pass': {name: value['all_pass']
                                for name, value in assessments.items()},
            'interpretation': 'Local engineering qualification only; one unrandomized run per policy/case, no statistical or LLM comparison.',
            'llm_status': 'unassessable_no_model_decisions'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--deterministic', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = compare(args.reference, args.deterministic)
    payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.out:
        if args.out.exists():
            p.error('comparison exists; refusing overwrite')
        args.out.write_text(payload)
    else:
        print(payload, end='')

if __name__ == '__main__':
    main()
