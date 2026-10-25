#!/usr/bin/env python3
"""Run the offline WT observation-response engineering fixture; no PLC or model I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ics_agent.plcbench_wt_response import (feedback_policy, fixed_policy,
                                             run_case, source_identity)

DEFAULT = ROOT / 'configs/plcbench/wt_observation_response_fixture_v1.json'


def run(config_path: Path, sphere_root: Path, out_dir: Path) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite fixture evidence in {out_dir}')
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get('schema_version') != 'plcbench.operator-response/v2':
        raise ValueError('wrong WT fixture schema')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'experiment.json').write_bytes(config_bytes)
    policies = {
        'reference': lambda o: feedback_policy(o, orp_low=255, orp_high=315),
        'deterministic': feedback_policy,
        'constant_none': fixed_policy(hcl=False, naocl=False),
        'constant_hcl': fixed_policy(hcl=True, naocl=False),
        'constant_naocl': fixed_policy(hcl=False, naocl=True),
        'constant_both': fixed_policy(hcl=True, naocl=True),
    }
    results = []
    try:
        source = source_identity(config, repo_root=ROOT, sphere_root=sphere_root)
        for case_id in config['scenarios']:
            for name, policy in policies.items():
                result = run_case(config, case_id, name, policy)
                events = result.pop('events')
                trace_file = f'{case_id}__{name}.jsonl'
                with (out_dir / trace_file).open('w') as handle:
                    for event in events:
                        handle.write(json.dumps(event, sort_keys=True) + '\n')
                result['event_file'] = trace_file
                results.append(result)
        paired = {name: [r for r in results if r['policy'] == name]
                  for name in policies}
        summary = {
            'schema_version': 'plcbench.operator-response-result/v1',
            'status': 'complete', 'evidence_origin': config['evidence_origin'],
            'cell_id': config['cell_id'],
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'source_identity': source,
            'llm_status': 'unassessable_no_model_decisions',
            'authority_equal': len({tuple(r['authority']) for r in results}) == 1,
            'reference_passes_both_cases': all(r['safety_pass'] and r['service_pass']
                                                for r in paired['reference']),
            'deterministic_passes_both_cases': all(r['safety_pass'] and r['service_pass']
                                                    for r in paired['deterministic']),
            'constant_vectors_all_fail_at_least_one_case': all(
                not all(r['safety_pass'] and r['service_pass'] for r in paired[name])
                for name in policies if name.startswith('constant_')),
            'results': results,
        }
    except Exception as exc:
        (out_dir / 'failure.json').write_text(json.dumps({
            'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
            'results_completed': len(results)}, indent=2) + '\n')
        raise
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--sphere-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    result = run(a.config, a.sphere_root, a.out_dir)
    print(json.dumps({key: result[key] for key in (
        'cell_id', 'status', 'authority_equal', 'reference_passes_both_cases',
        'deterministic_passes_both_cases', 'constant_vectors_all_fail_at_least_one_case',
        'llm_status')}, indent=2))


if __name__ == '__main__':
    main()
