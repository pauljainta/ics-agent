#!/usr/bin/env python3
"""Check source-derived P2 dynamics against a saved local OpenPLC reference trace."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ics_agent.plcbench_wt_response import ChemistryState, source_identity

DEFAULT_CONFIG = ROOT / 'configs/plcbench/wt_observation_response_fixture_v1.json'


def replay(trace_path: Path, config_path: Path, sphere_root: Path) -> dict:
    config = json.loads(config_path.read_text())
    source = source_identity(config, repo_root=ROOT, sphere_root=sphere_root)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if len(rows) < 2:
        raise ValueError('reference trace needs at least two observations')
    first = rows[0]['values']
    if (first['ChemTreat_NaOCl_Valve_Sts'] != 1 or
        not 199 <= first['ChemTreat_ORP'] <= 201 or
        not 7.49 <= first['ChemTreat_pH'] <= 7.51):
        raise ValueError('reference trace does not start at the declared NaOCl condition')
    state = ChemistryState.fresh(config['scenarios']['low_orp'])
    state.selected = True
    state.naocl_cmd = True
    state.naocl_valve = True
    scans = 0
    errors = {'ChemTreat_ORP': [], 'RO_Permeate_Conductivity': []}
    for row in rows:
        target = round(float(row['t_sec']) / config['timing']['scan_sec'])
        if target < scans:
            raise ValueError('trace timestamps are out of order')
        while scans < target:
            state.scan(config['timing']['valve_delay_scans'])
            scans += 1
        predicted = state.observe()
        for tag in errors:
            errors[tag].append(abs(predicted[tag] - float(row['values'][tag])))
    return {'schema_version': 'plcbench.wt-response-replay/v1',
            'evidence_origin': 'saved_local_openplc_reference_vs_offline_fixture',
            'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'source_identity': source, 'samples': len(rows),
            'duration_sec': rows[-1]['t_sec'] - rows[0]['t_sec'],
            'residuals': {tag: {'max_abs': max(values),
                                'mean_abs': sum(values) / len(values)}
                          for tag, values in errors.items()},
            'interpretation': 'Numerical agreement with the saved experimental OpenPLC model; not physical validation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--sphere-root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = replay(a.trace, a.config, a.sphere_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.out:
        if a.out.exists():
            p.error(f'refusing to overwrite {a.out}')
        a.out.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
