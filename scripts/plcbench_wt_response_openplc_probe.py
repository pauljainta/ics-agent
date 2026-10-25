#!/usr/bin/env python3
"""Run one experimental WT response case on a disposable local OpenPLC pair.

The host must restart the simulator process before each episode.
Fixture coils are runner-only and excluded from policy observations/actions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path

from ics_agent.plcbench_wd_storage_v2 import _request
from ics_agent.plcbench_wt_response import feedback_policy
from scripts.plcbench_wt_chemistry_probe import prepare_controller

ROOT = Path('/repo')
MAP = ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.json'
CONFIG = ROOT / 'configs/plcbench/wt_observation_response_openplc_v1.json'
SIM_ST = ROOT / 'testbeds/water_treatment/openplc/st/wt_simulator_response_v1.st'
CTRL = ('controller', 502)
SIM = ('simulator', 502)
FIXTURE = {'case_high_ph': 240, 'apply_pb': 241, 'apply_ack': 242,
           'active_high_ph': 243}
OBS = ('ChemTreat_pH', 'ChemTreat_ORP', 'ChemTreat_Conductivity',
       'RO_Permeate_Conductivity', 'ChemTreat_NaOCl_Valve_Sts',
       'ChemTreat_HCl_Valve_Sts', 'SYS_RUNNING', 'P2_Selected',
       'P2_NaOCl_Cmd', 'P2_HCl_Cmd')
ACTIONS = {'select_p2': ('P2_Selected', 1), 'naocl_on': ('P2_NaOCl_Cmd', 1),
           'naocl_off': ('P2_NaOCl_Cmd', 0), 'hcl_on': ('P2_HCl_Cmd', 1),
           'hcl_off': ('P2_HCl_Cmd', 0), 'observe': (None, None)}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(endpoint, kind, address):
    return _request(endpoint, kind, int(address))


def observe(mapping):
    values = {}
    for name in OBS:
        item = mapping[name]
        endpoint = SIM if item['owner'] == 'simulator' else CTRL
        kind = 'coil' if item['register_type'] == 'coil' else 'real'
        values[name] = read(endpoint, kind, item['address'])
    return values


def apply_case(case_id, config, mapping):
    for tag in ('P2_Selected', 'P2_NaOCl_Cmd', 'P2_HCl_Cmd'):
        if read(CTRL, 'coil', mapping[tag]['address']):
            raise RuntimeError(f'command not clean before fixture injection: {tag}')
    if read(SIM, 'coil', FIXTURE['apply_ack']):
        raise RuntimeError('fixture ack already set; fresh simulator restart required')
    _request(SIM, 'write', FIXTURE['case_high_ph'], case_id == 'high_ph')
    _request(SIM, 'write', FIXTURE['apply_pb'], True)
    deadline = time.monotonic() + 5
    while not read(SIM, 'coil', FIXTURE['apply_ack']):
        if time.monotonic() > deadline:
            raise RuntimeError('fixture apply was not acknowledged')
        time.sleep(0.1)
    _request(SIM, 'write', FIXTURE['apply_pb'], False)
    if bool(read(SIM, 'coil', FIXTURE['active_high_ph'])) != (case_id == 'high_ph'):
        raise RuntimeError('fixture active-case readback mismatch')
    expected = config['scenarios'][case_id]
    deadline = time.monotonic() + 5
    while True:
        row = observe(mapping)
        if (abs(row['ChemTreat_pH'] - expected['source_ph']) < 0.04 and
            abs(row['ChemTreat_ORP'] - (200 + 400 * expected['source_cl'])) < 3):
            return row
        if time.monotonic() > deadline:
            raise RuntimeError(f'fixture analyte readback mismatch: {row}')
        time.sleep(0.2)


def within(row, bands):
    return row['SYS_RUNNING'] == 1 and all(
        math.isfinite(float(row[tag])) and lo <= float(row[tag]) <= hi
        for tag, (lo, hi) in bands.items())


def run(args):
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes)
    mapping_file = json.loads(MAP.read_text())
    map_hash = hashlib.sha256(MAP.with_suffix('.yaml').read_bytes()).hexdigest()
    if mapping_file['source_yaml_sha256'] != map_hash:
        raise RuntimeError('bridge map hash mismatch')
    mapping = mapping_file['mappings']
    if not all(mapping[tag]['owner'] == 'controller' for tag, _ in ACTIONS.values() if tag):
        raise ValueError('response action escapes controller-owned P2 surface')
    if args.case not in config['scenarios']:
        raise ValueError('undeclared WT disturbance case')
    if (hashlib.sha256(SIM_ST.read_bytes()).hexdigest() !=
            config['realization']['sha256']):
        raise ValueError('live WT simulator differs from pinned realization')
    if args.duration != 180 and not args.smoke:
        raise ValueError('qualified episode must use declared 180 s horizon')
    if args.duration < 10:
        raise ValueError('smoke duration must be at least 10 s')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f'{args.case}__{args.policy}.jsonl'
    summary_path = out.with_suffix('.json')
    if out.exists() or summary_path.exists():
        raise FileExistsError('response case evidence exists; refusing overwrite')
    meta = {'schema_version': 'plcbench.wt-response-openplc/v1',
            'case_id': args.case, 'policy': args.policy, 'status': 'running',
            'generation': args.generation, 'started_utc': utc(),
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'map_yaml_sha256': map_hash,
            'simulator_st_sha256': hashlib.sha256(SIM_ST.read_bytes()).hexdigest(),
            'evidence_origin': 'isolated_local_openplc_experimental_fixture',
            'duration_sec': args.duration, 'smoke': args.smoke,
            'authority': list(ACTIONS), 'events': str(out.name)}
    try:
        meta['clean_readback'] = prepare_controller(mapping)
        before = apply_case(args.case, config, mapping)
        meta['fixture_readback'] = before
        if args.policy == 'reference':
            decide = lambda row: feedback_policy(row, orp_low=255, orp_high=315)
        elif args.policy == 'deterministic':
            decide = feedback_policy
        else:
            raise ValueError('unsupported policy')
        rows = []
        started = time.monotonic()
        steps = round(args.duration)
        with out.open('w') as handle:
            for step in range(steps):
                action = decide(before)
                tag, value = ACTIONS[action]
                now = time.monotonic() - started
                events = [
                    {'type': 'observation', 't_sec': now, 't_utc': utc(), 'values': before},
                    {'type': 'decision', 't_sec': now, 't_utc': utc(),
                     'policy': args.policy, 'action_id': action},
                ]
                if tag:
                    _request(CTRL, 'write', mapping[tag]['address'], bool(value))
                    readback = read(CTRL, 'coil', mapping[tag]['address'])
                    if readback != value:
                        raise RuntimeError(f'command readback mismatch for {tag}')
                events.append({'type': 'intervention', 't_sec': time.monotonic()-started,
                               't_utc': utc(), 'action_id': action, 'tag': tag,
                               'value': value, 'authorized': True})
                deadline = started + step + 1
                if time.monotonic() < deadline:
                    time.sleep(deadline - time.monotonic())
                after = observe(mapping)
                events.append({'type': 'plant_response', 't_sec': time.monotonic()-started,
                               't_utc': utc(), 'values': after,
                               'origin': meta['evidence_origin']})
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + '\n')
                handle.flush()
                rows.append(after)
                before = after
                if step % 30 == 29:
                    print(json.dumps({'case': args.case, 'step': step+1,
                        'pH': round(after['ChemTreat_pH'], 3),
                        'ORP': round(after['ChemTreat_ORP'], 2)}), flush=True)
        safety = sum(within(row, config['safety_envelope']) for row in rows) / len(rows)
        service_rows = rows[-min(len(rows), round(config['timing']['service_window_sec'])):]
        service = sum(within(row, config['service_target']) for row in service_rows) / len(service_rows)
        meta.update(status='complete', samples=len(rows), safety_score=safety,
                    service_score=service, safety_pass=safety == 1,
                    service_pass=service >= config['service_required_fraction'],
                    qualified=not args.smoke and len(rows) == 180,
                    final_readback=rows[-1])
    except Exception as exc:
        meta.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for tag in ('P2_HCl_Cmd', 'P2_NaOCl_Cmd', 'P2_Selected'):
            try:
                _request(CTRL, 'write', mapping[tag]['address'], False)
            except Exception:
                meta.setdefault('clear_failures', []).append(tag)
        meta['finished_utc'] = utc()
        summary_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--case', choices=('high_ph', 'low_orp'), required=True)
    p.add_argument('--policy', choices=('reference', 'deterministic'), default='deterministic')
    p.add_argument('--generation', required=True)
    p.add_argument('--duration', type=int, default=180)
    p.add_argument('--smoke', action='store_true')
    result = run(p.parse_args())
    print(json.dumps({key: result[key] for key in ('status', 'case_id', 'policy',
        'safety_score', 'service_score', 'qualified')}, sort_keys=True))

if __name__ == '__main__':
    main()
