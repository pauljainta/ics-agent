#!/usr/bin/env python3
"""Evidence probe for the isolated WT pair; never addresses a shared PLC."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
import time
from pathlib import Path

from ics_agent.plcbench_wd_storage_v2 import _request

MAP = Path('/repo/testbeds/water_treatment/openplc/configs/openplc_map.json')
CTRL = ('controller', 502)
SIM = ('simulator', 502)
OBS = ('ChemTreat_Conductivity', 'ChemTreat_pH', 'ChemTreat_ORP',
       'RO_Permeate_Conductivity')
COILS = ('SYS_RUNNING', 'P2_Selected', 'P2_NaOCl_Cmd',
         'ChemTreat_NaOCl_Valve', 'ChemTreat_NaOCl_Valve_Sts')


def read(endpoint, operation, address):
    return _request(endpoint, operation, int(address))


def observation(mapping):
    # Separate owner reads: no writable action is taken by this function.
    values = {name: read(SIM, 'real', mapping[name]['address']) for name in OBS}
    for name in COILS:
        endpoint = SIM if mapping[name]['owner'] == 'simulator' else CTRL
        values[name] = read(endpoint, 'coil', mapping[name]['address'])
    return values


def write(mapping, tag, value):
    if mapping[tag]['owner'] != 'controller' or tag not in {
        'HMI_Start_PB', 'HMI_Stop_PB', 'P2_Selected', 'P2_NaCl_Cmd',
        'P2_NaOCl_Cmd', 'P2_HCl_Cmd'}:
        raise ValueError('undeclared operator write')
    _request(CTRL, 'write', int(mapping[tag]['address']), bool(value))


def wait_clean(mapping):
    deadline = time.monotonic() + 20
    while True:
        v = observation(mapping)
        clean = (299 <= v['ChemTreat_Conductivity'] <= 301
                 and 7.49 <= v['ChemTreat_pH'] <= 7.51
                 and 199 <= v['ChemTreat_ORP'] <= 201
                 and 1.48 <= v['RO_Permeate_Conductivity'] <= 1.52
                 and v['P2_Selected'] == 0 and v['P2_NaOCl_Cmd'] == 0
                 and v['ChemTreat_NaOCl_Valve_Sts'] == 0)
        if clean:
            return v
        if time.monotonic() >= deadline:
            raise RuntimeError(f'full-restart clean-state readback timed out: {v}')
        time.sleep(0.5)


def prepare_controller(mapping):
    for tag in ('P2_Selected', 'P2_NaCl_Cmd', 'P2_NaOCl_Cmd', 'P2_HCl_Cmd',
                'HMI_Start_PB'):
        write(mapping, tag, False)
    write(mapping, 'HMI_Stop_PB', True)
    time.sleep(0.5)
    write(mapping, 'HMI_Stop_PB', False)
    wait_clean(mapping)
    write(mapping, 'HMI_Start_PB', True)
    time.sleep(0.5)
    write(mapping, 'HMI_Start_PB', False)
    deadline = time.monotonic() + 10
    while read(CTRL, 'coil', mapping['SYS_RUNNING']['address']) != 1:
        if time.monotonic() >= deadline:
            raise RuntimeError('controller did not reach RUNNING')
        time.sleep(0.2)
    return wait_clean(mapping)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--phase', choices=('zero_action', 'reference_naocl', 'restoration'), required=True)
    p.add_argument('--generation', required=True,
                   help='Docker simulator StartedAt identity supplied by host orchestrator')
    p.add_argument('--duration', type=float, default=70)
    p.add_argument('--interval', type=float, default=1)
    a = p.parse_args()
    if (a.phase != 'restoration' and a.duration < 35) or not 0.5 <= a.interval <= 2:
        p.error('requires >=35 s and 0.5-2 s sampling')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f'{a.phase}.jsonl'
    if out.exists():
        p.error('phase evidence exists; refusing overwrite')
    data = json.loads(MAP.read_text())
    yaml_path = MAP.with_suffix('.yaml')
    map_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    if data.get('source_yaml_sha256') != map_hash:
        raise RuntimeError('WT map copy fails source hash check')
    mapping = data['mappings']
    meta = {'schema_version': 'plcbench.wt-chemistry-probe/v1',
            'phase': a.phase, 'generation': a.generation,
            'map_yaml_sha256': map_hash, 'evidence_origin': 'isolated_local_openplc_pair',
            'duration_sec': a.duration, 'interval_sec': a.interval,
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    meta_path = a.out_dir / f'{a.phase}.json'
    try:
        meta['clean_readback'] = prepare_controller(mapping)
        if a.phase == 'restoration':
            meta['restored'] = True
            return
        if a.phase == 'reference_naocl':
            write(mapping, 'P2_Selected', True)
            write(mapping, 'P2_NaOCl_Cmd', True)
            meta['intervention'] = {'commands': ['P2_Selected=true', 'P2_NaOCl_Cmd=true'],
                                    't_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        started = time.monotonic()
        next_sample = started
        n = 0
        with out.open('w') as handle:
            while True:
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(next_sample - now)
                t = time.monotonic() - started
                v = observation(mapping)
                row = {'t_sec': t, 't_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                       'values': v}
                handle.write(json.dumps(row, sort_keys=True) + '\n')
                handle.flush()
                n += 1
                if n % 15 == 0:
                    print(json.dumps({'phase': a.phase, 't_sec': round(t, 2),
                        'ORP': round(v['ChemTreat_ORP'], 2),
                        'valve': v['ChemTreat_NaOCl_Valve_Sts']}), flush=True)
                if t >= a.duration:
                    break
                next_sample += a.interval
        meta['samples'] = n
        meta['last_readback'] = v
    except Exception as exc:
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if a.phase == 'reference_naocl':
            try:
                write(mapping, 'P2_NaOCl_Cmd', False)
                write(mapping, 'P2_Selected', False)
                meta['commands_cleared'] = True
            except Exception as exc:
                meta['clear_error'] = str(exc)
        meta['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
