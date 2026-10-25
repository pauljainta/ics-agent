#!/usr/bin/env python3
"""Run WT chemistry zero-action, NaOCl reference, and restoration on an isolated pair."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / 'configs/plcbench/wt_chemistry_local.compose.yaml'
PROJECT = 'plcbench-wt-chem'
WT = Path('sector-water/rovisys-treatment/usecases/full-system-p1-to-p6/implementations/openplc')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sphere-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path,
                   default=ROOT / 'runs/qualifications/wt_chemistry_pilot_v1')
    p.add_argument('--duration', type=int, default=70)
    a = p.parse_args()
    sphere = a.sphere_root.resolve(strict=True)
    if a.duration < 35:
        p.error('duration must be >=35 seconds')
    source_paths = {
        'controller_st': sphere / WT / 'st/wt_controller.st',
        'simulator_st': sphere / WT / 'st/wt_simulator.st',
        'canonical_map': sphere / WT / 'configs/openplc_map.yaml',
        'map_copy': ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.yaml',
        'map_json': ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.json',
        'bridge': ROOT / 'testbeds/water_distribution/openplc/scripts/modbus_bridge.py',
        'probe': ROOT / 'scripts/plcbench_wt_chemistry_probe.py',
        'compose': COMPOSE,
        'pilot_descriptor': ROOT / 'configs/plcbench/pilot_wt_dosing_v1.json',
    }
    hashes = {name: sha(path) for name, path in source_paths.items()}
    if hashes['canonical_map'] != hashes['map_copy']:
        raise RuntimeError('canonical WT map and local bridge map differ')
    data = json.loads(source_paths['map_json'].read_text())
    if data['source_yaml_sha256'] != hashes['map_copy']:
        raise RuntimeError('generated JSON map has stale YAML hash')
    a.out_root.mkdir(parents=True, exist_ok=True)
    attempt = a.out_root / f'attempt_{1 + len(list(a.out_root.glob("attempt_*")))}'
    attempt.mkdir()
    os.chmod(attempt, 0o777)  # probe runs as unprivileged UID 10001
    env = os.environ.copy()
    env.update(PLCBENCH_ROOT=str(ROOT), SPHERE_ROOT=str(sphere),
               PLCBENCH_ADMISSION_DIR=str(attempt))
    record = {'schema_version': 'plcbench.wt-admission/v1',
              'evidence_origin': 'isolated_local_openplc_pair',
              'attempt': attempt.name, 'source_hashes': hashes,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'phases': [], 'status': 'running'}
    (attempt / 'pilot_descriptor.json').write_bytes(source_paths['pilot_descriptor'].read_bytes())

    def save():
        (attempt / 'admission.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        index = []
        for path in sorted(a.out_root.glob('attempt_*/admission.json')):
            item = json.loads(path.read_text())
            index.append({'attempt': item['attempt'], 'status': item['status'],
                          'error': item.get('error'), 'path': str(path)})
        (a.out_root / 'attempts.json').write_text(json.dumps(index, indent=2) + '\n')

    def run(label, argv):
        print(label, flush=True)
        with (attempt / f'{label}.log').open('w') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    text=True, timeout=180 + a.duration)
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see {label}.log')
        return (attempt / f'{label}.log').read_text().strip()

    compose = ['docker', 'compose', '-p', PROJECT, '-f', str(COMPOSE)]
    try:
        images = {}
        for image in ('sphere-openplc:master', 'plcbench-native:py311-pymodbus386'):
            images[image] = run('image_' + image.replace(':', '_'),
                                ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])
        record['image_ids'] = images
        run('up', compose + ['up', '-d', '--wait', 'controller', 'simulator', 'bridge'])
        for phase in ('zero_action', 'reference_naocl', 'restoration'):
            # A full simulator process restart is required: chemistry flushing
            # alone cannot reset the private membrane_damage state.
            run(phase + '_sim_restart', compose + ['restart', 'simulator'])
            run(phase + '_sim_wait', compose + ['up', '-d', '--wait', 'simulator'])
            run(phase + '_bridge_restart', compose + ['restart', 'bridge'])
            generation = run(phase + '_generation',
                             ['docker', 'inspect', f'{PROJECT}-simulator-1',
                              '--format', '{{.State.StartedAt}}'])
            if any(x['generation'] == generation for x in record['phases']):
                raise RuntimeError('simulator generation was reused')
            probe = compose + ['run', '--rm', '--no-deps', 'probe', 'python',
                    '/repo/scripts/plcbench_wt_chemistry_probe.py', '--out-dir', '/evidence',
                    '--phase', phase, '--generation', generation,
                    '--duration', str(0 if phase == 'restoration' else a.duration)]
            run(phase + '_probe', probe)
            meta = json.loads((attempt / f'{phase}.json').read_text())
            if meta.get('error') or meta.get('generation') != generation:
                raise RuntimeError(f'{phase} proof invalid')
            record['phases'].append({'name': phase, 'generation': generation,
                                     'clean_readback': meta['clean_readback'],
                                     'samples': meta.get('samples', 0)})
            save()
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            run('down', compose + ['down'])  # preserve evidence and named volumes
        except Exception as exc:
            record['down_error'] = str(exc)
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print(attempt, flush=True)


if __name__ == '__main__':
    main()
