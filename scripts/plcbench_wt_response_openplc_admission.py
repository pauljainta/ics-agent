#!/usr/bin/env python3
"""Qualify two WT response cases on an isolated OpenPLC pair with fresh sim generations."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / 'configs/plcbench/wt_response_v1.compose.yaml'
PROJECT = 'plcbench-wt-response-v1'
WT = Path('sector-water/rovisys-treatment/usecases/full-system-p1-to-p6/implementations/openplc')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sphere-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path,
                   default=ROOT / 'runs/qualifications/wt_response_openplc_v1')
    p.add_argument('--duration', type=int, default=180)
    p.add_argument('--policy', choices=('deterministic', 'reference'),
                   default='deterministic')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if (args.duration != 180 and not args.smoke) or args.duration < 10:
        p.error('qualification requires 180 seconds; shorter runs require --smoke')
    sphere = args.sphere_root.resolve(strict=True)
    paths = {
        'controller_st': sphere / WT / 'st/wt_controller.st',
        'base_simulator_st': sphere / WT / 'st/wt_simulator.st',
        'fixture_simulator_st': ROOT / 'testbeds/water_treatment/openplc/st/wt_simulator_response_v1.st',
        'canonical_map': sphere / WT / 'configs/openplc_map.yaml',
        'map_copy': ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.yaml',
        'map_json': ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.json',
        'bridge': ROOT / 'testbeds/water_distribution/openplc/scripts/modbus_bridge.py',
        'probe': ROOT / 'scripts/plcbench_wt_response_openplc_probe.py',
        'compose': COMPOSE,
        'experiment': ROOT / 'configs/plcbench/wt_observation_response_openplc_v1.json',
    }
    hashes = {key: sha(path) for key, path in paths.items()}
    if hashes['canonical_map'] != hashes['map_copy']:
        raise RuntimeError('canonical WT map and local bridge map differ')
    if json.loads(paths['map_json'].read_text())['source_yaml_sha256'] != hashes['map_copy']:
        raise RuntimeError('generated WT map is stale')
    descriptor = json.loads(paths['experiment'].read_text())
    if hashes['fixture_simulator_st'] != descriptor['realization']['sha256']:
        raise RuntimeError('fixture simulator differs from live descriptor')
    if hashes['base_simulator_st'] != descriptor['realization']['derived_from']['sha256']:
        raise RuntimeError('base simulator differs from pinned source')
    args.out_root.mkdir(parents=True, exist_ok=True)
    attempt = args.out_root / f'attempt_{1 + len(list(args.out_root.glob("attempt_*")))}'
    attempt.mkdir()
    os.chmod(attempt, 0o777)
    (attempt / 'experiment.json').write_bytes(paths['experiment'].read_bytes())
    env = os.environ.copy()
    env.update(PLCBENCH_ROOT=str(ROOT), SPHERE_ROOT=str(sphere),
               PLCBENCH_ADMISSION_DIR=str(attempt))
    record = {'schema_version': 'plcbench.wt-response-admission/v1',
              'evidence_origin': 'isolated_local_openplc_experimental_fixture',
              'attempt': attempt.name, 'source_hashes': hashes,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'duration_sec': args.duration, 'smoke': args.smoke,
              'policy': args.policy,
              'cases': [], 'status': 'running'}

    def save():
        (attempt / 'admission.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')

    def run(label, argv, timeout=120):
        print(label, flush=True)
        with (attempt / f'{label}.log').open('w') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see {label}.log')
        return (attempt / f'{label}.log').read_text().strip()

    compose = ['docker', 'compose', '-p', PROJECT, '-f', str(COMPOSE)]
    try:
        record['image_ids'] = {image: run('image_' + image.replace(':', '_'),
            ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])
            for image in ('sphere-openplc:master', 'plcbench-native:py311-pymodbus386')}
        run('up', compose + ['up', '-d', '--wait', 'controller', 'simulator', 'bridge'])
        for case in ('high_ph', 'low_orp'):
            run(case + '_sim_restart', compose + ['restart', 'simulator'])
            run(case + '_sim_wait', compose + ['up', '-d', '--wait', 'simulator'])
            run(case + '_bridge_restart', compose + ['restart', 'bridge'])
            generation = run(case + '_generation', ['docker', 'inspect',
                f'{PROJECT}-simulator-1', '--format', '{{.State.StartedAt}}'])
            if any(item['generation'] == generation for item in record['cases']):
                raise RuntimeError('simulator generation reused')
            command = compose + ['run', '--rm', '--no-deps', 'probe', 'python',
                '/repo/scripts/plcbench_wt_response_openplc_probe.py',
                '--out-dir', '/evidence', '--case', case, '--policy', args.policy,
                '--generation', generation, '--duration', str(args.duration)]
            if args.smoke:
                command.append('--smoke')
            run(case + '_probe', command, timeout=args.duration + 90)
            meta = json.loads((attempt / f'{case}__{args.policy}.json').read_text())
            if (meta['status'] != 'complete' or meta['generation'] != generation or
                meta['policy'] != args.policy):
                raise RuntimeError(f'{case} evidence invalid')
            record['cases'].append({'case_id': case, 'policy': args.policy,
                'generation': generation,
                'qualified': meta['qualified'], 'safety_score': meta['safety_score'],
                'service_score': meta['service_score'], 'safety_pass': meta['safety_pass'],
                'service_pass': meta['service_pass'], 'samples': meta['samples'],
                'fixture_readback': meta['fixture_readback']})
            save()
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            run('down', compose + ['down'])
        except Exception as exc:
            record['down_error'] = str(exc)
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print(attempt, flush=True)

if __name__ == '__main__':
    main()
