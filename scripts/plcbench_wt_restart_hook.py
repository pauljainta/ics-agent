#!/usr/bin/env python3
"""Host-side WT full-restart hook for an isolated local Docker pair.

No arguments: plcbench_batch invokes this executable once per WT episode.
Set PLCBENCH_WT_SPHERE_ROOT to the isolated SPHERE checkout. This host process
uses Docker CLI; the native-code child never receives a Docker socket.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import struct
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ics_agent.plcbench_wt_restart import verify_wt_restart_proof
COMPOSE = ROOT / 'configs/plcbench/wt_chemistry_local.compose.yaml'
LOOPBACK = ROOT / 'configs/plcbench/wt_chemistry_loopback.compose.yaml'
PROXY = ROOT / 'scripts/plcbench_loopback_proxy.py'
PROJECT = 'plcbench-wt-chem'
ST_REL = Path('sector-water/rovisys-treatment/usecases/full-system-p1-to-p6/implementations/openplc/st/wt_simulator.st')
MAP_REL = Path('sector-water/rovisys-treatment/usecases/full-system-p1-to-p6/implementations/openplc/configs/openplc_map.yaml')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, env: dict[str, str], directory: Path, label: str) -> str:
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
    (directory / f'{label}.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'{label} exited {completed.returncode}; see {label}.log')
    return completed.stdout.strip()


def host_readback(map_json: Path, controller_port: int, simulator_port: int) -> dict:
    """Require actual Modbus replies through both host loopback proxies."""
    from pymodbus.client import ModbusTcpClient

    mappings = json.loads(map_json.read_text())['mappings']
    with ModbusTcpClient('127.0.0.1', port=controller_port, timeout=3) as controller:
        if not controller.connect():
            raise RuntimeError('WT controller loopback proxy is unreachable')
        running = controller.read_coils(int(mappings['SYS_RUNNING']['address']), count=1)
        if running.isError() or not running.bits or not running.bits[0]:
            raise RuntimeError('WT controller loopback status read failed')
    with ModbusTcpClient('127.0.0.1', port=simulator_port, timeout=3) as simulator:
        if not simulator.connect():
            raise RuntimeError('WT simulator loopback proxy is unreachable')
        ph = simulator.read_holding_registers(int(mappings['ChemTreat_pH']['address']), count=2)
        if ph.isError() or len(ph.registers) != 2:
            raise RuntimeError('WT simulator loopback pH read failed')
        ph_value = struct.unpack('>f', struct.pack('>HH', *ph.registers))[0]
        if not 7.49 <= ph_value <= 7.51:
            raise RuntimeError('WT simulator loopback pH is not clean')
    return {'SYS_RUNNING': 1, 'ChemTreat_pH': ph_value}


def restart() -> dict:
    sphere_value = os.environ.get('PLCBENCH_WT_SPHERE_ROOT')
    if not sphere_value:
        raise RuntimeError('PLCBENCH_WT_SPHERE_ROOT is required')
    sphere = Path(sphere_value).resolve(strict=True)
    source = sphere / ST_REL
    canonical_map = sphere / MAP_REL
    local_map = ROOT / 'testbeds/water_treatment/openplc/configs/openplc_map.yaml'
    map_json = local_map.with_suffix('.json')
    sim_hash = sha(source)
    expected = os.environ.get('PLCBENCH_WT_EXPECTED_SIM_SHA256')
    if expected and sim_hash != expected:
        raise RuntimeError('WT simulator source differs from expected hash')
    if sha(canonical_map) != sha(local_map):
        raise RuntimeError('canonical WT map differs from mounted bridge map')
    if json.loads(map_json.read_text()).get('source_yaml_sha256') != sha(local_map):
        raise RuntimeError('WT JSON bridge map is stale')
    evidence_root = Path(os.environ.get('PLCBENCH_WT_RESET_DIR',
                             ROOT / 'runs/qualifications/wt_restart_hook'))
    evidence_root.mkdir(parents=True, exist_ok=True)
    directory = evidence_root / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
                                 + '-' + uuid.uuid4().hex[:8])
    directory.mkdir()
    os.chmod(directory, 0o777)  # sidecar uses unprivileged UID 10001
    env = os.environ.copy()
    controller_port = int(env.get('PLCBENCH_WT_CONTROLLER_PORT', '19502'))
    simulator_port = int(env.get('PLCBENCH_WT_SIMULATOR_PORT', '19503'))
    if not (0 < controller_port < 65536 and 0 < simulator_port < 65536
            and controller_port != simulator_port):
        raise ValueError('WT loopback ports must be distinct valid TCP ports')
    env.update(PLCBENCH_ROOT=str(ROOT), SPHERE_ROOT=str(sphere),
               PLCBENCH_ADMISSION_DIR=str(directory))
    compose = ['docker', 'compose', '-p', PROJECT, '-f', str(COMPOSE),
               '-f', str(LOOPBACK)]
    record = {'status': 'running', 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'simulator_st_sha256': sim_hash, 'map_yaml_sha256': sha(local_map),
              'loopback_overlay_sha256': sha(LOOPBACK),
              'loopback_proxy_sha256': sha(PROXY),
              'controller_endpoint': f'127.0.0.1:{controller_port}',
              'simulator_endpoint': f'127.0.0.1:{simulator_port}'}
    path = directory / 'restart.json'
    try:
        run(compose + ['up', '-d', '--wait', 'controller', 'simulator', 'bridge',
                       'controller_proxy', 'simulator_proxy'],
            env=env, directory=directory, label='up')
        before = run(['docker', 'inspect', f'{PROJECT}-simulator-1',
                      '--format', '{{.State.StartedAt}}'], env=env,
                     directory=directory, label='before')
        run(compose + ['restart', 'simulator'], env=env, directory=directory, label='restart')
        run(compose + ['up', '-d', '--wait', 'simulator'], env=env,
            directory=directory, label='wait')
        run(compose + ['restart', 'bridge'], env=env,
            directory=directory, label='bridge_restart')
        after = run(['docker', 'inspect', f'{PROJECT}-simulator-1',
                     '--format', '{{.State.StartedAt}}'], env=env,
                    directory=directory, label='after')
        if before == after:
            raise RuntimeError('Docker simulator StartedAt did not change')
        run(compose + ['run', '--rm', '--no-deps', 'probe', 'python',
                       '/repo/scripts/plcbench_wt_chemistry_probe.py', '--out-dir', '/evidence',
                       '--phase', 'restoration', '--generation', after, '--duration', '0'],
            env=env, directory=directory, label='probe')
        probe = json.loads((directory / 'restoration.json').read_text())
        if not probe.get('restored') or probe.get('generation') != after:
            raise RuntimeError('WT restoration probe did not confirm generation')
        controller_binding = run(['docker', 'port', f'{PROJECT}-controller_proxy-1', '502/tcp'],
                                 env=env, directory=directory, label='controller_binding')
        simulator_binding = run(['docker', 'port', f'{PROJECT}-simulator_proxy-1', '502/tcp'],
                                env=env, directory=directory, label='simulator_binding')
        if (controller_binding != record['controller_endpoint']
                or simulator_binding != record['simulator_endpoint']):
            raise RuntimeError('WT proxy is not bound only to the declared host loopback ports')
        endpoint_readback = host_readback(map_json, controller_port, simulator_port)
        proof = {'proof_kind': 'wt-full-restart/v1', 'generation_id': after,
                 'started_at_before': before, 'started_at_after': after,
                 'controller_endpoint': record['controller_endpoint'],
                 'simulator_endpoint': record['simulator_endpoint'],
                 'loopback_overlay_sha256': record['loopback_overlay_sha256'],
                 'loopback_proxy_sha256': record['loopback_proxy_sha256'],
                 'host_endpoint_verified': True,
                 'host_port_bindings': {'controller': controller_binding,
                                        'simulator': simulator_binding},
                 'host_endpoint_readback': endpoint_readback,
                 'membrane_damage': 0,
                 'membrane_damage_basis': 'source-bound-permeate-readback/v1',
                 'simulator_st_sha256': sim_hash,
                 'clean_readback': probe['clean_readback'],
                 'evidence_path': str(directory)}
        verify_wt_restart_proof(proof, expected_simulator_sha256=sim_hash)
        record.update(status='complete', proof=proof)
        return proof
    except Exception as exc:
        record.update(status='failed', error={'type': type(exc).__name__,
                                               'message': str(exc)})
        raise
    finally:
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    try:
        print(json.dumps(restart(), sort_keys=True))
    except Exception as exc:
        print(f'WT restart hook failed: {exc}', file=sys.stderr)
        sys.exit(1)
