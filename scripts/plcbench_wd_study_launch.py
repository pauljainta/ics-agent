#!/usr/bin/env python3
"""Run the predeclared WD 50-pair cell in two fail-closed CHPC phases.

The plant and native-code broker stay on this host's loopback. CHPC only serves
the model over an SSH tunnel. Failed admissions remain in the phase's attempts
file and halt later measured execution for inspection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from ics_agent.plcbench_model_identity import probe_vllm, verify_live_target_deployment
from ics_agent.plcbench_wd_storage_v2 import reset_wd_storage_v2

REPO = Path(__file__).resolve().parents[1]
SSH_SOCKET = Path(os.environ.get('PLCBENCH_CHPC_SSH_SOCKET', '/nonexistent'))
LOGIN = os.environ.get('PLCBENCH_CHPC_LOGIN', '')
REMOTE = os.environ.get('PLCBENCH_CHPC_REMOTE', '')
ACCOUNT = os.environ.get('PLCBENCH_CHPC_ACCOUNT', '')
PARTITION = os.environ.get('PLCBENCH_CHPC_PARTITION', '')
LOGIN_IP = os.environ.get('PLCBENCH_CHPC_LOGIN_IP', '')
SCRIPT = 'serve_study_20261015.slurm'
MODEL_URL = 'http://127.0.0.1:18000'
CONFIG = REPO / 'configs/plcbench/wd_storage_mass_balance_v2.json'
BASE_DEPLOYMENT = REPO / 'runs/deployments/qwen_20261015.json'
KEY_FILE = REPO / 'runs/deployments/qwen_study_api_key'
DEFAULT_OUT = REPO / 'runs/studies/wd_storage_v2_50_pairs_20261015'


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ssh(*command: str) -> str:
    return subprocess.run(['ssh', '-S', str(SSH_SOCKET), '-o', 'BatchMode=yes',
                           LOGIN, *command], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def job_state(job_id: int) -> tuple[str, str]:
    output = ssh(f'squeue -h -j {job_id} -o "%T|%N"')
    if output:
        state, node = output.split('|', 1)
        return state, node
    state = ssh(f'sacct -n -j {job_id} --format=State -P | head -1')
    return state or 'MISSING', ''


def await_job(job_id: int, poll_sec: int) -> str:
    while True:
        state, node = job_state(job_id)
        print(json.dumps({'at_utc': stamp(), 'job_id': job_id,
                          'state': state, 'node': node}), flush=True)
        if state == 'RUNNING' and re.fullmatch(r'[A-Za-z0-9.-]+', node):
            return node
        if state != 'PENDING':
            raise RuntimeError(f'serving job {job_id} ended before study: {state}')
        time.sleep(poll_sec)


def start_tunnel(node: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ['ssh', '-S', str(SSH_SOCKET), '-o', 'BatchMode=yes',
         '-o', 'ExitOnForwardFailure=yes', '-N',
         '-L', f'127.0.0.1:18000:{node}:8001', LOGIN],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True)
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(f'SSH tunnel failed: {proc.stderr.read()[-500:]}')
    return proc


def await_model(key: str, *, timeout_sec: int = 900) -> dict:
    deadline = time.monotonic() + timeout_sec
    last_error = None
    while time.monotonic() < deadline:
        try:
            live = probe_vllm(MODEL_URL, 'plcbench', token=key, timeout=10)
            try:
                urllib.request.urlopen(MODEL_URL + '/v1/models', timeout=5)
            except urllib.error.HTTPError as exc:
                if exc.code not in (401, 403):
                    raise RuntimeError(f'unauthenticated model listing returned {exc.code}')
            else:
                raise RuntimeError('model listing is available without API key')
            try:
                urllib.request.urlopen(urllib.request.Request(
                    MODEL_URL + '/invocations', data=b'{}',
                    headers={'Content-Type': 'application/json'}), timeout=5)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise RuntimeError(f'undocumented route returned {exc.code}')
            else:
                raise RuntimeError('undocumented inference route is available')
            return live
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(10)
    raise RuntimeError(f'model failed authenticated health gate: {last_error}')


def verify_local_plant() -> dict:
    for service, port in [('controller_proxy', 19402), ('simulator_proxy', 19403)]:
        container = f'plcbench-wd-study-{service}-1'
        actual = subprocess.run(['docker', 'port', container, '502/tcp'],
                                check=True, capture_output=True, text=True).stdout.strip()
        if actual != f'127.0.0.1:{port}':
            raise RuntimeError(f'{container} not confined to loopback: {actual!r}')
    mapping = yaml.safe_load((REPO / 'testbeds/water_distribution/openplc/configs/'
                              'storage_v2/openplc_map.yaml').read_text())['mappings']
    return reset_wd_storage_v2(('127.0.0.1', 19402), ('127.0.0.1', 19403), mapping)


def freeze_deployment(job_id: int, live: dict, dest: Path) -> None:
    previous = json.loads(BASE_DEPLOYMENT.read_text())
    script_hash = hashlib.sha256((REPO / 'docs/baselines/chpc' / SCRIPT).read_bytes()).hexdigest()
    frozen = {k: previous[k] for k in (
        'model_source', 'weight_revision', 'vllm_version', 'runtime_digest',
        'runtime_kind', 'served_model', 'max_model_len', 'model_config_sha256')}
    frozen.update(live)
    frozen.update({'slurm_job_id': job_id, 'slurm_time_limit': '12:00:00',
                   'slurm_account': ACCOUNT, 'slurm_partition': PARTITION,
                   'job_script_sha256': script_hash,
                   'auth_relay_sha256': hashlib.sha256(
                       (REPO / 'scripts/plcbench_auth_relay.py').read_bytes()).hexdigest(),
                   'previous_deployment_sha256': hashlib.sha256(
                       BASE_DEPLOYMENT.read_bytes()).hexdigest(),
                   'stable_listing_captured_at_utc': live['checked_at_utc']})
    model = json.loads(CONFIG.read_text())['model']
    verify_live_target_deployment(model, frozen, live)
    with dest.open('x') as handle:
        json.dump(frozen, handle, indent=2)
        handle.write('\n')


def run_phase(root: Path, job_id: int, start: int, end: int, key: str) -> None:
    phase = root / f'phase_{start:02d}_{end:02d}'
    if phase.exists():
        raise FileExistsError(f'phase already exists: {phase}')
    probe = verify_local_plant()
    (root / f'plant_preflight_{start:02d}_{end:02d}.json').write_text(
        json.dumps({'checked_at_utc': stamp(), 'reset': probe}, indent=2) + '\n')
    deployment = root / f'deployment_{job_id}.json'
    if not deployment.exists():
        live = probe_vllm(MODEL_URL, 'plcbench', token=key)
        freeze_deployment(job_id, live, deployment)
    env = dict(os.environ, PYTHONPATH=str(REPO), PLCBENCH_API_KEY=key)
    command = [sys.executable, str(REPO / 'scripts/plcbench_batch.py'),
               '--config', str(CONFIG), '--controller', '127.0.0.1:19402',
               '--simulator', '127.0.0.1:19403', '--out-dir', str(phase),
               '--model-url', MODEL_URL, '--deployment-json', str(deployment),
               '--reps', '50', '--start-rep', str(start), '--end-rep', str(end),
               '--run']
    with (root / f'phase_{start:02d}_{end:02d}.log').open('x') as handle:
        completed = subprocess.run(command, cwd=REPO, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'phase {start}-{end} failed; inspect its attempts and log')
    attempts = json.loads((phase / 'attempts.json').read_text())
    if len(attempts) != 2 * (end - start + 1) or any(
            item['status'] != 'completed' for item in attempts):
        raise RuntimeError(f'phase {start}-{end} has incomplete attempts')


def submit_next() -> int:
    export = f'ALL,PLCBENCH_STUDY_ROOT={REMOTE},PLCBENCH_LOGIN_IP={LOGIN_IP}'
    output = ssh(f'cd {shlex.quote(REMOTE)} && sbatch --parsable '
                 f'--account={shlex.quote(ACCOUNT)} '
                 f'--partition={shlex.quote(PARTITION)} '
                 f'--export={shlex.quote(export)} {shlex.quote(SCRIPT)}')
    return int(output.split(';', 1)[0])


def cancel(job_id: int) -> None:
    state, _ = job_state(job_id)
    if state in {'RUNNING', 'PENDING'}:
        ssh(f'scancel {job_id}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-job-id', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--poll-sec', type=int, default=60)
    args = parser.parse_args()
    if not SSH_SOCKET.is_socket():
        parser.error('PLCBENCH_CHPC_SSH_SOCKET must name an active SSH control socket')
    if not LOGIN or not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', LOGIN):
        parser.error('PLCBENCH_CHPC_LOGIN must be a user@host SSH destination')
    if not re.fullmatch(r'/[A-Za-z0-9/._-]+', REMOTE):
        parser.error('PLCBENCH_CHPC_REMOTE must be an absolute scratch path')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', ACCOUNT) or not re.fullmatch(r'[A-Za-z0-9_-]+', PARTITION):
        parser.error('PLCBENCH_CHPC_ACCOUNT and PLCBENCH_CHPC_PARTITION are required')
    try:
        login_ip = ipaddress.IPv4Address(LOGIN_IP)
    except ValueError:
        parser.error('PLCBENCH_CHPC_LOGIN_IP must be the current login-node IPv4 source')
    if login_ip.is_unspecified or login_ip.is_loopback or login_ip.is_multicast:
        parser.error('PLCBENCH_CHPC_LOGIN_IP must be a specific non-loopback source')
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError('empty private API key')
    index = root / 'execution.json'
    if index.exists():
        manifest = json.loads(index.read_text())
        if (manifest.get('status') != 'waiting_for_model'
                or manifest.get('jobs') != [args.first_job_id]
                or any(root.glob('phase_*'))):
            raise FileExistsError(f'existing study cannot be restarted safely: {root}')
    else:
        manifest = {'schema_version': 'plcbench.study-execution/v1',
                    'cell_id': 'wd-storage-mass-balance-v2', 'declared_pairs': 50,
                    'phases': [[1, 25], [26, 50]], 'started_at_utc': stamp(),
                    'order': [{'rep': rep,
                               'arms': ['native', 'matched'] if rep % 2
                               else ['matched', 'native']} for rep in range(1, 51)],
                    'status': 'waiting_for_model', 'jobs': [args.first_job_id]}
        index.write_text(json.dumps(manifest, indent=2) + '\n')
    job_id = args.first_job_id
    try:
        for start, end in ((1, 25), (26, 50)):
            node = await_job(job_id, args.poll_sec)
            tunnel = start_tunnel(node)
            try:
                await_model(key)
                manifest['status'] = f'running_{start}_{end}'
                index.write_text(json.dumps(manifest, indent=2) + '\n')
                run_phase(root, job_id, start, end, key)
            finally:
                tunnel.terminate()
                tunnel.wait(timeout=10)
            cancel(job_id)
            if end < 50:
                job_id = submit_next()
                manifest['jobs'].append(job_id)
                manifest['status'] = 'waiting_for_second_model'
                index.write_text(json.dumps(manifest, indent=2) + '\n')
        phase_dirs = [root / 'phase_01_25', root / 'phase_26_50']
        report_path = root / 'report.json'
        report_env = dict(os.environ, PYTHONPATH=str(REPO))
        subprocess.run([sys.executable, str(REPO / 'scripts/plcbench_report.py'),
                        *map(str, phase_dirs), '--output', str(report_path)],
                       cwd=REPO, env=report_env, check=True)
        design = json.loads(report_path.read_text())['design']
        if design != {'declared_pairs': 50, 'complete_pairs': 50,
                      'native_first_pairs': 25, 'matched_first_pairs': 25}:
            raise RuntimeError(f'final report has incomplete or unbalanced design: {design}')
        manifest['report_path'] = str(report_path)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        manifest['finished_or_updated_at_utc'] = stamp()
        index.write_text(json.dumps(manifest, indent=2) + '\n')
        if manifest['status'] == 'failed':
            cancel(job_id)


if __name__ == '__main__':
    main()
