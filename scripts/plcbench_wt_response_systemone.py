#!/usr/bin/env python3
"""Export or run System One decisions for the offline WT fixture; never contacts a PLC."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D3_CHECKPOINT_REVISION = 'cf15549a78fc52598ddd1951f321c58372bbd64f'
sys.path.insert(0, str(ROOT))

from ics_agent.plcbench_systemone import SystemOneChoicePolicy, canonical
from ics_agent.plcbench_wt_response import ChemistryState, run_case, source_identity


def http_json(url: str, *, body: dict | None, key: str | None, timeout: float) -> dict:
    headers = {'Accept': 'application/json'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(url, data=canonical(body) if body is not None else None,
                                     headers=headers, method='POST' if body is not None else 'GET')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as reply:
        result = json.loads(reply.read(1_000_001))
    if not isinstance(result, dict):
        raise ValueError('System One endpoint returned a non-object')
    return result


def model_ids(inventory: dict) -> set[str]:
    rows = inventory.get('models', inventory.get('data', []))
    if not isinstance(rows, list):
        raise ValueError('invalid System One model inventory')
    return {row if isinstance(row, str) else row.get('id', row.get('name'))
            for row in rows if isinstance(row, str) or isinstance(row, dict)} - {None}


def run(policy_path: Path, sphere_root: Path, out_dir: Path, *, mode: str,
        endpoint: str | None = None, key_file: Path | None = None,
        deployment_json: Path | None = None,
        timeout: float = 60.0) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite evidence in {out_dir}')
    spec_bytes = policy_path.read_bytes()
    spec = json.loads(spec_bytes)
    fixture_bytes = (ROOT / spec['fixture_ref']).read_bytes()
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    if fixture_sha != spec['fixture_sha256']:
        raise ValueError('WT fixture differs from pinned System One policy contract')
    fixture = json.loads(fixture_bytes)
    source = source_identity(fixture, repo_root=ROOT, sphere_root=sphere_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'policy.json').write_bytes(spec_bytes)
    (out_dir / 'experiment.json').write_bytes(fixture_bytes)
    summary = {
        'schema_version': 'plcbench.systemone-wt-result/v1', 'mode': mode,
        'status': 'running', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'policy_sha256': hashlib.sha256(spec_bytes).hexdigest(),
        'fixture_sha256': fixture_sha, 'source_identity': source,
        'adapter_sha256': hashlib.sha256((ROOT / 'ics_agent/plcbench_systemone.py').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evidence_origin': fixture['evidence_origin'], 'results': [],
    }
    try:
        if deployment_json:
            deployment_bytes = deployment_json.read_bytes()
            deployment = json.loads(deployment_bytes)
            if (deployment.get('schema_version') != 'plcbench.d3-deployment/v1'
                    or deployment.get('model_id') != spec['model']['model_id']
                    or deployment.get('model_source') != 'vllm-sr/d3'
                    or deployment.get('weight_revision') != D3_CHECKPOINT_REVISION
                    or not deployment.get('server_sha256')):
                raise ValueError('D3 deployment identity does not match the decision policy')
            (out_dir / 'deployment.json').write_bytes(deployment_bytes)
            summary['deployment_identity_sha256'] = hashlib.sha256(deployment_bytes).hexdigest()
            summary['deployment_identity_status'] = 'provided_unverified_endpoint_binding'
        elif mode != 'export':
            summary['deployment_identity_status'] = 'missing_unassessable_runtime_identity'
        key = key_file.read_text().strip() if key_file else None
        if mode != 'export':
            if not endpoint or not endpoint.startswith(('http://', 'https://')):
                raise ValueError('a System One HTTP endpoint is required')
            base = endpoint.rstrip('/')
            summary['endpoint'] = base
            inventory = http_json(base + '/v1/models', body=None, key=key, timeout=timeout)
            if spec['model']['model_id'] not in model_ids(inventory):
                raise ValueError('declared model is absent from System One inventory')
            summary['model_inventory'] = inventory
            summary['model_inventory_sha256'] = hashlib.sha256(canonical(inventory)).hexdigest()
        for case_id, scenario in fixture['scenarios'].items():
            policy = SystemOneChoicePolicy(spec, fixture, lambda request: http_json(
                base + '/v1/systemone', body=request, key=key, timeout=timeout)) if mode != 'export' else None
            if mode == 'export':
                request = SystemOneChoicePolicy(spec, fixture, lambda _: {}).request(
                    ChemistryState.fresh(scenario).observe())
                filename = f'{case_id}.systemone.json'
                (out_dir / filename).write_text(json.dumps(request, indent=2, sort_keys=True) + '\n')
                summary['results'].append({'case_id': case_id, 'request_file': filename,
                                           'assessment_status': 'unassessable_no_model_response'})
            elif mode == 'preflight':
                observation = ChemistryState.fresh(scenario).observe()
                request = policy.request(observation)
                (out_dir / f'{case_id}.systemone.json').write_text(
                    json.dumps(request, indent=2, sort_keys=True) + '\n')
                action = policy(observation)
                summary['results'].append({'case_id': case_id, 'action_id': action,
                                           'decisions': policy.records,
                                           'assessment_status': 'one_step_only_safety_service_unassessable'})
            else:
                result = run_case(fixture, case_id, spec['policy_id'], policy)
                events = result.pop('events')
                trace = f'{case_id}.events.jsonl'
                with (out_dir / trace).open('w') as handle:
                    for event in events:
                        handle.write(json.dumps(event, sort_keys=True) + '\n')
                result.update({'event_file': trace, 'decisions': policy.records,
                               'assessment_status': 'assessed_offline_fixture'})
                summary['results'].append(result)
        summary['status'] = 'exported' if mode == 'export' else 'complete'
    except Exception as exc:
        summary['status'] = 'failed'
        summary['assessment_status'] = ('unassessable_model_not_admitted' if not summary['results']
                                        else 'unassessable_incomplete_model_run')
        summary['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        raise
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path,
                        default=ROOT / 'configs/plcbench/wt_response_systemone_policy_v1.json')
    parser.add_argument('--sphere-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['export', 'preflight', 'full'], default='export')
    parser.add_argument('--endpoint', help='System One base URL; only used in preflight/full')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--deployment-json', type=Path,
                        help='CHPC job identity snapshot to embed in this bundle')
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()
    summary = run(args.policy, args.sphere_root, args.out_dir, mode=args.mode,
                  endpoint=args.endpoint, key_file=args.api_key_file,
                  deployment_json=args.deployment_json, timeout=args.timeout)
    print(json.dumps({'status': summary['status'], 'mode': summary['mode'],
                      'cases': len(summary['results'])}))


if __name__ == '__main__':
    main()
