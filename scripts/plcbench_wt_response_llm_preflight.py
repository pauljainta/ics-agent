#!/usr/bin/env python3
"""One read-only Qwen decision per hidden WT fixture case; no plant action."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ics_agent.plcbench_model_identity import probe_vllm, verify_live_target_deployment
from ics_agent.plcbench_native import build_openai_generate
from ics_agent.plcbench_wt_response import ChemistryState, source_identity
from ics_agent.plcbench_wt_response_llm import LLMResponsePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT / 'configs/plcbench/wt_response_llm_policy_v1.json')
    parser.add_argument('--sphere-root', type=Path, required=True)
    parser.add_argument('--deployment-json', type=Path, required=True)
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('output directory must be empty')
    spec_bytes = args.policy.read_bytes()
    spec = json.loads(spec_bytes)
    fixture_path = ROOT / spec['fixture_ref']
    fixture_bytes = fixture_path.read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != spec['fixture_sha256']:
        raise ValueError('WT fixture differs from pinned LLM policy contract')
    fixture = json.loads(fixture_bytes)
    source = source_identity(fixture, repo_root=ROOT, sphere_root=args.sphere_root)
    model = spec['model']
    frozen = json.loads(args.deployment_json.read_text())
    token = args.api_key_file.read_text().strip()
    live = probe_vllm(args.model_url, model['model_id'], token=token)
    verify_live_target_deployment(model, frozen, live)
    generate = build_openai_generate(model['model_id'], base_url=args.model_url,
                                     api_key=token, temperature=model['temperature'],
                                     max_tokens=model['max_tokens'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = {'schema_version': 'plcbench.wt-response-llm-preflight/v1',
              'status': 'running', 'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'policy_id': spec['policy_id'],
              'policy_sha256': hashlib.sha256(spec_bytes).hexdigest(),
              'fixture_sha256': spec['fixture_sha256'],
              'adapter_sha256': hashlib.sha256(
                  (ROOT / 'ics_agent/plcbench_wt_response_llm.py').read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_identity': source,
              'deployment_identity_sha256': frozen['models_identity_sha256'],
              'evidence_origin': 'experimental_source_derived_offline_fixture',
              'model_invoked': True, 'plant_interventions': 0,
              'plant_response_observed': False, 'measured_episode': False,
              'safety_score': None, 'service_score': None,
              'cases': []}
    for case_id, scenario in fixture['scenarios'].items():
        observation = ChemistryState.fresh(scenario).observe()
        policy = LLMResponsePolicy(spec, fixture, generate)
        case = {'case_id': case_id, 'observation': observation,
                'authority': list(policy.actions), 'status': 'running'}
        try:
            case['action_id'] = policy(observation)
            case['status'] = 'parsed_authorized_decision'
        except Exception as exc:
            case['status'] = 'unassessable_model_decision'
            case['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        case['decision_evidence'] = policy.decisions
        record['cases'].append(case)
    record['status'] = ('contract_passed' if all(case['status'] == 'parsed_authorized_decision'
                                                 for case in record['cases'])
                        else 'unassessable')
    (args.out_dir / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'],
                      'cases': [{key: case.get(key) for key in ('case_id', 'status', 'action_id')}
                                for case in record['cases']],
                      'evidence': str(args.out_dir / 'preflight.json')}))


if __name__ == '__main__':
    main()
