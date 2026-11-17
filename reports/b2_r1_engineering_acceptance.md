# B2-R1 Engineering Acceptance

## Verdict

**READY for future, separately initiated read-only C0-C4 research evaluation.**

This acceptance does not claim improved Qwen reasoning. No Qwen/OpenRouter request,
simulator write, action credential, or action token was created.

## Frozen Harness Baseline

- Commit: `65e96bf` (`fix(scadaver): harden and freeze production ICS agent harness`)
- Pre-commit suite: `251 passed, 1 skipped`
- Baseline production manifest: `76b61043...4067`
- Original B2 config: `9ebf5087...4b5b`
- Original B2 prompt: `57ac35a4...a100`
- Integrated evaluator: `6f137b04...4050`
- E5 evaluator: `22a6c8e7...b553`
- Action broker: `06cc238f...0cf`

Immutable historical B2 files under
`Artifact/Results/scadaver_sphere/reasoning_ablation_b2_multiseed_20261116`
were read for analysis and were not modified.

## B2-R1 Implementation

New modules:

- `b2_r1.py`: profiles, prompt rendering, schemas, current-episode provenance, and
  component-isolated validation.
- `b2_r1_acceptance.py`: no-model/no-write C0-C4 production-path acceptance.
- `b2_r1_campaign.py`: durable, fail-closed, read-only campaign orchestration.

Integrated changes:

- `integrated_runner.py` loads B2-R1 profiles without changing B0-B4 behavior,
  registers worker observations privately, and persists initial-condition metadata.
- `reasoning_metrics.py` computes error-recovery, evidence, hypothesis, contradiction,
  uncertainty, temporal, and causal-support diagnostics from actual trace records.
- `production_runtime_manifest_v1.json` is versioned as
  `production-runtime-parity-v2` and pins the new runtime/config/prompt files.

The original B2 files, tool schemas, evaluator, and broker were unchanged.

## Test Results

Local results:

- B2-R1 focused suite: `32 passed`.
- Relevant agent/inference/evaluator/security slice: `213 passed`.
- Complete repository suite: `283 passed, 1 skipped` in 5.95 seconds.
- The skip is an existing optional environment-dependent test, not a B2-R1 failure.
- Local `.venv` has `pymodbus 3.15.0`; local manifest tests substitute the installed
  dependency version while preserving every source/config hash. Production remains
  pinned to and validated with `pymodbus 3.6.9`.

Deployed Access-VM results:

| Gate | Result |
|---|---|
| Runtime file/import parity | PASS |
| Production dependency `pymodbus==3.6.9` | PASS |
| 19-tool catalog | PASS |
| Frozen B0 scientific configuration | PASS |
| Frozen production acceptance | PASS (`EXPERIMENT_READY=true`) |
| C0 deterministic production mock | PASS |
| C1 deterministic production mock | PASS |
| C2 deterministic production mock | PASS |
| C3 deterministic production mock | PASS |
| C4 deterministic production mock | PASS |
| Post-run operator verification | PASS (`TESTBED_READY=true`) |

Deployed counters: OpenRouter calls `0`, deployed writes `0`, action tokens `0`.
The final deployed manifest SHA-256 is
`00cbda710c9186531d1c9b20d3fcef728e0a6dc885cc8f537b1a7f2e6dd14f8c`.

## Preservation and Safety Checks

- C0 prompt and schema are exactly equal to original B2.
- C0 model-visible state-tool output is regression-tested against the original B2
  manager, including no R1 validation payload.
- C1-C4 expose only their selected components.
- All original 19 investigative tools remain available.
- Future campaign execution explicitly removes `SCADAVER_ENABLE_SITL_ACTIONS` and
  `SCADAVER_SITL_ACTION_TOKEN`, omits `--enable-actions`, and creates no authorization.
- The broker and action authorization boundaries are unchanged.
- The causal evaluator and all scoring criteria are unchanged.
- No PLC, simulator, network, or topology file changed.
- No historical result was overwritten.

## Evidence Artifacts

Machine-readable deployed evidence is preserved under `reports/evidence/b2_r1/`.
The final exact-candidate records are:

- `b2_r1_final_parity.json`
- `b2_r1_final_20261117T002029Z_pre_verify.json`
- `b2_r1_final_20261117T002029Z_production/production_acceptance.json`
- `b2_r1_final_20261117T002029Z_profiles/b2_r1_acceptance.json`
- `b2_r1_final_20261117T002029Z_post_verify.json`

The final transport archive SHA-256 was
`4e95b9dae46dddc45e955605a3bab57d410c198e2872d8323f94f114ffc2676e`.
An earlier interrupted production acceptance directory contains only a partial
parity artifact and is not treated as evidence. An earlier behavior-equivalent
acceptance bundle was superseded; only the exact-candidate records above are
included as authoritative evidence.

## Exact Validation Commands

```bash
cd /home/researcher/scadaver-lab
python3 -m experiments.scadaver_sphere.deployment_parity
python3 -m experiments.scadaver_sphere.production_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-dir /tmp/new-production-acceptance
python3 -m experiments.scadaver_sphere.b2_r1_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-root /tmp/new-b2-r1-acceptance
python3 -m experiments.scadaver_sphere.operator_workflow verify
```

## Remaining Limitations

- Deterministic mock acceptance proves plumbing, schema, provenance, isolation, and
  safety boundaries, not Qwen effectiveness.
- No live C0-C4 result exists yet, by design.
- A future study must preregister seeds, condition order, and any matched-state band.
- The current testbed is continuously evolving; the campaign records rather than
  resets initial state.
- Correctness and causal sufficiency remain independent evaluator judgments.
