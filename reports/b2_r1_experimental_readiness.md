# B2-R1 C0-C4 Experimental Readiness

## Final verdict

`B2_R1_RESEARCH_READY=false`

The code, deployment, testbed, and deterministic production paths pass, but the
review corrections are not yet committed. A live campaign requires an exact
40-character `ICS_AGENT_SOURCE_COMMIT`; supplying the old `0af8608` revision would
misidentify the reviewed source. The sole blocking step is to review and commit
the targeted files listed in the code-review inventory, redeploy that commit, and
repeat parity. This verdict does not authorize or validate E5, simulator writes,
or a full E1-E5 success claim.

## Readiness matrix

| Check | Result |
|---|---|
| Original B2 preserved | PASS: prompt/config/evaluator/broker hashes and historical artifacts unchanged |
| C0 matches B2 behavior | PASS: exact prompt, schema, and model-visible state response tests |
| C1 isolated | PASS: R1-A only |
| C2 isolated | PASS: R1-B only |
| C3 isolated | PASS: R1-C only |
| C4 combined mechanisms correct | PASS |
| Reasoning-state logic valid | PASS with advisory-use limitation |
| Observation references valid | PASS: current-episode and successful-support enforcement |
| Agent autonomy preserved | PASS: no target/tool/answer selection by harness |
| Read-only execution enforced | PASS |
| Tool interfaces healthy | PASS: deployed production acceptance |
| Testbed healthy | PASS: Access-VM `TESTBED_READY=true` |
| Deployment parity | PASS on Access VM with pinned `pymodbus 3.6.9` |
| Full regression suite | PASS; one unrelated Docker integration skipped |
| Production-path mocks | PASS for C0-C4; zero model calls/writes/tokens |
| Evaluation rubric frozen | PASS: `b2-r1-readonly-evidence-v1` |
| Future campaign commands verified | PASS; live commands not executed |
| Reviewed source frozen in Git | BLOCKED: targeted review corrections are uncommitted |

## Deterministic validation

- Full local suite: PASS twice, `292 passed, 1 skipped` in 25.40 and 5.35
  seconds; the skip is the explicitly opt-in Docker container integration test.
- Failure injections cover timeout, protocol incompatibility, repeated failed
  probes, invalid/cross-episode citations, contradiction retention, missing test
  evidence, revision, budget exhaustion, malformed/null/truncated responses,
  missing dependency, deployment hash mismatch, worker transport failure,
  interrupted campaign state, and unexpected action request.
- Production acceptance: PASS with `openrouter_calls=0`, `deployed_writes=0`.
- C0-C4 deterministic production mocks: PASS for all five conditions with
  `openrouter_calls=0`, `deployed_writes=0`, `action_tokens_created=0`.
- Historical B2 checksum verification: 35/35 unchanged.

### Final deployed evidence

The final Access-VM pass used deployment manifest SHA-256
`a525f6a0ea8386d0804e77b1a1698c29233e153337769275521dc6631f32c5c3`.
Downloaded copies were independently parsed from `/tmp/b2-r1-review-evidence/`:

| Artifact | SHA-256 | Result |
|---|---|---|
| `b2-r1-review-final-parity.json` | `f8cdce6f7108ea30a7411da70cff1ea3f1dd798c907feffb23012f717ef4448a` | PASS |
| `production_acceptance.json` | `868090fcefa768df97d1bd666e5683abf72cec5fe49214f5a67c9174a59d0467` | All readiness gates true; zero calls/writes |
| `b2_r1_acceptance.json` | `31b8cea41ea192569a801b168e5b8f1622b1bfef69d04da4f578d09b44b1b6bc` | C0-C4 PASS; zero calls/writes |
| `b2-r1-review-final-health.json` | `33d3dcf459cfd4ee1e08baa0aae48357ab794a1dc46807097af2e619cbb79b15` | `TESTBED_READY=true` |

The final health check found exactly one OpenPLC core, legacy controller inactive,
all seven exposed services healthy, no action environment/token/latch, and 11
confirmed closed-loop intervals with zero inconsistent or indeterminate intervals.
These checks were read-only and made no OpenRouter request.

The workstation virtualenv has a different `pymodbus` version and is not the
scientific runtime. Access-VM parity is authoritative.

## Operator runbook

Run on the Access VM from `/home/researcher/scadaver-lab` unless noted.

### 1. Inspect frozen profiles

```bash
for C in C0 C1 C2 C3 C4; do
  python3 -m json.tool "experiments/scadaver_sphere/config/b2_r1_v1/$C.json"
done
```

### 2. Verify deployment parity

```bash
cd /home/researcher/scadaver-lab
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.deployment_parity
```

### 3. Verify read-only testbed health

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify
```

Expected: `TESTBED_READY=true`, OpenPLC core healthy, legacy controller inactive,
seven services healthy, no active action state.

### 4. Run no-model production validation and C0-C4 mocks

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.production_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-dir "/tmp/b2-r1-production-$STAMP"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.b2_r1_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-root "/tmp/b2-r1-c0-c4-$STAMP"
```

### 5. Prepare one balanced pilot block

```bash
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export ICS_AGENT_SOURCE_COMMIT='<40-character reviewed commit hash>'
ROOT="$HOME/Artifact/Results/scadaver_sphere/b2_r1_pilot/block_1"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.b2_r1_campaign prepare \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT"
```

The OpenRouter key must already be provided through the configured private
environment. Never print it or place it in a result directory.

### 6. Launch one future read-only condition

Obtain the matched raw-level band from the accepted initial-condition record and
use the same preregistered band for all episodes.

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.b2_r1_campaign run \
  --execute-live \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT" \
  --condition C0 \
  --seed 101 \
  --matched-level-min-raw '<predeclared-minimum>' \
  --matched-level-max-raw '<predeclared-maximum>'
```

Repeat with the block order and seeds in `reports/b2_r1_pilot_plan.json`, using a
new results root per block. Do not pass `--enable-actions`; the campaign CLI does
not expose it.

### 7. Monitor safely

```bash
cat "$ROOT/campaign_state.json"
pgrep -af 'b2_r1_campaign|integrated_runner'
```

Inspect per-episode `runner_stdout.log`, `runner_stderr.log`, and `episode.json`.
Usage, costs, termination, tool errors, compaction, and reproducibility metadata
are incremental or preserved in the final episode artifact.

### 8. Stop a read-only episode

Send `SIGINT` to the exact campaign PID, then wait for cleanup and run operator
verification. Do not start a replacement in the same results root.

```bash
kill -INT '<exact-campaign-pid>'
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify
```

The campaign records `INTERRUPTED`; automatic replacement is prohibited.

### 9. Collect and compare

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.b2_r1_campaign report \
  --results-root "$ROOT" \
  --report-dir "$ROOT/comparison"
```

Preserve `episode.json`, deployment/preflight/postflight artifacts, raw runner
logs, campaign state, and generated JSON/CSV. Analyze primary and diagnostic
metrics according to the frozen rubric; never infer E5 success.

## Stop conditions

Do not launch if parity, production acceptance, matched initial state, worker
health, or source-commit validation fails. Do not repair or rerun a scientifically
poor episode. Technical aborts require a separately documented operator decision
and a new results root.

After a narrow reviewed commit is created, replace the placeholder source commit
with that commit's full 40-character hash, deploy the exact committed files, and
rerun deployment parity. If those checks pass with no new Critical/High finding,
the verdict may be changed to `B2_R1_RESEARCH_READY=true` without altering the
scientific conditions.
