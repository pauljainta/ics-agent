# Reasoning-ablation campaign operator runbook

Run these commands on the Access VM from `/home/researcher/scadaver-lab`.
`validate` and `prepare` reject action environment variables and never initialize
the live model client.

```bash
export ROOT="$HOME/scadaver-results/reasoning_ablation_v1"
export SCADAVER_WORKER_URL=http://127.0.0.1:18777

# Inspect existing B0-B4 episodes.
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  status --results-root "$ROOT"

# Verify source parity and deployed read-only health.
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.deployment_parity
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify

# Run deterministic B1-B4 production-path mocks with no provider or deployed action.
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  validate --results-root "$ROOT" --production-smoke

# Build the idempotent future execution plan.
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  prepare --results-root "$ROOT"

# Generate comparison outputs from available authentic episodes.
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  report --results-root "$ROOT" --report-dir "$ROOT/comparisons"
```

The explicit `run --execute-live` invocation authorizes one episode under the frozen
simulator-only policy. The runner creates and destroys its private per-episode packet
and token automatically. It refuses any condition that already has an episode. For
the currently eligible B4 condition:

```bash
set -a; source "$HOME/.config/ics-agent/openrouter.env"; set +a
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  run --execute-live --condition B4 --results-root "$ROOT"
```

Monitor without changing state:

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow monitor \
  --out /tmp/B4-monitor.jsonl --interval-seconds 10 --iterations 720
```

To stop an active runner, send `SIGINT` to its exact PID, do not start a replacement,
and preserve the partial directory. Then unset action variables and verify recovery:

```bash
kill -INT "$(pgrep -f 'experiments.scadaver_sphere.integrated_runner.*B4.json')"
unset SCADAVER_SITL_ACTION_TOKEN SCADAVER_ENABLE_SITL_ACTIONS
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify
```
# Hardened production acceptance

Run these commands on the Access VM from `/home/researcher/scadaver-lab`. They do not call a model, create an action token, or write to the simulator.

```bash
export PYTHONPATH=$PWD

# Source/import/config/dependency parity
python3 -m experiments.scadaver_sphere.deployment_parity

# Read-only service, OpenPLC core, legacy-controller, worker, and process checks
python3 -m experiments.scadaver_sphere.operator_workflow verify
python3 -m experiments.scadaver_sphere.operator_workflow baseline

# Complete fail-closed gate, including B0-B4 production mocks
ACCEPTANCE=/tmp/scadaver-production-acceptance-$(date -u +%Y%m%dT%H%M%SZ)
python3 -m experiments.scadaver_sphere.production_acceptance \
  --worker-url http://127.0.0.1:18777 \
  --out-dir "$ACCEPTANCE"
test "$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["EXPERIMENT_READY"]).lower())' "$ACCEPTANCE/production_acceptance.json")" = true
```

Repository regressions from the repository host:

```bash
cd /home/researcher/Agentic_ICS/ics-agent
PYTHONPATH=$PWD .venv/bin/pytest -q
```

Inspect budget and failed-tool diagnostics without editing an episode:

```bash
python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); print(json.dumps({"statistics":v.get("statistics"),"failed_tools":[x for x in v.get("diagnostics",[]) if x.get("kind") in {"unsuccessful_tool_call","model_request_not_admitted"}]},indent=2))' EPISODE/episode.json
```

Check the broker remains disabled outside a live episode:

```bash
python3 -m experiments.scadaver_sphere.operator_workflow status
# Require enabled=false, token_present=false, emergency_latch_present=false.
```

Monitor and safely stop a future campaign:

```bash
python3 -m experiments.scadaver_sphere.operator_workflow monitor \
  --out /tmp/scadaver-monitor.jsonl --interval-seconds 10 --iterations 60
pgrep -af 'reasoning_campaign|integrated_runner'
# Stop only the identified runner PID with SIGTERM; never kill PLC/simulator processes globally.
kill -TERM RUNNER_PID
python3 -m experiments.scadaver_sphere.operator_workflow verify
```

Future B2 launch uses the campaign runner, which now performs the complete gate automatically and passes its immutable acceptance record to the live runner before inference or action authorization:

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.reasoning_campaign \
  run --execute-live --condition B2 --seed SEED \
  --worker-url http://127.0.0.1:18777 \
  --results-root /home/researcher/scadaver-results/reasoning_ablation_b2_future
```

Do not use the direct integrated runner for a live model without a fresh successful `--acceptance-record`. After any run, verify testbed recovery with `operator_workflow verify` and inspect transaction evidence before considering the episode complete.
