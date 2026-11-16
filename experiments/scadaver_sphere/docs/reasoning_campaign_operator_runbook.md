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
