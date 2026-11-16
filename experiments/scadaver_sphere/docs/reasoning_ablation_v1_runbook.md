# Qwen reasoning ablation v1 runbook

This study compares one continuous Qwen E1-E5 episode under cumulative reasoning
conditions B0-B4. B0 uses the frozen integrated prompts and tools without additions.
B1-B4 add only model-controlled reasoning-state recording and generic guidance.
The evaluator, investigation tools, action broker, one-request policy, testbed, model,
budgets, and finalization behavior remain unchanged.

## Before every condition

Run the read-only readiness check from the Access VM deployment. Do not run
`operator_workflow` directly on the external experiment host: the industrial
subnet and node-management SSH names are intentionally reachable through the
attached XDC and Access VM only.

```bash
mrg xdc ssh -x scadaver.lgarcia access
cd /home/researcher/scadaver-lab
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.deployment_parity
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify
SMOKE="$HOME/scadaver-results/prelaunch_validation/prelaunch-$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.production_path_smoke \
  --worker-url http://127.0.0.1:18777 --out-root "$SMOKE"
```

All three commands must exit successfully before creating an authorization packet
or action token. The integrated runner repeats the source/config parity gate before
loading authorization state or initializing the model client.

Obtain a separate authorization for exactly one condition. Copy that condition's
template outside Git, set `authorization_status` to `AUTHORIZED`, and assign a
unique authorization ID. Create a fresh action token using the established private
operator procedure. Never reuse an authorization or token across conditions.

## Execute one condition

Replace `B0` consistently with the separately authorized condition. Run the
episode on the Access VM, where the frozen harness and industrial network are
available. The output directory must be new and empty.

```bash
cd /home/researcher/scadaver-lab
set -a
source "$HOME/.config/ics-agent/openrouter.env"
set +a
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export SCADAVER_ENABLE_SITL_ACTIONS=1
export SCADAVER_SITL_ACTION_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
OBJECTIVE="$(python3 -c 'import json; print(json.load(open("experiments/scadaver_sphere/config/first_qwen_objective.json"))["agent_facing_objective"])')"
RUN_DIR="$HOME/scadaver-results/reasoning_ablation_v1/B0/episode-$(date +%Y%m%dT%H%M%S)"
trap 'unset SCADAVER_SITL_ACTION_TOKEN SCADAVER_ENABLE_SITL_ACTIONS' EXIT INT TERM
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.integrated_runner \
  --execute-live-model \
  --enable-actions \
  --worker-url "$SCADAVER_WORKER_URL" \
  --config experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json \
  --reasoning-config experiments/scadaver_sphere/config/reasoning_ablation_v1/B0.json \
  --authorization-file "$HOME/.config/ics-agent/B0.authorization.json" \
  --objective "$OBJECTIVE" \
  --out-dir "$RUN_DIR"
```

In a separate Access-VM terminal, monitor without changing state:

```bash
cd /home/researcher/scadaver-lab
python3 -m experiments.scadaver_sphere.operator_workflow monitor \
  --out /tmp/B0-readiness-monitor.jsonl --interval-seconds 10 --iterations 720
```

The emergency-stop path is `/tmp/scadaver-action-emergency-stop`. Follow the
established operator procedure to activate it if health or restoration fails; do
not clear it automatically. After the run, unset `SCADAVER_ENABLE_SITL_ACTIONS`
and `SCADAVER_SITL_ACTION_TOKEN`, preserve the complete result directory, verify
that no transaction is incomplete, and perform `operator_workflow verify` again.
Execute conditions only in B0, B1, B2, B3, B4 order. A poor scientific result is
not grounds for retrying.

## Offline fixture validation

Fixture validation does not authorize or perform deployed writes and is not evidence
of model autonomy:

```bash
pytest -q tests/test_reasoning_ablation.py
```
