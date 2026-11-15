# First continuous Qwen E1-E5 operator runbook

This procedure prepares and launches exactly one continuous agent episode. The
preparation commands are read-only. Do not execute Section C without separate
written authorization. Run repository commands from
`/home/researcher/Agentic_ICS/ics-agent` and Access-VM commands from
`$HOME/scadaver-lab`.

## A. Start and verify the existing testbed

Confirm the existing realization and open the worker tunnel. Keep the tunnel
terminal open:

```bash
mrg list xdcs lgarcia -j
mrg xdc ssh -x scadaver.lgarcia -L 18777:127.0.0.1:18777 access
```

On the Access VM, start only missing managed services and verify the complete
deployment. Healthy services are not restarted. A missing bootstrap-managed
independent endpoint is a hard failure rather than being silently recreated.

```bash
cd "$HOME/scadaver-lab"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow start \
  | tee /tmp/scadaver-start.json
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify \
  | tee /tmp/scadaver-verify.json
test "$(python3 -c 'import json; print(str(json.load(open("/tmp/scadaver-verify.json"))["TESTBED_READY"]).lower())')" = true
```

`TESTBED_READY=true` requires one compiled OpenPLC core, the registered IEC
program, HTTP/RPC/Modbus health, legacy controller inactivity, all seven
industrial ports, a `heterogeneous_e4` worker session, disabled action mode,
no token, and a clear emergency latch. Systemd state alone is insufficient.

Capture the continuously evolving initial state:

```bash
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow baseline \
  | tee /tmp/scadaver-first-qwen-baseline.json
```

The simulator has no deterministic reset claim. The recorded level, flow,
inlet, scan counter, timestamps, controller state, and service health are the
episode's actual initial conditions.

## B. Prepare the frozen experiment

On the repository host:

```bash
git switch scadaver
git describe --tags --exact-match HEAD
git status --short
python3 -m json.tool experiments/scadaver_sphere/config/qwen_openrouter_model_v1.json
python3 -m json.tool experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json
python3 -m json.tool experiments/scadaver_sphere/config/first_qwen_objective.json
sha256sum \
  experiments/scadaver_sphere/config/qwen_openrouter_model_v1.json \
  experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json \
  experiments/scadaver_sphere/config/first_qwen_objective.json \
  experiments/scadaver_sphere/private/integrated_e1_e5_first_qwen_spec.json
```

The profile is one 7,200-second episode with 250 model calls, 500 tool calls,
2,000,000 cumulative input tokens, 150,000 cumulative output tokens, a 4,096
normal response ceiling, an 8,192 final-submission ceiling, five reserved
completion turns, and a USD 25 operator-selected ceiling. The per-request
context remains 262,144 documented model tokens; cumulative budget is not a
context-window claim.

Run the no-model dry-run against deployed read-only services and the local E5
fixture in a new directory:

```bash
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
OBJECTIVE="$(python3 -c 'import json; print(json.load(open("experiments/scadaver_sphere/config/first_qwen_objective.json"))["agent_facing_objective"])')"
DRY_RUN="/home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/prelaunch_dry_run_$(date +%Y%m%dT%H%M%S)"
.venv/bin/python -m experiments.scadaver_sphere.integrated_runner \
  --config experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json \
  --evaluation-spec experiments/scadaver_sphere/private/integrated_e1_e5_first_qwen_spec.json \
  --worker-url "$SCADAVER_WORKER_URL" --objective "$OBJECTIVE" \
  --out-dir "$DRY_RUN" --mock --mock-action
```

## C. Separately authorize one autonomous action

Do not edit the tracked template. Copy it to a private, mode-`0600` path,
review every field, set a unique authorization ID, operator, issue/expiry
times, and change `authorization_status` to `AUTHORIZED` only after written
approval:

```bash
install -m 0600 experiments/scadaver_sphere/config/first_qwen_e5_authorization.template.json \
  "$HOME/.config/ics-agent/first-qwen-authorization.json"
${EDITOR:?set EDITOR} "$HOME/.config/ics-agent/first-qwen-authorization.json"
```

Re-run `operator_workflow verify` immediately before authorization. Then, in
the launch shell only, create a fresh ephemeral token and enable the one fixed
broker operation. Never write the token to disk:

```bash
export SCADAVER_ENABLE_SITL_ACTIONS=1
export SCADAVER_SITL_ACTION_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

The model-facing tool requires Qwen to select an endpoint, address, value, and
duration from its evidence. The broker, schema, configuration, and authorization
packet independently enforce one `bounded_binary_output` request, a fixed
private simulator/coil allowlist, maximum 1.25 seconds, restoration, controller
recovery, safe envelope, and fail-closed emergency behavior. No automatic
action retry exists.

## D. Run exactly one continuous Qwen episode

On the Access VM, after loading the ignored OpenRouter key file and completing
Sections A-C:

```bash
cd "$HOME/scadaver-lab"
set -a
source "$HOME/.config/ics-agent/openrouter.env"
set +a
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export SCADAVER_ENABLE_SITL_ACTIONS=1
export SCADAVER_SITL_ACTION_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
OBJECTIVE="$(python3 -c 'import json; print(json.load(open("experiments/scadaver_sphere/config/first_qwen_objective.json"))["agent_facing_objective"])')"
RUN_DIR="$HOME/scadaver-results/first_qwen_e1_e5_$(date +%Y%m%dT%H%M%S)"
trap 'unset SCADAVER_SITL_ACTION_TOKEN SCADAVER_ENABLE_SITL_ACTIONS' EXIT INT TERM
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.integrated_runner \
  --config experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json \
  --evaluation-spec experiments/scadaver_sphere/private/integrated_e1_e5_first_qwen_spec.json \
  --worker-url "$SCADAVER_WORKER_URL" --objective "$OBJECTIVE" \
  --authorization-file "$HOME/.config/ics-agent/first-qwen-authorization.json" \
  --out-dir "$RUN_DIR" --execute-live-model --enable-actions
```

This is one `run_agent` invocation, one initial objective, one message context,
one tool catalog, one global budget, one raw trace, and one retrospective
evaluation. E1-E5 are evaluator dimensions, not forced stages.

## E. Monitor without influencing the agent

In a separate Access-VM terminal:

```bash
cd "$HOME/scadaver-lab"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow monitor \
  --out "$HOME/scadaver-results/operator-health-$(date +%Y%m%dT%H%M%S).jsonl" \
  --interval-seconds 10 --iterations 720
```

On the run host, inspect only operator artifacts:

```bash
watch -n 5 'test -f "$RUN_DIR/episode.json" && jq ".status,.statistics,.diagnostics[-1]" "$RUN_DIR/episode.json" || true'
```

The health monitor records services, controller core, simulator reachability,
worker state, broker environment, and latch state. It never sends results to
the model.

## F. Emergency interruption

Interrupt the single runner with `Ctrl-C` or `TERM`; do not relaunch it. On the
Access VM, fail closed before recovery:

```bash
printf '%s\n' 'operator emergency stop; review required' > /tmp/scadaver-action-emergency-stop
pkill -TERM -f 'experiments.scadaver_sphere.integrated_runner' || true
unset SCADAVER_SITL_ACTION_TOKEN SCADAVER_ENABLE_SITL_ACTIONS
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow status \
  | tee /tmp/scadaver-emergency-status.json
```

The latch is intentionally not auto-cleared. Preserve the partial run
directory and obtain fresh authorization before any later run.

## G. Post-run restoration and final health

After normal completion, immediately invalidate permissions and remove the
private authorization file only after archiving its non-secret hash:

```bash
unset SCADAVER_SITL_ACTION_TOKEN SCADAVER_ENABLE_SITL_ACTIONS
cd "$HOME/scadaver-lab"
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow recover \
  | tee /tmp/scadaver-post-run-recover.json
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow baseline \
  | tee /tmp/scadaver-post-run-baseline.json
```

`recover` never clears a latch. If the latch is present, inspect transaction
evidence and follow the separately authorized recovery process. Do not start
the legacy controller as an accepted substitute for OpenPLC.

## H. Evaluate and archive

The runner stores the independent scorecard in `episode.json`. Validate and
archive without changing it:

```bash
python3 -m json.tool "$RUN_DIR/episode.json" >/dev/null
jq '{status,evaluation,statistics,diagnostics,inference_requests,authorization,config}' \
  "$RUN_DIR/episode.json" > "$RUN_DIR/scorecard.json"
find "$RUN_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/SHA256SUMS"
```

Do not delete the raw messages, observations, action journal, request usage,
or failure diagnostics. Compare against the frozen deterministic dry-run only
after the live episode has ended.

## Intentional stop procedure

The testbed normally remains available. Intentional shutdown is an operator
maintenance action, not an experiment cleanup step. Stop the agent and worker
first, ensure no transaction is active, then stop OpenPLC before simulator and
observation services. Never enable the legacy controller during shutdown.
