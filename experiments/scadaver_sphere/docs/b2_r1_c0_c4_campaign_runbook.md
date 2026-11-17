# B2-R1 C0-C4 Campaign Runbook

Run all industrial checks and experiments on the existing Access VM. The external
repository host is not expected to route to `198.18.4.0/24`.

## Connect

From the repository host, confirm the control plane and open the Access VM:

```bash
mrg show realization scadaverfingerprint.openplcwater.lgarcia --json
mrg show xdc scadaver.lgarcia --json
mrg xdc ssh -x scadaver.lgarcia access
```

The XDC must report materialization
`scadaverfingerprint.openplcwater.lgarcia`. Continue on `access` only.

## Common environment

```bash
cd /home/researcher/scadaver-lab
export PYTHONPATH=$PWD
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export ICS_AGENT_SOURCE_COMMIT='<full reviewed runtime commit from the handoff>'
ROOT="$HOME/Artifact/Results/scadaver_sphere/b2_r1_c0_c4_$(date -u +%Y%m%dT%H%M%SZ)"
```

The ignored OpenRouter environment file is needed only immediately before a live
`run` command. Never print or copy its contents into the results directory.

## Prepare and validate

The predeclared matched level band below covers the validated normal closed-loop
operating region. Do not change it after observing research outcomes.

```bash
python3 scripts/run_b2_r1_campaign.py prepare \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT" \
  --matched-level-min-raw 400 \
  --matched-level-max-raw 500

python3 scripts/run_b2_r1_campaign.py validate \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT"
```

Both commands are deterministic, read-only, and model-free. Preparation starts
only missing supported services, then reruns complete verification and C0-C4
production mocks.

## Status

```bash
python3 scripts/run_b2_r1_campaign.py status \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT"
```

## Run one condition

```bash
set -a
source "$HOME/.config/ics-agent/openrouter.env"
set +a
python3 scripts/run_b2_r1_campaign.py run --execute-live \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT" \
  --condition C2 --seed 101 --trial-id exploratory-C2-seed-101 \
  --matched-level-min-raw 400 --matched-level-max-raw 500
```

## Run the frozen 15-episode pilot

Use a new, empty result root. The runner applies the reviewed block order and
seeds from `config/b2_r1_v1/pilot_plan.json`.

```bash
set -a
source "$HOME/.config/ics-agent/openrouter.env"
set +a
python3 scripts/run_b2_r1_campaign.py run --execute-live --all \
  --worker-url "$SCADAVER_WORKER_URL" \
  --results-root "$ROOT" \
  --matched-level-min-raw 400 --matched-level-max-raw 500
```

`--max-episodes N` may be used only for a prospectively bounded subset of the
frozen schedule. Never use it to select runs after observing outcomes.

## Monitor and stop

```bash
cat "$ROOT/campaign_state.json"
pgrep -af 'b2_r1_campaign|integrated_runner'
```

To stop, send `SIGINT` to the exact campaign PID, wait for cleanup, and verify the
testbed. The interrupted identity must not be reused.

```bash
kill -INT '<exact-campaign-pid>'
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.operator_workflow verify
```

## Report

```bash
python3 scripts/run_b2_r1_campaign.py report \
  --results-root "$ROOT" \
  --report-dir "$ROOT/comparisons"
```

The comparison explicitly scopes conclusions to read-only E1-E3 and descriptive
planning. E5 and complete E1-E5 success are not outcomes of this campaign.
