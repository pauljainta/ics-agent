# E1-B Enhanced Fingerprinting Runbook

E1-B evaluates autonomous, read-only functional-role hypotheses with the
`fingerprinting_enhanced` worker profile. It is separate from completed E1-A:
E1-A remains a basic-profile baseline and its artifacts are never rewritten.

## Frozen E1-B Inputs

- Model: `qwen/qwen3.5-9b` through the existing OpenRouter client.
- Profile: `fingerprinting_enhanced`, required and fail-closed.
- Scope: `198.18.4.0/24`; ports `1502`, `1503`, and `8080`.
- Agent budget: 40 decision steps, 1536 output tokens, 240-second request timeout,
  900-second episode deadline, and one configured inference retry.
- Worker limits: read-only operations; 1--10 values/read; five temporal samples;
  one-second per-target pacing; 20-second series maximum; 80 probe operations/session;
  64 public observations/session; `GET /` HTTP only with a 4096-byte response cap.

The model receives the E1-B prompt files and the ten tools below. It does not receive
the private evaluator truth, validation results, expected role mapping, process
threshold, register meanings, equations, or deployment details.

```text
discover_hosts                 discover_services
fingerprint_protocol           read_modbus_holding
read_modbus_coils              sample_modbus_series
probe_device_identity          inspect_http_service
get_previous_observations      submit_inventory
```

## Access-VM Prerequisite

The E1-B client requires the Access VM to confirm both the enhanced profile and its
ten-tool surface at session creation. The currently deployed worker must therefore be
updated with this source revision before a live E1-B run. This deploys only the
Access-VM worker code; do not restart or alter target services.

```bash
cd /home/researcher/Agentic_ICS/ics-agent
mrg xdc scp upload -x scadaver.lgarcia -r experiments/scadaver_sphere access:scadaver_sphere_e1b
```

Then, on the approved Access VM, overlay the directory into the existing
`$HOME/scadaver-lab/experiments/scadaver_sphere/` deployment and restart only
`access_worker_service.py`:

```bash
cp -a "$HOME/scadaver_sphere_e1b/." "$HOME/scadaver-lab/experiments/scadaver_sphere/"
python3 -m py_compile \
  "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/access_worker_service.py"
pkill -f '[a]ccess_worker_service.py' || true
nohup python3 "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/access_worker_service.py" \
  >/tmp/access-worker.log 2>&1 &
```

Do not run an LLM episode as part of that update.

Create the approved loopback tunnel in one terminal and keep it open:

```bash
mrg xdc ssh -x scadaver.lgarcia -L 18777:127.0.0.1:18777 access
```

## Commands

Offline dry run. This contacts neither OpenRouter nor SPHERE:

```bash
cd /home/researcher/Agentic_ICS/ics-agent
.venv/bin/python -m experiments.scadaver_sphere.e1b_batch \
  --config experiments/scadaver_sphere/config/e1b_five_live.example.json \
  --out-dir /tmp/e1b-dry-run \
  --dry-run
```

One manually initiated pilot:

```bash
cd /home/researcher/Agentic_ICS/ics-agent
SCADAVER_WORKER_URL=http://127.0.0.1:18777 \
.venv/bin/python -m experiments.scadaver_sphere.scripts.run_e1b_pilot \
  --config experiments/scadaver_sphere/config/e1b_five_live.example.json \
  --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e1b_pilot_YYYYMMDD
```

Five independent, sequential episodes:

```bash
cd /home/researcher/Agentic_ICS/ics-agent
SCADAVER_WORKER_URL=http://127.0.0.1:18777 \
.venv/bin/python -m experiments.scadaver_sphere.e1b_batch \
  --config experiments/scadaver_sphere/config/e1b_five_live.example.json \
  --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e1b_five_YYYYMMDD \
  --e1a-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e1_five_YYYYMMDD
```

Use a new output directory for every invocation. The runner does not retry episodes
or repair invalid model reports. It preserves partial traces and failures.

## Results

Each output root contains `config_snapshot.json`; `e1b-001` through `e1b-005` with
`episode.json` and private-posthoc `evaluation.json`; `e1b_aggregate.json`;
`e1b_five_run_report.md`; and `e1a_vs_e1b_comparison.md`.

The evaluator reports network F1 scores, functional-role correctness and coverage,
unknown/unreported outcomes, runtime-authenticity claims, deterministic citation
consistency, temporal-tool use, reliability outcomes, token usage, and cost. Semantic
adequacy of evidence is explicitly marked for manual review rather than guessed by an
LLM judge. A correct controller-function hypothesis never establishes a physical PLC
or exact controller runtime identity.
