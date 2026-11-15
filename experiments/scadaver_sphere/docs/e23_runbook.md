# E2/E3 operator commands

Run from `ics-agent`. Set `SCADAVER_WORKER_URL` to the existing loopback
tunnel and set `OPENROUTER_API_KEY` only for live runs. Never place the key in
arguments or result directories.

```bash
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export OPENROUTER_API_KEY='...'
```

Read-only readiness check:

```bash
.venv/bin/python -m experiments.scadaver_sphere.e23_readiness --experiment e2
```

One E2 episode, or five independent E2 episodes:

```bash
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e2 --config experiments/scadaver_sphere/config/e2_live.example.json --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e2_one_YYYYMMDD
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e2 --config experiments/scadaver_sphere/config/e2_live.example.json --episodes 5 --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e2_five_YYYYMMDD
```

One E3 telemetry-only, one E3 operational-metadata, or five telemetry E3 episodes:

```bash
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e3 --condition telemetry --config experiments/scadaver_sphere/config/e3_telemetry_live.example.json --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e3_telemetry_one_YYYYMMDD
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e3 --condition metadata --config experiments/scadaver_sphere/config/e3_metadata_live.example.json --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e3_metadata_one_YYYYMMDD
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e3 --condition telemetry --config experiments/scadaver_sphere/config/e3_telemetry_live.example.json --episodes 5 --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e3_telemetry_five_YYYYMMDD
```

Re-evaluate saved artifacts and regenerate their summary without API calls:

```bash
.venv/bin/python -m experiments.scadaver_sphere.e23_results --experiment e2 --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e2_five_YYYYMMDD
.venv/bin/python -m experiments.scadaver_sphere.e23_results --experiment e3 --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/e3_telemetry_five_YYYYMMDD
```

Mock validation never calls OpenRouter or requires an API key:

```bash
.venv/bin/python -m experiments.scadaver_sphere.e23_runner --experiment e2 --config experiments/scadaver_sphere/config/e2_live.example.json --out-dir /tmp/scadaver_e2_mock --mock
```

The simulator is continuous. No automatic reset is performed between runs;
each artifact records a read-only preflight and the agent must interpret the
state it observes. Restarting the simulator is an operator-only deployment
operation and is intentionally outside the agent runner.
