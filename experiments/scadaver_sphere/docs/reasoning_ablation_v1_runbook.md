# Qwen reasoning ablation v1 runbook

This study compares one continuous Qwen E1-E5 episode under cumulative reasoning
conditions B0-B4. B0 uses the frozen integrated prompts and tools without additions.
B1-B4 add only model-controlled reasoning-state recording and generic guidance.
The evaluator, investigation tools, action broker, one-request policy, testbed, model,
budgets, and finalization behavior remain unchanged.

## Before every condition

Run the read-only readiness check from the Access VM deployment:

```bash
cd /home/researcher/scadaver-lab
python3 -m experiments.scadaver_sphere.operator_workflow verify
```

Obtain a separate authorization for exactly one condition. Copy that condition's
template outside Git, set `authorization_status` to `AUTHORIZED`, and assign a
unique authorization ID. Create a fresh action token using the established private
operator procedure. Never reuse an authorization or token across conditions.

## Execute one condition

Replace `B0` consistently with the separately authorized condition. The output
directory must be new and empty.

```bash
cd /home/researcher/Agentic_ICS/ics-agent
export SCADAVER_WORKER_URL=http://127.0.0.1:18777
export SCADAVER_ENABLE_SITL_ACTIONS=1
export SCADAVER_SITL_ACTION_TOKEN='EPHEMERAL_PRIVATE_TOKEN'
.venv/bin/python -m experiments.scadaver_sphere.integrated_runner \
  --execute-live-model \
  --enable-actions \
  --worker-url "$SCADAVER_WORKER_URL" \
  --config experiments/scadaver_sphere/config/integrated_e1_e5_first_qwen.json \
  --reasoning-config experiments/scadaver_sphere/config/reasoning_ablation_v1/B0.json \
  --authorization-file /private/operator/B0.authorization.json \
  --objective "$(python3 -c 'import json; print(json.load(open("experiments/scadaver_sphere/config/first_qwen_objective.json"))["objective"])')" \
  --out-dir /home/researcher/Agentic_ICS/Artifact/Results/scadaver_sphere/reasoning_ablation_v1/B0/episode-001
```

Unset action variables immediately afterward, preserve the complete result directory,
and perform the read-only readiness check again. Execute conditions only in B0, B1,
B2, B3, B4 order. A poor scientific result is not grounds for retrying.

## Offline fixture validation

Fixture validation does not authorize or perform deployed writes and is not evidence
of model autonomy:

```bash
pytest -q tests/test_reasoning_ablation.py
```
