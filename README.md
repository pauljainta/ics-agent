# ICS Agent

`ics-agent` is the agent-side repository in the shared `Agentic_ICS` workspace.
Today it is primarily the entry point for scripted cyber-physical wargaming of
the water-treatment OpenPLC setup, with the local Qwen tooling kept as optional
infrastructure for future LLM-based agents.

## What Is Implemented

The current implemented functionality is:

- a scripted multi-scenario wargame runner in `agents/scripted_wargame.py`
- support for four suite types:
  - `readiness`
  - `normal`
  - `attack`
  - `recheck`
- integration with the water-treatment testbed in the sibling
  `sphere-usecases` repository
- suite and per-scenario artifact generation under the workspace `artifacts/`
  directory
- optional local Qwen runner scripts for future autonomous-agent work

This repo does not yet contain a full adaptive red-team or blue-team agent with
planning, memory, tool selection, or autonomous strategy generation. The
current wargame layer is scripted and scenario-driven.

## Operational Status

The workspace has already been made operational on this machine without
changing the system Python version.

Important context:

- the scripted runner in this repo drives the host-side testbed controller
- the host-side testbed now supports a native Modbus-backed fallback runtime in
  the sibling `sphere-usecases` repo
- that fallback is used when Docker is unavailable, while keeping the same
  action and observation surface for scenarios and agents

Validated suites on this machine so far:

- `attack`: 12 scenarios passed
- `normal`: 10 scenarios passed
- `recheck`: 2 scenarios passed

## Repository Layout

### Scripted wargame entry point

- `agents/scripted_wargame.py`

Responsibilities:

- discover scenario groups
- call the water-treatment testbed controller
- run suites end to end
- write suite summaries and scenario outputs

### Optional local model utilities

- `models/qwen_test.py`
- `scripts/setup_env.sh`
- `scripts/download_qwen_model.sh`
- `scripts/run_local_qwen.sh`

These utilities are still useful for future experiments, but they are not
required for the current scripted water-treatment runs.

## Workspace Assumptions

This repository is intended to live beside these sibling repositories:

- `ics-agent/`
- `sphere-usecases/`
- `cps-enclave-model/`

The scripted runner derives the workspace root dynamically from its own file
location. Tracked source code should not contain machine-specific absolute
paths.

## How The Scripted Wargame Works

`agents/scripted_wargame.py` imports the testbed controller from the
water-treatment OpenPLC implementation and runs curated scenario groups.

Scenario sources:

- `readiness`
  - canonical scenarios from the OpenPLC implementation
- `normal`
  - nominal scenarios under `artifacts/agent_smoke/normal`
- `attack`
  - adversarial scenarios under `artifacts/agent_smoke/attack`
- `recheck`
  - follow-up regression scenarios under `artifacts/agent_smoke/recheck`

Typical outputs:

- `artifacts/scripted_agents/`
- `artifacts/scripted_agents_attack/`
- `artifacts/scripted_agents_normal/`
- `artifacts/scripted_agents_recheck/`

Generated summaries now prefer workspace-relative paths instead of
machine-specific absolute paths.

## Running It

From the workspace root:

```bash
python3 ics-agent/agents/scripted_wargame.py --suite readiness
python3 ics-agent/agents/scripted_wargame.py --suite attack
python3 ics-agent/agents/scripted_wargame.py --suite normal
python3 ics-agent/agents/scripted_wargame.py --suite recheck
python3 ics-agent/agents/scripted_wargame.py --suite all
```

Custom output root:

```bash
python3 ics-agent/agents/scripted_wargame.py \
  --suite attack \
  --output-root artifacts/my_scripted_runs
```

## Local Qwen Runner

The local Qwen flow remains available for future autonomous-agent experiments.

Setup:

```bash
bash scripts/setup_env.sh
cp -f .env.example .env
bash scripts/download_qwen_model.sh
```

Run:

```bash
bash scripts/run_local_qwen.sh
```

CPU-only run:

```bash
QWEN_DEVICE=cpu bash scripts/run_local_qwen.sh
```

Prompt test:

```bash
source .venv/bin/activate
set -a
source .env
set +a
python models/qwen_test.py --offline --prompt "List three PLC tags for tank level control."
```

Precision values:

- `auto`
- `bf16`
- `fp16`
- `8bit`
- `4bit`

## Git Hygiene

Before committing this repository:

- keep tracked code free of hardcoded user-specific paths
- do not commit `.env` files
- do not commit model weights or local caches
- do not commit generated artifacts unless you intentionally want frozen
  example outputs in version control

Note:

- runtime artifacts may still include scenario metadata for traceability
- tracked source files should stay portable and path-clean

## What This Repo Means Right Now

`ics-agent` is no longer just a stale note about a local Qwen model. It now
documents and exposes the implemented scripted wargaming layer for the
water-treatment CPS environment, while leaving room for future autonomous
red-team and blue-team agents.
