# ICS Agent

`ics-agent` is now the config-driven experiment and agent orchestration repo for
the OpenPLC water-treatment wargaming stack in this workspace.

It reuses the working host-side OpenPLC testbed from the sibling
`sphere-usecases` repo, but moves experiment control, scenario metadata,
agent-role swapping, metrics, reporting, and results storage into this repo.

## What Is In Scope

The repo now covers these AGENTS.md requirements in a reusable structure:

- end-to-end OpenPLC testbed orchestration through a shared environment adapter
- scripted online roles
- agentic role scaffolding for Qwen 3.5 with LangChain / LangGraph execution
- offline analysis roles operating only on stored run artifacts
- config-driven scenario catalog and experiment matrices
- combination-matrix execution with named experimental conditions
- metrics and report generation for each run
- workspace-level `Agentic_ICS/Results/` output kept outside the `ics-agent` git repo
- automated validation and pytest coverage for config loading and orchestration

## Repo Layout

- `ics_agent/`
  The Python package for runtime orchestration, environment integration,
  metrics, reporting, and role implementations.
- `configs/scenarios/`
  Repo-local execution scenario YAMLs. Add new scenarios here.
- `configs/scenario_catalog.yaml`
  Suite defaults plus per-scenario metadata overrides. Add intent,
  prerequisites, expected outcomes, metrics, and ground truth here.
- `configs/agent_profiles.yaml`
  Configurable scripted, agentic, and offline role profiles.
- `configs/experiments/`
  Experiment launch configs, including smoke and full matrix examples.
- `configs/prompts/`
  Agentic prompt templates kept in files for reproducibility.
- `tests/`
  Unit and orchestration tests.
- `../Results/`
  Machine-readable and human-readable outputs for completed runs. This lives at
  the workspace root so results stay outside the `ics-agent` git repo.
- `agents/scripted_wargame.py`
  Legacy scripted runner kept for compatibility with older flows.

## Scenario Model

A scenario now has two layers:

1. `configs/scenarios/<suite>/*.yaml`
   This is the execution spec: duration, initial conditions, timeline, and
   expectations.
2. `configs/scenario_catalog.yaml`
   This is the metadata schema: intent, prerequisites, attacker actions,
   defender responses, expected outcomes, success/failure conditions, metrics,
   safety bounds, and ground truth.

That split keeps scenario mechanics reusable while letting the same scenario be
run with scripted, agentic, or offline analyses.

## Agent Model

Profiles in `configs/agent_profiles.yaml` define each role:

- `attacker`
- `defender`
- `observer`
- `planner`
- `operator`
- `offline_analysis`

Each role can be swapped at runtime by changing the experiment config, not code.

Implemented role families:

- `scripted`
  Deterministic scenario-plan or rule-based behavior
- `agentic`
  Qwen-oriented role wrapper that defaults to a LangChain / LangGraph decision
  flow with cached model loading, candidate-action selection, memory windows,
  and strict backend enforcement so agentic profiles fail explicitly when the
  local model path or dependencies are unavailable
- `offline`
  Same role interface, but executed only after a run from stored artifacts

## Running

Validate the catalog and configs:

```bash
python3 -m ics_agent.cli validate
```

List scenarios:

```bash
python3 -m ics_agent.cli list-scenarios
```

Run the core smoke matrix:

```bash
python3 -m ics_agent.cli run experiments/smoke_core_matrix.yaml
```

Bootstrap the local Qwen runtime for the all-agentic path:

```bash
bash scripts/setup_env.sh
bash scripts/download_qwen_model.sh
QWEN_MODEL_PATH=./models/local_models/qwen3.5-9b .venv/bin/python -m ics_agent.cli run experiments/smoke_full_agentic.yaml
```

Run mixed observer / planner smoke combinations:

```bash
python3 -m ics_agent.cli run experiments/smoke_mixed_roles.yaml
```

Run the full combination matrix:

```bash
python3 -m ics_agent.cli run experiments/full_matrix.yaml
```

Results are written under:

```text
../Results/<experiment_name>/<experiment_id>/<condition>/<suite>/<scenario_id>/trial-001/
```

Each run directory includes:

- `observations.csv`
- `actions.json`
- `decisions.json`
- `alerts.json`
- `system_events.json`
- `offline_analysis.json`
- `metrics.json`
- `summary.json`
- `summary.md`
- `config_snapshot.yaml`
- `scenario_resolved.yaml`

## Replay / Offline Mode

Replay mode lets offline agents operate only on saved logs.

Workflow:

1. Run a live experiment and choose one run directory.
2. Copy or link that run directory into `../Results/replay_seed/<name>/`.
3. Run:

```bash
python3 -m ics_agent.cli run experiments/smoke_offline_replay.yaml
```

## Testing

Run:

```bash
pytest tests -q
```

The test suite validates:

- scenario catalog loading
- metrics calculations
- fake-environment orchestration
- replay-mode execution

## Notes

- No sensitive paths or secrets are hardcoded into tracked source.
- Prompt templates, role choices, and scenario metadata live in files.
- The OpenPLC environment adapter preserves a stable role interface while
  swapping implementation families.
