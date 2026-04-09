# ICS Agent

`ics-agent` is now the config-driven experiment and agent orchestration repo for
the OpenPLC water-treatment wargaming stack in this workspace.

It reuses the working host-side OpenPLC testbed from the sibling
`sphere-usecases` repo, but moves experiment control, scenario metadata,
agent-role swapping, metrics, reporting, and results storage into this repo.

## Current Status

As of 2026-04-09, the attack-only agentic path has been verified end to end.

- Repo/package naming is intentional:
  the git repo is `ics-agent/`, while the Python package inside it is
  `ics_agent/`. There is no separate required `ics-agnt` package in the
  runtime path.
- Verified attack-only agentic artifacts live under:
  `../Results/full_matrix/<experiment_id>/agentic_agents/attack/`
- Final verified attack summary:
  `18/18` attack scenarios completed, `18/18` passed, observer detection fired
  in `18/18`, defender-side detection fired in `15/18`, defender interventions
  executed in `3/18`, and aggregate metrics were written to
  `../Results/full_matrix/<experiment_id>/attack_metrics_summary.json`
- Key fixes applied during verification:
  reusable role-specific operational manuals are loaded before runs, blue-team
  live context now includes richer telemetry and short-horizon relationships,
  defender actions are no longer deferred behind pending attacker playbook
  steps, structured output parsing is hardened to reduce playbook fallbacks,
  observer and defender metrics are split cleanly, and redundant scheduled
  playbook actions are skipped once the plant is already in the requested state

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

## Operational Knowledge

Agentic profiles can now preload reusable water-treatment manuals before the
run starts. These manuals live under `configs/knowledge/` and are attached per
role through `configs/agent_profiles.yaml`.

- `shared_testbed_manual.md`
  Common process semantics, tags, actions, alarms, and mitigation doctrine.
- `defender_manual.md`
  Blue-team intervention guidance for Process 1 shutdowns and Process 2
  containment.
- `observer_manual.md`
  Alerting and anomaly-triage guidance.
- `operator_manual.md`
  Safe operator-side playbook execution guidance.
- `attacker_manual.md`
  Action-surface guidance for adversarial playbook steps.
- `offline_analyst_manual.md`
  Stored-artifact interpretation guidance for post-run analysis.

These manuals are operational references, not hidden scenario keys. The live
agent context still omits future timeline steps, hidden ground-truth labels, or
scenario-specific answer keys from defender, observer, planner, and operator
prompts.

## Attack-Only Improvements

The attack-only `agentic_agents` path was reworked in these areas:

- blue-team context construction now exposes active alarms, commands, state
  relationships, short-horizon trends, and derived risk flags
- defender and observer now reason on post-action state, so they can react to
  the freshest attacker effects
- defender intervention is no longer blocked simply because attacker playbook
  steps remain pending
- structured-output parsing accepts stricter JSON while still recovering from
  common fenced, YAML-like, or `<think>`-wrapped backend responses
- diagnoses are normalized into short labels so observer and defender metrics do
  not get polluted by free-form sentences
- attack metrics now separate observer detection, defender detection, and
  defender intervention instead of blending them into one blue-team score

Implemented role families:

- `scripted`
  Deterministic scenario-plan or rule-based behavior
- `agentic`
  Qwen-oriented role wrapper that uses a strict GPU-backed LangChain /
  LangGraph decision flow, generic action-space construction, contextual memory
  windows, and explicit backend enforcement so agentic profiles fail instead of
  silently degrading into heuristics
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

Run the verified attack-only all-agentic experiment:

```bash
QWEN_MODEL_PATH=./models/local_models/qwen3.5-9b .venv/bin/python -m ics_agent.cli run experiments/attack_full_agentic.yaml
```

The latest verified attack-only agentic output is:

```text
../Results/full_matrix/<experiment_id>/agentic_agents/attack/
```

Run mixed observer / planner smoke combinations:

```bash
python3 -m ics_agent.cli run experiments/smoke_mixed_roles.yaml
```

Run the full combination matrix:

```bash
python3 -m ics_agent.cli run experiments/full_matrix.yaml
```

`experiments/full_matrix.yaml` is the fresh combined run: it executes all
scenario suites with three conditions:

- `scripted_baseline`
- `fully_agentic`
- `hybrid_blue_team`

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
- Agentic roles require a CUDA-capable GPU and do not fall back to CPU or
  heuristic decision policies.
