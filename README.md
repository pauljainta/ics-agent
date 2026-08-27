# ics-agent

`ics-agent` is the implementation repository for bounded cyber-physical attacker-defender wargaming over OpenPLC water-system testbeds. The repo contains the agent policies, scenario definitions, experiment runners, runtime backends, and documentation needed to execute the current water-treatment and water-distribution experiments without depending on external source repositories.

The paper source is kept separately in `ics-agent-paper`; this repo produces experiment outputs under the workspace-level `Artifact/Results/` tree.

## Repository Map

- [`configs/`](configs/README.md): tracked experiment YAMLs, shared profile registries, public infrastructure templates, and shared knowledge manuals.
- [`ics_agent/`](ics_agent/README.md): Python package implementing agents, orchestration, runtime backends, metrics, reporting, and experiment controllers.
- [`scripts/`](scripts/README.md): executable entry points for the current experiments and model-variant runs.
- [`models/`](models/README.md): local model-weight location and model download notes.
- [`testbeds/`](testbeds/README.md): water-treatment and water-distribution scenario corpora, prompts, knowledge, OpenPLC assets, and runtime specs.
- [`vendor/`](vendor/README.md): vendored OpenPLC support needed by the treatment backend.
- [`tests/`](tests/README.md): pytest tests and fixtures.
- [`water_treatment_scenario.md`](water_treatment_scenario.md): human-readable water-treatment process and scenario overview.
- [`water_distribution.md`](water_distribution.md): human-readable water-distribution process and scenario overview.

## What The Framework Runs

The core loop is:

1. Load a scenario from a testbed catalog.
2. Instantiate attacker, defender, observer, planner, operator, and offline-analysis roles from profile YAML.
3. Start the selected backend, usually the SPHERE/OpenPLC backend for full experiments.
4. Build bounded observations and candidate actions for each role.
5. Ask the selected policy, such as agentic, scripted, or rule-based, to choose admissible actions.
6. Apply validated actions to the process runtime.
7. Record trajectories, metrics, runtime artifacts, summaries, and progress logs.

The default current agentic comparison uses black-box, gray-box, white-box, and target-blind high-domain/high-cyber attackers against the full-containment agentic defender. Additional runners cover context ablations, defender-authority ablations, deterministic scripted attackers, deterministic rule-based defenders, and alternate local model checkpoints.

## Environment Setup

Use one repo-local virtual environment for experiments:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the venv interpreter explicitly when running long experiments:

```bash
.venv/bin/python scripts/run_water_treatment_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py
```

## Local Configuration

Do not commit credentials, private reservation names, private hostnames, IP addresses, SSH keys, passwords, or filled-in local infrastructure files.

Public examples:

- [`.env.example`](.env.example) shows local model and runtime environment variable names.
- [`configs/infrastructure/sphere.example.yaml`](configs/infrastructure/sphere.example.yaml) shows the shape of a SPHERE runtime profile using placeholders only.

Typical local setup:

```bash
cp .env.example .env
cp configs/infrastructure/sphere.example.yaml configs/private/sphere.local.yaml
```

Then edit the copied files locally. `configs/private/` and `.env` are ignored by Git.

## Model Weights

Model weights are intentionally not tracked. Store local checkpoints under [`models/local_models/`](models/local_models/README.md), or use:

```bash
.venv/bin/python scripts/download_local_model.py --model qwen3_14b
.venv/bin/python scripts/download_local_model.py --model phi4_mini_instruct
```

The exact supported model variants are implemented in [`ics_agent/experiments/model_variant_runner.py`](ics_agent/experiments/model_variant_runner.py).

## Running Current Experiments

Before SPHERE/OpenPLC experiments, make sure the local private infrastructure YAML is filled in and any required SPHERE/MRG session is active.

Water-treatment seeded attacker-knowledge comparison:

```bash
.venv/bin/python scripts/run_water_treatment_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py
```

Water-distribution seeded attacker-knowledge comparison:

```bash
.venv/bin/python scripts/run_water_distribution_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py
```

Water-treatment defender-authority ablation:

```bash
.venv/bin/python scripts/run_water_treatment_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py
```

Water-distribution defender-authority ablation:

```bash
.venv/bin/python scripts/run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py
```

Qwen3-14B model-variant runs:

```bash
.venv/bin/python scripts/run_qwen3_14b_water_treatment_model_variant.py
.venv/bin/python scripts/run_qwen3_14b_water_distribution_model_variant.py
```

Phi-4-mini model-variant runs:

```bash
.venv/bin/python scripts/run_phi4_mini_water_treatment_model_variant.py
.venv/bin/python scripts/run_phi4_mini_water_distribution_model_variant.py
```

Most long-running controllers are resume-aware at the scenario or trial level. If a run stops, rerun the same command after fixing the underlying issue. The runner should skip completed runs and purge failed or incomplete trials where that behavior is implemented.

## Results Layout

Experiment outputs are written outside the repo source tree under `Artifact/Results/` relative to the workspace root. Typical layouts include:

- `Artifact/Results/Water-treatment/...`
- `Artifact/Results/water_distribution_openplc/...`
- `Artifact/Results/Qwen3-14B/...`
- `Artifact/Results/Phi-4-mini-instruct/...`

Results are not source code and should not be committed to this repo.

## Safety And Secret Hygiene

- Keep real infrastructure values in `.env` or `configs/private/*.yaml`.
- Keep model weights in `models/local_models/`.
- Do not add generated `__pycache__`, logs, large result folders, model weights, private configs, or local credentials.
- Use placeholder values in tracked examples.
