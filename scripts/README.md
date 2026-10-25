# scripts

Executable wrappers for experiments and setup. Run them from the repo root with the repo virtual environment:

```bash
.venv/bin/python scripts/<script-name>.py
```

## Setup Scripts

- [`download_local_model.py`](download_local_model.py): downloads supported local model checkpoints into [`../models/local_models/`](../models/local_models/README.md).
- [`phase35_knowledge_pipeline.py`](phase35_knowledge_pipeline.py): builds sanitized domain-primer artifacts from source manuals for target-blind domain/cyber profiles.

## Current Agentic Experiment Scripts

- [`run_water_treatment_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py`](run_water_treatment_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py): treatment seeded attacker-knowledge comparison.
- [`run_water_distribution_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py`](run_water_distribution_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py): distribution seeded attacker-knowledge comparison.
- [`run_water_treatment_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py`](run_water_treatment_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py): treatment white-box context ablation.
- [`run_water_distribution_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py`](run_water_distribution_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py): distribution white-box context ablation.
- [`run_water_treatment_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py`](run_water_treatment_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py): treatment defender-authority ablation.
- [`run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py`](run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py): distribution defender-authority ablation.
- [`run_water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py): treatment target-blind high-domain/high-cyber single-family run.
- [`run_water_distribution_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py): distribution target-blind high-domain/high-cyber single-family run.
- [`run_water_treatment_single_agentic_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_single_agentic_attacker_vs_agentic_full_containment_defender.py): treatment single agentic pair helper.

## Scripted Baseline Scripts

- [`run_water_treatment_scripted_black_box_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_scripted_black_box_attacker_vs_agentic_full_containment_defender.py)
- [`run_water_treatment_scripted_gray_box_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_scripted_gray_box_attacker_vs_agentic_full_containment_defender.py)
- [`run_water_distribution_scripted_black_box_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_scripted_black_box_attacker_vs_agentic_full_containment_defender.py)
- [`run_water_distribution_scripted_gray_box_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_scripted_gray_box_attacker_vs_agentic_full_containment_defender.py)

## Rule-Based Defender Baselines

- [`run_rule_based_defender_baseline.py`](run_rule_based_defender_baseline.py): parameterized runner for rule-based defender modes.

Example:

```bash
.venv/bin/python scripts/run_rule_based_defender_baseline.py controller \
  --testbed water_treatment \
  --attacker black_box \
  --defender-mode active_intervention
```

## Model-Variant Scripts

- [`run_qwen3_14b_water_treatment_model_variant.py`](run_qwen3_14b_water_treatment_model_variant.py)
- [`run_qwen3_14b_water_distribution_model_variant.py`](run_qwen3_14b_water_distribution_model_variant.py)
- [`run_phi4_mini_water_treatment_model_variant.py`](run_phi4_mini_water_treatment_model_variant.py)
- [`run_phi4_mini_water_distribution_model_variant.py`](run_phi4_mini_water_distribution_model_variant.py)

Smoke mode validates configs and model availability without running scenarios:

```bash
.venv/bin/python scripts/run_qwen3_14b_water_treatment_model_variant.py --smoke-only
```

## Backend Smoke

- [`run_water_treatment_sphere_openplc_backend_smoke.py`](run_water_treatment_sphere_openplc_backend_smoke.py): treatment SPHERE/OpenPLC backend smoke helper.

## PLCBench handoff

Start with the [JP handoff guide](../docs/baselines/plcbench_jp_handoff.md).
[`plcbench_package_pilot.py`](plcbench_package_pilot.py) packages only the
three saved pilot groups into a separate archive with source/config hashes;
it refuses to overwrite an existing archive. The original `runs/` bundles
remain ignored and unchanged.

## Results

Scripts write results under `Artifact/Results/` outside the repo source tree. Do not commit result folders.
