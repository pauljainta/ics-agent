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
- [`run_water_distribution_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py): distribution target-semantic ablation. This varies only the attacker target-system context payload while keeping the agentic full-containment defender fixed.
- [`run_water_treatment_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py): treatment target-semantic ablation. This runs the same eight semantic variants against the restored Water Treatment full-containment defender, supports `--seed`, and writes seed-specific progress metadata.
- [`run_process_defender_capability_rerun.py`](run_process_defender_capability_rerun.py): harmonized legacy capability rerunner. It preserves the existing attacker Kt/interface matrices but regenerates configs with the process-only full-containment defender semantics used by the newer state-aware experiments.
- [`run_water_treatment_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py`](run_water_treatment_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py): treatment defender-authority ablation.
- [`run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py`](run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py): distribution defender-authority ablation.
- [`run_water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py): treatment target-blind high-domain/high-cyber single-family run.
- [`run_water_distribution_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.py): distribution target-blind high-domain/high-cyber single-family run.
- [`run_water_treatment_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py`](run_water_treatment_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py): treatment autonomous-ReAct white-box attacker against the full-containment agentic defender.
- [`run_water_distribution_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py`](run_water_distribution_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py): distribution autonomous-ReAct white-box attacker against the full-containment agentic defender.
- [`run_water_treatment_autonomous_react_black_box_attacker_vs_silent_defender_smoke.py`](run_water_treatment_autonomous_react_black_box_attacker_vs_silent_defender_smoke.py): treatment autonomous-ReAct black-box attacker smoke against a silent defender.
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

## Target-Semantic Ablation

Validate the water-distribution target-semantic ablation grid without starting the backend:

```bash
.venv/bin/python scripts/run_water_distribution_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py --validate-only
```

Run the seed-101 water-distribution target-semantic ablation:

```bash
.venv/bin/python scripts/run_water_distribution_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py
```

Validate the water-treatment target-semantic ablation grid without starting the backend:

```bash
.venv/bin/python scripts/run_water_treatment_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py --validate-only --seed 101
```

Run the seed-101 water-treatment target-semantic ablation:

```bash
.venv/bin/python scripts/run_water_treatment_agentic_target_semantic_ablation_attacker_vs_agentic_full_containment_defender.py --seed 101
```

## Harmonized Process-Defender Capability Reruns

Validate the full Water Treatment legacy capability plan without starting the backend:

```bash
.venv/bin/python scripts/run_process_defender_capability_rerun.py \
  --testbed wt \
  --suite all-capability \
  --critical-impact-audit \
  --seed 101 \
  --dry-run
```

`--dry-run` and `--validate-only` both inspect the generated plan without
starting the backend or running scenarios. The plan lists the selected testbed,
suite, scenario ids, condition names, defender mode, seed, result root, and
expected run count.

Run the Water Treatment Kt or interface reruns:

```bash
.venv/bin/python scripts/run_process_defender_capability_rerun.py \
  --testbed wt \
  --suite kt \
  --critical-impact-audit \
  --seed 101

.venv/bin/python scripts/run_process_defender_capability_rerun.py \
  --testbed wt \
  --suite interface \
  --critical-impact-audit \
  --seed 101
```

Run the Water Distribution Kt or interface reruns with the approved local XDC override.
Store the concrete XDC name only in an ignored local env file or shell variable,
not in tracked code:

```bash
export ICS_AGENT_WD_PROCESS_DEFENDER_XDC="<your-water-distribution-xdc>"
```

```bash
.venv/bin/python scripts/run_process_defender_capability_rerun.py \
  --testbed wd \
  --suite kt \
  --critical-impact-audit \
  --xdc "$ICS_AGENT_WD_PROCESS_DEFENDER_XDC" \
  --seed 101

.venv/bin/python scripts/run_process_defender_capability_rerun.py \
  --testbed wd \
  --suite interface \
  --critical-impact-audit \
  --xdc "$ICS_AGENT_WD_PROCESS_DEFENDER_XDC" \
  --seed 101
```

## Backend Smoke

- [`run_water_treatment_sphere_openplc_backend_smoke.py`](run_water_treatment_sphere_openplc_backend_smoke.py): treatment SPHERE/OpenPLC backend smoke helper.

## Results

Scripts write results under `Artifact/Results/` outside the repo source tree. Do not commit result folders.
