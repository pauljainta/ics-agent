# ics_agent/experiments

This package contains experiment-specific controllers. These modules are called by the wrapper scripts in [`../../scripts/`](../../scripts/README.md).

## Files

- [`model_variant_runner.py`](model_variant_runner.py): runs current attacker/defender profiles while changing only the local LLM checkpoint.
- [`rule_based_defender_baseline.py`](rule_based_defender_baseline.py): controller for deterministic rule-based defender baselines.
- [`water_treatment/`](water_treatment/README.md): treatment-specific seeded, ablation, and scripted-baseline controllers.
- [`water_distribution/`](water_distribution/README.md): distribution-specific seeded, ablation, and scripted-baseline controllers.
- [`__init__.py`](__init__.py): package marker.

## Resume Behavior

Long-running controllers keep progress files in their result folders. When a controller is resume-aware, rerunning the same command should skip completed trials and continue from the next missing or failed item. Check the per-controller README and progress logs before deleting results.
