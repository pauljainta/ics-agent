# configs/experiments

Experiment YAMLs in this folder are the main declarative entry points for the orchestrator. They specify which scenarios to run, which actor profiles to use, which backend to start, and where results should be written.

## File Families

- `water_treatment_agentic_*`: water-treatment agentic attacker and defender comparisons.
- `water_treatment_scripted_*`: water-treatment deterministic scripted attacker baselines.
- `water_treatment_*_ablation*`: treatment context, domain, and defender-authority ablations.
- `water_distribution_*_vs_agentic_defender.yaml`: distribution attacker-knowledge comparisons.
- `water_distribution_*_ablation*.yaml`: distribution context, domain, and defender-authority ablations.
- `water_distribution_scripted_*`: distribution deterministic scripted attacker baselines.
- `water_treatment_sphere_smoke.yaml`: backend smoke validation for treatment.

## How They Connect

The wrapper scripts in [`../../scripts/`](../../scripts/README.md) load these YAMLs through [`../../ics_agent/config.py`](../../ics_agent/config.py) and [`../../ics_agent/orchestrator.py`](../../ics_agent/orchestrator.py). Profiles are resolved from [`../profiles/shared.yaml`](../profiles/shared.yaml) and testbed-local profile registries such as [`../../testbeds/water_treatment/profiles.yaml`](../../testbeds/water_treatment/profiles.yaml).

## Minimal YAML Shape

```yaml
experiment_name: water_treatment
testbed: water_treatment
repeats: 1
resume: true
scenario_selectors:
  - suite: attack
conditions:
  - name: black_box_attacker__agentic_defender
    profiles:
      attacker: black_box_attacker
      defender: agentic_defender
      observer: agentic_observer
      planner: scripted_planner
      operator: scripted_operator
      offline_analysis: scripted_offline_analyst
environment:
  backend: sphere_openplc
```

Keep infrastructure-specific fields in local private config overlays where possible.
