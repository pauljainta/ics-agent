# configs/profiles

This folder contains shared role-profile definitions.

## Files

- [`shared.yaml`](shared.yaml): common scripted roles, rule-based roles, and role defaults reused by testbed-specific profile registries.

## How Profiles Resolve

Profiles are loaded from:

1. [`shared.yaml`](shared.yaml)
2. the selected testbed registry, such as [`../../testbeds/water_treatment/profiles.yaml`](../../testbeds/water_treatment/profiles.yaml) or [`../../testbeds/water_distribution/profiles.yaml`](../../testbeds/water_distribution/profiles.yaml)

The resolution logic is implemented in [`../../ics_agent/paths.py`](../../ics_agent/paths.py), [`../../ics_agent/config.py`](../../ics_agent/config.py), and [`../../ics_agent/orchestrator.py`](../../ics_agent/orchestrator.py).

## Profile YAML Shape

```yaml
profiles:
  scripted_operator:
    role: operator
    policy: scripted
  agentic_defender:
    role: defender
    policy: agentic
    llm:
      backend: qwen_langgraph
      temperature: 0.0
```

Keep model and infrastructure overrides in experiment YAMLs or local private config rather than duplicating them across profiles.
