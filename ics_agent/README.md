# ics_agent

This is the Python package for the wargaming framework.

## Main Modules

- [`agents/`](agents/README.md): agent policies and action-space construction.
- [`experiments/`](experiments/README.md): long-running experiment controllers and model-variant runners.
- [`action_semantics.py`](action_semantics.py): action normalization, surfaces, and containment metadata.
- [`cli.py`](cli.py): package CLI helpers.
- [`config.py`](config.py): YAML loading and config validation.
- [`environment.py`](environment.py): local and SPHERE/OpenPLC runtime backends.
- [`infrastructure.py`](infrastructure.py): infrastructure config loading and profile resolution.
- [`knowledge.py`](knowledge.py): manual/context loading.
- [`metrics.py`](metrics.py): metric calculations.
- [`mitre.py`](mitre.py): MITRE-style staged interaction metadata.
- [`orchestrator.py`](orchestrator.py): scenario execution loop, role coordination, result writing, aggregation.
- [`paths.py`](paths.py): repo-relative path helpers and result/config roots.
- [`remote_roles.py`](remote_roles.py): remote role-service proxy support.
- [`remote_service.py`](remote_service.py): role service process used when roles execute through a remote host.
- [`reporting.py`](reporting.py): JSON/YAML/CSV output helpers.
- [`scenarios.py`](scenarios.py): scenario catalog and selector handling.
- [`types.py`](types.py): shared dataclasses and type definitions.
- [`utils.py`](utils.py): common utility helpers.
- [`water_distribution_remote_runtime.py`](water_distribution_remote_runtime.py): distribution-specific remote runtime support.

## Execution Flow

Scripts in [`../scripts/`](../scripts/README.md) call experiment controllers in [`experiments/`](experiments/README.md), which in turn use [`orchestrator.py`](orchestrator.py). The orchestrator loads YAML config, resolves scenarios from [`../testbeds/`](../testbeds/README.md), instantiates role agents from [`agents/`](agents/README.md), starts [`environment.py`](environment.py), and writes result bundles.
