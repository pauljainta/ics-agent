# configs

This directory contains tracked configuration used by the experiment orchestrator.

## Contents

- [`experiments/`](experiments/README.md): experiment YAMLs that bind scenarios, profiles, runtime settings, and result roots.
- [`infrastructure/`](infrastructure/README.md): public infrastructure templates with placeholders only.
- [`knowledge/`](knowledge/README.md): shared knowledge manuals used by multiple profiles.
- [`profiles/`](profiles/README.md): shared role-profile registry.
- [`scenario_catalog.yaml`](scenario_catalog.yaml): legacy/shared scenario catalog reference. Current testbed-specific catalogs live under [`testbeds/`](../testbeds/README.md).

## Private Config Pattern

Local infrastructure values belong in `configs/private/`, which is ignored by Git. Do not commit filled-in files from that folder.

Example:

```bash
cp configs/infrastructure/sphere.example.yaml configs/private/sphere.local.yaml
```

Then edit `configs/private/sphere.local.yaml` locally with the active reservation details. Keep placeholders in tracked examples.

Safe shape for a private YAML file:

```yaml
profiles:
  water_treatment:
    merge_server: your-local-merge-server
    experiment: your-experiment.project
    realization: your-realization
    revision: your-revision
    xdc: your-xdc
    controller_process_ip: controller-process-host
    simulator_process_ip: simulator-process-host
```

Use symbolic placeholders like the example above in documentation and Git. Never commit real infrastructure identifiers.
