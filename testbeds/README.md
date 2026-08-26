# Testbeds

This folder contains the execution, deployment, knowledge assets, and runtime
implementations for the ICS experiments. It is required for the supported
testbed-backed runs. It is not a disposable test-data directory, and it is
separate from the pytest unit-test directory at `../tests/`.

The `runtime/` package contains the Python implementations shared by the
orchestrator and testbed backends. The `water_treatment/` and
`water_distribution/` folders contain domain-specific assets.

## Top-level testbeds

### `water_distribution/`

The active water-distribution testbed implementation and all of its runtime
assets.

- `tag_contract.yaml`: Canonical tag names and tag metadata used to keep the
  testbed, agents, scenarios, and mappings consistent.
- `scenario_catalog.yaml`: Catalog and metadata for water-distribution attack,
  readiness, and smoke scenarios.
- `prompts/`: Testbed-specific attacker, defender, observer, and ablation prompt
  material loaded by relevant experiment profiles.
- `knowledge/`: Testbed manuals and context primers. These are runtime inputs
  for the corresponding agent profiles, not merely explanatory documents.
- `scenarios/attack/`: Active water-distribution attack scenario definitions.
- `scenarios/readiness/`: Readiness and startup scenario definitions.
- `scenarios/smoke/`: Short smoke-run scenario definitions.
- `docs/process-models/`: Process-model knowledge loaded by context-aware
  configurations. The files describe the supply line, distribution grid, and
  return-water process models.
- `docs/assumptions.md`: Testbed assumptions and operating boundaries.
- `docs/*-pass.md`: Historical validation and experiment notes. These support
  reproducibility and project documentation but are not required to launch a
  normal run.
- `openplc/`: Active local and remote OpenPLC implementation.
  - `README.md`: OpenPLC setup and operation guide.
  - `configs/openplc_map.yaml`: PLC/Modbus mapping used by the bridge and
    validation workflows.
  - `projects/controller_project/beremiz.xml`: Controller project metadata.
  - `projects/controller_project/plc.xml`: Controller PLC project definition.
  - `projects/simulator_project/beremiz.xml`: Simulator project metadata.
  - `projects/simulator_project/plc.xml`: Simulator PLC project definition.
  - `st/wd_controller.st`: Active controller IEC 61131-3 source.
  - `st/wd_simulator.st`: Active simulator IEC 61131-3 source.
  - `scenarios/*.yaml`: OpenPLC-level idle, startup, demand, and alarm inputs.
  - `scripts/modbus_bridge.py`: Executable controller/simulator bridge.
  - `scripts/validation_harness.py`: OpenPLC invariant and validation harness.
  - `scripts/requirements.txt`: Dependencies for the OpenPLC helper scripts.
- `sphere/`: Deployment records and operational notes for the remote testbed.
  These document deployment state and are useful for deployment/reproduction,
  but are not required for local experiment execution.

### `water_treatment/`

Water-treatment scenario, domain knowledge, and experiment documentation used
by the water-treatment profiles.

- `scenarios/attack/`: Water-treatment attack scenario definitions.
- `scenarios/normal/`: Nominal process scenario definitions.
- `scenarios/readiness/`: Startup and readiness scenario definitions.
- `scenarios/recheck/`: Recheck and regression scenario definitions.
- `scenarios/smoke/`: Short smoke-run scenario definitions.

- `knowledge/phase35/water_treatment_domain_primer.md`: Domain primer used by
  the phase-specific treatment profile.
- `docs/2026-08-11-swat-informed-domain-ablation-pass.md`: Domain-ablation
  methodology and validation record.
- `docs/2026-08-20-scripted-gray-box-baseline-pass.md`: Scripted gray-box
  baseline validation record.
- `docs/2026-08-20-scripted-white-box-baseline-pass.md`: Scripted white-box
  baseline validation record.

## Runtime dependency rule

A selected experiment normally needs only one testbed's relevant subset. For
example, a water-distribution run needs the distribution testbed assets, while
a water-treatment run needs the treatment assets and the treatment runtime
implementation elsewhere in `ics_agent/`. The entire `testbeds/` directory is
kept because experiment selection, profile loading, deployment, and
reproducibility are all configuration-driven.

## Retired content

The former `water_distribution/openplc/st/legacy/` controller and simulator
sources were removed after verifying that no production script, deployment
configuration, or documentation referenced them. The active sources are under
`openplc/st/`.

Generated Python bytecode, logs, and runtime artifacts may appear below this
folder during execution. They are ignored and should not be committed. The
tracked source files and YAML/XML/Markdown assets above are the source of truth.

Do not place real credentials, personal usernames, private host addresses,
private keys, or deployment secrets in tracked testbed files. Use local ignored
configuration for environment-specific values.
