# Test Suite

This directory contains unit, integration, regression, and contract tests for
`ics-agent`. These files are for validating the codebase only. They are not the
actual experiment launchers and are not required at runtime to run experiments.
The experiment entrypoints are under `../scripts/`.

## Test files

- `conftest.py`: Shared pytest fixtures, including fake OpenPLC modules.
- `test_config.py`: Loads local-only values from the ignored `test.env` file and
  provides safe placeholder defaults.
- `test_agentic.py`: Agent context, reproducibility metadata, and JSON recovery.
- `test_metrics.py`: Metric semantics for runtime and symbolic containment.
- `test_orchestrator.py`: Orchestrator events, containment, and advisory response.
- `test_environment.py`: Environment actions, resets, transport configuration, and
  backend contracts using fakes and temporary paths.
- `test_water_distribution.py`: Water-distribution testbed behavior and bounds.
- `test_phase35_profiles.py`: Profile, prompt, and knowledge-file constraints.
- `test_rule_based_defender_baseline.py`: Rule-based baseline profiles and behavior.
- `test_scripted_black_box_baseline.py`: Treatment scripted black-box runner
  bookkeeping, preflight, determinism, and resume contracts.
- `test_scripted_gray_box_baseline.py`: Treatment scripted gray-box runner
  bookkeeping, preflight, determinism, and resume contracts.
- `test_scripted_white_box_baseline.py`: Treatment scripted white-box runner
  bookkeeping, preflight, determinism, and resume contracts.
- `test_seeded_backbone.py`: Treatment seeded-backbone manifests and helpers.
- `test_white_box_ablation.py`: Treatment white-box information-ablation rules.
- `test_defender_authority_ablation.py`: Treatment defender-authority runner
  profiles, condition pairs, ledgers, and summaries.
- `test_water_treatment_domain_ablation.py`: Treatment domain-ablation metadata,
  hashes, ledgers, and methodology checks.
- `test_water_distribution_domain_ablation.py`: Distribution domain-ablation
  metadata, hashes, ledgers, and audit checks.
- `test_water_distribution_defender_authority_ablation.py`: Distribution
  authority-ablation conditions, ledgers, and metadata.
- `test_water_distribution_white_box_ablation.py`: Distribution white-box
  information-ablation rules.
- `test_water_distribution_scripted_black_box_baseline.py`: Distribution scripted
  black-box bookkeeping, preflight, determinism, and resume contracts.
- `test_water_distribution_scripted_gray_box_baseline.py`: Distribution scripted
  gray-box bookkeeping, preflight, determinism, and resume contracts.
- `test_water_distribution_scripted_white_box_baseline.py`: Distribution scripted
  white-box bookkeeping, preflight, determinism, and resume contracts.
- `test_water_distribution_seeded_backbone.py`: Distribution seeded-backbone
  manifests, helpers, command construction, and resume-state handling.

## Configure local test values

`test.env` is ignored by Git. It contains placeholder values only and is safe to
replace locally when a test needs deployment-specific fixtures. Do not commit
`test.env`, credentials, private keys, real usernames, private IP addresses,
external hostnames, or deployment identifiers.

Example format:

```dotenv
ICS_TEST_MERGE_SERVER=replace-with-local-merge-server
ICS_TEST_TREATMENT_EXPERIMENT=replace-with-local-treatment-experiment
ICS_TEST_TREATMENT_REALIZATION=replace-with-local-treatment-realization
ICS_TEST_DISTRIBUTION_REALIZATION=replace-with-local-distribution-realization
ICS_TEST_REVISION=replace-with-local-revision
ICS_TEST_MERGE_USER=replace-with-local-user
ICS_TEST_XDC_HOST=replace-with-local-xdc-host
ICS_TEST_XDC_REF=replace-with-local-xdc-ref
ICS_TEST_ACCESS_IP=test-access-node
ICS_TEST_ENGINEERING_IP=test-engineering-node
ICS_TEST_CONTROLLER_IP=test-controller-node
ICS_TEST_SIMULATOR_IP=test-simulator-node
```

The same values may be supplied as environment variables. Environment variables
named `ICS_TEST_*` override values in `test.env`.

For ordinary unit tests, the placeholder defaults are sufficient. Tests that
perform a real external preflight should use sanitized test fixtures or explicit
local values and should only be run in an approved environment.

## Run tests

From the repository root:

```bash
python3.10 -m pytest -q tests
```

If unrelated globally installed pytest plugins cause collection errors, disable
plugin autoloading for this run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.10 -m pytest -q tests
```

To run one file:

```bash
python3.10 -m pytest -q tests/test_environment.py
```

To run one test:

```bash
python3.10 -m pytest -q tests/test_environment.py::test_block_sensor_write_blocks_later_attacker_sensor_writes
```

The repository targets Python 3.10.x. Install dependencies from
`../requirements.txt` before running the suite. The tests use mocks, fakes, and
temporary paths; they do not launch the actual experiment matrix.
