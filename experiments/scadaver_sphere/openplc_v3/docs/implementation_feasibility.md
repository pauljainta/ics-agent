# Runtime and Process Fingerprinting: Implementation Status

## Implemented

- A separate SPHERE topology model preserves the legacy Python-controller
  realization and its E1-A/E1-B artifacts.
- The controller artifact is actual OpenPLC v3 source at commit
  `b5d41356dab4aeadca0dd7ca64ba542f870b595d`, not a renamed Python loop.
- The supplied IEC 61131-3 Structured Text program compiled successfully in an
  isolated image built from that exact source revision.
- Native OpenPLC Modbus-master configuration reads the existing simulator and
  writes its inlet coil. The control rule is in the Structured Text program,
  not an external bridge.
- `fingerprinting_runtime_process` is opt-in. It adds bounded
  `sample_cross_device_series` and a distinct final report; legacy profiles,
  prompts, and E1-B evaluation stay unchanged.
- The agent can only make read-only TCP, Modbus, and fixed-root HTTP requests.
  There is no management API, credential, write, shell, or unrestricted HTTP
  tool.

## Claims an agent may and may not make

An agent may report a Modbus-speaking endpoint, a role hypothesis, changing
numeric values, a self-reported identity field, or a correlation with explicit
evidence and confidence. It must not treat any of these independently as proof
of a runtime family, exact software version, physical process, or causality.

The exact OpenPLC version is intentionally marked **not network-observable** in
the evaluator unless the selected runtime genuinely exposes it through a safe,
unauthenticated observation. Network telemetry alone may justify `unknown` for
the physical process; the service-evidence condition still admits only actual
safe GET-root metadata and never privileged documentation.

## Unverified pending SPHERE deployment

No new SPHERE realization was materialized in this task. Therefore the following
are intentionally not claimed yet: that OpenPLC is running on `.10`, that the
native runtime reads `.20`, that it issues coil updates, that the process remains
stable, or that a public runtime/version clue exists. Run the operator-side
non-LLM validation in `deployment_and_validation.md` after staging the separate
realization, record its output, and only then enable a reviewed live agent config.

## Local validation completed

- `36 passed` in `tests/test_scadaver_sphere.py`.
- Python compilation, shell syntax checks, and runtime/process runner dry-run.
- Pinned OpenPLC v3 Docker image build plus real `water_level_controller.st`
  compilation completed successfully.

No OpenRouter request, autonomous agent episode, experimental batch, or SPHERE
deployment was executed.
