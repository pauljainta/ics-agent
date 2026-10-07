# Live pilot qualification record

Date: 2026-10-07

Status: **BLOCKED - no live trial run**.

Scope: the water-treatment CTF operator pilot in
`ctf_water_treatment_operator_pilot.yaml`. This record is not a qualification
of hydro/power, distribution, a student-facing CTF, or physical hardware.

## Evidence collected

| Gate | Result | Evidence |
| --- | --- | --- |
| Scenario and profile validation | Pass | `python3 -B -m ics_agent.cli validate --testbed water_treatment` recognized the pilot scenario and profiles. |
| Operator policy | Pass | `tests/test_ctf_operator_pilot.py`: routine start, one missed-check note, reset of policy state, and rejection of an operator sensor write. |
| Existing repository suite | Baseline-limited | 165 passed, 3 failed. The same three failed on an untouched checkout: one model-decoding metadata expectation and two host Python 3.9 / vendored Python 3.10 bytecode/cache checks. |
| Private SPHERE profile | Blocked | Neither `ICS_AGENT_SPHERE_CONFIG` nor the default ignored `configs/private/sphere.local.yaml` exists in this checkout. Resolving the pilot config fails before connection. |
| Agent execution host | Blocked | The example infrastructure profile assigns no remote roles, and the local checkout has no cached model. The agentic backend requires a CUDA GPU. |
| Isolated realization and ownership | Unknown | No current reservation, XDC, or exclusive-use evidence was available for this run. Do not infer isolation from an old experiment name. |
| Full reset and repeatability | Unqualified | The default treatment `soft_reseed` path clears external controls and reseeds levels, but does not validate all controller/process state. No postflight run or repeat run was possible. |
| Live effect and evidence | Not run | No remote actions, packet capture, controller transition, process consequence, or postflight observation was collected. |

## Before a staff live trial

1. Supply an ignored private infrastructure profile for a currently reserved,
   isolated **simulated OpenPLC** realization. Verify its experiment, revision,
   XDC, controller/simulator hosts, and exclusive-use status without placing
   credentials in the repository.
2. Assign the attacker role to a GPU-capable remote node with the required
   Qwen model and Python dependencies; verify the remote role service can load
   the model before any plant-facing action.
3. Verify the effective action allowlist, remote process ownership, and reset
   mode. A reset must restore controller state, injected controls, simulator
   state, and baseline observations; a level-only reseed is not sufficient
   evidence by itself.
4. Capture initial controller/process state, then run one bounded trial through
   the configured action path. Record proposal, validation, execution, operator
   note, controller response, process effect, and timestamps separately.
5. Reset and repeat from a matched initial state. Stop on ambiguous ownership,
   failed reset, stale observations, or unexpected access to physical hardware.

Only after those checks pass should the pilot be described as live-qualified.
