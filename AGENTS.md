# Codex CLI Task List

Use this file as the execution brief for the next Codex CLI run.

## Goal

Prepare and validate a minimal **Option B** SPHERE deployment for the
OpenPLC water-treatment testbed.

In this Option B design:

- the **attacker reasons on this server**
- the **defender reasons on this server**
- SPHERE provides the **environment-side VMs**
- the attacker does **not** begin with direct plant access
- the attacker must be able to operate through a staged SPHERE-side path

This task is **not** the full 18-scenario experiment yet.

The immediate objective is:

1. prepare or deploy the required SPHERE-side infrastructure
2. run a **small unit test / feasibility test**
3. verify that the staged path is usable
4. if successful, commit and push only the relevant code/config/docs changes

## Scope

This run is for **infrastructure preparation and a minimal feasibility check**
only.

Do **not** run the full 18-scenario wargaming suite in this task.

Do **not** modify the paper repo in this task.

## Working Directories

- Workspace root: `/home/researcher/Agentic_ICS`
- Main code repo: `/home/researcher/Agentic_ICS/ics-agent`
- SPHERE repo: `/home/researcher/Agentic_ICS/sphere-usecases`

## Architecture Assumptions

Reasoning stays on this server with the local GPU.

- attacker reasoning host: **this server**
- defender reasoning host: **this server**
- SPHERE nodes: **execution and environment roles only**

SPHERE VMs do **not** need GPUs for this task.

## Target Option B Topology

Use `wargamestable.openplcwater.lgarcia` as the baseline plant deployment, but
do **not** mutate the stable baseline in-place if that would risk breaking the
known-good setup.

Preferred approach:

1. inspect the current `openplcwater.lgarcia` experiment and existing
   realizations/revisions
2. preserve the known-good `wargamestable` plant backend
3. create or prepare a **derived Option B variant** if extra environment-side
   roles are required

The desired environment-side roles are:

- `jump/access` VM
- `hmi/engineering` VM
- existing OpenPLC `plant backend` VM(s)

If the current stable realization already provides enough structure to model the
jump/access and HMI roles cleanly, reuse it. Otherwise create a new revision or
derived realization rather than overwriting the stable baseline.

## Infrastructure Objective

Prepare the minimum environment needed so that the attacker can operate through
a staged path:

1. local attacker reasoning host
2. SPHERE-side jump/access VM
3. SPHERE-side HMI/engineering VM
4. SPHERE-side plant backend

The defender still reasons locally on this server and may observe/interact
through the environment backend without requiring its own dedicated SPHERE GPU
VM.

## Deployment And Preparation Tasks

1. Inspect the current Merge / Launch state for:
   - experiment `openplcwater.lgarcia`
   - realization `wargamestable.openplcwater.lgarcia`
   - any relevant newer revisions or realizations
2. Determine whether the existing stable deployment can support the Option B
   staging roles with only minimal changes.
3. If not, create or prepare a derived revision/realization that adds the
   minimum required environment-side roles without discarding the stable plant
   backend.
4. Materialize or attach the required XDC and ensure the environment-side VMs
   are reachable.
5. Apply only the minimum configuration or deployment changes required in:
   - `sphere-usecases/water-treatment/implementations/openplc`
   - `ics-agent` SPHERE backend/config files
6. Keep all authentication local-only and never print or commit secrets.

## Feasibility Test

Run a **small unit test** after the infrastructure is ready.

The test should verify the staged architecture, not full attacker performance.

### Test intent

Prove that:

- the local attacker can obtain observations from the SPHERE-side path
- the local attacker can issue a harmless remote action through that path
- the plant backend remains reachable
- the local defender can still observe the environment

### Minimum acceptable unit test

Use a harmless or nominal interaction sequence such as:

1. confirm XDC attachment and SSH reachability
2. confirm jump/access VM is reachable
3. confirm HMI/engineering VM is reachable from the prepared environment
4. confirm plant backend is reachable from the HMI/engineering role or through
   the configured backend path
5. perform one harmless read-like or nominal control-path validation
6. confirm the local `ics-agent` stack can receive the observation and complete
   one end-to-end loop without requiring direct plant-starting attack logic

Prefer a test that does **not** alter safety-critical state.

Examples of acceptable success signals:

- successful staged connectivity from local orchestrator to jump/access role and
  onward to the control-facing role
- successful local observation collection from the SPHERE backend
- successful execution of one benign environment action or no-op validation
- successful defender-side observation pass

If the current code needs a dedicated smoke config for this, create one with a
clear name such as:

- `configs/experiments/sphere_optionb_smoke.yaml`

or a similarly named helper script under `scripts/`.

## Success Criteria

The feasibility test is successful only if all of the following are true:

1. the environment-side roles needed for Option B exist and are reachable
2. the attacker can reason locally while acting through the staged SPHERE-side
   path
3. the defender can still reason locally with usable observations
4. no SPHERE GPU dependency was introduced
5. the plant backend remains intact and reachable
6. the test result is recorded in a progress log

## Result And Progress Logging

Write progress logs under:

- `/home/researcher/Agentic_ICS/Artifact/Results/New/_progress_optionb`

Keep at minimum:

- `phase_status.json`
- `portal_inspection.json`
- `deployment_notes.md`
- `unit_test_result.json`
- `last_successful_step.txt`

Log at least:

1. portal inspection complete
2. topology decision complete
3. deployment/preparation complete
4. unit test started
5. unit test passed or failed
6. relevant code committed/pushed

If the test fails, leave the environment and logs in a state that makes it easy
to resume from the first incomplete step.

## Validation

Run validation from `/home/researcher/Agentic_ICS/ics-agent`.

At minimum:

- `.venv/bin/python -m ics_agent.cli validate`
- `.venv/bin/python -m pytest tests/test_environment.py tests/test_orchestrator.py tests/test_agentic.py`

If a small smoke config or helper script is added, run it as the final
feasibility check.

## Git Requirement

If the infrastructure preparation and small unit test succeed:

### `ics-agent`

- inspect `git status`
- stage only relevant code/config/docs changes
- do **not** stage results
- do **not** stage progress logs
- do **not** stage secrets or auth files
- commit and push if changes were required

### `sphere-usecases`

- inspect `git status`
- stage only relevant code/config/docs changes
- do **not** stage results
- do **not** stage progress logs
- do **not** stage secrets or auth files
- commit and push if changes were required

If one repo stayed unchanged, do not create an empty commit for it.

## Secret Handling

Never print, commit, or copy into tracked files:

- Merge usernames or passwords
- token material
- SSH private keys
- local secret env files

Use existing local authentication only.

## Definition Of Done

This task is complete only when:

1. the current SPHERE state was inspected and recorded
2. a clear Option B topology decision was made
3. the required environment-side infrastructure was prepared or deployed
4. a small staged-path feasibility test was run
5. the test outcome was recorded in logs
6. relevant code/config/docs changes were committed and pushed if necessary
7. no results, progress logs, or secrets were committed
