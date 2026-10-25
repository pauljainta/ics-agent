# PLCBench engineering handoff for JP

This branch contains a **local PLCBench-inspired implementation** for
ICS-Agent comparisons. No author-supplied PLCBench artifact has been verified.
The [provenance and deviation record](plcbench_takeover_alignment.md) is the
starting point for any claim about the paper. This guide identifies what JP can
reproduce now and what still needs an admission gate or experiment.

## Start here

Use Python **3.10.x**. For the offline scorer, fixtures, and focused tests, the
small pinned dependency set avoids downloading local-model training/serving
packages. In a clean checkout:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-plcbench-offline.txt
PYTHONPATH=. python -m pytest -q tests/test_plcbench_*.py
PYTHONPATH=. python scripts/plcbench_fixture_studies.py fixed > /tmp/plcbench-fixed.json
PYTHONPATH=. python scripts/plcbench_fixture_studies.py factorial > /tmp/plcbench-factorial.json
PYTHONPATH=. python scripts/plcbench_fixture_studies.py adaptive > /tmp/plcbench-adaptive.json
```

Use `requirements.txt` for the full ICS-Agent runtime and live model paths.
The fixture scripts are offline software checks. The fixed fixture marks the
LLM policy unassessable when no model decision file is supplied. They do not
start a PLC or run a measured study. Docker/plant integration tests have
separate prerequisites described in the
[takeover runbook](plcbench_takeover_runbook.md).

For this handoff, an isolated official `python:3.10-slim` image at digest
`sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d`
installed `requirements-plcbench-offline.txt` and ran the focused suite with
the checkout and pinned SPHERE source mounted read-only: **114 passed, 1
skipped**. The skip was the opt-in Docker integration. Without the separate
SPHERE checkout, the same suite reported 112 passed, 3 skipped. One run also
emitted a test-server thread shutdown warning after passing; no assertion
failed. The Python 3.10 environment re-scored 30/30 pilot bundles and ran the
fixed, factorial, and adaptive offline fixtures. These checks do not qualify
an external controller or model endpoint. Set `PLCBENCH_SPHERE_ROOT` to the
separately pinned SPHERE checkout to run the two source-dependent tests.

The private PR carries a companion `plcbench-pilot-30-*.tgz` archive, outside
Git history. It contains `runs/wd-batch`, `runs/wt-lot`, and `runs/wt-dosing`
(five repetitions × two arms × three objectives), an inventory, and per-file
SHA-256 manifest. After checking the archive SHA-256 printed in the PR:

```sh
tar -xzf /path/to/plcbench-pilot-30-*.tgz -C .
shasum -a 256 -c PILOT_MANIFEST.sha256
PYTHONPATH=. python -m ics_agent.plcbench_evidence --pilot-root runs
```

Expected re-score: **30 episodes, 30 matching stored verdicts, 0
unassessable**, using the three versioned pilot objectives. The original
bundles lack their objective snapshots, so this is retrospective re-scoring
with external versioned configs. Missing evidence without an objective is
unassessable. Do not overwrite the bundles or substitute the proposed 5 s/30 s
objectives for the pilot 3 s/10 s objectives. The [pilot summary](../../runs/COMPARISON_SUMMARY.md)
lists the observed flags and the menu, information, and execution confounds.

## Repository map and status

| Area | Entry point | Handoff status |
| --- | --- | --- |
| Six-flag scoring and re-score | `ics_agent/plcbench_scorer.py`, `ics_agent/plcbench_evidence.py` | Offline, testable; 30 saved pilot verdicts reproduced with explicit objectives. |
| Versioned experiments and runner | `configs/plcbench/`, `scripts/plcbench_batch.py`, `scripts/plcbench_report.py` | Resolved bundle metadata, reset/admission records, balanced 50-pair schedule, and per-cell Wilson intervals implemented; measured pairs absent. |
| WD storage v2 | `configs/plcbench/wd_storage_mass_balance_v2.json` | Experimental positive control passed local mechanistic admission; trivial `stop` policy can attain the endpoint. Original proposed WD storage v1 failed admission. |
| WT chemistry / response | `configs/plcbench/wt_chemistry_dosing_v2.json`, `configs/plcbench/wt_observation_response_fixture_v1.json` | Chemistry requires full simulator restart. Reference and deterministic response policies passed local offline/OpenPLC gates; no measured model comparison. Membrane equation is experimental. |
| Qwen deployment | `configs/plcbench/qwen25_vllm_deployment.example.json`, `docs/baselines/chpc/` | Health and identity tooling exists; the October 15 service smoke was not a measured episode. A fresh deployment identity is required. |
| D3 System One | `scripts/plcbench_wt_response_systemone.py`, `docs/baselines/chpc/d3_trial_20261021.md` | Full checkpoint completed a two-case, single-run offline WT fixture on CHPC. It did not contact a PLC and has no paired effect estimate or cryptographic served-weight attestation. |
| Extensions | CPS-VR, SPHERE, Mattermost, swarm, Rockwell adapter | Read-only/local/offline contracts only. See the takeover runbook; external checkouts and their source identities are separate from this PR. |

The CPS-VR observer change is local commit `3585439` on
`codex/plcbench-cpsvr` in `iotrustlab/CPS-VR`; the SPHERE descriptor/model
annotation is local commit `43e7b45` on `codex/plcbench-sphere` in
`iotrustlab/sphere-usecases`. Neither branch has a tracked upstream in its
checkout, so JP must receive those commits separately before running the
external integration. The ICS-Agent read-only import contracts and tests are
in this PR. The Mattermost fixture is isolated and synthetic; the Rockwell
binding is offline only.

The first 50-pair WD study launch wrote a failed execution record after a CHPC
`squeue` timeout, **before any pair bundles were produced**. The saved
`execution.json` remains local and is not a result. No physical Rockwell
transaction, Unity Editor playback, independent native-confinement review, or
paper draft is part of this handoff.

## First experiment for JP

1. Re-score the archive and reproduce the three offline fixture outputs above.
   The 30-pilot native/menu association is descriptive. The menu supplied
   grounded operator semantics, and a deterministic policy can choose the two
   stop endpoints; this is not evidence of an LLM advantage.
2. Use the [ICS-Agent comparison protocol](plcbench_ics_agent_comparison.md) to
   choose one shared plant realization and predeclare the objective, canonical
   observations, authority, model, timing, reset, and denominators. Run the
   deterministic control before interpreting any agent comparison.
3. Re-admit the chosen cell and a **fresh** serving deployment before a measured
   batch. The planned design is 50 pairs per cell with 25 native-first and 25
   menu-first orders. Keep WD and WT, pilot and proposed objectives, and
   safety and service scores separate. Failed admissions and resets belong in
   the attempt ledger, never silently outside the denominator.

The historical WD study supervisor is parameterized for a new environment.
Before using `scripts/plcbench_wd_study_launch.py`, supply the private
`PLCBENCH_CHPC_SSH_SOCKET`, `PLCBENCH_CHPC_LOGIN`,
`PLCBENCH_CHPC_REMOTE`, `PLCBENCH_CHPC_ACCOUNT`,
`PLCBENCH_CHPC_PARTITION`, and `PLCBENCH_CHPC_LOGIN_IP` variables outside
Git. Submit the parameterized Slurm template with the same confirmed
allocation, scratch directory, and login source IP. Use a fresh job ID and
output directory; the October 15 failed execution is not resumable evidence
for a new study.

For a WD v2 local Docker qualification, follow the
[admission instructions](plcbench_wd_storage_v2.md). For WT, follow the
[response fixture](plcbench_wt_response_fixture.md) and
[restart boundary](plcbench_wt_restart_boundary.md). These commands need
locally built images and a SPHERE source checkout; the archive alone does not
provide them. The Rockwell path remains offline until program identity,
replay, restoration, and hardware arrangements are verified.

## Handoff integrity

- The PR is the code review surface. `runs/` remains ignored; no key, model
  weight, deployment log, private endpoint, or raw research result belongs in
  Git history.
- The pilot archive is immutable evidence for this handoff, with its archive
  SHA-256 in the PR and file hashes inside it. It contains only the three
  pilot groups. Other local qualification and D3 records are **not** included.
- Use the source commit, objective hash, model/runtime identity, realization
  hash, reset proof, and evidence-origin field in every new result. `unassessable`
  is an explicit outcome when a required part is absent.
