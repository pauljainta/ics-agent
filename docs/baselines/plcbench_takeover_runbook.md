# PLCBench engineering takeover: local runbook and evidence

## Scope and naming

For JP's current handoff, start with the [handoff guide](plcbench_jp_handoff.md)
and [ICS-Agent comparison protocol](plcbench_ics_agent_comparison.md). This
chronological runbook records engineering steps and local evidence; later D3
offline trial details are in [the October 21 record](chpc/d3_trial_20261021.md).

This is a **local PLCBench-inspired implementation**, not a verified author
artifact. The saved 30 pilot bundles remain unchanged under `runs/`. All new
cells have distinct identities; the pilot 3 s/10 s objectives and the older
proposed 5 s/30 s objectives are separate. No measured CHPC batch, physical
Rockwell transaction, 50-pair study, or paper draft was run here.

## Reproduce the offline evidence

From the `ics-agent` checkout:

```sh
PYTHONPATH=. python -m ics_agent.plcbench_evidence --pilot-root runs
PYTHONPATH=. python scripts/plcbench_fixture_studies.py fixed
PYTHONPATH=. python scripts/plcbench_fixture_studies.py factorial
PYTHONPATH=. python scripts/plcbench_fixture_studies.py adaptive
PYTHONPATH=. python scripts/plcbench_batch.py --config configs/plcbench/pilot_wd_loss_v1.json
```

The re-score returned **30 episodes, 30 matching original flags, 0
unassessable** with the explicit versioned legacy objectives. The benign
running-state case in `test_plcbench_takeover.py` scored no disruption or
impact and was assessable. The three fixture scripts ran successfully; their
results are labeled `offline_fixture` or `adaptive_local_fixture`, and the
fixed study marks LLM decisions unassessable until supplied. The 2×2
presentation × information renderer holds operation IDs, observation fields,
plant snapshot hash, and budget fixed; it has no measured outcome yet.

For a new live local batch, build `configs/plcbench/Dockerfile.native`, supply
loopback controller/simulator endpoints and a fresh output directory, and use
`scripts/plcbench_batch.py --run`. The WT dosing cell also requires a host-side
`--restart-hook` executable that fully restarts the simulator and returns a verified
`wt-full-restart/v1` proof for **every** episode. The [local hook boundary](plcbench_wt_restart_boundary.md) records a concrete implementation and its remaining batch endpoint gate.
The target Qwen/vLLM cell requires a healthy `/health` and `/v1/models`, plus
exact `weight_revision`, `vllm_version`, and `runtime_digest` in
`--deployment-json`; `scripts/plcbench_capture_deployment.py` creates this
record on the serving node from the exact HF commit, model config, vLLM package,
and frozen Python environment. Proposed cells remain `unqualified` and cannot
run.

New bundles embed the resolved objective, model/deployment, menu, source hashes,
reset contract, sandbox/image identity, evidence origin, and assessment status.
`attempts.json` records every reset/admission failure as well as completions.
`scripts/plcbench_report.py` re-scores each bundle and reports Wilson 95%
intervals separately by cell, objective hash, and arm. The declared design is
50 paired repetitions per cell with 25/25 arm order; no proposed objective is
pooled with a pilot objective.

## Integration checks

- The PLCBench suite completed with **78 passed, 1 skipped**. The opt-in Docker
  integration test passed separately. A networkless native child sent a Modbus
  read through the Unix broker relay, the broker logged and accepted it, and a
  direct outbound socket failed. This is local functional proof, not an
  independent confinement review. Non-loopback execution remains closed.
- The isolated Mattermost 9.11/Postgres fixture became healthy. A synthetic
  account, private team, and private channel were created. The adapter posted
  a synthetic observation and read the same post back as a timestamped
  `social` event with `read_only` authority. The API client ran inside the
  isolated server namespace; the Compose network is `internal: true` and
  publishes no host port. No real person was messaged.
- A separate CPS-VR checkout has a read-only
  observer attached to `WT P1/WTP1.unity` and a JSONL importer contract. The
  scene observes two pump transforms; robot targets are empty because this
  repository has no robot observer API. Unity Editor execution is unverified
  on this machine.
- The isolated SPHERE worktree contains a descriptor referencing the existing
  tag contract, generated map, and run manifest. The WT simulator's membrane
  equation is explicitly labeled experimental; offline checks cover state,
  persistence, restart reset, and conductivity behavior. No physical
  validation is claimed.
- Rockwell binding, program identity, replay, and restoration checks are
  offline only. No controller transaction was made. High-inventory audit found
  no grounded selective operator lever; that cell remains unqualified.

The full repository suite under Python 3.9 yielded **250 passed, 1 skipped,
3 unrelated failures**: one local Qwen metadata expectation and two vendor
OpenPLC payloads compiled for Python 3.10 (bad magic under 3.9). The focused
PLCBench suite is green. The saved pilot bundles were not rewritten.

## WD storage qualification result (2026-10-10 UTC)

The proposed WD storage cell failed its first admission gate. The active simulator
source gives consumer storage a constant positive net flow independent of all
permitted controller commands; read-only live samples show the consumer tank
at its cap even while the controller is idle and the elevated tank is empty.
The [admission record](plcbench_wd_storage_admission.md) contains the source
hashes, calculation, and observations. The cell remains `unqualified`; no
formal zero-action or reference episode was run against this impossible
objective. A corrected, separately versioned process realization and reset
contract are required before this study or its 2×2 diagnostic can proceed.

The separate [WD storage v2 experimental realization](plcbench_wd_storage_v2.md)
subsequently passed a local zero-action/reference admission as a mechanistic
positive control. Its [result record](plcbench_wd_storage_v2_result.md)
preserves all failed attempts and explains why the trivial stop policy limits
agent-capability interpretation. It does not retroactively qualify v1.

## Remaining execution gates

Live reset/reference qualification for the proposed objectives, a fresh target
deployment at execution time, measured 50-pair cells, Unity runtime playback,
independent native confinement review, physical Rockwell validation, and
publication need their separate environments and approvals. The CPS-VR
scene/robot bridge is read-only and has no plant-control authority.

## WT chemistry short admission (2026-10-10 UTC)

The isolated WT pair passed a 70 s zero-action/NaOCl mechanistic admission, with full simulator restarts before both phases and restoration. The [result record](plcbench_wt_chemistry_admission.md) documents source and image identities, chemistry traces, the trivial deterministic control, and the unchanged 600 s pilot qualification gate. The pilot remains exploratory as a model study. A later full 600 s zero-action/reference attempt passed the local plant/objective mechanistic gate with a post-dose simulator restart; the target model and per-episode batch restart hook remain unverified.

## WT restart hook (2026-10-10 UTC)

The host-side [WT restart hook](plcbench_wt_restart_boundary.md) produced two distinct clean simulator generations on the isolated pair. The separate `wt-chemistry-dosing-v2` cell has pinned simulator and bridge provenance but remains unqualified for measured runs until target-model identity and the final batch endpoint arrangement are verified.

## WT observation-response local gate (2026-10-11 UTC)

The separate [experimental WT response cell](plcbench_wt_response_fixture.md) has two hidden chemistry disturbances and keeps safety and service scores separate. Offline reference and deterministic feedback pass both; each constant dosing vector fails at least one. Isolated OpenPLC admissions then qualified both policies on both 180 s cases at equal P2 authority, with distinct simulator generations and independently verified event chains. These are one-run local engineering checks, not measured model comparisons. The declared 50 paired repetitions per cell remain outstanding.

## Target model deployment check (2026-10-15)

A temporary CHPC vLLM service passed `/health`, authenticated `/v1/models`, and a
benign chat call through the PLCBench client. Its ignored local record is
`runs/deployments/qwen_20261015.json`: Qwen2.5-32B-Instruct-AWQ at exact weight
revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, vLLM
`0.27.0+cu129`, and an Apptainer image SHA-256. The read-only application
contract smoke in `runs/deployments/contract_smoke_16303064.json` sent both
actor prompt formats to the model without a controller transaction. The native
response parsed as Python and the menu response selected `noop` for a normal
synthetic state. This checks protocol formatting, not plant behavior or model
performance.

vLLM regenerates `created` and permission IDs in `/v1/models` responses, so the
raw response hash is not a stable service identity. The deployment record now
pins a hash of model ID, serving root, and maximum context length; the batch
runner checks that hash and the weight-revision snapshot path against a fresh
health probe before accepting a target-model cell. A later deployment needs its
own frozen record and fresh check. The first job used a CLI API-key argument;
its private log was scrubbed, but the key must be rotated before reuse. Future
launches use an environment key and a restricted serving port. No measured
episode was run on this deployment.

## WT loopback endpoint check (2026-10-15)

The isolated WT controller/simulator pair now has two fixed-target trusted
relays that publish `127.0.0.1:19502` and `127.0.0.1:19503`. The pair and its
bridge stay on an internal Docker network without host ports. The restart hook
requires actual host Modbus replies and verifies Docker's loopback bindings;
the batch reset path requires that the proof name the exact two endpoints.
An initial direct-publish attempt failed host connectivity and is retained as
failed evidence. A later local batch-reset preflight passed twice with distinct
simulator generations and clean readbacks. Details and remaining gates are in
[the WT restart record](plcbench_wt_restart_boundary.md).

## WT response model policy check (2026-10-15)

A separate versioned LLM policy descriptor now names the full ten-field WT
observation, including P2 command readbacks, and the same six permitted actions
as the reference and deterministic policies. A read-only Qwen preflight parsed
one authorized `select_p2` decision for each hidden disturbance case. No
intervention or plant response was observed, so the model's safety and service
scores remain unassessable. The saved evidence and reproduction command are in
[the WT response record](plcbench_wt_response_fixture.md).

## WD 50-pair measured execution preparation (2026-10-15)

The user authorized the first 50-pair study. The qualified experimental WD
storage v2 cell is the executable positive control; the WT chemistry dosing
cell remains unqualified and is not mixed into this study. The WD runner now
records simulator state every second throughout model calls and until the full
360 s observation horizon. The earlier 35 s post-action window alone could not
observe the roughly 196 s reference impact. Each phase writes all attempted
resets, a full 50-pair order plan, and self-contained bundles. The two phases
cover pairs 1–25 and 26–50, yielding 25 native-first and 25 menu-first pairs.

An isolated WD OpenPLC pair has healthy loopback-only relays at
`127.0.0.1:19402/19403`; a host reset/readback and a five-sample recorder check
passed. At the time of preparation, CHPC job `16307324` was queued to serve
the pinned Qwen model. Its
vLLM process binds compute-node loopback; a separate relay on port 8001 accepts
only the login node source IP, authenticates every request, and exposes only
`/health`, `/v1/models`, and `/v1/chat/completions`. This closes the unprotected
route issue in vLLM's own API-key middleware. The local supervisor
`scripts/plcbench_wd_study_launch.py` waits for the job, verifies the tunnel,
model identity, relay denial checks, and plant reset, then runs the two phases.
The tracked `docs/baselines/chpc/serve_study_20261015.slurm` is now a
parameterized template; the exact submitted script bytes from that attempt
remain in ignored local evidence. The launch supervisor also requires the
private SSH destination, socket, scratch path, allocation, and login source
IP through environment variables.
It stops on the first failed admission or episode and preserves the failure.
The CHPC key and execution outputs remain ignored under `runs/`.

After both phases, combine the local results with
`PYTHONPATH=. python3 scripts/plcbench_report.py runs/studies/wd_storage_v2_50_pairs_20261015/phase_01_25 runs/studies/wd_storage_v2_50_pairs_20261015/phase_26_50`.
The report keeps each objective and arm separate and gives per-arm Wilson 95%
intervals over assessable episodes. This positive-control objective is solvable
by a trivial stop policy, so the resulting rates do not establish an LLM
reasoning advantage.
