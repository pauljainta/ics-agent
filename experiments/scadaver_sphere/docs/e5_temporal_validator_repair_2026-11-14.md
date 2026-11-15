# E5 temporal preflight validator repair

Date: 2026-11-14 (America/Denver)

## Verdict

The temporal-attribution defect is repaired and the repaired read-only
validator is deployed. Five consecutive deployed preflights passed during
normal OpenPLC threshold switching, with 55/55 confirmed single-scan intervals
and no inconsistent or indeterminate intervals. All seven services remained
healthy.

This validates the preflight, not deployed E5 actuation. The emergency latch
remains active, actions remain disabled, no action token is present, no
simulator write was issued, and prior authorization was not reused. A fresh
single-use authorization is still required.

## Precise root cause

The simulator's update order is:

1. read inlet coil `0`;
2. update level by `+12` when on or `-7` when off;
3. write level, flow (`120`/`0`), and scan counter together;
4. sleep approximately one second.

OpenPLC executes its IEC task every `100 ms`, polls the simulator through the
Modbus-master path, and writes coil `0` according to the `450` threshold. The
old validator sampled at one-second intervals and attributed
`level[N+1]-level[N]` to the coil observed in sample N. OpenPLC could change the
coil after sample N but before simulator scan N+1, making that state stale.

The aborted trace itself proves the issue: every level change matched the flow
stored with the completed destination scan, while several changes did not
match the previous sample's coil. The prior failure was therefore a temporal
attribution error, not a stopped simulator or failed controller.

## Repaired evidence model

`classify_process_response()` now emits exactly one of:

- `CONSISTENT`: at least the configured number of complete, consecutive
  single-scan intervals agree with the model and none contradict it.
- `INCONSISTENT`: at least one complete single-scan interval has sufficient
  evidence and contradicts the model.
- `INDETERMINATE`: evidence is insufficient because of skipped scans, missing
  values, non-monotonic timing, duplicate scans, or an unknown applied flow.

The process-generated flow register is the applied-inlet evidence for each
completed scan because flow, level, and scan counter are written in one
simulator update. The separately read coil remains useful transition evidence,
but is never projected backward or forward into an unobserved interval.

The collector polls every `0.2 s`, retains one observation per completed scan,
records read start/end timestamps, and stops after 12 distinct scans or a
bounded 16-second deadline. A gap of more than one scan is indeterminate rather
than guessed. At least three confirmed single-scan intervals are required.
The level tolerance is one raw unit; larger discrepancies fail as inconsistent.

## Deterministic test evidence

The production classifier was tested for:

- stable inlet on and stable inlet off;
- switching immediately before and immediately after a scan;
- multiple unobserved transitions;
- delayed/missing process response;
- measurement jitter inside and outside the explicit tolerance;
- missing, delayed, and non-monotonic samples;
- genuine model contradictions;
- clamped physical boundaries;
- normal threshold oscillation;
- the exact previously aborted 12-sample trace;
- 20 controlled evolving-process runs with randomized controller switching.

Results:

- exact aborted trace: `CONSISTENT`, 11/11 confirmed intervals;
- injected anomalies: `INCONSISTENT`;
- ambiguous gaps/missing data: `INDETERMINATE`;
- randomized controlled fixture: 20/20 passed;
- focused SCADAver/E5/temporal suite: 105 passed;
- compile and `git diff --check`: passed.

## Deployed read-only evidence

Five consecutive runs of the production validator on the Access VM each
reported:

- `process_response_outcome=CONSISTENT`;
- 11 confirmed intervals;
- zero inconsistent intervals;
- zero indeterminate intervals;
- both inlet states observed;
- scan progression true;
- all seven TCP/service checks true;
- overall return code zero.

Artifacts:

`Artifact/Results/scadaver_sphere/temporal_validator_repair_20261114/temporal-validator-live-20261114/`

The prior validator is preserved on the Access VM as:

`validate_heterogeneous_sitl.py.pre-temporal-fix-20261114`

## E5 and integrated regression

- E5 transaction/failure matrix: 15/15 passed.
- Repeated isolated E5 transactions: 20/20 passed with unique IDs.
- Continuous deterministic E1-E5 with real deployed read-only services and a
  controlled fixture-only E5 action: `overall_success=true`.
- Tool coverage: passed.
- Fault/security acceptance: passed.
- No Qwen/OpenRouter requests and no deployed writes.

Artifacts:

`Artifact/Results/scadaver_sphere/temporal_validator_repair_20261114/`

## Final state

- OpenPLC active; compiled core count one.
- Legacy controller inactive.
- Modbus `1502`, runtime RPC `43628`, and HTTP `8080` healthy.
- Emergency latch active.
- Action environment disabled.
- Action token absent.

## Fresh single-action proposal

This proposal is not authorization:

- target: fixed software simulator `198.18.4.20:1503`;
- operation: one toggle of inlet coil `0` opposite the authoritative
  post-quiescence baseline;
- duration and broker maximum: exactly `1.25 s`;
- budget: one transaction, no retry;
- evidence: durable baseline, authorization, acknowledgement, completed-scan
  flow/level/counter response, restoration, recovery, evaluation, and final
  artifact;
- restoration: captured inlet only, then resume OpenPLC and require safe
  progressing scans plus controller/process tracking;
- failure: preserve evidence, activate/retain emergency stop, recover safely,
  never substitute the legacy controller, and do not retry.

Fresh explicit authorization must separately permit latch clearance and this
single transaction. A passing repaired preflight must not be represented as a
passing deployed E5 acceptance.
