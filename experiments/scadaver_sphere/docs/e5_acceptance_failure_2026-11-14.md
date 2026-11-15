# E5 deployed acceptance failure and safety recovery

Date: 2026-11-14 (America/Denver)

## Verdict

The one freshly authorized E5 simulator action was attempted exactly once and
failed its restoration-verification gate. It was not retried. The persistent
emergency-stop latch is active, the controller was recovered through the
operator safety path, and the testbed is healthy. The integrated E1-E5
acceptance and Qwen experiment remain blocked.

## Pre-action gates

All required gates passed before the attempt:

- The broker was fixed to `198.18.4.20:1503`, coil `0`, one action, and `0.25 s`.
- OpenPLC was active with one core and ports `8080`, `43628`, and `1502` live.
- The legacy controller was inactive.
- A no-actuation quiesce/resume rehearsal closed and restored port `1502`.
- All seven TCP services, three Modbus endpoints, OPC UA, S7, and the genuine
  closed loop passed.
- The simulator baseline and timestamps were captured.

## Attempt and failure

The broker reached restoration verification and raised
`BrokerDenied: restoration verification failed; emergency stop activated`.
The deployed implementation restored the captured inlet and level, then
required exact equality for inlet, level, and derived flow. The simulator
continues to advance once per second while OpenPLC is quiesced, so exact level
equality is not a physically stable restoration criterion. The same failure
branch also lacked a detailed audit write and left OpenPLC quiesced.

No automatic retry occurred. Because the deployed runner raised before writing
`result.json`, the acknowledged and during snapshots were not retained and a
valid independent E5 effect assessment cannot be reconstructed.

## Safety recovery

The emergency-stop file was created. Read-only inspection found OpenPLC
inactive, legacy inactive, and simulator scanning. The established recovery
procedure resumed OpenPLC without another simulator write. Final checks showed:

- OpenPLC active and Modbus `1502` open.
- Legacy controller inactive.
- Simulator scan progression and safe-range values.
- All seven services healthy.
- OpenPLC lifecycle, 12-sample closed loop, OPC UA, and S7 passed.

## Engineering correction

> Superseded: the initial one-scan tolerance described below was subsequently
> replaced by the explicit exact-snapshot and operational-recovery models in
> `e5_pipeline_repair_report_2026-11-14.md`. It is retained here only as a
> chronological record of the immediate failure response.

The broker now permits at most one documented physical scan (`12` raw level
units) between restoration write and read-back while still requiring exact
inlet and derived-flow restoration. Larger drift remains fail-closed. A
restoration mismatch now records before/restored evidence and attempts
controller recovery while retaining the emergency-stop latch. The E5 runner
also writes a failure `result.json`. These fixes were deployed without clearing
the latch or issuing another action.

Regression result: `71 passed` for `tests/test_scadaver_sphere.py`; focused
E5/lifecycle tests: `10 passed`. The deployed read-only fault/security suite
also passed all `15/15` cases after recovery.

## Evidence

Local evidence is preserved under
`Artifact/Results/scadaver_sphere/e5_acceptance_20261114_final_failed/`:

- `e5-controller-recovery-preflight.json`
- `e5-failure-recovery.json`
- `e5-post-failure-health.json`
- `scadaver-action-emergency-stop`
- `acceptance_summary.json`

The authorization is consumed. Any future deployed action requires operator
review, explicit emergency-stop clearance, and a new authorization.
