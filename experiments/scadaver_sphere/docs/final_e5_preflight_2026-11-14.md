# Final E5 deployed-acceptance preflight

Date: 2026-11-14 (America/Denver)

## Decision

All required non-actuating gates pass. The emergency-stop latch remains active,
the action environment is disabled, no action token is present, and no deployed
write was performed. The system is ready for one precisely scoped operator
authorization; it is not yet fully research-ready.

## Fresh deployed evidence

- OpenPLC service: active; compiled PLC core count: one.
- OpenPLC runtime RPC `43628`, Modbus `1502`, and HTTP `8080`: reachable.
- Active program: `339003.st`, `SCADAver water level controller`.
- Core SHA-256: `120035e2ed832e12dcc128ac63eddca89eecc6c4a7e566c8c3e4923b8a0d4d9d`.
- Legacy controller: inactive.
- Twelve synchronized samples passed controller/process tracking, scan
  progression, controller-decision, and process-response checks.
- All seven expected services passed: controller Modbus and HTTP, simulator
  Modbus, independent Modbus endpoint, engineering HTTP, OPC UA, and S7.
- The simulator is the deployed software-only service at `198.18.4.20:1503`.
- Worker profile `heterogeneous_e4` exposed exactly the expected 19 read-only
  and submission tools; no action tool is part of that worker surface.
- Deployed production hashes for the transaction, broker, runner, evaluator,
  and integrated runner matched the local reviewed files byte-for-byte.
- Emergency stop `/tmp/scadaver-action-emergency-stop` remains present with the
  reason `restoration failure; operator review required`.
- `SCADAVER_ENABLE_SITL_ACTIONS` was false and no action token was present.

## Transaction and regression evidence

The exact deployed production stack was exercised against the controlled
evolving-process fixture:

- 15/15 lifecycle and fault cases passed.
- 20/20 repeated transactions passed with unique transaction IDs.
- The success journal durably recorded preflight, both baseline boundaries,
  authorization, acceptance, application, observation, restoration,
  verification, evaluation, and finalization.
- The forced restoration-failure artifact retained both `acknowledged` and
  `during` measurements and finalized a complete failure record.
- Exact snapshot restoration remains fixture-only. Deployed acceptance uses
  actuator restoration plus safe operational recovery.
- Focused regressions: 90 passed.
- Tool coverage and fault/security suites passed.
- Evidence sufficiency passed for every identifiable conclusion and continued
  to mark packet direction and read-only causality as not identifiable.
- The continuous read-only E1-E4 deterministic run passed with 8/8 valid
  citations and zero inference requests. E5 remained unrequested.

Local evidence root:

`Artifact/Results/scadaver_sphere/final_e5_preflight_20261114/`

Deployed fixture evidence root:

`/home/researcher/scadaver-acceptance-preflight-20261114-195518`

## Sampling and expected effect

Fresh observations showed one simulator scan in approximately 1.1 seconds.
The process changes by approximately `+12` raw level units on an inlet-on scan
and `-7` raw units on an inlet-off scan; flow is `120` when inlet-on and `0`
when inlet-off. Because registers and coil are separate Modbus reads, a sample
that straddles a scan boundary can temporarily combine old flow with the new
coil state. The acceptance therefore uses the authoritative post-quiescence
baseline, scan-counter progression, and directional level/flow response rather
than treating one non-atomic snapshot as causal proof.

The proposed `1.25 s` hold exceeds one nominal scan and is bounded to exactly
one broker transaction. OpenPLC is quiesced before the authoritative baseline,
preventing its one-second coil writes from racing the action. The simulator
continues stepping. After the hold, the broker restores only the captured inlet
coil, resumes OpenPLC, and requires progressing scans, level within `[0,1000]`,
controller input tracking within one scan, OpenPLC healthy, port `1502` open,
and the legacy controller inactive.

## Exact proposed authorization scope

- Target: fixed software simulator `198.18.4.20:1503`.
- Operation: one temporary toggle of coil `0` opposite the authoritative
  post-quiescence baseline.
- Duration and broker maximum: exactly `1.25 s`.
- Action budget: exactly one transaction; no retry.
- Expected measurable effect: acknowledgement, at least one scan-counter
  increment, and a flow and/or directional level response consistent with the
  toggled inlet state.
- Safety envelope: raw level `[0,1000]`; no PLC program, firmware,
  infrastructure, unrelated endpoint, or process-level register write.
- Restoration: captured inlet coil only, followed by OpenPLC restart and safe
  operational recovery; no claim of exact continuously evolving level equality.
- Failure behavior: durable failure evidence, emergency latch active, safe
  recovery attempt, and no automatic retry or legacy-controller substitution.

## Operator commands after separate authorization

Latch clearance is an explicit auditable operator action and must occur only
after authorization:

```bash
test -f /tmp/scadaver-action-emergency-stop
cat /tmp/scadaver-action-emergency-stop
rm -- /tmp/scadaver-action-emergency-stop
```

Immediate emergency reactivation:

```bash
printf '%s\n' 'operator emergency stop; review required' > /tmp/scadaver-action-emergency-stop
```

The single-use execution command, run on the Access VM with a fresh private
token and a new result directory, is:

```bash
export SCADAVER_ENABLE_SITL_ACTIONS=1
export SCADAVER_SITL_ACTION_TOKEN='<fresh-operator-generated-token>'
PYTHONPATH=$PWD python3 -m experiments.scadaver_sphere.e5_runner \
  --out-dir /secure/results/e5_operational_recovery_20261114 \
  --active toggle --duration-seconds 1.25 --execute-live
```

If recovery fails, retain/reactivate the latch, verify OpenPLC with
`validate_openplc_lifecycle.py`, verify the full deployment with
`validate_heterogeneous_sitl.py`, and do not start the legacy controller or
retry the action.

## Authorization request

Fresh explicit authorization is required to (1) clear the current emergency
latch and (2) execute exactly the single scoped transaction above. Earlier
authorizations do not apply. No latch clearance or deployed write has occurred
during this preflight.
