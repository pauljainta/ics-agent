# Final E5 acceptance: preflight abort

Date: 2026-11-14 (America/Denver)

## Outcome

The authorized deployed transaction was **not executed**. The repeated
read-only preflight failed its mandatory closed-loop
`process_response_consistent` predicate, so execution stopped before emergency
latch clearance, token provisioning, controller quiescence, or any simulator
write. No retry was attempted.

## What passed

- OpenPLC active with one compiled PLC core.
- Legacy controller inactive.
- Runtime RPC `43628`, Modbus `1502`, and HTTP `8080` reachable.
- All seven expected industrial/application services reachable.
- Controller input and simulator level synchronized within one scan.
- Controller decisions included both inlet states.
- Simulator scan counter advanced.
- Simulator temporal observations changed.
- OPC UA, S7, and all three Modbus endpoints responded.

## Failed predicate and diagnosis

The twelve-sample trace showed the expected simulator increments (`+12` while
inlet-on and `-7` while inlet-off), but the validator computes each interval's
direction from the *previous* sample's coil state. OpenPLC can change that coil
between samples. In the captured trace, the current sample's flow and inlet
state consistently explained the arriving level change, while several changes
were scored against stale previous-sample state. The validator therefore
reported:

```json
{
  "synchronized": true,
  "controller_decisions_observed": true,
  "scan_progress": true,
  "process_response_consistent": false,
  "passed": false
}
```

This is a temporal-attribution defect in the safety validator. It does not
justify bypassing the failed gate or retrying until a favorable phase alignment
occurs.

## Preserved evidence and final state

Deployed evidence:

`/home/researcher/scadaver-lab/final-e5-aborted-preflight-20261114-2136/`

- `deployed_preflight.json` SHA-256:
  `2fb201bd6e93591984490c98e7dcaa96238cae0ea6d77d803d67d709a10bd7c9`
- `abort_summary.json` SHA-256:
  `a173f9b9dfd5928b0d764c264823a9d3fc0f9dda21d99d0b420bcea416538293`

Final state was independently checked:

- transaction executed: false;
- simulator write executed: false;
- action environment enabled: false;
- action token present: false;
- emergency latch: active and unchanged;
- OpenPLC: active;
- PLC core count: one;
- legacy controller: inactive;
- Modbus `1502`, runtime RPC `43628`, and HTTP `8080`: healthy.

## Readiness

**NOT FULLY READY.** Fixture and read-only E1-E4 validation remain successful,
but deployed E5 acceptance is still unproven. The closed-loop validator must be
corrected to use temporally valid actuator attribution (or synchronized
transition evidence), validated offline, and reviewed before requesting a new
single-use authorization. The authorization associated with this attempt must
not be reused automatically.
