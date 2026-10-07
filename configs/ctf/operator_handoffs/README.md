# CTF operator handoff bindings

These are **consumer bindings** for Operation Alpine Shield, not new utility
source models. Canonical tags and ownership remain in `sphere-usecases` at the
recorded revision. The two fictional operator nodes, deadlines, permitted
evidence age, and regional dependency narrative are CTF-authored choices.

`hydro_power.yaml` records a pre-reset equipment-status handoff. The hydro
source says reset depends on gate, breaker, and speed conditions, but the
controller remains the authority for the actual reset. The independent
sync-check relay is not a workflow shortcut or an attacker-controlled control.

`water_distribution.yaml` records a grid-status review before proposing
service restoration. The water-distribution contract exposes modeled tank,
valve, and pump feedback, but it does not define a field worker or an HMI
restoration command; those roles are teaching abstractions. Both maps use
simulator observations, **not** independent physical field measurements.

The generic `ics_agent.operator_handoffs` tracker records request,
acknowledgement, evidence-backed completion, expiry, and rejected completion.
It never issues a PLC command or decides physical safety. A handoff completion
means the declared evidence was delivered on time from the declared modeled
path; an independent guardrail/controller still assesses its values, authority,
and service consequence. A missed, late, stale, or wrong-source check remains
visible in the event log.

To cross-check the copied tag names against a local `sphere-usecases` checkout:

```sh
python3 -B -m ics_agent.operator_handoffs_check /path/to/sphere-usecases
```

This mapping does **not** connect hydro output to water pumps. The regional
power-to-water condition remains a labeled, frozen CTF input until a separate
composed model is executed and qualified.
