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

`examples/` contains **authored offline traces**, not captured plant runs:

- `hydro_checked.yaml`: evidence delivered before a reset recommendation;
- `hydro_skipped.yaml`: a recommendation made before any acknowledgement;
- `water_distribution_echo.yaml`: an HMI echo rejected as equipment feedback,
  followed by a mistaken recommendation anyway.

Replay a trace without any model, network, or controller connection:

```sh
python3 -B -m ics_agent.operator_handoffs_replay \
  configs/ctf/operator_handoffs/hydro_power.yaml \
  configs/ctf/operator_handoffs/examples/hydro_skipped.yaml
```

The JSON records a SHA-256 digest of the mapping and scenario inputs, and
distinguishes evidence delivery from the requester's recommendation.
`recommend_operation` is an **operator statement in the simulation**, not an
actuator command or authorization. `verified_current` means only that declared
evidence was delivered from declared modeled paths within the authored age
limit; it is not a safety verdict. The example values and timing are teaching
fixtures; they do not establish safe thresholds or realistic human error rates.

To cross-check the copied tag names against a local `sphere-usecases` checkout:

```sh
python3 -B -m ics_agent.operator_handoffs_check /path/to/sphere-usecases
```

This mapping does **not** connect hydro output to water pumps. The regional
power-to-water condition remains a labeled, frozen CTF input until a separate
composed model is executed and qualified.
