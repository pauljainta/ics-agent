# CTF live-agent operator pilot

This is a staff-only qualification slice for the fictional CTF utility setting. It
reuses the existing online attacker and SPHERE/OpenPLC environment. It does not
deploy a student instance or authorize actions against physical controllers.

Run configuration: `ctf_water_treatment_operator_pilot.yaml`.

## Role model

- `attacker`: existing gray-box online agent, limited to its configured action
  surface. The pilot scenario provides no future attack timeline.
- `operator`: deterministic rule-based stand-in for a remote human operator. It
  starts a routine run and, if the displayed raw-water level falls below 450
  while the controller reports RUNNING, records that it did not request an
  independent check. That threshold is an authored trigger for this pilot, not
  a value sourced from the RoviSys FDS or a calibrated human-behavior model.
- `defender` and `observer`: silent profiles because the existing orchestrator
  requires these role slots. They do not detect or contain the attack. A student
  team or a separately specified guardrail may take the defense role later.

Operator decisions carry `actor_kind: simulated_human_operator` and a workflow
role. The missed-check event is a note, not evidence that a real person saw an
alarm. These traces must never be presented as measured human behavior. The
proposal's human/agent workflow-node idea motivates the representation; this
pilot does not implement PRISM's inference or guardrail claims.

## Qualification

Offline validation:

```sh
python3 -B -m pytest -q -p no:cacheprovider tests/test_ctf_operator_pilot.py
python3 -B -m ics_agent.cli validate --testbed water_treatment
```

Staff may run `python3 -B -m ics_agent.cli run
configs/experiments/ctf_water_treatment_operator_pilot.yaml` only after
checking the configured SPHERE realization, isolation, reset path, action
allowlist, model/CUDA availability, and log destination. Do not run it against
an occupied or shared experiment.

An end-to-end pass must show an agent proposal, action validation, controller
result, process observation, operator decision, and reset as separate events.
It must also show that an operator cannot write a sensor or perform a
security-defender action. Repeat the same initial state to check stability.

This water-treatment pilot is not the three-utility CTF. Hydro/power and
distribution now have source-checked, CTF-authored operator handoff maps and
offline replays under `configs/ctf/operator_handoffs/`; they are not connected
to this pilot or to an action broker. Their live observation/action bindings,
regional dependency inputs, team isolation, concurrency, student views, and
Take 1/Take 2 grading remain separate release gates. In particular, this
pilot's omitted verification note is not yet a timed handoff in a live run.
