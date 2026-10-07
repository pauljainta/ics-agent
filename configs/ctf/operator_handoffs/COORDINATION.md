# CTF handoffs alongside `feat/plcbench-baseline`

Reviewed the local PLCBench worktree through `a169071` on 2026-10-07. That branch
already owns the Ollama and OpenAI-compatible/vLLM model backends, native
actor sandbox and broker, six-flag scorer, run harness, and distribution
live-plant binding and Modbus map work. Its local checkout is ahead of `origin/main`; this CTF
work does not edit that worktree or duplicate those components.

This branch owns only:

- a non-actuating request/acknowledgement/evidence-completion tracker;
- CTF-authored operator-role bindings for hydro/power and distribution;
- authored offline replays of checked, omitted, and wrong-source handoffs;
- read-only checks of tag names and tag ownership against a pinned
  `sphere-usecases` revision.

The actor/model backend is independent of operator handoffs. For a future
joined experiment, the runner should convert observed controller/simulator
samples into `EvidenceSample` values with their actual observation timestamps
and declared source paths, then record handoff events in the run bundle.
Do not synthesize a field observation by copying the command or an HMI echo.
Do not let a handoff-complete event directly call the PLC broker: completion
does not establish that values are safe, that an actor is authorized, or that
the modeled service outcome is acceptable.
The replay module intentionally imports no plant binding, broker, or model
backend; joining it to a live run is future integration work.

The branch's WD process-object bindings and proposed impact thresholds remain
its own experimental work. The CTF handoff map only identifies evidence needed
for an operator review, and the fictional power-to-water dependency remains a
frozen scenario input rather than a live feeder. There is no hydro actor
binding in `feat/plcbench-baseline` as reviewed, so no hydro live-agent claim
follows from this map.

Before combining branches, keep the `ics_agent.operator_handoffs` API and
`configs/ctf/operator_handoffs/` data distinct from PLCBench scoring and
broker authority. Re-run the source-map checker after any source revision or
tag-contract change. Ollama `llama3.1` can be a local pipeline smoke; model
hosting on Cisco or CHPC is a separate capacity and deployment decision, not
an assumption in the handoff model.
