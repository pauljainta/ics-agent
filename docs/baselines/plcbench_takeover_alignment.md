# PLCBench engineering alignment

The current `ics_agent/plcbench_*` implementation is a locally built,
PLCBench-inspired reference. Its provenance is the `ics-agent` Git history, not
an identified author artifact. PLCBench arXiv v1 describes a separate, planned
artifact release; this checkout has no verified import commit or license for
author code. Record any later author artifact by repository, revision, license,
and changed files before calling a condition a reproduction.

## Equivalence and deviation record

| Axis | Paper-described condition | Current local condition | Classification |
| --- | --- | --- | --- |
| Code origin | Author framework and planned artifact snapshot | Local `ics_agent/plcbench_*` commits on `feat/plcbench-baseline` | Inspired implementation; author code not verified |
| Workloads | Four closed-loop S1–S4 workloads | WD run-stop; WT run-stop and ChemTreat excursion | New workload/objectives |
| Controller | Four commercial PLCs and a SoftPLC reference | Local OpenPLC controller/simulator pairs | Software realization only |
| Model | Five paper-level model families, exact deployment in manifests | Saved pilot: Ollama `qwen3-coder:30b`, temperature 0, 512-token ceiling; later Qwen/vLLM protocol smoke and full D3 single-run offline fixture have separate identities | Stand-in pilot and distinct engineering trials; do not pool |
| Initial state and reset | Verified per-episode controller and process reset | Saved pilot had driver-level run reset and chemistry flush; new runner requires a fresh simulator generation and zero-damage proof for WT dosing | Pilot limitation; new contract still needs live qualification |
| Observation and authority | Native tools and declared process feedback | Menu arm receives named candidates; native arm runs free-form code via broker | Bundled interface and information differences |
| Episode budget | 100 actions, 3600 s | 8–10 actions, 600 s; additional settle window | Adapted |
| Evaluation | Six flags from independent evidence | Six local flags from transaction log and modeled simulator trace | Similar rule shape; different realization |
| Incomplete evidence | Admission, reset and invalidation records | Saved pilot has 30 bundles; objective is absent from each `meta.json`; exceptions may omit episodes | Identity and denominator gaps |

The saved 30-episode pilot is an exploratory interface diagnostic. Neither the
menu result nor its six-flag table is a matched causal estimate or an author
reproduction. A menu-only deterministic `stop` policy is implemented as a
pilot control in `ics_agent/plcbench_study.py`; it has not been run as a
measured cell.

## Reuse boundary

- PLCBench engineering reuses the six-flag scorer, broker, local run bundles,
  and OpenPLC observer from this repository.
- SPHERE qualification supplies source/program identity, `tag_contract.yaml`,
  the generated backend map, and run-manifest evidence through a small adapter.
- CTF fixtures may reuse admission, reset, and evidence validation, but no
  actor-private menu, hidden answer, model result, or offensive metric becomes
  a grading dependency.

The first handoff case is a benign offline trace: a normal running state with
no controller writes must produce no disruption or impact. It is exercised by
`tests/test_plcbench_takeover.py::test_benign_offline_evidence_case` and
requires no controller or model runtime.
