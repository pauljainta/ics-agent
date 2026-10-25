# WT observation-response engineering cell

This is a separate, experimental WT study cell. It does not change the saved pilot bundles or their original 3 s warning / 10 s impact objective. The base simulator comes from the pinned SPHERE WT Structured Text, but the two disturbances and their initialization are PLCBench additions. They are not physically validated chemistry. The results are local engineering qualifications, not a model comparison or paper result.

## Offline gate

`configs/plcbench/wt_observation_response_fixture_v1.json` freezes two hidden cases: source pH 8.1 with ORP 280 mV, and source pH 7.5 with ORP 200 mV. Each policy sees the same P2 chemistry measurements, valve status, running status, and its own command readbacks, and can write only P2 select, HCl, and NaOCl commands. The source-derived state update uses the pinned P2 mixing, valve delay, and experimental membrane equation. It scored all 180 one-second decisions with separate safety and final-60-second service windows. This reduced-order runner holds the P4 dechlorination valve off, matching the saved full pilot reference trace; it does not simulate the complete plant.

The offline evidence is local at `runs/qualifications/wt_response_fixture_v1/attempt_3/`. Both the reference and deterministic feedback policies passed safety and service in both cases. All four constant HCl/NaOCl command vectors failed at least one case. Each trace records observation, decision, authorized intervention, and resulting state. The `summary.json` SHA-256 is `200412e486d0e2326e35266faef6bbda83c737460f2c350db706efd820d54c2a`. An LLM result is **unassessable** because no model decisions were made.

A separate replay compared the source-derived NaOCl trajectory with the saved 600-second isolated OpenPLC pilot trace: maximum absolute ORP residual was 1.56 mV and permeate-conductivity residual was 0.0033 µS/cm over 601 samples. This checks the implementation of those equations for that condition; it does not validate the experimental membrane model against a physical system.

## Isolated OpenPLC realization

`testbeds/water_treatment/openplc/st/wt_simulator_response_v1.st` derives from the pinned SPHERE simulator and adds four fixture-only coils at 240–243. An apply edge selects one of the two cases and initializes its source and P2 chemistry state. It does **not** reset `membrane_damage`; each episode requires a new simulator process generation and a clean sensor readback. The fixture selector and acknowledgement are excluded from the bridge map and actor observations/actions. The controller still owns only the three P2 command coils. `configs/plcbench/wt_observation_response_openplc_v1.json` identifies this realization separately from the offline fixture.

The isolated Compose pair has no host ports and uses an internal network. `scripts/plcbench_wt_response_openplc_probe.py` checks fixture acknowledgement, case-specific sensor readback, controller command readback, and timestamped observation → decision → intervention → plant-response events. `scripts/plcbench_wt_response_openplc_admission.py` restarts the simulator before each case and records its `StartedAt` identity, source hashes, logs, traces, and distinct safety/service scores. The local runner does not perform physical controller transactions.

The finalized live descriptor completed as local `runs/qualifications/wt_response_openplc_v1/attempt_2/` on 2026-10-10 UTC. Each case had a separate simulator `StartedAt` generation and a clean baseline (pH 7.5, ORP 200 mV, permeate conductivity about 1.5 µS/cm), followed by the expected fixture readback. The standalone assessor verified 180 observation → decision → intervention → response chains per case and recomputed the scores from saved traces.

| Case | First fixture pH / ORP | Safety score | Final-60-second service score |
| --- | --- | ---: | ---: |
| High pH | 8.1 / 280 mV | 1.0 | 1.0 |
| Low ORP | 7.5 / 200 mV | 1.0 | 1.0 |

The independent `assessment.json` SHA-256 is `b4891068cd6b33368184f63d93e7a480d1160d6db9806b6ade69e5eee8fc872f`. Its `all_pass=true` qualifies the two deterministic local fixture episodes only. A subsequent reference-policy run is described below. No LLM decisions were made, so an LLM verdict remains **unassessable**. None of these traces establishes physical chemistry validity or an advantage over the deterministic policy.

## Reference-policy local gate

A separate local `attempt_3` used the **same live descriptor hash** (`ad77a2cb3af591616d18bf7f9cde92c4baed579db340f7940843123d3a190aef`) with the reference feedback policy. It uses the same P2 authority and pH threshold as the deterministic policy; its ORP on/off thresholds are 255/315 mV rather than 260/320 mV. Both cases used new simulator generations and passed the independent event-chain, reset, timing, safety, and service checks. The reference `assessment.json` SHA-256 is `949accf5942f04c953bf32c59a137f2a943a6ca2d13ffacef671455b0e40782e`.

| Case | Reference safety / service | Deterministic safety / service | Maximum reference / deterministic sample gap |
| --- | ---: | ---: | ---: |
| High pH | 1.0 / 1.0 | 1.0 / 1.0 | 2.118 / 1.216 s |
| Low ORP | 1.0 / 1.0 | 1.0 / 1.0 | 1.224 / 1.214 s |

The high-pH reference trace had one retried Modbus read. Its 2.118 s maximum gap passed the assessor's 2.5 s local integrity limit; the observed timing is in the comparison record. The two probe source hashes differ only by a corrected explanatory docstring. The comparison SHA-256 is `8f9b3b0844da0c252cba51b8d6e601cb9ef61e099560d2a296e0fc46be4a1c42` at local `runs/qualifications/wt_response_openplc_v1/reference_vs_deterministic_v2.json`.

This is **one unrandomized run per policy and case**. It is an engineering gate, not a 50-pair study, a causal comparison, or evidence of an LLM result.

## Target-model policy interface preflight (2026-10-15)

The separate `configs/plcbench/wt_response_llm_policy_v1.json` pins the offline
fixture hash and explicitly lists all ten fields the working feedback policies
receive, including the three P2 command readbacks. Its six action IDs exactly
match the reference and deterministic policies' permitted P2 authority. The
`LLMResponsePolicy` adapter rejects missing observations, malformed responses,
and actions outside that authority; it never substitutes an action for an
invalid model response.

Using the health-checked Qwen2.5-32B deployment, the read-only preflight sent
one initial observation for each hidden case. Both replies parsed as the
authorized `select_p2` action. The local record is
`runs/qualifications/wt_response_llm_preflight_v1/20261015_v2/preflight.json`
(SHA-256 `d56e3ea5411c073f4200b624675637b305f370e0368b876d2e2d23be961b0c1c`).
It records the prompts, replies, policy and source hashes, and frozen model
identity. No action was applied to a plant, so safety and service outcomes for
the LLM remain **unassessable**. The earlier deterministic/reference evidence
and its config identities are unchanged.

With a fresh, pinned Qwen deployment, repeat the contract check from this
worktree with a new output directory:

```sh
python3 scripts/plcbench_wt_response_llm_preflight.py --sphere-root /path/to/sphere-usecases --deployment-json /path/to/pinned-deployment.json --model-url http://127.0.0.1:8000 --api-key-file /path/to/private-key --out-dir runs/qualifications/wt_response_llm_preflight_v1/new_attempt
```

Reproduce the offline gate and local OpenPLC gate from the PLCBench worktree:

```sh
python3 scripts/plcbench_wt_response_fixture.py --sphere-root /path/to/pinned/sphere-usecases --out-dir runs/qualifications/wt_response_fixture_v1/new_attempt
python3 scripts/plcbench_wt_response_openplc_admission.py --sphere-root /path/to/pinned/sphere-usecases
python3 scripts/plcbench_wt_response_openplc_assess.py runs/qualifications/wt_response_openplc_v1/attempt_N
python3 scripts/plcbench_wt_response_openplc_admission.py --sphere-root /path/to/pinned/sphere-usecases --policy reference
python3 scripts/plcbench_wt_response_openplc_assess.py runs/qualifications/wt_response_openplc_v1/attempt_M
python3 scripts/plcbench_wt_response_openplc_compare.py --reference runs/qualifications/wt_response_openplc_v1/attempt_M --deterministic runs/qualifications/wt_response_openplc_v1/attempt_N
```

Each output directory or numbered attempt is new; evidence files are never overwritten. The OpenPLC qualification requires the local `sphere-openplc:master` and `plcbench-native:py311-pymodbus386` images. The target Qwen/vLLM deployment, paired 50-repetition measurements, and physical validation remain separate gates.
