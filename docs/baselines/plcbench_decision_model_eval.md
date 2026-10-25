# System One decision-model trial for PLCBench

For the pinned full D3 CHPC service, see [the deployment runbook](chpc/D3_SYSTEMONE.md).

Status: **single-run offline engineering trial completed 2026-10-21**. The
[CHPC result record](chpc/d3_trial_20261021.md) reports 180 D3 decisions in
each of two fixed WT cases, with an independently checked observation →
decision → intervention → response chain. Safety/service scores were 1.00/1.00
for high pH and 1.00/0.95 for low ORP. These two fixed cases do not estimate
performance differences or establish experienced-operator realism. The endpoint
binding to served weights was recorded but not cryptographically attested.
This is a local PLCBench-inspired fixture, not an author-supplied PLCBench
artifact. The WT realization is experimental, source-derived software behavior;
it is not a physically validated plant.

The decision model receives the same ten observations, six P2 actions, safety envelope,
service target, one-second decision interval, and clean state per case as the existing
WT response policies. Its System One `choice` question asks for one action; the adapter
checks the complete probability vector and action ID before the offline fixture can
apply the action. No live controller or broker address is reachable from this runner.
The hidden case name is absent from the request. A one-step preflight has no safety or
service verdict; only a full trajectory produces separate scores and the verified
observation → decision → intervention → response trace.

## Inspect the exact requests

The pinned SPHERE source used by this fixture is the separate
`codex/plcbench-sphere` worktree. A newer SPHERE checkout has a different ST hash
and fails the source gate as intended.

```sh
python3 scripts/plcbench_wt_response_systemone.py \
  --sphere-root /path/to/pinned/sphere-usecases \
  --out-dir /private/tmp/plcbench-systemone-export --mode export
```

Open either `*.systemone.json` in DecisionUI's editable JSON request panel. DecisionUI
is a view for requests and probabilities; the saved evidence and scoring stay in this
runner. The UI source reviewed for this contract is
[`nsssayom/decisionUI`](https://github.com/nsssayom/decisionUI) at
`b4e10937501595cabeafcf1c5f3f77e5e6263a1f`. Its core request uses
`POST /v1/systemone` and its model inventory uses `GET /v1/models`.

## Query a local decision endpoint for a new trial

Use a System One endpoint serving the exact model ID declared in
`configs/plcbench/wt_response_systemone_policy_v1.json`. If the student used a
different D3 variant, copy this policy to a new versioned identity and change its
model ID before running. Keep API keys in a file outside the bundle; the runner records
the model inventory, endpoint, policy/fixture/source hashes, request hashes, complete
probability vectors, and action IDs, but never the key.

```sh
python3 scripts/plcbench_wt_response_systemone.py \
  --sphere-root /path/to/pinned/sphere-usecases \
  --out-dir /private/tmp/plcbench-systemone-preflight --mode preflight \
  --endpoint http://127.0.0.1:8080
```

`--mode full` runs both 180-decision case trajectories and emits separate safety and
service scores. A full run needs a recorded weight/revision and server/runtime identity
in addition to the `/v1/models` response before it can be used as a measured model
comparison. The adapter rejects missing/extra options, nonfinite or unnormalized
probabilities, and a reported choice that disagrees with the largest probability.
There is no deterministic action substitution after a malformed model response.

For any later paired comparison, freeze model identity, presentation, observation set,
action authority, time budget, and source/reset checks per cell. Compare D3 to the
reference, deterministic, constant, and Qwen policies within the same WT objective;
do not pool the WD storage objective or treat a binary classification success as a
plant-control result. Record failed calls and reset failures as outcomes.
