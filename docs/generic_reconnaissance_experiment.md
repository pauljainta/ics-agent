# Generic-Knowledge Reconnaissance Experiment

This runner prepares the 39-scenario reconnaissance experiment over the current OpenPLC water testbeds without changing the existing attacker/defender implementations.

The attacker uses the current Qwen 3.5 profile family with generic ICS cyber knowledge only. Target-specific plant semantics are withheld through the `no_target_semantics` target-system context setting, normalized observation (`norm-O`), and normalized interface (`norm-G`). The defender and observer use silent scripted no-op profiles, so they do not detect, contain, shut down, or otherwise alter the run.

The same runner also supports a capability-progression extension that keeps the silent defender and current Qwen policy fixed while adding three attacker capability profiles. This extension tests whether target-system context, operational grounding, and runtime observation reduce the attacker action cost of reaching critical impact.

## Scenario Scope

- Water Treatment legacy scenarios: 18 scenarios from `testbeds/water_treatment/scenarios/attack`.
- Water Treatment state-contingent scenarios: 6 scenarios from `testbeds/water_treatment/scenarios/stateaware`.
- Water Distribution legacy scenarios: 11 scenarios from `testbeds/water_distribution/scenarios/attack`.
- Water Distribution state-contingent scenarios: 4 scenarios from `testbeds/water_distribution/scenarios/stateaware`.
- Total full manifest: 39 scenarios.

## Infrastructure Contract

The runner enforces fixed XDC mapping and does not fall back to discovery:

- Water Treatment uses `example-xdc-09`.
- Water Distribution uses `example-xdc-06`.

Concrete SPHERE/MergeTB deployment coordinates must come from ignored local infrastructure config, for example `configs/private/sphere.local.yaml` or the path named by `ICS_AGENT_SPHERE_CONFIG`. Do not put private hostnames, IP addresses, reservations, or credentials in tracked configs.

## Outputs

By default, outputs are organized under:

```bash
Artifact/Results/generic_reconnaissance_qwen/seed_101
```

The capability-progression extension writes new profile outputs under:

```bash
Artifact/Results/generic_reconnaissance_qwen_capability_progression/seed_101
```

The three new profile subfolders are `full_kt_norm_g_norm_o`, `full_kt_high_g_norm_o`, and `full_kt_high_g_high_o`. The original no-context baseline remains in `Artifact/Results/generic_reconnaissance_qwen/seed_101`; the progression runner does not copy, overwrite, or rerun it.

The runner writes:

- `_manifests/seed_<seed>/generic_reconnaissance_manifest.json`
- `_manifests/seed_<seed>/generic_reconnaissance_manifest.csv`
- `_generated_configs/seed_<seed>/<testbed>/*.yaml` when `--write-configs`, `--preflight-only`, or `--execute` is used
- per-trial core artifacts from the existing orchestrator
- per-trial recon audit artifacts:
  - `attacker_visible_trace.jsonl`
  - `evaluator_private_trace.jsonl`
  - `recon_event_trace.jsonl`
  - `knowledge_acquisition.json`
  - `critical_impact_evidence.json`
  - `action_accounting.json`
  - `timing_metrics_recon.json`
  - `recon_audit_summary.json`
- aggregate files:
  - `recon_scenario_level_results.csv`
  - `recon_scenario_level_results.json`
  - `recon_aggregate_summary.csv`
  - `recon_aggregate_summary.json`

The audit files keep attacker-visible records separate from evaluator-private records. The post-processor never invents hidden chain-of-thought, unlogged candidate sets, or unavailable critical-impact attribution.

## Capability Profiles

- Profile 0, existing baseline: `no_target_semantics_norm_g_norm_o`. No target-system semantics, normalized operational grounding, normalized runtime observation, and silent defender.
- Profile 1: `full_kt_norm_g_norm_o`. Full target-system context, normalized operational grounding, normalized runtime observation, and silent defender.
- Profile 2: `full_kt_high_g_norm_o`. Full target-system context, high operational grounding, normalized runtime observation, and silent defender.
- Profile 3: `full_kt_high_g_high_o`. Full target-system context, high operational grounding, high runtime observation, and silent defender.

`Full Kt` reuses the existing target-semantic ablation `full` variant. `Norm-G` uses `normalized_interface`. `High-G` reuses the existing high-grounding mechanism for each suite: legacy scenarios use `legacy_white_interface`, while state-aware scenarios use `grounded_interface`. `Norm-O` uses `normalized_observation`; `High-O` uses `high_observation`. These switches affect attacker-visible context and candidate presentation, not the underlying allowed action types or validator.

## Non-Executing Validation

Prepare and inspect the full manifest without live SPHERE calls:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --dry-run --write-configs
```

Plan the three new profiles without rerunning the baseline:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py \
  --profile all-new \
  --dry-run \
  --write-configs
```

This plans exactly `39 scenarios x 3 profiles = 117` future episodes. It does not load Qwen, run XDC preflight, reset the plant, or execute attack actions.

Validate only one testbed or family:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --testbed wt --family legacy --dry-run --write-configs
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --testbed wd --family stateaware --dry-run --write-configs
```

Run non-mutating MRG/XDC auth preflight only:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --preflight-only
```

Preflight the capability-progression configs without executing episodes:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py \
  --profile all-new \
  --preflight-only
```

## Future Live Execution

Only run live experiments when the infrastructure is confirmed ready:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --execute
```

Run the three new profiles only after explicit approval:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py \
  --profile all-new \
  --execute
```

The runner has scenario-level resume because generated configs set `resume: true`. If a live run is interrupted, rerun the same command with the same seed and output root.

Aggregate existing outputs without running anything:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --aggregate-only
```

Aggregate the three new profile roots without running anything:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py \
  --profile all-new \
  --aggregate-only
```

Generate the four-profile comparison against the existing baseline:

```bash
.venv/bin/python scripts/run_generic_reconnaissance_experiment.py --compare-only
```

Comparison outputs are written under:

```bash
Artifact/Results/generic_reconnaissance_qwen_capability_progression/seed_101/_comparison
```

The comparison writer produces `profile_summary.csv`, `profile_summary.json`, `per_scenario_comparison.csv`, `per_scenario_comparison.json`, `paired_action_costs.csv`, `timing_comparison.csv`, `impact_comparison.csv`, and `comparison_report.md`.

## Metrics and Counting

The runner records:

- executed attacker actions
- attempted attacker actions when blocked-action events are present
- reconnaissance actions
- effect-capable attacker actions
- blocked actions
- first effect-capable action time
- effective containment cutoff time if any
- critical-impact evidence and attribution source

The capability-progression comparison treats action cost as the primary endpoint. Overall action averages are reported alongside impact counts because a failed short run is not evidence of a more efficient successful attack. Paired action-cost outputs are scenario-level, and common-success interpretation should be done only after verifying comparable endpoints.

For legacy scenarios, critical impact uses the existing metrics layer when attacker attribution is present. If only a generic safety-violation counter is available, the audit marks the attribution caveat instead of claiming hidden certainty. For state-contingent scenarios, critical impact follows the evaluator-defined false-to-true critical predicate transition when that evidence is present in the artifacts.

## Caveats

- The default dry run does not load the model and does not connect to SPHERE.
- `--preflight-only` checks MRG/XDC reachability but does not reset the plant or issue attack actions.
- The live runner preserves the existing PLC/controller dynamics, action validation, scenario expectations, and evaluator logic.
- Budgets are recorded in generated configs and enforced where the existing profile/orchestrator supports those fields.
