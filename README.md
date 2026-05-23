# ICS Agent

`ics-agent` is now the config-driven experiment and agent orchestration repo for
the OpenPLC water-treatment wargaming stack in this workspace.

It reuses the working host-side OpenPLC testbed from the sibling
`sphere-usecases` repo, but moves experiment control, scenario metadata,
agent-role swapping, metrics, reporting, and results storage into this repo.

## Current Status

As of 2026-04-15, the attack-only agentic defender refresh has been
implemented and revalidated end to end.

- Repo/package naming is intentional:
  the git repo is `ics-agent/`, while the Python package inside it is
  `ics_agent/`. There is no separate required `ics-agnt` package in the
  runtime path.
- Fresh verified attack-only agentic artifacts live under:
  `../Artifact/Results/full_matrix/20260415T083154-133ee0fb/agentic_agents/attack/`
- Fresh verified attack summary:
  `18/18` attack scenarios completed, `10/18` passed, blue-team detection
  fired in `16/18`, defender-side detection fired in `14/18`, defender
  interventions executed in `14/18`, containment landed before a safety
  violation in `14/18`, `8/18` scenarios recovered, and aggregate metrics were
  written to
  `../Artifact/Results/full_matrix/20260415T083154-133ee0fb/attack_metrics_summary.json`
- Key fixes applied during verification:
  live observer output is now handed to the defender in the same loop cycle,
  defender reasoning cooldown is bypassed on materially new evidence,
  persistent incident memory tracks `suspected -> confirmed -> containing ->
  stabilized`, defender action candidates are contextualized around telemetry
  and alarms, Process 2 and raw-water doctrine is stronger, a narrow bounded
  last-mile guardrail can issue obvious containment without replacing the
  Qwen/LangGraph path, and observer/defender/intervention metrics are split
  cleanly

## 2026-05-21 Formalized Agentic Refresh

The current working tree now formalizes the live roles around:

`observe -> update structured role memory -> choose bounded candidate action -> act or hold -> record rationale`

Key implementation changes in this refresh:

- the OpenPLC environment remains autonomous and external to the agents
- attacker, observer, defender, and planner prompts now use concise live state
  plus structured memory instead of replaying long transcripts
- the practical black-box attacker now has bounded `inspect_surface`
  reconnaissance actions for tags, registers, I/O image, alarm bits,
  actuator-state bits, and Process 2 control-plane inspection
- attacker memory tracks objective progress, tactic phase, leverage
  hypothesis, prior action effects, reconnaissance findings, and inferred
  defender attention
- observer memory tracks anomaly hypotheses, evidence tokens, suspicion level,
  and recommended attention target
- defender memory tracks incident hypothesis, severity, confidence, lifecycle
  stage, prior mitigations, containment effectiveness, and escalation need
- nominal Process 2 operations and nominal refill-response behavior now have
  explicit context guards so they do not get mislabeled as attacks
- metrics now separate nominal and attack semantics and expose canonicalized
  trajectories for honest repeated-run diversity measurement
- attacker success is now scored from actual safety-bound violation / unsafe
  state achievement rather than authored scenario `passed`, while the old
  scenario-expectation signal remains available separately

Fresh verified normal-suite artifacts for this refresh:

- fresh result root for this rerun:
  `../Artifact/Results/qwen35_9b_formalized_agentic_pair/`
- root:
  `../Artifact/Results/qwen35_9b_formalized_agentic_pair/formalized_agentic_normal/20260522T041232-0a543781/`
- results:
  `10/10` nominal scenarios completed and passed, `0` false-positive
  episodes, `0` false-intervention episodes, and `0` nominal safety
  violations

Attack and repeated-run experiment note:

- the GPU-backed local Qwen path is working and the formalized attack suite
  starts cleanly without backend/parser failure
- fresh attack-family outputs for this rerun are written under
  `../Artifact/Results/qwen35_9b_formalized_agentic_pair/formalized_agentic_attack/`
  and
  `../Artifact/Results/qwen35_9b_formalized_agentic_pair/practical_black_box_attack_pairs/`
  instead of being overwritten

## 2026-05-23 Practical Black-Box Attacker Refresh

The practical black-box attacker is now explicitly biased toward
`recon -> choose surface -> commit -> sustain/escalate -> reorient only on stall`
instead of flat one-step probing.

Key behavior changes in this refresh:

- bounded `inspect_surface` reconnaissance is treated as a small budget rather
  than a default loop, and repeated reads of the same surface are sharply
  down-ranked
- structured attacker memory now keeps discovered controls, reconnaissance
  findings, an observed effect map, candidate leverage hypotheses, explicit
  exploit commitment state, and simple precondition state
- commitment is surface-based and short-horizon: once a surface is chosen, the
  attacker is pushed to stay on it for several exploit steps unless observed
  effect stalls
- sensor manipulation now supports staged escalation and sustain behavior
  rather than a single conservative bias level
- attacker success metrics continue to treat actual safety-bound violation as
  the primary success signal while preserving action counts, recon counts,
  trajectories, and the older scenario-expectation signal separately
- pair folders now write dedicated aggregate summaries alongside the run
  folders, including `condition_summary.json`,
  `attack_metrics_summary.json`, `nominal_metrics_summary.json`, and
  `trajectory_metrics_summary.json`

The current full `18 x 3` attack-pair rerun writes to:

- `../Artifact/Results/qwen35_9b_formalized_agentic_pair/practical_black_box_attack_pairs/20260523T041616-10d187ec/`

Expected pair folders under that experiment root are:

- `practical_black_box_attacker__agentic_defender`
- `practical_black_box_attacker__rule_based_defender`
- `practical_black_box_attacker__scripted_defender`

Verified completed outcome for that rerun:

- all `54/54` attack episodes completed and wrote `metrics.json`,
  `alerts.json`, `actions.json`, and `summary.json`
- all three pair folders wrote `condition_summary.json`,
  `attack_metrics_summary.json`, `nominal_metrics_summary.json`, and
  `trajectory_metrics_summary.json`
- experiment-level `experiment_summary.json`, `attack_metrics_summary.json`,
  `nominal_metrics_summary.json`, and `trajectory_metrics_summary.json` were
  written at the experiment root
- against the agentic defender, the redesigned attacker caused safety-bound
  violations in `4/18` scenarios with average recon usage reduced to `0.5556`
  actions per run
- against the rule-based defender, the attacker caused `0/18` safety-bound
  violations
- against the scripted defender, the attacker also caused `0/18`
  safety-bound violations in this rerun, so the exploit redesign improved
  commitment behavior and reduced recon dependence, but it did not yet
  translate into successful unsafe-state achievement against the weakest
  defense pairing

## Current Performance Note

Using the fresh 2026-04-15 attack-only `agentic_agents` batch and the current
stored scripted baseline, the current tradeoff is:

- `agentic_agents` is now stronger on blue-team detection coverage, defender
  coverage, and recovery count, and it reacts materially earlier on most
  detected scenarios
- `scripted_baseline` is still stronger on overall safety control and alert
  discipline, especially on raw-water alarm scenarios

Current attack-batch snapshot:

- `agentic_agents`: `16/18` detected, `14/18` defender-detected, `14/18`
  mitigated, `14/18` contained before a safety violation, `8/18` recoveries,
  average detection time about `0.38s`, average mitigation time about `0.61s`,
  `105` total safety violations, and `130` blue-team alerts
- `scripted_baseline`: `13/18` detected, `13/18` defender-detected, `13/18`
  mitigated, `6/18` recoveries, average detection time about `0.50s`, average
  mitigation time about `0.50s`, `50` total safety violations, and `26`
  blue-team alerts

So, as of now, the refreshed agentic defender is much better than the earlier
agentic batches at noticing attacks and taking action, but it is still not
better than the scripted defender overall when the goal is to keep the plant in
the safest state.

This comparison is useful but not perfectly apples-to-apples: the baseline is a
stored 2026-04-09 batch rather than a fresh same-code rerun, and attack-suite
`passed` is not a pure blue-team score because successful containment can still
leave attack-effect checks such as `must_change` or `ever_nonzero` unsatisfied.

## What Is In Scope

The repo now covers these AGENTS.md requirements in a reusable structure:

- end-to-end OpenPLC testbed orchestration through a shared environment adapter
- scripted online roles
- agentic role scaffolding for Qwen 3.5 with LangChain / LangGraph execution
- offline analysis roles operating only on stored run artifacts
- config-driven scenario catalog and experiment matrices
- combination-matrix execution with named experimental conditions
- metrics and report generation for each run
- workspace-level `Agentic_ICS/Artifact/` runtime assets kept outside the
  `ics-agent` git repo
- automated validation and pytest coverage for config loading and orchestration

## Repo Layout

- `ics_agent/`
  The Python package for runtime orchestration, environment integration,
  metrics, reporting, and role implementations.
- `configs/prompts/`
  Agentic prompt templates kept in files for reproducibility.
- `../Artifact/Configs/`
  Live runtime config files such as `agent_profiles.yaml` and
  `scenario_catalog.yaml`.
- `../Artifact/Experiments/`
  Live experiment launch YAMLs.
- `../Artifact/Scenarios/`
  Live scenario execution YAMLs grouped by suite.
- `tests/`
  Unit and orchestration tests.
- `../Artifact/Results/`
  Machine-readable and human-readable outputs for completed runs.

## Scenario Model

A scenario now has two layers:

1. `../Artifact/Scenarios/<suite>/*.yaml`
   This is the execution spec: duration, initial conditions, timeline, and
   expectations.
2. `../Artifact/Configs/scenario_catalog.yaml`
   This is the metadata schema: intent, prerequisites, attacker actions,
   defender responses, expected outcomes, success/failure conditions, metrics,
   safety bounds, and ground truth.

That split keeps scenario mechanics reusable while letting the same scenario be
run with scripted, agentic, or offline analyses.

## Agent Model

Profiles in `../Artifact/Configs/agent_profiles.yaml` define each role:

- `attacker`
- `defender`
- `observer`
- `planner`
- `operator`
- `offline_analysis`

Each role can be swapped at runtime by changing the experiment config, not code.

## Operational Knowledge

Agentic profiles can now preload reusable water-treatment manuals before the
run starts. These manuals live under `configs/knowledge/` and are attached per
role through `../Artifact/Configs/agent_profiles.yaml`.

- `shared_testbed_manual.md`
  Common process semantics, tags, actions, alarms, and mitigation doctrine.
- `defender_manual.md`
  Blue-team intervention guidance for Process 1 shutdowns and Process 2
  containment.
- `observer_manual.md`
  Alerting and anomaly-triage guidance.
- `operator_manual.md`
  Safe operator-side playbook execution guidance.
- `attacker_manual.md`
  Action-surface guidance for adversarial playbook steps.
- `offline_analyst_manual.md`
  Stored-artifact interpretation guidance for post-run analysis.

These manuals are operational references, not hidden scenario keys. The live
agent context still omits future timeline steps, hidden ground-truth labels, or
scenario-specific answer keys from defender, observer, planner, and operator
prompts.

## Attack-Only Improvements

The attack-only `agentic_agents` path was reworked in these areas:

- blue-team live context now rebuilds between observer and defender passes, so
  same-cycle observer alerts reach the defender immediately
- blue-team context construction now exposes active alarms, commands, state
  relationships, short-horizon trends, recent alerts, incident state, and
  derived risk flags
- defender reasoning cooldown is now event-driven and is bypassed on new
  observer alerts, attacker actions, active alarms, worsening risk flags, and
  repeated suspicious Process 2 evidence
- defender incident state now persists across turns and escalates through
  `suspected -> confirmed -> containing -> stabilized`
- defender candidate actions are now contextual and evidence-linked instead of
  being limited to generic stop / estop / deselect suggestions
- a narrow bounded defender guardrail can issue last-mile containment when the
  agentic path clearly recognizes a dangerous incident but fails to emit the
  obvious safety action; the primary decision path remains Qwen + LangChain /
  LangGraph
- defender prompt and manual doctrine now make Process 2 deselection,
  controlled stop, and estop expectations more explicit without leaking hidden
  scenario ground truth
- structured-output parsing accepts stricter JSON while still recovering from
  common fenced, YAML-like, or `<think>`-wrapped backend responses
- diagnoses are normalized into short labels so observer and defender metrics do
  not get polluted by free-form sentences
- attack metrics now separate observer recognition/detection, defender
  recognition/detection, blue-team alerts, planner alerts, and defender
  intervention timing / containment instead of blending them into one score

Implemented role families:

- `scripted`
  Deterministic scenario-plan or rule-based behavior
- `agentic`
  Qwen-oriented role wrapper that uses a strict GPU-backed LangChain /
  LangGraph decision flow, generic action-space construction, contextual memory
  windows, and explicit backend enforcement so agentic profiles fail instead of
  silently degrading into heuristics
- `offline`
  Same role interface, but executed only after a run from stored artifacts

## Running

Validate the catalog and configs:

```bash
python3 -m ics_agent.cli validate
```

List scenarios:

```bash
python3 -m ics_agent.cli list-scenarios
```

Experiment names are resolved from `../Artifact/Experiments/`.

Run the core smoke matrix:

```bash
python3 -m ics_agent.cli run smoke_core_matrix.yaml
```

Bootstrap the local Qwen runtime for the all-agentic path:

```bash
bash scripts/setup_env.sh
bash scripts/download_qwen_model.sh
QWEN_MODEL_PATH=./models/local_models/qwen3.5-9b .venv/bin/python -m ics_agent.cli run smoke_full_agentic.yaml
```

Run the verified attack-only all-agentic experiment:

```bash
QWEN_MODEL_PATH=./models/local_models/qwen3.5-9b .venv/bin/python -m ics_agent.cli run attack_full_agentic.yaml
```

The latest verified attack-only agentic output is the 2026-04-15 run
`20260415T083154-133ee0fb`:

```text
../Artifact/Results/full_matrix/20260415T083154-133ee0fb/agentic_agents/attack/
```

Compared with the most recent matching scripted baseline run
`20260409T080533-2789bcab`, this fresh agentic attack-only run improved pass
coverage, blue-team detection coverage, defender detection coverage, defender
action coverage, and recovery outcomes, but it is still not better than the
scripted baseline overall because safety performance regressed. Fresh aggregate
results were `10/18` passed, `16/18` detected, `14/18` defender-detected,
`14/18` mitigated, `14/18` contained before a safety violation, `8/18`
recoveries, and `105` total safety violations. The scripted baseline was
`7/18` passed, `13/18` detected, `13/18` defender-detected, `13/18`
mitigated, `6/18` recoveries, and `50` total safety violations.

Remaining caveats: pass count on the attack suite is partially driven by
attack-effect checks, so early containment can still leave a run marked
`passed: false`; `wt-attack-p2-hcl-drain` still produces excessive alert
volume; `wt-attack-p2-select-chatter` still repeats Process 2 deselection; and
raw-water scenarios `wt-attack-sensor-rw-cascade-low-hh` and
`wt-attack-sensor-rw-hh-alarm` still accumulate `53` and `52` safety
violations before shutdown.

Run mixed observer / planner smoke combinations:

```bash
python3 -m ics_agent.cli run smoke_mixed_roles.yaml
```

Run the full combination matrix:

```bash
python3 -m ics_agent.cli run full_matrix.yaml
```

`Artifact/Experiments/full_matrix.yaml` is the fresh combined run: it executes all
scenario suites with three conditions:

- `scripted_baseline`
- `fully_agentic`
- `hybrid_blue_team`

Results are written under:

```text
../Artifact/Results/<experiment_name>/<experiment_id>/<condition>/<suite>/<scenario_id>/trial-001/
```

Each run directory includes:

- `observations.csv`
- `actions.json`
- `decisions.json`
- `alerts.json`
- `system_events.json`
- `offline_analysis.json`
- `metrics.json`
- `summary.json`
- `summary.md`
- `config_snapshot.yaml`
- `scenario_resolved.yaml`

## Replay / Offline Mode

Replay mode lets offline agents operate only on saved logs.

Workflow:

1. Run a live experiment and choose one run directory.
2. Copy or link that run directory into `../Artifact/Results/replay_seed/<name>/`.
3. Run:

```bash
python3 -m ics_agent.cli run smoke_offline_replay.yaml
```

## Testing

Run:

```bash
pytest tests -q
```

The test suite validates:

- scenario catalog loading
- metrics calculations
- fake-environment orchestration
- replay-mode execution

## Notes

- No sensitive paths or secrets are hardcoded into tracked source.
- Prompt templates, role choices, and scenario metadata live in files.
- The OpenPLC environment adapter preserves a stable role interface while
  swapping implementation families.
- Agentic roles require a CUDA-capable GPU and do not fall back to CPU or
  heuristic decision policies.
