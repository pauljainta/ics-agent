# Water Distribution Seeded Reset-Correction Pass

- Date: `2026-08-06`
- Scope: seeded water-distribution black-box and gray-box correction only
- Canonical result root: `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b`

## Bug Identified

- The SPHERE soft-reset path bypassed the remote testbed reset path that clears remote sensor override state.
- The deployed simulator could continue integrating stale or background runtime state immediately after reseed, so the first recorded seeded observations no longer matched the authored scenario initial conditions.
- That produced the stale `t=0` contamination pattern in the seeded black-box and gray-box families.

## Code Paths Changed

- `ics_agent/testbeds/water_distribution.py`
  - remote reset now clears remote overrides, primes a simulator hold state, and zeroes simulator-side command inputs before reseeding
  - remote `hmi_start` releases the simulator hold, and remote `hmi_reset` reasserts it
- `ics_agent/water_distribution_remote_runtime.py`
  - simulator now skips process evolution on seed-apply scans
  - simulator now holds seeded levels until the hold coil is released
- `ics_agent/environment.py`
  - seeded SPHERE reset now validates the post-reset initial observation against authored scenario values and fails loudly on mismatch
- `ics_agent/orchestrator.py`
  - the validated post-reset observation is recorded as the first `t=0` observation
- `ics_agent/water_distribution_seeded_backbone.py`
  - cached environment validation now requires an explicit passed initial-state validation payload
- `scripts/run_water_distribution_seeded_resetfix_rerun.py`
  - helper driver for rerunning only the corrected seeded black-box and gray-box families without touching the paper repo

## Initial-State Validation

- Validation fields: `Supply_Tank_Level`, `Grid_Elev_Tank_Level`, `RWS_Tank_Level`
- Behavior on failure: raise, abort the run, and log expected vs observed values in the run error path
- Behavior on success: emit an `initial_state_validated` system event before `wargame_t0`

## Targeted Sanity Checks

- Black-box sanity scenario: `wd-attack-grid-low-spoof`
- Gray-box sanity scenario: `wd-attack-grid-oscillation-spoof`
- Sanity result root: `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b_sanity_20260806`
- Result:
  - both runs passed initial-state validation
  - the first recorded observations matched the authored scenario initial conditions
  - the stale pre-existing `t=0` safety-violation pattern did not appear in the sanity runs

## Branch Cleanup And Rerun

- Deleted seeded result branches:
  - `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b/black_box_attacker__agentic_defender`
  - `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b/gray_box_attacker__agentic_defender`
- Left untouched:
  - `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b/white_box_attacker__agentic_defender`
  - all water-treatment roots
- Full corrected rerun status as of `2026-08-06`:
  - launched via tmux session `wd_seeded_resetfix_20260806`
  - driver: `scripts/run_water_distribution_seeded_resetfix_rerun.py`
  - destination: `Artifact/Results/water_distribution_openplc/seeded_backbone_qwen35_9b`

## Security

- No secrets or credentials were committed in this pass.
