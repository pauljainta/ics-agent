Defender-side prior knowledge for the water-treatment blue team.

Process semantics:
- `RW_Tank_Level` is the raw-water inventory signal for `WT-P1`.
- `P2_Selected=1` enables Process 2 manual control and expands the chemical-dosing attack surface.
- `ChemTreat_*_Valve_Sts=1` plus falling `ChemTreat_*_Level` implies active manual chemical movement.

Safety thresholds and alarm meaning:
- Treat `RW_Tank_Level <= 450` as the raw-water low protection band.
- Treat `RW_Tank_Level >= 1180` as the raw-water high-high danger band.
- `Alarm_RW_Tank_LL` means raw-water low-low protection has been breached.
- `Alarm_RW_Tank_HH` means high-high raw-water behavior is present and can justify immediate containment.

Suspicious defender-side patterns:
- Unplanned `P2_Selected=1` is suspicious.
- Unplanned Process 2 manual valve activity is stronger evidence.
- Falling chemical inventory during unplanned Process 2 use is strong containment evidence.
- Abrupt tank-level jumps or repeated command-status mismatch can indicate telemetry or actuator integrity problems.

Mitigation doctrine:
- Prefer `select_process active=false` first when the incident is isolated to Process 2 manual dosing.
- Escalate from Process 2 deselection to `hmi_stop` when Process 2 evidence persists after prior containment or when multiple chemical paths are affected.
- Prefer `hmi_stop` for raw-water low protection issues and most telemetry-integrity incidents with active alarms.
- Reserve `hmi_estop` for immediate high-high overflow danger or cases where a controlled stop is too slow.

Escalation doctrine:
- Do not wait through a full reasoning cooldown once a matched high-risk pattern is present.
- If prior containment was attempted and the same dangerous pattern persists, escalate to the next stronger mitigation.
