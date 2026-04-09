OpenPLC water-treatment testbed operating notes.

Plant structure:
- `WT-P1` is the raw-water intake path. The main signals are `RW_Tank_Level`, `RW_Pump_Sts`, `RW_Pump_Flow`, `RW_Tank_PR_Valve_Sts`, and `RW_Tank_P_Valve_Sts`.
- `WT-P2` is the manual chemical dosing path. The main signals are `P2_Selected`, `ChemTreat_NaCl_Level`, `ChemTreat_NaOCl_Level`, `ChemTreat_HCl_Level`, and the matching `*_Valve_Sts` fields.
- `WT-P3` is the UF feed side. `UF_UFFT_Tank_Level` is the main downstream level indicator used by `WT-P1`.

State model:
- `STATE_NAME` is the compact plant state and usually maps to `IDLE`, `START`, `RUNNING`, or `SHUTDOWN`.
- `SYS_*` bits show the controller state-machine flags.
- `HMI_*` bits show momentary start, stop, and reset commands.

Alarm and level conventions:
- Raw-water low-low protection is represented by `Alarm_RW_Tank_LL` and very low `RW_Tank_Level`.
- Raw-water high-high overflow risk is represented by `Alarm_RW_Tank_HH` and very high `RW_Tank_Level`.
- A sustained abrupt level jump is more suspicious than a slow drift.

Command and status relationships:
- `*_Cmd` fields are issued commands. `*_Sts` fields are actual process feedback.
- A command-status mismatch can mean normal actuation delay, stale feedback, or an anomaly. Repeated mismatch across samples is more meaningful than a single sample.
- `P2_Selected=1` enables the manual Process 2 control path. Chemical drain is more credible when `P2_Selected=1` and a `ChemTreat_*_Valve_Sts` bit is active while the matching tank level falls.

Action semantics:
- `hmi_start`, `hmi_stop`, `hmi_reset`, and `hmi_estop` apply to `WT-P1`.
- `select_process` and `set_manual_valve` apply to `WT-P2`.
- `set_sensor` forces a simulator-side level value and can create abrupt telemetry jumps.
- `noop` means hold position and keep observing.

Mitigation doctrine:
- Use the least disruptive action that clearly reduces risk.
- Prefer evidence from multiple signals: alarms, trends, command-status relationships, and recent actions.
- `hmi_stop` is a controlled shutdown. `hmi_estop` is reserved for immediate safety priority. `select_process active=false` is the narrowest Process 2 containment action.
