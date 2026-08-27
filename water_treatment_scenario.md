# Water Treatment Testbed and Attack Scenario Catalog


## Process Structure

### WT-P1: Raw-Water Intake Path

`WT-P1` is the main start/stop process path. It represents the raw-water intake
and transfer behavior that most of the Process 1 attacks manipulate.

Key telemetry and state:

- `RW_Tank_Level`: raw-water tank level
- `RW_Pump_Sts`: raw-water pump status
- `RW_Pump_Flow`: raw-water pump flow
- `RW_Tank_PR_Valve_Sts`: refill / process response valve status
- `RW_Tank_P_Valve_Sts`: related valve status on the raw-water path
- `STATE_NAME`: compact controller state such as `IDLE`, `START`, `RUNNING`,
  or `SHUTDOWN`
- `Alarm_RW_Tank_LL`: low-low raw-water alarm / protection condition
- `Alarm_RW_Tank_HH`: high-high raw-water alarm / overflow condition

Operational meaning:

- Low raw-water level can block startup or trigger protective refill behavior.
- High-high raw-water level can indicate overflow risk.
- Repeated abrupt jumps in `RW_Tank_Level` are more suspicious than slow drift.

### WT-P2: Manual Chemical Dosing Path

`WT-P2` models manual chemical dosing and process selection. It is the main
surface used by the chemical drain and process hijack attacks.

Key telemetry and state:

- `P2_Selected`: whether Process 2 manual control is enabled
- `ChemTreat_NaCl_Level`: sodium chloride tank level
- `ChemTreat_NaOCl_Level`: sodium hypochlorite tank level
- `ChemTreat_HCl_Level`: hydrochloric acid tank level
- `ChemTreat_*_Valve_Sts`: manual valve status for each chemical
- `P2_*_Cmd`: command bits associated with Process 2 manual dosing

Operational meaning:

- Process 2 abuse is more credible when `P2_Selected=1`, a manual valve is
  active, and the matching tank level decreases.
- `select_process active=false` is the narrowest containment action for Process
  2 attacks.

### WT-P3: UF Feed / Downstream Dependency

`WT-P3` is represented primarily through downstream level state rather than a
full separate operator workflow in these scenarios.

Key telemetry:

- `UF_UFFT_Tank_Level`

Operational meaning:

- It acts as a downstream condition relevant to Process 1 permissives.
- High downstream level can block or disrupt Process 1 operation even if the
  raw-water path itself looks nominal.

## Action Surface

The shared environment exposes a bounded cyber-physical action space.

Plant control actions:

- `hmi_start`: start `WT-P1`
- `hmi_stop`: controlled stop of `WT-P1`
- `hmi_reset`: reset `WT-P1`
- `hmi_estop`: emergency stop of `WT-P1`

Process 2 actions:

- `select_process`: enable or disable `WT-P2`
- `set_manual_valve`: open or close a Process 2 chemical valve

Simulator-side manipulation:

- `set_sensor`: force a sensor value, typically to spoof a level signal

Passive action:

- `noop`: hold position and continue observing

## Safety Model

The global safety bounds used by the scenario catalog are:

- `RW_Tank_Level`: `0` to `1200`
- `UF_UFFT_Tank_Level`: `0` to `1200`
- `ChemTreat_NaCl_Level`: `0` to `1000`
- `ChemTreat_NaOCl_Level`: `0` to `1000`
- `ChemTreat_HCl_Level`: `0` to `1000`

Attacker success is evaluated from actual unsafe-state / safety-bound violation,
not only from authored scenario completion checks.

## Attack Scenario Families

The `18` attack scenarios fall into three broad families:

- HMI manipulation attacks against `WT-P1`
- Process 2 hijack and chemical-drain attacks against `WT-P2`
- Sensor spoofing attacks against `WT-P1` and the downstream UF dependency

## HMI Manipulation Attacks

### 1. `wt-attack-hmi-chatter`

- Target process: `WT-P1`
- Duration: `45s`
- Core idea: repeatedly alternate `hmi_start` and `hmi_stop`
- Sequence: start at `5s`, stop at `12s`, restart at `18s`, stop again at
  `25s`, restart at `31s`, final stop at `38s`
- Intended effect: create operator/control confusion and repeated plant state
  transitions without directly spoofing sensor values

### 2. `wt-attack-hmi-forced-estop`

- Target process: `WT-P1`
- Duration: `35s`
- Core idea: force an unauthorized emergency stop while the plant is running
- Sequence: `hmi_start` at `5s`, `hmi_estop` at `15s`
- Intended effect: abrupt availability disruption and immediate transition to
  `SHUTDOWN`

### 3. `wt-attack-hmi-forced-stop`

- Target process: `WT-P1`
- Duration: `35s`
- Core idea: force an unauthorized controlled stop during runtime
- Sequence: `hmi_start` at `5s`, `hmi_stop` at `15s`
- Intended effect: halt production in a less abrupt but still unauthorized way

## Process 2 Hijack and Chemical-Drain Attacks

### 4. `wt-attack-p2-dual-hijack`

- Target process: `WT-P2`
- Duration: `96s`
- Core idea: chain two chemical actions to create a multi-stage compromise
- Sequence: start `WT-P1`, enable `WT-P2`, open `NaCl`, close it, then open
  `NaOCl`, close it, deselect `WT-P2`, then stop `WT-P1`
- Intended effect: drain multiple chemical inventories while remaining within a
  plausible operator-style Process 2 workflow

### 5. `wt-attack-p2-hcl-drain`

- Target process: `WT-P2`
- Duration: `88s`
- Core idea: drain the hydrochloric acid tank through manual valve abuse
- Sequence: start `WT-P1`, enable `WT-P2`, open `HCl`, later close it, deselect
  `WT-P2`, then stop `WT-P1`
- Intended effect: unauthorized depletion of `ChemTreat_HCl_Level`

### 6. `wt-attack-p2-multi-chemical-drain`

- Target process: `WT-P2`
- Duration: `95s`
- Core idea: open three manual chemical valves in sequence
- Sequence: start `WT-P1`, enable `WT-P2`, open `NaCl`, then `NaOCl`, then
  `HCl`, later close them all, deselect `WT-P2`, then stop `WT-P1`
- Intended effect: broad multi-tank depletion with simultaneous chemical
  misuse

### 7. `wt-attack-p2-nacl-drain`

- Target process: `WT-P2`
- Duration: `85s`
- Core idea: drain the sodium chloride tank through Process 2 manual control
- Sequence: start `WT-P1`, enable `WT-P2`, open `NaCl`, later close it,
  deselect `WT-P2`, then stop `WT-P1`
- Intended effect: unauthorized depletion of `ChemTreat_NaCl_Level`

### 8. `wt-attack-p2-naocl-drain`

- Target process: `WT-P2`
- Duration: `88s`
- Core idea: drain the sodium hypochlorite tank through Process 2 manual
  control
- Sequence: start `WT-P1`, enable `WT-P2`, open `NaOCl`, later close it,
  deselect `WT-P2`, then stop `WT-P1`
- Intended effect: unauthorized depletion of `ChemTreat_NaOCl_Level`

### 9. `wt-attack-p2-process-hijack-estop`

- Target process: `WT-P2` with `WT-P1` availability impact
- Duration: `60s`
- Core idea: begin a Process 2 hijack and then force an emergency stop
- Sequence: start `WT-P1`, enable `WT-P2`, open `NaCl`, then issue
  `hmi_estop` at `24s`
- Intended effect: combine chemical misuse with abrupt plant shutdown

### 10. `wt-attack-p2-select-chatter`

- Target process: `WT-P2`
- Duration: `44s`
- Core idea: repeatedly toggle `P2_Selected`
- Sequence: start `WT-P1`, toggle Process 2 on/off multiple times between `8s`
  and `28s`, then stop `WT-P1`
- Intended effect: control confusion, distraction, and repeated Process 2
  enable/disable churn without necessarily draining a tank

## Sensor Spoofing Attacks

### 11. `wt-attack-sensor-rw-cascade-low-hh`

- Target process: `WT-P1`
- Duration: `50s`
- Core idea: spoof a low raw-water level and then a high-high level in the same
  run
- Sequence: start `WT-P1`, set `RW_Tank_Level=390` at `14s`, then
  `RW_Tank_Level=1210` at `24s`, then stop at `38s`
- Intended effect: force multiple controller reactions, including refill-like
  logic and overflow-style alarm behavior

### 12. `wt-attack-sensor-rw-hh-alarm`

- Target process: `WT-P1`
- Duration: `40s`
- Core idea: spoof a high-high raw-water level
- Sequence: start `WT-P1`, set `RW_Tank_Level=1250` at `15s`, then stop at
  `30s`
- Intended effect: trigger `Alarm_RW_Tank_HH` and an overflow-risk condition

### 13. `wt-attack-sensor-rw-low-block`

- Target process: `WT-P1`
- Duration: `35s`
- Core idea: force a low raw-water reading before startup
- Sequence: set `RW_Tank_Level=200` at `2s`, then attempt `hmi_start` at `5s`
- Intended effect: block transition into `RUNNING` by violating startup
  permissives

### 14. `wt-attack-sensor-rw-low-disrupt`

- Target process: `WT-P1`
- Duration: `40s`
- Core idea: spoof a low raw-water level during runtime
- Sequence: start `WT-P1`, set `RW_Tank_Level=200` at `15s`, then stop at
  `30s`
- Intended effect: disrupt an already running process and trigger
  low-level protection behavior

### 15. `wt-attack-sensor-rw-oscillation`

- Target process: `WT-P1`
- Duration: `46s`
- Core idea: oscillate the raw-water level between low and high values
- Sequence: start `WT-P1`, then set `RW_Tank_Level` to `420`, `930`, and `410`
  across the run, then stop at `34s`
- Intended effect: create repeated, conflicting process cues that can confuse
  defender diagnosis

### 16. `wt-attack-sensor-rw-refill-spoof`

- Target process: `WT-P1`
- Duration: `40s`
- Core idea: spoof a low raw-water level to induce unnecessary refill response
- Sequence: start `WT-P1`, set `RW_Tank_Level=400` at `15s`, then stop at `30s`
- Intended effect: make the controller believe refill behavior is needed, often
  visible through `RW_Tank_PR_Valve_Sts`

### 17. `wt-attack-sensor-uf-high-block`

- Target process: `WT-P1` with downstream dependency on `WT-P3`
- Duration: `35s`
- Core idea: force a high downstream UF level before startup
- Sequence: set `UF_UFFT_Tank_Level=1000` at `2s`, then attempt `hmi_start` at
  `5s`
- Intended effect: block `RUNNING` through a downstream permissive constraint

### 18. `wt-attack-sensor-uf-high-disrupt`

- Target process: `WT-P1` with downstream dependency on `WT-P3`
- Duration: `40s`
- Core idea: spoof a high downstream UF level during runtime
- Sequence: start `WT-P1`, set `UF_UFFT_Tank_Level=1000` at `15s`, then stop at
  `30s`
- Intended effect: disrupt a running process by making the downstream side look
  saturated or unsafe

## Summary

The current `18`-scenario suite is intentionally diverse:

- HMI attacks stress command integrity and availability
- Process 2 attacks stress manual control abuse and chemical inventory loss
- Sensor spoofing attacks stress telemetry trust, permissive logic, and alarm
  interpretation

Together, these scenarios provide a compact cyber-physical benchmark for
comparing scripted, rule-based, and agentic attackers and defenders over the
same OpenPLC water-treatment environment.
