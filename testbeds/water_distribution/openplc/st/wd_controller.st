(*
 * Water Distribution UC0 Controller
 * SPHERE CPS Enclave - OpenPLC Implementation
 *
 * IEC 61131-3 Structured Text (OpenPLC/matiec compatible)
 *
 * This controller manages:
 *   - Supply Line: Supply tank level control, pump operation
 *   - Distribution Grid: Elevated tank filling via valve control
 *   - Return Water System: RWS tank level control, pump operation
 *
 * Architecture:
 *   Controller PLC <-> Modbus Bridge <-> Simulator PLC
 *
 * I/O Mapping (configured in OpenPLC web interface):
 *   Inputs from Simulator (via Bridge):
 *     ai_supply_tank_level -> %IW70
 *     ai_supply_pump_flow -> %IW71
 *     ai_supply_naocl_level -> %IW72
 *     ai_supply_nh4cl_level -> %IW73
 *     ai_grid_elev_level -> %IW74
 *     ai_grid_consum_level -> %IW75
 *     ai_rws_tank_level -> %IW76
 *     di_supply_pump_sts -> %IX2.0
 *     di_supply_mixer_sts -> %IX2.1
 *     di_supply_valve_sts -> %IX2.2
 *     di_grid_elev_valve_sts -> %IX2.3
 *     di_rws_pump_sts -> %IX2.4
 *     di_rws_valve_sts -> %IX2.5
 *     hmi_start -> %IX0.0
 *     hmi_stop -> %IX0.1
 *
 *   Outputs to Simulator (via Bridge):
 *     ao_supply_pump_speed -> %QW100
 *     ao_rws_pump_speed -> %QW101
 *     do_supply_valve -> %QX5.0
 *     do_grid_elev_valve -> %QX5.1
 *     do_rws_valve -> %QX5.2
 *     do_sys_idle -> %QX7.0
 *     do_sys_running -> %QX7.1
 *     do_alarm_supply_hh -> %QX8.0
 *)

(* System State Type - Using INT since iec2c doesn't support ENUM well *)
(* 0=IDLE, 1=START, 2=RUNNING, 3=SHUTDOWN *)

PROGRAM MainProgram
  VAR
    (* ==== located I/O (Modbus-exposed; hydro holding-register/coil pattern).
       Same addresses as wd_simulator.st. The controller writes the commands and
       system/alarm status it owns, and only reads the simulator-owned sensors
       and status plus the operator HMI coils. ==== *)
    (* sensors from simulator (simulator-owned; controller reads) *)
    ai_supply_tank_level AT %MD5 : REAL;    (* Supply tank level [mm] *)
    ai_supply_pump_flow AT %MD6 : REAL;     (* Supply pump flow [L/min] *)
    ai_supply_naocl_level AT %MD7 : REAL;   (* NaOCl tank level [mm] *)
    ai_supply_nh4cl_level AT %MD8 : REAL;   (* NH4Cl tank level [mm] *)
    ai_grid_elev_level AT %MD9 : REAL;      (* Elevated tank level [mm] *)
    ai_grid_consum_level AT %MD10 : REAL;   (* Consumer tank level [mm] *)
    ai_rws_tank_level AT %MD11 : REAL;      (* RWS tank level [mm] *)
    (* status from simulator (controller reads) *)
    di_supply_pump_sts AT %QX20.0 : BOOL;       (* Supply pump running *)
    di_supply_mixer_sts AT %QX20.1 : BOOL;      (* Supply mixer running *)
    di_supply_valve_sts AT %QX20.2 : BOOL;      (* Supply valve open *)
    di_grid_elev_valve_sts AT %QX20.3 : BOOL;   (* Elevated tank valve open *)
    di_rws_pump_sts AT %QX20.4 : BOOL;          (* RWS pump running *)
    di_rws_valve_sts AT %QX20.5 : BOOL;         (* RWS valve open *)
    (* HMI operator inputs (external master writes; controller reads) *)
    hmi_start AT %QX0.0 : BOOL;             (* Start button *)
    hmi_stop AT %QX0.1 : BOOL;              (* Stop button *)
    (* commands to simulator (controller-owned; controller writes) *)
    ao_supply_pump_speed AT %MD0 : REAL;    (* Supply pump speed [%] *)
    ao_rws_pump_speed AT %MD1 : REAL;       (* RWS pump speed [%] *)
    do_supply_valve AT %QX5.0 : BOOL;       (* Supply valve command *)
    do_grid_elev_valve AT %QX5.1 : BOOL;    (* Elevated tank valve command *)
    do_rws_valve AT %QX5.2 : BOOL;          (* RWS valve command *)
    (* system state + alarm outputs (controller-owned) *)
    do_sys_idle AT %QX7.0 : BOOL;           (* System idle state *)
    do_sys_running AT %QX7.1 : BOOL;        (* System running state *)
    do_alarm_supply_hh AT %QX8.0 : BOOL;    (* Supply tank HH alarm *)
  END_VAR

  VAR
    (* System State Machine - 0=IDLE, 1=START, 2=RUNNING, 3=SHUTDOWN *)
    sys_state : INT := 0;
    permissive_supply_ok : BOOL := FALSE;
    permissive_grid_ok : BOOL := FALSE;
    permissives_ready : BOOL := FALSE;

    (* Control Setpoints *)
    supply_tank_low_sp : REAL := 500.0;         (* Start pump below this [mm] *)
    supply_tank_high_sp : REAL := 1200.0;       (* Stop pump above this [mm] *)
    supply_tank_hh_sp : REAL := 1200.0;         (* HH alarm threshold [mm] *)

    grid_elev_low_sp : REAL := 600.0;           (* Open valve below this [mm] *)
    grid_elev_high_sp : REAL := 1600.0;         (* Close valve above this [mm] *)

    rws_tank_low_sp : REAL := 300.0;            (* Start pump below this [mm] *)
    rws_tank_high_sp : REAL := 1200.0;          (* Stop pump above this [mm] *)

    (* Pump Speed Setpoints *)
    supply_pump_normal_speed : REAL := 75.0;    (* Normal operating speed [%] *)
    rws_pump_normal_speed : REAL := 50.0;       (* Normal operating speed [%] *)

    (* Internal State *)
    supply_pump_cmd : BOOL := FALSE;
    rws_pump_cmd : BOOL := FALSE;
  END_VAR

  (* State constants *)
  (* STATE_IDLE = 0; STATE_START = 1; STATE_RUNNING = 2; STATE_SHUTDOWN = 3 *)

  CASE sys_state OF

    0: (* STATE_IDLE - System is idle, waiting for start command *)
      do_sys_idle := TRUE;
      do_sys_running := FALSE;

      (* All outputs off in idle *)
      ao_supply_pump_speed := 0.0;
      ao_rws_pump_speed := 0.0;
      do_supply_valve := FALSE;
      do_grid_elev_valve := FALSE;
      do_rws_valve := FALSE;

      (* Transition: Start button pressed *)
      IF hmi_start AND NOT hmi_stop THEN
        sys_state := 1;
      END_IF;

    1: (* STATE_START - Starting up, check permissives *)
      do_sys_idle := FALSE;
      do_sys_running := FALSE;

      (* Check permissives *)
      permissive_supply_ok := (ai_supply_tank_level > 300.0);
      permissive_grid_ok := (ai_grid_elev_level < 5500.0);
      permissives_ready := permissive_supply_ok AND permissive_grid_ok;

      (* Transition: Permissives met -> RUNNING *)
      IF permissives_ready THEN
        sys_state := 2;
      END_IF;

      (* Transition: Stop pressed -> SHUTDOWN *)
      IF hmi_stop THEN
        sys_state := 3;
      END_IF;

    2: (* STATE_RUNNING - Normal operation *)
      do_sys_idle := FALSE;
      do_sys_running := TRUE;

      (* SUPPLY LINE CONTROL *)
      (* Level-based pump control *)
      IF ai_supply_tank_level <= supply_tank_low_sp THEN
        supply_pump_cmd := TRUE;
      ELSIF ai_supply_tank_level >= supply_tank_high_sp THEN
        supply_pump_cmd := FALSE;
      END_IF;

      (* Set pump speed *)
      IF supply_pump_cmd THEN
        ao_supply_pump_speed := supply_pump_normal_speed;
        do_supply_valve := TRUE;
      ELSE
        ao_supply_pump_speed := 0.0;
        do_supply_valve := FALSE;
      END_IF;

      (* Supply tank HH alarm *)
      IF ai_supply_tank_level >= supply_tank_hh_sp THEN
        do_alarm_supply_hh := TRUE;
      ELSE
        do_alarm_supply_hh := FALSE;
      END_IF;

      (* DISTRIBUTION GRID CONTROL *)
      IF ai_grid_elev_level <= grid_elev_low_sp THEN
        do_grid_elev_valve := TRUE;
      ELSIF ai_grid_elev_level >= grid_elev_high_sp THEN
        do_grid_elev_valve := FALSE;
      END_IF;

      (* RETURN WATER SYSTEM CONTROL *)
      IF ai_rws_tank_level >= rws_tank_high_sp THEN
        rws_pump_cmd := TRUE;
      ELSIF ai_rws_tank_level <= rws_tank_low_sp THEN
        rws_pump_cmd := FALSE;
      END_IF;

      IF rws_pump_cmd THEN
        ao_rws_pump_speed := rws_pump_normal_speed;
        do_rws_valve := TRUE;
      ELSE
        ao_rws_pump_speed := 0.0;
        do_rws_valve := FALSE;
      END_IF;

      (* Transition: Stop pressed -> SHUTDOWN *)
      IF hmi_stop THEN
        sys_state := 3;
      END_IF;

    3: (* STATE_SHUTDOWN - Shutting down *)
      do_sys_idle := FALSE;
      do_sys_running := FALSE;

      (* Stop all pumps *)
      ao_supply_pump_speed := 0.0;
      ao_rws_pump_speed := 0.0;
      supply_pump_cmd := FALSE;
      rws_pump_cmd := FALSE;

      (* Close all valves *)
      do_supply_valve := FALSE;
      do_grid_elev_valve := FALSE;
      do_rws_valve := FALSE;

      (* Clear alarms *)
      do_alarm_supply_hh := FALSE;

      (* Clear permissives *)
      permissive_supply_ok := FALSE;
      permissive_grid_ok := FALSE;
      permissives_ready := FALSE;

      (* Transition: Stop released -> IDLE *)
      IF NOT hmi_stop THEN
        sys_state := 0;
      END_IF;

  END_CASE;

END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK Main(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM Inst0 WITH Main : MainProgram;
  END_RESOURCE
END_CONFIGURATION
