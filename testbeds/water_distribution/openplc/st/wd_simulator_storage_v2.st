(*
 * Water Distribution storage v2 experimental simulator
 * SPHERE CPS Enclave - OpenPLC Implementation
 *
 * IEC 61131-3 Structured Text (OpenPLC/matiec compatible)
 *
 * This simulator models the physical dynamics of:
 *   - Supply Line: Supply tank, chemical tanks, pump, valves
 *   - Distribution Grid: Elevated tank, consumer tank, valves
 *   - Return Water System: RWS tank, pump, valves
 *
 * Architecture:
 *   Controller PLC <-> Modbus Bridge <-> Simulator PLC (this)
 *
 * I/O Mapping (configured in OpenPLC web interface):
 *   Command Inputs (from Controller via Bridge):
 *     cmd_supply_pump_speed -> %QW200
 *     cmd_rws_pump_speed -> %QW201
 *     cmd_supply_valve -> %QX3.0
 *     cmd_grid_elev_valve -> %QX3.1
 *     cmd_rws_valve -> %QX3.2
 *
 *   Sensor Outputs (to Controller via Bridge):
 *     out_supply_tank_level -> %QW300
 *     out_supply_pump_flow -> %QW301
 *     out_supply_naocl_level -> %QW302
 *     out_supply_nh4cl_level -> %QW303
 *     out_grid_elev_level -> %QW304
 *     out_grid_consum_level -> %QW305
 *     out_rws_tank_level -> %QW306
 *
 *   Status Outputs (to Controller via Bridge):
 *     out_supply_pump_sts -> %QX40.0
 *     out_supply_mixer_sts -> %QX40.1
 *     out_supply_valve_sts -> %QX40.2
 *     out_grid_elev_valve_sts -> %QX40.3
 *     out_rws_pump_sts -> %QX40.4
 *     out_rws_valve_sts -> %QX40.5
 *)

(* Experimental low-storage fixture. Model time is wall time at the 100 ms scan. *)
PROGRAM MainProgram
  VAR
    (* ==== located I/O (Modbus-exposed; hydro holding-register/coil pattern).
       REAL at %MD (two holding registers each); BOOL at %QX coils. Both PLCs
       declare every tag at the same address; the simulator writes the ones it
       owns (sensors/status) and only reads the controller-owned commands. ==== *)
    (* commands from controller (controller-owned; simulator reads only) *)
    cmd_supply_pump_speed AT %MD0 : REAL;   (* Supply pump speed cmd [%] *)
    cmd_rws_pump_speed AT %MD1 : REAL;      (* RWS pump speed cmd [%] *)
    cmd_supply_valve AT %QX5.0 : BOOL;      (* Supply valve command *)
    cmd_grid_elev_valve AT %QX5.1 : BOOL;   (* Elevated tank valve cmd *)
    cmd_rws_valve AT %QX5.2 : BOOL;         (* RWS valve command *)
    (* Fixture-only reset is written directly to the simulator, never bridged to the actor. *)
    fixture_reset AT %QX30.0 : BOOL;
    out_grid_to_consumer_flow AT %MD12 : REAL;
    out_consumer_served_flow AT %MD13 : REAL;
    out_fixture_reset_generation AT %MD14 : REAL;
    out_fixture_demand_flow AT %MD15 : REAL;
    out_total_spill_flow AT %MD16 : REAL;
    (* sensor outputs (simulator-owned; simulator writes) *)
    out_supply_tank_level AT %MD5 : REAL;   (* Supply tank level [mm] *)
    out_supply_pump_flow AT %MD6 : REAL;    (* Supply pump flow [L/min] *)
    out_supply_naocl_level AT %MD7 : REAL;  (* NaOCl tank level [mm] *)
    out_supply_nh4cl_level AT %MD8 : REAL;  (* NH4Cl tank level [mm] *)
    out_grid_elev_level AT %MD9 : REAL;     (* Elevated tank level [mm] *)
    out_grid_consum_level AT %MD10 : REAL;  (* Consumer tank level [mm] *)
    out_rws_tank_level AT %MD11 : REAL;     (* RWS tank level [mm] *)
    (* status outputs (simulator-owned; simulator writes) *)
    out_supply_pump_sts AT %QX20.0 : BOOL;      (* Supply pump running *)
    out_supply_mixer_sts AT %QX20.1 : BOOL;     (* Supply mixer running *)
    out_supply_valve_sts AT %QX20.2 : BOOL;     (* Supply valve open *)
    out_grid_elev_valve_sts AT %QX20.3 : BOOL;  (* Elevated tank valve open *)
    out_rws_pump_sts AT %QX20.4 : BOOL;         (* RWS pump running *)
    out_rws_valve_sts AT %QX20.5 : BOOL;        (* RWS valve open *)
  END_VAR

  VAR
    (* Simulation Constants - Timing *)
    DT_SEC : REAL := 0.1;                       (* Simulation timestep [s] *)
    MOTORIZED_VALVE_DELAY : INT := 600;         (* 60s at 100ms scan *)
    PUMP_SPINUP_DELAY : INT := 20;              (* 2s at 100ms scan *)

    (* Physical parameters - Supply *)
    SUPPLY_TANK_AREA_M2 : REAL := 2.0;
    SUPPLY_TANK_MAX_MM : REAL := 1500.0;
    SUPPLY_INFLOW_LPM : REAL := 150.0;
    SUPPLY_PUMP_MAX_LPM : REAL := 200.0;

    (* Physical parameters - Chemical tanks *)
    NAOCL_TANK_AREA_M2 : REAL := 0.1;
    NAOCL_DOSE_LPM : REAL := 0.038;
    NH4CL_TANK_AREA_M2 : REAL := 0.1;
    NH4CL_DOSE_LPM : REAL := 0.019;

    (* Physical parameters - Grid *)
    ELEV_TANK_AREA_M2 : REAL := 0.3;
    ELEV_TANK_MAX_MM : REAL := 2000.0;
    CONSUM_TANK_AREA_M2 : REAL := 0.5;
    CONSUM_TANK_MAX_MM : REAL := 2000.0;
    GRID_DEMAND_LPM : REAL := 100.0;
    ELEV_DRAIN_LPM : REAL := 100.0;

    (* Physical parameters - RWS *)
    RWS_TANK_AREA_M2 : REAL := 0.3;
    RWS_TANK_MAX_MM : REAL := 1500.0;
    RWS_PUMP_MAX_LPM : REAL := 38.0;
    RETURN_FRACTION : REAL := 0.9;

    (* Tank levels [mm] *)
    supply_tank_level : REAL := 450.0;
    naocl_tank_level : REAL := 800.0;
    nh4cl_tank_level : REAL := 800.0;
    grid_elev_level : REAL := 600.0;
    grid_consum_level : REAL := 500.0;
    rws_tank_level : REAL := 500.0;

    (* Valve states with delay modeling *)
    supply_valve_timer : INT := 0;
    supply_valve_state : BOOL := FALSE;
    grid_elev_valve_timer : INT := 0;
    grid_elev_valve_state : BOOL := FALSE;
    rws_valve_timer : INT := 0;
    rws_valve_state : BOOL := FALSE;

    (* Pump states with delay modeling *)
    supply_pump_timer : INT := 0;
    supply_pump_state : BOOL := FALSE;
    supply_pump_actual_speed : REAL := 0.0;
    rws_pump_timer : INT := 0;
    rws_pump_state : BOOL := FALSE;
    rws_pump_actual_speed : REAL := 0.0;

    (* Flow rates [L/min] *)
    supply_pump_flow : REAL := 0.0;
    rws_pump_flow : REAL := 0.0;
    grid_demand_actual : REAL := 0.0;
    return_flow : REAL := 0.0;

    (* Mixer state *)
    mixer_running : BOOL := TRUE;

    fixture_reset_prev : BOOL := FALSE;
    grid_to_consumer_flow : REAL := 0.0;
    consumer_served_flow : REAL := 0.0;
    total_spill_flow : REAL := 0.0;
    available_flow : REAL := 0.0;
    (* Temporary calculation variables *)
    delta_level : REAL := 0.0;
    flow_in : REAL := 0.0;
    flow_out : REAL := 0.0;
  END_VAR

  (* A rising edge resets every persistent process state, including actuator delays.
     The harness resets controller state separately and verifies read-back. *)
  IF fixture_reset AND NOT fixture_reset_prev THEN
    supply_tank_level := 450.0;
    naocl_tank_level := 800.0;
    nh4cl_tank_level := 800.0;
    grid_elev_level := 600.0;
    grid_consum_level := 500.0;
    rws_tank_level := 500.0;
    supply_valve_timer := 0;
    grid_elev_valve_timer := 0;
    rws_valve_timer := 0;
    supply_valve_state := FALSE;
    grid_elev_valve_state := FALSE;
    rws_valve_state := FALSE;
    supply_pump_timer := 0;
    rws_pump_timer := 0;
    supply_pump_state := FALSE;
    rws_pump_state := FALSE;
    supply_pump_actual_speed := 0.0;
    rws_pump_actual_speed := 0.0;
    supply_pump_flow := 0.0;
    rws_pump_flow := 0.0;
    grid_to_consumer_flow := 0.0;
    consumer_served_flow := 0.0;
    return_flow := 0.0;
    mixer_running := TRUE;
    out_fixture_reset_generation := out_fixture_reset_generation + 1.0;
  END_IF;
  fixture_reset_prev := fixture_reset;

  (* =====================================================================
     VALVE DELAY MODELING
     ===================================================================== *)

  (* Supply Tank Valve *)
  IF cmd_supply_valve AND NOT supply_valve_state THEN
    supply_valve_timer := supply_valve_timer + 1;
    IF supply_valve_timer >= MOTORIZED_VALVE_DELAY THEN
      supply_valve_state := TRUE;
      supply_valve_timer := MOTORIZED_VALVE_DELAY;
    END_IF;
  ELSIF NOT cmd_supply_valve AND supply_valve_state THEN
    supply_valve_timer := supply_valve_timer - 1;
    IF supply_valve_timer <= 0 THEN
      supply_valve_state := FALSE;
      supply_valve_timer := 0;
    END_IF;
  ELSIF cmd_supply_valve AND supply_valve_state THEN
    supply_valve_timer := MOTORIZED_VALVE_DELAY;
  ELSE
    supply_valve_timer := 0;
  END_IF;

  (* Grid Elevated Tank Valve *)
  IF cmd_grid_elev_valve AND NOT grid_elev_valve_state THEN
    grid_elev_valve_timer := grid_elev_valve_timer + 1;
    IF grid_elev_valve_timer >= MOTORIZED_VALVE_DELAY THEN
      grid_elev_valve_state := TRUE;
      grid_elev_valve_timer := MOTORIZED_VALVE_DELAY;
    END_IF;
  ELSIF NOT cmd_grid_elev_valve AND grid_elev_valve_state THEN
    grid_elev_valve_timer := grid_elev_valve_timer - 1;
    IF grid_elev_valve_timer <= 0 THEN
      grid_elev_valve_state := FALSE;
      grid_elev_valve_timer := 0;
    END_IF;
  ELSIF cmd_grid_elev_valve AND grid_elev_valve_state THEN
    grid_elev_valve_timer := MOTORIZED_VALVE_DELAY;
  ELSE
    grid_elev_valve_timer := 0;
  END_IF;

  (* RWS Tank Valve *)
  IF cmd_rws_valve AND NOT rws_valve_state THEN
    rws_valve_timer := rws_valve_timer + 1;
    IF rws_valve_timer >= MOTORIZED_VALVE_DELAY THEN
      rws_valve_state := TRUE;
      rws_valve_timer := MOTORIZED_VALVE_DELAY;
    END_IF;
  ELSIF NOT cmd_rws_valve AND rws_valve_state THEN
    rws_valve_timer := rws_valve_timer - 1;
    IF rws_valve_timer <= 0 THEN
      rws_valve_state := FALSE;
      rws_valve_timer := 0;
    END_IF;
  ELSIF cmd_rws_valve AND rws_valve_state THEN
    rws_valve_timer := MOTORIZED_VALVE_DELAY;
  ELSE
    rws_valve_timer := 0;
  END_IF;

  (* =====================================================================
     PUMP DELAY MODELING
     ===================================================================== *)

  (* Supply Pump *)
  IF cmd_supply_pump_speed > 0.0 THEN
    IF NOT supply_pump_state THEN
      supply_pump_timer := supply_pump_timer + 1;
      IF supply_pump_timer >= PUMP_SPINUP_DELAY THEN
        supply_pump_state := TRUE;
      END_IF;
    END_IF;
    IF supply_pump_state THEN
      IF supply_pump_actual_speed < cmd_supply_pump_speed THEN
        supply_pump_actual_speed := supply_pump_actual_speed + 1.0;
        IF supply_pump_actual_speed > cmd_supply_pump_speed THEN
          supply_pump_actual_speed := cmd_supply_pump_speed;
        END_IF;
      ELSIF supply_pump_actual_speed > cmd_supply_pump_speed THEN
        supply_pump_actual_speed := supply_pump_actual_speed - 1.0;
        IF supply_pump_actual_speed < cmd_supply_pump_speed THEN
          supply_pump_actual_speed := cmd_supply_pump_speed;
        END_IF;
      END_IF;
    END_IF;
  ELSE
    supply_pump_timer := 0;
    supply_pump_state := FALSE;
    supply_pump_actual_speed := 0.0;
  END_IF;

  (* RWS Pump *)
  IF cmd_rws_pump_speed > 0.0 THEN
    IF NOT rws_pump_state THEN
      rws_pump_timer := rws_pump_timer + 1;
      IF rws_pump_timer >= PUMP_SPINUP_DELAY THEN
        rws_pump_state := TRUE;
      END_IF;
    END_IF;
    IF rws_pump_state THEN
      IF rws_pump_actual_speed < cmd_rws_pump_speed THEN
        rws_pump_actual_speed := rws_pump_actual_speed + 1.0;
        IF rws_pump_actual_speed > cmd_rws_pump_speed THEN
          rws_pump_actual_speed := cmd_rws_pump_speed;
        END_IF;
      ELSIF rws_pump_actual_speed > cmd_rws_pump_speed THEN
        rws_pump_actual_speed := rws_pump_actual_speed - 1.0;
        IF rws_pump_actual_speed < cmd_rws_pump_speed THEN
          rws_pump_actual_speed := cmd_rws_pump_speed;
        END_IF;
      END_IF;
    END_IF;
  ELSE
    rws_pump_timer := 0;
    rws_pump_state := FALSE;
    rws_pump_actual_speed := 0.0;
  END_IF;

  (* Volumes are liters because one millimeter across one square meter is one
     liter. Q [L/min] * DT_SEC / 60 changes volume in liters. Each transfer is
     bounded by available upstream volume; overflow is recorded as spill. *)
  grid_demand_actual := GRID_DEMAND_LPM;
  out_fixture_demand_flow := grid_demand_actual;
  total_spill_flow := 0.0;

  IF rws_pump_state AND rws_valve_state THEN
    rws_pump_flow := (rws_pump_actual_speed / 100.0) * RWS_PUMP_MAX_LPM;
    available_flow := rws_tank_level * RWS_TANK_AREA_M2 * 60.0 / DT_SEC;
    IF rws_pump_flow > available_flow THEN rws_pump_flow := available_flow; END_IF;
  ELSE
    rws_pump_flow := 0.0;
  END_IF;

  IF supply_pump_state AND supply_valve_state AND grid_elev_valve_state THEN
    supply_pump_flow := (supply_pump_actual_speed / 100.0) * SUPPLY_PUMP_MAX_LPM;
    available_flow := (supply_tank_level * SUPPLY_TANK_AREA_M2 * 60.0 / DT_SEC)
                      + SUPPLY_INFLOW_LPM + rws_pump_flow;
    IF supply_pump_flow > available_flow THEN supply_pump_flow := available_flow; END_IF;
  ELSE
    supply_pump_flow := 0.0;
  END_IF;

  grid_to_consumer_flow := ELEV_DRAIN_LPM;
  available_flow := (grid_elev_level * ELEV_TANK_AREA_M2 * 60.0 / DT_SEC)
                    + supply_pump_flow;
  IF grid_to_consumer_flow > available_flow THEN
    grid_to_consumer_flow := available_flow;
  END_IF;

  consumer_served_flow := grid_demand_actual;
  available_flow := (grid_consum_level * CONSUM_TANK_AREA_M2 * 60.0 / DT_SEC)
                    + grid_to_consumer_flow;
  IF consumer_served_flow > available_flow THEN
    consumer_served_flow := available_flow;
  END_IF;
  return_flow := consumer_served_flow * RETURN_FRACTION;

  supply_tank_level := supply_tank_level
    + ((SUPPLY_INFLOW_LPM + rws_pump_flow - supply_pump_flow)
       / SUPPLY_TANK_AREA_M2) * DT_SEC / 60.0;
  IF supply_tank_level < 0.0 THEN supply_tank_level := 0.0; END_IF;
  IF supply_tank_level > SUPPLY_TANK_MAX_MM THEN
    total_spill_flow := total_spill_flow
      + (supply_tank_level - SUPPLY_TANK_MAX_MM) * SUPPLY_TANK_AREA_M2 * 60.0 / DT_SEC;
    supply_tank_level := SUPPLY_TANK_MAX_MM;
  END_IF;

  grid_elev_level := grid_elev_level
    + ((supply_pump_flow - grid_to_consumer_flow)
       / ELEV_TANK_AREA_M2) * DT_SEC / 60.0;
  IF grid_elev_level < 0.0 THEN grid_elev_level := 0.0; END_IF;
  IF grid_elev_level > ELEV_TANK_MAX_MM THEN
    total_spill_flow := total_spill_flow
      + (grid_elev_level - ELEV_TANK_MAX_MM) * ELEV_TANK_AREA_M2 * 60.0 / DT_SEC;
    grid_elev_level := ELEV_TANK_MAX_MM;
  END_IF;

  grid_consum_level := grid_consum_level
    + ((grid_to_consumer_flow - consumer_served_flow)
       / CONSUM_TANK_AREA_M2) * DT_SEC / 60.0;
  IF grid_consum_level < 0.0 THEN grid_consum_level := 0.0; END_IF;
  IF grid_consum_level > CONSUM_TANK_MAX_MM THEN
    total_spill_flow := total_spill_flow
      + (grid_consum_level - CONSUM_TANK_MAX_MM) * CONSUM_TANK_AREA_M2 * 60.0 / DT_SEC;
    grid_consum_level := CONSUM_TANK_MAX_MM;
  END_IF;

  rws_tank_level := rws_tank_level
    + ((return_flow - rws_pump_flow) / RWS_TANK_AREA_M2) * DT_SEC / 60.0;
  IF rws_tank_level < 0.0 THEN rws_tank_level := 0.0; END_IF;
  IF rws_tank_level > RWS_TANK_MAX_MM THEN
    total_spill_flow := total_spill_flow
      + (rws_tank_level - RWS_TANK_MAX_MM) * RWS_TANK_AREA_M2 * 60.0 / DT_SEC;
    rws_tank_level := RWS_TANK_MAX_MM;
  END_IF;

  IF supply_pump_state THEN
    naocl_tank_level := naocl_tank_level
      - (NAOCL_DOSE_LPM / NAOCL_TANK_AREA_M2) * DT_SEC / 60.0;
    nh4cl_tank_level := nh4cl_tank_level
      - (NH4CL_DOSE_LPM / NH4CL_TANK_AREA_M2) * DT_SEC / 60.0;
  END_IF;

  out_supply_tank_level := supply_tank_level;
  out_supply_pump_flow := supply_pump_flow;
  out_supply_naocl_level := naocl_tank_level;
  out_supply_nh4cl_level := nh4cl_tank_level;
  out_grid_elev_level := grid_elev_level;
  out_grid_consum_level := grid_consum_level;
  out_rws_tank_level := rws_tank_level;
  out_grid_to_consumer_flow := grid_to_consumer_flow;
  out_consumer_served_flow := consumer_served_flow;
  out_total_spill_flow := total_spill_flow;
  out_supply_pump_sts := supply_pump_state;
  out_supply_mixer_sts := mixer_running;
  out_supply_valve_sts := supply_valve_state;
  out_grid_elev_valve_sts := grid_elev_valve_state;
  out_rws_pump_sts := rws_pump_state;
  out_rws_valve_sts := rws_valve_state;

END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK Main(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM Inst0 WITH Main : MainProgram;
  END_RESOURCE
END_CONFIGURATION
