(* Native OpenPLC v3 program.  %IW100 is populated by the v3 Modbus-master
   holding-register map; %QX100.0 is emitted by the same native map as coil 0. *)
PROGRAM water_level_controller
  VAR
    level AT %IW100 : UINT;
    inlet AT %QX100.0 : BOOL;
  END_VAR

  IF level < 450 THEN
    inlet := TRUE;
  ELSE
    inlet := FALSE;
  END_IF;
END_PROGRAM

(* OpenPLC v3's supplied compile script expects these generated names. *)
CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK control_task(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM controller_instance WITH control_task : water_level_controller;
  END_RESOURCE
END_CONFIGURATION
