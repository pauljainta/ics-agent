(* OpenPLC V2: private process I/O plus an expiring supervisory request.
   Device 0 maps process HR0 -> %IW100 and %QX100.0 -> process coil 0.
   Device 1 maps localhost supervisory HR0..1 -> %IW200..201.
   The local service expires requests independently; normal automatic control
   resumes as soon as %IW200 is zero. *)
PROGRAM water_level_controller_v2
  VAR
    level AT %IW100 : UINT;
    supervisory_active AT %IW200 : UINT;
    supervisory_value AT %IW201 : UINT;
    inlet AT %QX100.0 : BOOL;
    override_visible AT %QW0 : UINT;
    inlet_visible AT %QW1 : UINT;
  END_VAR

  IF supervisory_active = 1 THEN
    inlet := supervisory_value <> 0;
    override_visible := 1;
  ELSE
    inlet := level < 450;
    override_visible := 0;
  END_IF;
  IF inlet THEN
    inlet_visible := 1;
  ELSE
    inlet_visible := 0;
  END_IF;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK control_task(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM controller_instance WITH control_task : water_level_controller_v2;
  END_RESOURCE
END_CONFIGURATION
