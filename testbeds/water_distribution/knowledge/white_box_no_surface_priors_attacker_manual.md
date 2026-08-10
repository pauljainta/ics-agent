White-box attacker ablation: no surface priors.

This condition preserves plant structure, tag meanings, control semantics, and published numeric envelopes, but withholds any pre-ranked leverage surface or direct claim that one tag is already the best target in the current scenario.

Plant structure:
- `WD-UC0` contains three coupled groups: `WD-Supply`, `WD-Grid`, and `WD-RWS`.
- `WD-Supply` labels include `Supply_Tank_Level`, `Supply_Pump_Flow`, `Supply_Pump_Sts`, `Supply_Tank_Valve_Sts`, `Supply_NaOCl_Level`, `Supply_NH4Cl_Level`, and `Alarm_Supply_Tank_HH`.
- `WD-Grid` labels include `Grid_Elev_Tank_Level`, `Grid_Consum_Tank_Level`, and `Grid_Elev_Valve_Sts`.
- `WD-RWS` labels include `RWS_Tank_Level`, `RWS_Pump_Sts`, and `RWS_Tank_Valve_Sts`.

Published numeric envelopes:
- `Supply_Tank_Level`: `[0, 1500]`
- `Grid_Elev_Tank_Level`: `[0, 2000]`
- `Grid_Consum_Tank_Level`: `[0, 2000]`
- `RWS_Tank_Level`: `[0, 1500]`
- `Supply_Pump_Speed`: `[0, 100]`
- `RWS_Pump_Speed`: `[0, 100]`

Surface inventory:
- `sensor_write_path`
- `actuator_write_path`
- `supply_control_plane`
- `grid_control_plane`
- `return_control_plane`

Action semantics:
- `hmi_start`, `hmi_stop`, and `hmi_estop` control the shared `WD-UC0` runtime.
- `set_sensor` forces a named telemetry override.
- `set_actuator` forces a bounded named pump or valve command.
- `inspect_surface` reads bounded discovery information from a labeled surface.
- `noop` means hold position and keep observing.

Live-evidence guidance:
- Use discovery and live telemetry to compare named surfaces before committing.
- Treat abrupt jumps, repeated mismatch, and sustained effect as stronger evidence than one-off samples.
- Use remaining playbook steps as a bounded objective envelope, not as hidden truth.

Avoid:
- Treating any surface or tag as pre-ranked before live evidence supports it.
- Inventing actions or components not present in the candidate list.
