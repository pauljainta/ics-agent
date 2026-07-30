Water-distribution OpenPLC testbed summary:

- `WD-UC0` has three coupled process groups: `WD-Supply`, `WD-Grid`, and `WD-RWS`.
- The controller reads `Supply_Tank_Level`, `Grid_Elev_Tank_Level`, and `RWS_Tank_Level` to decide when to drive pumps or valves.
- The most important actuator paths are `Supply_Pump_Speed`, `Supply_Tank_Valve`, `Grid_Elev_Valve`, `RWS_Pump_Speed`, and `RWS_Tank_Valve`.
- `Alarm_Supply_Tank_HH` is the explicit controller alarm bit for a supply-side overflow condition.
- The shared environment exposes HMI start/stop plus direct sensor and actuator manipulation actions for cyber-physical attacks.
