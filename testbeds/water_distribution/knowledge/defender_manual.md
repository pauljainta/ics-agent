Defender operating notes for water distribution:

- Compare controller commands against actuator status before assuming a level excursion is purely process-driven.
- Treat abrupt supply, grid, or return-water level jumps as likely spoofing when they do not match recent pump or valve behavior.
- Prefer targeted containment first: `block_sensor_write` for deception and `block_actuator_write` for forced pumps or valves.
- Escalate to `hmi_stop` or `hmi_estop` when the unsafe condition continues after containment or when `Alarm_Supply_Tank_HH` is active.
