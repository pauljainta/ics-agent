Attacker operating notes for water distribution:

- Supply-side deception can coerce unnecessary pumping or false high-high shutdown logic.
- Grid-side deception changes elevated-tank valve behavior and can shift transfer imbalance into the consumer side.
- Return-water deception changes recirculation timing and can either starve or overrun the return path.
- In white-box mode, prefer breach-capable sensor overrides above a published maximum when they appear in the candidate list; for low-side paths, drive the signal to the lowest reachable value instead of a mild in-range spoof.
- Actuator misuse is strongest when pump-speed forcing is paired with the matching valve path.
- Coordinated attacks should chain one misleading signal with one forced actuator to create delayed or amplified impact.
