General water-treatment domain primer for a target-blind attacker.

- Water-treatment processes usually chain intake, storage, pretreatment, filtration or membrane stages, polishing, and transfer stages.
- Tanks buffer inventory and create lag, so upstream depletion or downstream starvation may appear several control cycles after an action.
- Pumps move water between stages, and valves gate or redirect paths; persistent command and feedback mismatch is more meaningful than a single sample.
- Manual or maintenance control paths can widen authority when they are explicitly enabled, especially around chemical handling or bypass operations.
- Level, flow, pressure, and quality indicators should be interpreted together because a single abrupt signal jump can be false while multi-signal drift is harder to dismiss.
- Chemical dosing changes may influence later stages indirectly, with delayed physical consequences rather than immediate effect.
- Alarms, permissives, and interlocks can automatically clamp motion, halt transfer, or switch the process into a safer operating mode.
- Harmful patterns include overfill, drain-down, starvation, oscillation, sustained loss of inventory, and process imbalance that propagates from one stage to another.
