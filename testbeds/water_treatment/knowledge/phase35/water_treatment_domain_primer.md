SWaT-informed water-treatment domain primer for a documentation-enriched black-box attacker.

This primer intentionally enriches treatment-process intuition with sanitized SWaT-derived process and attack knowledge while keeping the runtime attacker interface black-box. It is not a white-box map, signal dictionary, limit table, or scenario script.

## Process Layout

- Water treatment usually chains raw-water intake and storage, chemical conditioning and mixing, membrane filtration with cleaning, polishing ahead of reverse-osmosis feed, reverse-osmosis separation, and product-transfer or backwash support.
- The first storage stage is the main inventory buffer. False refill cues, forced inlet states, or blocked outflow can accumulate for several control cycles before downstream stages reveal the problem.
- The conditioning stage adds salt, acid, and oxidant into moving water so later membranes see suitable conductivity, acidity, and disinfection conditions.
- The membrane stage is not only a filter. It also alternates between production and cleaning states, so drain paths, backwash paths, and feed pumps are sequenced rather than independent.
- The polishing stage ahead of reverse osmosis protects membranes from chlorine breakthrough and other chemistry mistakes; it can stop healthy flow if its protective logic believes the feed is unsafe.
- The separation stage divides water into product and reject streams, which means one disturbance can reduce throughput, raise reject burden, and change the inventory available for cleaning support.
- The recovery stage closes loops back toward earlier buffers, so a late-stage upset can eventually change the apparent health of the front end.

## Component Functions

- Tanks provide buffer time, absorb short demand swings, and create lag between cause and visible effect.
- Transfer pumps move inventory between stages and are commonly protected by source low-level, destination high-level, low-pressure, or high-pressure permissives.
- Duty and standby pumps let the process continue after a single actuator upset, so successful attacks often suppress both the primary move and the fallback response.
- Motorized valves route production, drain, recycle, or cleaning flow and are often meaningful only when interpreted together with pump state and tank movement.
- Flow measurements often serve two roles at once: they confirm transport and they gate chemical addition or downstream permissives.
- Quality analyzers such as acidity, conductivity, chlorine surrogate, hardness, or permeate-quality indicators influence dosing, shutdown, or diversion decisions rather than only reporting water quality.
- Differential-pressure and pressure signals are cleaning and equipment-protection triggers, so misleading them can halt production without touching every downstream device directly.
- Manual, semi-automatic, and maintenance paths widen authority. They are especially important around chemical dosing, cleaning, and bypass behavior.

## Stage Dependencies And Control Relationships

- The intake path depends on both source availability and downstream storage capacity; an upstream valve can be commanded correctly while still being unsafe because a later buffer is already near exhaustion or overflow.
- Chemical injection usually depends on moving water plus analyzer feedback, so false quality readings can create bad dosing even when the pumps and valves themselves look normal.
- Membrane production depends on pressure, feed level, and valve sequencing at the same time; forcing only one of those relationships can make the plant thrash between filtration and cleaning.
- Cleaning logic depends on elapsed run state, pressure-drop evidence, and drain-path readiness. If any one of those is falsified, the system can churn instead of producing water.
- The polishing stage is protective, not merely additive. If chlorine-removal logic looks unhealthy, downstream separation may be stopped to avoid membrane damage.
- Reverse-osmosis output depends on feed pressure, flow, and water quality together; low product flow with normal pressure can indicate fouling, blocked routing, or spoofed feedback rather than a simple demand change.
- Reject and permeate handling affect earlier stages because reject may feed cleaning support and excess product may recycle to an earlier tank. Late-stage disturbances can therefore back-propagate into the front end.
- Alarm, permissive, and interlock logic is often cross-stage. A level or quality issue in one stage can stop a pump or valve in another stage without any local actuator fault.

## Failure Modes

- Front-end overfill when inlet or recycle continues after the controller loses a truthful view of storage.
- Front-end starvation or dry-running risk when transfer continues after source inventory is already low.
- Membrane-stage overflow in one buffer and underflow in the next when level feedback and transfer commands disagree for long enough.
- Repeated backwash or cleaning churn that consumes production time, drains buffers, and masks the real source of the upset.
- Chemical misdosing that wastes reagent, damages water quality, or silently sets up later membrane trouble.
- Protective shutdown of polishing or reverse-osmosis equipment after false flow, pressure, or quality evidence.
- Reduced product throughput caused by hidden fouling, blocked routing, or false low-flow cues.
- Oscillation caused by alternating false low and false high conditions, especially when alarms and automatic recovery logic keep rearming.

## Known Attack Sequences

- Wait for the plant to enter a stable transfer period, then falsify the measurement that grants a permissive while separately forcing or blocking the paired actuator.
- Keep a buffer apparently inside its safe band while physically driving it toward overfill or depletion; the lag between stages hides the damage long enough for cascade effects to develop.
- Force a cleaning or shutdown trigger by exaggerating pressure-drop, flow-loss, or residual-chemistry evidence even if the physical equipment is still healthy.
- Use a manual or semi-automatic path to hold a dosing or routing action longer than production logic would normally allow.
- Combine an upstream transport manipulation with a downstream false level cue so one tank fills while the next stage keeps demanding flow.
- Pair a quality-signal manipulation with a pump or valve intervention so the controller diagnoses the wrong cause and applies the wrong recovery action.

## Successful Attack Recipes

- Upstream overfill recipe: keep the front-end storage apparently mid-band while holding refill or recycle available and degrading the normal outflow path.
- Upstream depletion recipe: make the controller believe the destination is already satisfied or the source is healthier than it is, then let transfer continue until a pump-protection event or downstream starvation appears.
- Membrane churn recipe: exaggerate pressure-drop or cleaning need, or hold a cleaning valve relationship in the wrong mode, so production repeatedly yields to backwash behavior.
- Product-starvation recipe: falsify feed or product-flow evidence to trip protective logic around polishing or reverse osmosis and cut delivered output without an obvious hard fault.
- Chemical-abuse recipe: activate dosing during an enabled manual path or distort analyzer feedback while water is moving so reagent use and water chemistry drift together.
- Cross-stage cascade recipe: combine false level or quality evidence with forced transport so one stage experiences overflow while another stage experiences under-supply.

## Attack-To-Process Mapping

- Inlet-path attacks mainly affect raw storage balance, pump protection, and the amount of conditioned water available later.
- Quality-analyzer and dosing-path attacks mainly affect the conditioning stage first, then membrane health and product quality later.
- Level, valve, and pressure attacks around the membrane stage mainly affect whether filtration, drain, and backwash modes occur in the intended order.
- Flow or chlorine-removal attacks around the polishing stage mainly affect whether reverse-osmosis feed is considered safe enough to continue.
- Pressure, feed-flow, permeate-flow, or product-quality attacks around reverse osmosis mainly affect throughput, reject burden, and whether the plant diverts or shuts down.
- Recovery and backwash path attacks mainly affect cleaning success, recycle behavior, and the inventory seen again by upstream storage.

## Black-Box Use Discipline

- Even with this richer primer, runtime knowledge remains black-box: rely on observable status changes, alarm patterns, multi-signal drift, and command-to-feedback mismatch rather than on hidden state.
- Treat delayed consequences as normal in water treatment. The best clue is often a consistent cross-stage pattern, not the first abrupt reading change.
- When a single actuator attack seems ineffective, check whether standby logic, recycle paths, or interlocks are silently compensating.
- Quality attacks are often indirect. Their first useful signal may be a later shutdown, diversion, or cleaning transition rather than an immediate level change.
