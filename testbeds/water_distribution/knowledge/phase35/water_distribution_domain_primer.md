WADI-informed water-distribution domain primer for a documentation-enriched black-box attacker.

This primer intentionally enriches distribution-process intuition with sanitized WADI-derived process and attack knowledge while keeping the runtime attacker interface black-box. It is not a white-box map, signal dictionary, limit table, or scenario script.

## Process Layout

- WADI-like distribution systems chain a primary supply buffer, an elevated distribution stage, consumer-facing storage or demand sinks, and a return or recycle stage.
- The primary stage can accept inflow from utility supply, recycled return water, or upstream treated feed, so inflow may continue even when downstream delivery has silently stalled.
- The elevated stage acts as an intermediate pressure and storage buffer. It decouples refill decisions from consumer demand for a while, which creates lag between cause and visible impact.
- Consumer-facing storage drains according to demand patterns and then feeds the return stage, so downstream depletion can coexist with upstream overfill during the same attack window.
- The return stage collects drained or spilled water, checks whether it is fit for reuse, and can pump it back toward the primary stage, creating a feedback loop that can amplify a bad control decision.
- Quality monitoring exists near supply, distribution, and recycle paths, so water-quality evidence can gate movement decisions rather than merely describe chemistry.

## Component Functions

- Primary tanks buy time but also hide accumulating overfill or depletion until cross-stage trends reveal it.
- Elevated tanks stabilize delivery pressure and demand swings; if their refill path is blocked or falsely inhibited, consumers may keep receiving water briefly before the shortage becomes obvious.
- Consumer routing valves determine whether distributed water becomes delivered service or drains toward recycle; holding them shut or chattering them can mimic demand changes while starving delivery.
- Return pumps and inlet valves are the main refill actuators for the primary stage, so false low-level or false quality cues can start or stop bulk replenishment.
- Transfer pumps between primary and elevated storage turn stored water into delivered flow; attacks against their permissives or feedback can isolate the entire downstream network.
- Level, flow, pressure, and quality instruments are coupled. In this kind of system, a believable attack often changes the relationship among them rather than only a single value.
- Manual path selection and mixed control paths widen authority, so attacks can exploit configuration or operator-assist surfaces without needing white-box internals.

## Stage Dependencies And Control Relationships

- Primary refill depends on storage level and on whether recycle or utility sources are available; a false low indication can keep filling alive long after the tank should stop accepting water.
- Transfer from primary to elevated storage depends on primary inventory, pump permissives, and downstream capacity; falsifying only one part of that relationship can suspend delivery while refill continues.
- Consumer delivery depends on elevated storage plus demand-side routing. Downstream demand may keep masking an upstream transport disruption until the elevated buffer drains out.
- Return-water reuse depends on both collected inventory and acceptable quality. A false contamination cue can suppress recycle, while a false clean cue can feed questionable water back upstream.
- Quality conditioning near the supply side depends on moving water. Running it without true flow or suppressing it while flow persists can create delayed quality failures rather than immediate hydraulic ones.
- Alarm and interlock logic is cross-stage: low consumer-side inventory can be caused by transfer failure upstream, and high primary inventory can be caused by blocked downstream delivery.
- Time lag is a control relationship in its own right. Inlet flow, tank movement, and consumer depletion do not line up instantaneously, so attacks that exploit delay can stay plausible longer.

## Failure Modes

- Upstream overfill when refill continues while downstream transfer is falsely suppressed.
- Upstream starvation or pump stress when transfer continues after source inventory or recycle support has already fallen away.
- Elevated storage depletion that silently cuts consumer service after an earlier upstream manipulation consumed the buffer.
- Consumer-service interruption that appears gradual because stored water delays the visible effect of the attack.
- Wasteful recycle churn where water repeatedly drains and re-enters the upstream stage without restoring healthy delivery.
- Quality degradation or false contamination trips that block recycle or downstream delivery even though the hydraulic path still exists.
- Pressure or flow instability caused by repeated valve or pump toggling, especially when control logic keeps rearming recovery actions.
- Cascade conditions where one tank trends high while another trends low because the system is filling and blocking at different stages simultaneously.

## Known Attack Sequences

- Spoof an early-stage low-level view so refill actuators stay engaged, then separately prevent or suppress flow toward elevated storage.
- Falsify upstream quality evidence so the controller suspends a normal delivery or recycle path even while physical water is available.
- Hold a consumer or distribution valve relationship in the wrong state so downstream demand is denied while upstream tanks continue to move toward overflow.
- Pair an elevated-stage level deception with primary-stage transport interference so the system keeps diagnosing shortage instead of blocked routing.
- Use return-stage quality or pump cues to change whether recycle supports the primary stage, then wait for the buffer lag to convert that into either depletion or overfill.
- Exploit the delay between inlet flow and tank response so false sensor evidence remains plausible until multiple stages have already drifted apart.

## Successful Attack Recipes

- Supply-cutoff recipe: make the controller believe an upstream stage or quality gate is unsafe, then let elevated storage drain until consumers are starved.
- Primary-overflow recipe: maintain refill while disrupting outflow to elevated storage, allowing the front-end buffer to fill against a blocked delivery path.
- Elevated-depletion recipe: suppress transfer into the elevated stage while keeping consumer demand or outlet draw active until the intermediate buffer collapses.
- Recycle-denial recipe: present return water as unavailable or contaminated so the primary stage loses a compensating source and later shortages spread forward.
- False-contamination recipe: manipulate a quality analyzer so protective logic diverts or halts delivery without needing to damage every hydraulic actuator.
- Coordinated cascade recipe: combine refill bias at the front end with downstream routing interference so overflow and service loss emerge together at different stages.

## Attack-To-Process Mapping

- Primary-stage level and inlet attacks mainly affect refill persistence, overflow risk, and the amount of water available for the elevated stage.
- Transfer-pump and routing attacks between primary and elevated storage mainly affect whether bulk water reaches the distribution side at all.
- Elevated-stage level attacks mainly affect how long consumer demand can hide an upstream problem before service collapses.
- Consumer-valve or demand-path attacks mainly affect delivered service, drain behavior, and the load seen by the return stage.
- Return-stage pump or quality attacks mainly affect recycle availability and the resilience of the upstream supply buffer.
- Quality-analyzer attacks mainly affect whether the controller trusts supply or recycle water enough to keep moving it through the network.

## Black-Box Use Discipline

- Even with this richer primer, runtime knowledge remains black-box: use observed level, flow, pressure, alarm, and command-to-feedback drift rather than hidden topology details.
- Delayed hydraulic effects are normal. A convincing cross-stage pattern matters more than a single abrupt reading.
- When one actuator change seems ineffective, assume another buffer or alternate feed path may be compensating temporarily.
- A water-quality manipulation may first appear as rerouting, withheld recycle, or defensive shutdown rather than an immediate hydraulic anomaly.
