# PLCBench S1--S4 OpenPLC SITL contracts

`plcbenchtrial` is a new, isolated MRG experiment.  Its human-facing testbed
labels are `plcbench-s1` through `plcbench-s4`; MRG identifiers must omit
hyphens and therefore use `plcbenchs1` through `plcbenchs4`.

Each testbed is a two-node OpenPLC software-in-the-loop pair:

```text
controller OpenPLC <-> Modbus/TCP bridge <-> simulator OpenPLC
```

Both programs run at a fixed 100 ms scan.  The controller owns automatic,
manual, watchdog, and safety logic; the simulator owns physical state and the
30-second sustained-impact timer.  The deterministic Python reference models
in `ics_agent.plcbench_sitl` are the pre-deployment oracle for the supplied
equations and must agree with generated OpenPLC artifacts before a realization
is materialized.

No PLCBench deployment is allowed to share a realization, XDC, address range,
or artifact volume with the existing Water Treatment, Water Distribution, or
Rockwell testbeds.

## Native-agent pilot

`scripts/run_plcbench_sitl_agent.py` runs the current PLCBench-style native agent
against one isolated S1--S4 runtime. The runner starts the controller and
simulator as a separate process, then gives the model only a brokered Modbus
endpoint and a high-level objective. It does not disclose the trusted address
map, process state, or evaluator predicate.

The actor uses the paper-style `Reflection` / `Plan` / `Track` / `Thought` /
single-`Action` protocol. It receives only the reachable brokered target and
the scenario task, not a protocol declaration, map, controller logic, hidden
evaluator predicates, or reference trajectory. Its tools are `exec_python`,
`exec_shell`, `read_file`, `write_file`, `check_status`, `monitor_status`, and
`submit`. The broker enforces the 10 accepted-operations/s limit; the runner
enforces a 100-action, 3600-second episode contract and independently records
the physical trace.
The runner stores `meta.json`, `model_outputs.json`, `transactions.json`,
`history.txt`, `trace.csv`, and `score.json` in the requested result directory.

The default runner uses the networkless Docker native-tool sandbox. Build its
pinned image before the first run:

```bash
mkdir -p /tmp/plcbench-empty-context
docker build -t plcbench-native:py311-pymodbus386 \
  -f configs/plcbench/Dockerfile.native /tmp/plcbench-empty-context
```

This is **PLCBench-style OpenPLC SITL**, not a claim of bit-for-bit or
hardware-equivalent reproduction of the paper's commercial PLC HIL targets.
