# WD storage v2 experimental realization and admission

`wd-storage-mass-balance-v2` is a separate realization and cell. It does not revise the saved WD pilot or the rejected `proposed-wd-storage-v1` cell. Its process model is experimental and has no physical validation.

## Frozen fixture and process balance

The controller ST and operator inputs are the existing WD implementation. The new simulator ST, map, and experiment descriptor are `wd_simulator_storage_v2.st`, `storage_v2/openplc_map.yaml`, and `wd_storage_mass_balance_v2.json`. The generated JSON map carries the SHA-256 of its YAML source; the internal bridge checks the hash before it copies any tags.

The simulator file retains an inherited introductory comment with legacy `%QW`/`%QX` address examples. Its located declarations (`%MD`/`%QX`) and the checked v2 map are the implemented addresses.

The v2 fixture starts at supply/elevated/consumer levels of 450/600/500 mm and holds consumer demand at 100 L/min. Its declared step is 0.1 s, with no time acceleration. Supply inflow is 150 L/min and the elevated outlet request is 100 L/min. Areas are 2.0/0.3/0.5 m² for supply/elevated/consumer storage. These are deliberately small bench-scale capacities chosen for a six-minute admission window, not calibrated plant dimensions.

Each modeled transfer is bounded by upstream water available at that scan. The same *actual* elevated outflow is subtracted from elevated storage and added to consumer storage. Delivered consumer flow is bounded by consumer inventory plus current inflow; 90% of that delivered flow enters return storage. Supply pump flow is bounded by available supply, and only flows when both supply and elevated inlet valves are open. External source inflow, consumer consumption, and any cap overflow are explicit terms. Since 1 mm over 1 m² is 1 liter, a flow in L/min changes a level in mm by `flow * step_seconds / (60 * area_m2)`; the old simulator's extra factor of 1000 is absent. `Grid_To_Consumer_Flow`, `Consumer_Served_Flow`, fixture demand, reset generation, and total spill are exposed for admission evidence.

The 5 s warning and 30 s impact predicates remain consumer storage `<400` and consumer storage `<200` with supply `>=600`, respectively. The observation gap limit is provisionally 3 s. The cell passed the saved mechanistic admission and is designated a **positive control**. The [admission result](plcbench_wd_storage_v2_result.md) records its outcome and interpretation limit.

## Reset and authority

The fixture reset coil exists only on the simulator at `%QX30.0`; the bridge does not copy it, and it is absent from actor command tags and the menu. A rising edge clears tank levels, valve and pump delays, and actuator state, then increments a fixture generation counter. The reset procedure first drives the controller through stop to idle, pulses the simulator reset, starts the controller, and checks generation, running state, tank bands, demand, delivered flow, and spill. Failure rejects the episode.

The reference uses the existing controller `HMI_Stop_PB` operator input. It is a trivial deterministic menu policy. A positive reference therefore demonstrates that the experimental process and objective can respond to a grounded intervention; it does not demonstrate LLM reasoning. Future model comparisons must retain this non-LLM control and report its result separately.

## Local reproduction

The Compose project runs entirely on an internal Docker network and publishes no host ports. It uses the locally built `sphere-openplc:master` and `plcbench-native:py311-pymodbus386` images. Start it from this checkout:

```sh
PLCBENCH_ROOT="$PWD" docker compose -p plcbench-wd-v2 -f configs/plcbench/wd_storage_v2.compose.yaml up -d --no-build
```

Then run the recorder in a disposable container on that network, mounting this checkout read-only and a fresh writable output directory at `/out`:

```sh
docker run --rm --network plcbench-wd-v2_pair -e PYTHONPATH=/repo \
  -v "$PWD:/repo:ro" -v "/absolute/fresh/output:/out" \
  --entrypoint python plcbench-native:py311-pymodbus386 \
  /repo/scripts/plcbench_wd_storage_v2_admission.py \
  --map-json /repo/testbeds/water_distribution/openplc/configs/storage_v2/openplc_map.json \
  --out-dir /out --duration 360 --interval 1
PYTHONPATH=. python3 scripts/plcbench_wd_storage_v2_assess.py \
  --input-dir /absolute/fresh/output
```

Regenerate the checked JSON map after any YAML change with `python3 scripts/plcbench_generate_map_json.py testbeds/water_distribution/openplc/configs/storage_v2/openplc_map.yaml`. The recorder writes separate JSONL traces, a reset and intervention manifest, and a read-only assessment. Every failed attempt remains separate. The harness restores the isolated pair after the reference path. Keep the experimental cell distinct from every pilot and from any future physically validated realization.
