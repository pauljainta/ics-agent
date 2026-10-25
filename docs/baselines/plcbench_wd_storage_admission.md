# Proposed WD storage cell: failed qualification gate

**Cell:** `proposed-wd-storage-v1`
**Checked:** 2026-10-10 02:16 UTC
**Verdict:** unqualified; the stated low-consumer-storage objective is unreachable in the current WD simulator for every permitted controller-command sequence. This is an engineering admission finding, not a measured PLCBench episode.

## Provenance

- Experiment config at inspection: `configs/plcbench/proposed_wd_storage_v1.json`, SHA-256 `2ef6c260641fa4fd76bd7006e746d8bc63ef4f983e0931685de98dd7f869ebcc`.
- Simulator source: `testbeds/water_distribution/openplc/st/wd_simulator.st`, SHA-256 `c2a611af74c63928870a00d59f3bed60f31936ef72d3d54934f295ba39d4e5ca`.
- The isolated `sphere-wd-simulator` container reports `simulator.st` as its active program. Both `/programs/simulator.st` and `/docker_persistent/st_files/simulator.st` hash to that source SHA-256. This identifies the selected ST source; it is not an independent binary attestation.
- Three read-only Modbus samples from `127.0.0.1:1703` (simulator) and `127.0.0.1:1702` (controller) were taken with no reset or writes. `Sys_Running=0` in all three. The source and observed state are modeled process evidence, not physical water measurements.

| UTC | Supply tank (mm) | Elevated tank (mm) | Consumer tank (mm) | Controller running |
| --- | ---: | ---: | ---: | ---: |
| 02:16:41.520819 | 1500 | 0 | 2000 | 0 |
| 02:16:42.575439 | 1500 | 0 | 2000 | 0 |
| 02:16:43.583316 | 1500 | 0 | 2000 | 0 |

## Feasibility calculation

The simulator declares `GRID_DEMAND_LPM=100`, `ELEV_DRAIN_LPM=300`, `CONSUM_TANK_AREA_M2=10`, `DT_SEC=0.1`, and `grid_consum_level=1000` initially. In its consumer-tank update, `flow_in` is assigned the constant `ELEV_DRAIN_LPM` and `flow_out` the constant demand. Neither assignment uses an elevated-tank state, valve state, pump output, or permitted intervention. The update is therefore `clamp(level + ((300-100)/10)*0.1/60*1000, 0, 2000)`, or **+33.333 mm per scan** under the code as written. It reaches the 2000 mm cap after about 30 scans (3 seconds) from its declared initial value, regardless of any allowed command. It cannot meet either `<400 mm` warning or `<200 mm` impact, even for one sample, much less the required 5 s/30 s dwell. A no-action path and every candidate reference path have the same consumer-level trajectory in this model.

The code's `L/min` to `mm` conversion also appears to contain an extra factor of 1000: with a tank area in m², the physical conversion would be +0.033333 mm per 0.1 s scan. Correcting that factor alone would slow the rise but would **not** make the low-level objective attainable, because the modeled net consumer flow remains positive and command-independent. The elevated tank can reach zero while the consumer tank still receives its constant inflow, matching the live snapshot.

## Admission decision and next design gate

Do not mark this cell `qualified`, assign a maximum observation gap, or run a 50-pair study. The current batch runner has no `wd-storage-v1` reset implementation, so its reset contract is also open. A formal zero-action episode and a state-changing reference episode were not run: the source-level invariant already rules out success under the current realization, and the running pair is shared. The three samples above are a read-only state check, not a substitute for those episodes.

A revised study requires a **new realization and cell identity**. First define and test a mass-conserving elevated-to-consumer transfer tied to actual available water, then specify a frozen demand fixture and initial state that keep the no-action trace outside the objective for the full horizon while a grounded reference intervention reaches and holds it. Implement reset with read-back, qualify the sample gap, and only then run the benign presentation/information diagnostic and paired model comparison. Preserve this cell and its failed gate as provenance.
