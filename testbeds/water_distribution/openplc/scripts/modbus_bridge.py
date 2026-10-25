#!/usr/bin/env python3
"""Controller <-> simulator Modbus bridge (map-derived, use-case agnostic).

Both PLCs declare every tag at the same address (see configs/openplc_map.yaml)
and write only the ones they own. This bridge copies, each cycle, every
controller-owned block to the simulator and every simulator-owned block to the
controller, so the controller's commands drive the simulator's physics and the
simulator's sensors/status reach the controller. One contiguous block per
(owner, register type); REAL analog lives in holding registers (%MD), digital in
coils (%QX). There are no input registers / discrete inputs (a master cannot
write those).

The transfers are derived entirely from a backend map, so the same bridge serves
any use-case pair whose map follows that layout; ``--map`` selects it (default:
this testbed's Water Distribution map). Examples:

    python modbus_bridge.py --controller localhost:1702 --simulator localhost:1703
    python modbus_bridge.py --map /path/to/wt/openplc_map.yaml \\
        --controller localhost:1802 --simulator localhost:1803
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

try:
    from pymodbus.client import ModbusTcpClient
except ImportError as exc:  # pragma: no cover
    sys.exit(f"needs pyyaml and pymodbus: {exc}")

log = logging.getLogger("modbus_bridge")
DEFAULT_MAP = Path(__file__).resolve().parent.parent / "configs" / "openplc_map.yaml"


def build_blocks(mappings: dict) -> list[dict]:
    """One copy-block per (owner, register_type): src=owner, dst=peer."""
    blocks = []
    for owner in ("controller", "simulator"):
        for rt in ("coil", "holding_register"):
            entries = [v for v in mappings.values()
                       if v.get("owner") == owner and v["register_type"] == rt]
            if not entries:
                continue
            lo = min(e["address"] for e in entries)
            hi = max(e["address"] + int(e.get("words", 1)) for e in entries)
            blocks.append({"owner": owner, "kind": rt, "lo": lo, "count": hi - lo})
    return blocks


def parse_hp(s: str, default_port: int) -> tuple[str, int]:
    if ":" in s:
        h, p = s.rsplit(":", 1)
        return h, int(p)
    return s, default_port


def main() -> None:
    ap = argparse.ArgumentParser(description="controller<->simulator Modbus bridge (map-derived)")
    ap.add_argument("--map", default=os.environ.get("BRIDGE_MAP", str(DEFAULT_MAP)),
                    help="backend map YAML (default: this testbed's WD map)")
    ap.add_argument("--controller", default=os.environ.get("CONTROLLER_ADDR", "localhost:1702"))
    ap.add_argument("--simulator", default=os.environ.get("SIMULATOR_ADDR", "localhost:1703"))
    ap.add_argument("--cycle-ms", type=int, default=100)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if Path(a.map).suffix == ".json":
        data = json.loads(Path(a.map).read_text())
        source_hash = data.get("source_yaml_sha256")
        if source_hash:
            source = Path(a.map).with_suffix(".yaml")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != source_hash:
                sys.exit("generated JSON map does not match source YAML")
        mappings = data["mappings"]
    else:
        import yaml
        mappings = yaml.safe_load(Path(a.map).read_text())["mappings"]
    blocks = build_blocks(mappings)
    ch, cp = parse_hp(a.controller, 502)
    sh, sp = parse_hp(a.simulator, 502)
    ctrl, sim = ModbusTcpClient(ch, port=cp, timeout=2), ModbusTcpClient(sh, port=sp, timeout=2)
    for _ in range(30):
        if ctrl.connect() and sim.connect():
            break
        time.sleep(2)
    else:
        sys.exit("cannot connect to both PLCs")
    log.info("Connected. Controller %s:%d Simulator %s:%d", ch, cp, sh, sp)
    log.info("Blocks: %s", blocks)

    clients = {"controller": ctrl, "simulator": sim}
    cycles = errors = 0
    cycle_s = a.cycle_ms / 1000.0
    try:
        while True:
            t0 = time.monotonic()
            for b in blocks:
                src = clients[b["owner"]]
                dst = clients["simulator" if b["owner"] == "controller" else "controller"]
                try:
                    if b["kind"] == "coil":
                        rr = src.read_coils(b["lo"], count=b["count"])
                        if not rr.isError():
                            dst.write_coils(b["lo"], list(rr.bits[:b["count"]]))
                    else:
                        rr = src.read_holding_registers(b["lo"], count=b["count"])
                        if not rr.isError():
                            dst.write_registers(b["lo"], list(rr.registers))
                except Exception as exc:  # keep the loop alive on transient errors
                    errors += 1
                    log.debug("block %s error: %s", b, exc)
            cycles += 1
            if cycles % 100 == 0:
                log.info("cycles=%d errors=%d last=%.1fms", cycles, errors,
                         (time.monotonic() - t0) * 1000)
            sleep = cycle_s - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()
        sim.close()


if __name__ == "__main__":
    main()
