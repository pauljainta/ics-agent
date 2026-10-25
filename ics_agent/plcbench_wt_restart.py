"""Validate a WT simulator restart proof before admitting an episode."""
from __future__ import annotations

import math
from typing import Any, Mapping


def verify_wt_restart_proof(proof: Mapping[str, Any], *,
                            previous_generation: str | None = None,
                            expected_simulator_sha256: str | None = None,
                            expected_controller_endpoint: str | None = None,
                            expected_simulator_endpoint: str | None = None) -> str:
    """Check process generation and source-bound clean chemistry evidence.

    membrane_damage is private simulator state. The hook reports zero only when
    a new process generation and the model's zero-damage permeate readback agree.
    """
    if not isinstance(proof, Mapping) or proof.get('proof_kind') != 'wt-full-restart/v1':
        raise RuntimeError('WT restart proof kind missing or unsupported')
    generation = proof.get('generation_id')
    if not isinstance(generation, str) or not generation or generation == previous_generation:
        raise RuntimeError('simulator restart did not produce a new generation')
    if proof.get('started_at_after') != generation or proof.get('started_at_before') == generation:
        raise RuntimeError('simulator process restart not evidenced')
    if proof.get('membrane_damage') != 0 or proof.get('membrane_damage_basis') != 'source-bound-permeate-readback/v1':
        raise RuntimeError('simulator restart did not verify membrane_damage=0')
    source = proof.get('simulator_st_sha256')
    if not isinstance(source, str) or len(source) != 64:
        raise RuntimeError('WT simulator source hash missing')
    if expected_simulator_sha256 and source != expected_simulator_sha256:
        raise RuntimeError('WT simulator source differs from experiment identity')
    if (expected_controller_endpoint is not None
            and proof.get('controller_endpoint') != expected_controller_endpoint):
        raise RuntimeError('WT restart proof refers to a different controller endpoint')
    if (expected_simulator_endpoint is not None
            and proof.get('simulator_endpoint') != expected_simulator_endpoint):
        raise RuntimeError('WT restart proof refers to a different simulator endpoint')
    if ((expected_controller_endpoint is not None or expected_simulator_endpoint is not None)
            and proof.get('host_endpoint_verified') is not True):
        raise RuntimeError('WT restart proof lacks verified host endpoints')
    if expected_controller_endpoint is not None or expected_simulator_endpoint is not None:
        bindings = proof.get('host_port_bindings')
        if not isinstance(bindings, Mapping):
            raise RuntimeError('WT restart proof lacks host port bindings')
        if (expected_controller_endpoint is not None
                and bindings.get('controller') != expected_controller_endpoint):
            raise RuntimeError('WT controller proxy is not loopback-bound as declared')
        if (expected_simulator_endpoint is not None
                and bindings.get('simulator') != expected_simulator_endpoint):
            raise RuntimeError('WT simulator proxy is not loopback-bound as declared')
    values = proof.get('clean_readback')
    if not isinstance(values, Mapping):
        raise RuntimeError('WT restart clean readback missing')
    bands = {'ChemTreat_Conductivity': (299.0, 301.0),
             'ChemTreat_pH': (7.49, 7.51),
             'ChemTreat_ORP': (199.0, 201.0),
             'RO_Permeate_Conductivity': (1.48, 1.52)}
    try:
        clean = all(math.isfinite(float(values[name])) and lo <= float(values[name]) <= hi
                    for name, (lo, hi) in bands.items())
        clean = clean and all(values[name] == expected for name, expected in {
            'SYS_RUNNING': 1, 'P2_Selected': 0, 'P2_NaOCl_Cmd': 0,
            'ChemTreat_NaOCl_Valve_Sts': 0}.items())
    except (KeyError, TypeError, ValueError):
        clean = False
    if not clean:
        raise RuntimeError('WT restart clean-state readback failed')
    return generation
