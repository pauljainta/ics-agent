"""Admission checks for the WT full-simulator restart hook contract."""
import pytest

from ics_agent.plcbench_wt_restart import verify_wt_restart_proof


SOURCE = 'a' * 64
CLEAN = {
    'ChemTreat_Conductivity': 300.0,
    'ChemTreat_pH': 7.5,
    'ChemTreat_ORP': 200.0,
    'RO_Permeate_Conductivity': 1.5,
    'SYS_RUNNING': 1,
    'P2_Selected': 0,
    'P2_NaOCl_Cmd': 0,
    'ChemTreat_NaOCl_Valve_Sts': 0,
}


def proof():
    return {'proof_kind': 'wt-full-restart/v1', 'generation_id': 'new',
            'started_at_before': 'old', 'started_at_after': 'new',
            'controller_endpoint': '127.0.0.1:19502',
            'simulator_endpoint': '127.0.0.1:19503',
            'host_endpoint_verified': True,
            'host_port_bindings': {'controller': '127.0.0.1:19502',
                                   'simulator': '127.0.0.1:19503'},
            'membrane_damage': 0,
            'membrane_damage_basis': 'source-bound-permeate-readback/v1',
            'simulator_st_sha256': SOURCE, 'clean_readback': dict(CLEAN)}


def test_accepts_new_generation_and_clean_source_bound_readback():
    assert verify_wt_restart_proof(proof(), previous_generation='old',
                                   expected_simulator_sha256=SOURCE,
                                   expected_controller_endpoint='127.0.0.1:19502',
                                   expected_simulator_endpoint='127.0.0.1:19503') == 'new'


@pytest.mark.parametrize('change,reason', [
    ({'started_at_before': 'new'}, 'process restart'),
    ({'generation_id': 'old'}, 'new generation'),
    ({'membrane_damage_basis': 'claimed'}, 'membrane_damage'),
    ({'simulator_st_sha256': 'b' * 64}, 'source differs'),
    ({'controller_endpoint': '127.0.0.1:1'}, 'different controller endpoint'),
    ({'simulator_endpoint': '127.0.0.1:2'}, 'different simulator endpoint'),
    ({'host_port_bindings': {'controller': '0.0.0.0:19502',
                             'simulator': '127.0.0.1:19503'}}, 'not loopback-bound'),
    ({'clean_readback': {**CLEAN, 'RO_Permeate_Conductivity': 2.4}}, 'clean-state'),
])
def test_rejects_unsupported_or_dirty_proof(change, reason):
    candidate = {**proof(), **change}
    with pytest.raises(RuntimeError, match=reason):
        verify_wt_restart_proof(candidate, previous_generation='old',
                                expected_simulator_sha256=SOURCE,
                                expected_controller_endpoint='127.0.0.1:19502',
                                expected_simulator_endpoint='127.0.0.1:19503')


def test_batch_reset_rejects_hook_source_drift_before_modbus(monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts.plcbench_batch import reset_for

    monkeypatch.setattr('scripts.plcbench_batch.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(proof())))
    spec = {'reset': {'kind': 'wt-chemistry-flush-v1',
                      'expected_simulator_sha256': 'b' * 64},
            'command_tags': []}
    reset = reset_for(spec, ('127.0.0.1', 1), ('127.0.0.1', 2), {},
                      restart_hook='/unused/hook')
    with pytest.raises(RuntimeError, match='source differs'):
        reset()
