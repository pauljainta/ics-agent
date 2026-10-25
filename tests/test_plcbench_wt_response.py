"""Behavioral checks for the experimental WT observation-response cell."""
import json
from pathlib import Path

import pytest

from ics_agent.plcbench_wt_response import (feedback_policy, fixed_policy,
                                             run_case, verify_chain)

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     'configs/plcbench/wt_observation_response_fixture_v1.json').read_text())


def test_feedback_passes_both_hidden_cases_at_equal_authority():
    results = [run_case(CONFIG, case, 'deterministic', feedback_policy)
               for case in CONFIG['scenarios']]
    assert len({tuple(result['authority']) for result in results}) == 1
    assert all(result['safety_pass'] and result['service_pass'] for result in results)
    assert all(result['observation_decision_intervention_response_verified'] for result in results)
    assert all(result['evidence_origin'] == 'experimental_source_derived_offline_fixture'
               for result in results)


def test_constant_dosing_vectors_do_not_satisfy_both_cases():
    for hcl, naocl in ((False, False), (True, False), (False, True), (True, True)):
        results = [run_case(CONFIG, case, 'constant', fixed_policy(hcl=hcl, naocl=naocl))
                   for case in CONFIG['scenarios']]
        assert not all(result['safety_pass'] and result['service_pass']
                       for result in results)
    safe_but_unserved = run_case(CONFIG, 'high_ph', 'none',
                                 fixed_policy(hcl=False, naocl=False))
    assert safe_but_unserved['safety_pass'] and not safe_but_unserved['service_pass']


def test_undeclared_intervention_fails_closed():
    with pytest.raises(ValueError, match='unauthorized'):
        run_case(CONFIG, 'low_orp', 'bad', lambda _observation: 'stop_plant')


def test_event_chain_rejects_decision_intervention_mismatch():
    result = run_case(CONFIG, 'low_orp', 'deterministic', feedback_policy)
    events = result['events']
    assert verify_chain(events)
    events[2]['action_id'] = 'hcl_on'
    assert not verify_chain(events)


def test_openplc_fixture_coils_are_outside_actor_surface_and_bridge_map():
    from scripts.plcbench_wt_response_openplc_probe import ACTIONS, FIXTURE, OBS
    root = Path(__file__).resolve().parents[1]
    fixture = (root / 'testbeds/water_treatment/openplc/st/'
               'wt_simulator_response_v1.st').read_text()
    mapping = json.loads((root / 'testbeds/water_treatment/openplc/configs/'
                          'openplc_map.json').read_text())['mappings']
    mapped_coils = {v['address'] for v in mapping.values()
                    if v['register_type'] == 'coil'}
    assert set(FIXTURE.values()).isdisjoint(mapped_coils)
    assert all(tag not in OBS for tag in FIXTURE)
    assert {tag for tag, _ in ACTIONS.values() if tag} == {
        'P2_Selected', 'P2_NaOCl_Cmd', 'P2_HCl_Cmd'}
    assert all(mapping[tag]['owner'] == 'controller' for tag, _ in ACTIONS.values() if tag)
    for key, address in FIXTURE.items():
        assert f'AT %QX30.{address - 240}' in fixture
    assert 'membrane_damage := 0' not in fixture


def test_openplc_descriptor_pins_fixture_and_full_restart():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    descriptor = json.loads((root / 'configs/plcbench/'
        'wt_observation_response_openplc_v1.json').read_text())
    source = root / descriptor['realization']['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == descriptor['realization']['sha256']
    assert descriptor['realization']['fixture_io'] == {
        'case_high_ph': 240, 'apply_pb': 241,
        'apply_ack': 242, 'active_high_ph': 243}
    assert descriptor['reset']['kind'] == 'full_simulator_process_restart_per_episode'
    assert descriptor['cell_id'] != CONFIG['cell_id']
    assert descriptor['evidence_origin'] == 'isolated_local_openplc_experimental_fixture'
