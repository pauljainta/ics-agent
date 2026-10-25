"""Failure modes for WT chemistry admission evidence."""
import json
from pathlib import Path

from scripts.plcbench_wt_chemistry_assess import assess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    'ChemTreat_Conductivity': 300.0,
    'ChemTreat_pH': 7.5,
    'ChemTreat_ORP': 200.0,
    'RO_Permeate_Conductivity': 1.5,
    'SYS_RUNNING': 1,
    'ChemTreat_NaOCl_Valve_Sts': 0,
}


def setup_attempt(path, generations):
    spec = (ROOT / 'configs/plcbench/pilot_wt_dosing_v1.json').read_bytes()
    (path / 'pilot_descriptor.json').write_bytes(spec)
    (path / 'admission.json').write_text(json.dumps({
        'status': 'complete', 'source_hashes': {},
        'phases': [{'name': name, 'generation': generation,
                    'clean_readback': BASELINE}
                   for name, generation in zip(
                       ('zero_action', 'reference_naocl', 'restoration'), generations)]}))


def test_missing_phase_is_unassessable(tmp_path):
    setup_attempt(tmp_path, ('a', 'b'))
    result = assess(tmp_path)
    assert result['status'] == 'unassessable'
    assert 'phase proof' in result['reason']


def test_reused_simulator_generation_is_unassessable(tmp_path):
    setup_attempt(tmp_path, ('a', 'a', 'b'))
    result = assess(tmp_path)
    assert result['status'] == 'unassessable'
    assert result['reset_verified'] is False
    assert 'restart' in result['reason']


def test_permeate_readback_rejects_hidden_membrane_damage(tmp_path):
    setup_attempt(tmp_path, ('a', 'b', 'c'))
    data = json.loads((tmp_path / 'admission.json').read_text())
    data['phases'][1]['clean_readback']['RO_Permeate_Conductivity'] = 2.4
    (tmp_path / 'admission.json').write_text(json.dumps(data))
    result = assess(tmp_path)
    assert result['status'] == 'unassessable'
    assert result['reset_verified'] is False


def test_missing_frozen_objective_is_unassessable(tmp_path):
    result = assess(tmp_path)
    assert result['status'] == 'unassessable'
    assert 'objective missing' in result['reason']
