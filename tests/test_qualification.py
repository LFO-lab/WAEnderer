from copy import deepcopy
import pytest
from stable_audio_wanderer.qualification import evaluate_transport


def passing_report():
    return {'audio_device':'physical output', 'clock':'physical output (muted)',
            'navigation':'production', 'requested_seconds':600, 'duration_seconds':600.1,
            'underruns':0, 'error':None, 'rendered_blocks':25000,
            'windows':{str(w):{'decode_ms':{'count':10}} for w in range(2,33,2)},
            'transitions':[{'mode':mode, 'step':step, 'settled_ms':20}
                for mode in ('random','manual','reorganized') for step in range(17)]}


def test_physical_production_campaign_passes():
    assert evaluate_transport(passing_report())['passed']


@pytest.mark.parametrize('changes', [
    {'clock':'silent software callback'}, {'audio_device':None},
    {'navigation':'replay'}, {'requested_seconds':599}, {'duration_seconds':599},
    {'underruns':1}, {'underruns':None}, {'error':'decode failed'},
    {'rendered_blocks':0}, {'duration_seconds':float('nan')}, {'duration_seconds':float('inf')}, {'windows':{}}, {'transitions':[]}])
def test_incomplete_or_failed_evidence_never_qualifies(changes):
    report = passing_report()
    report.update(changes)
    assert not evaluate_transport(report)['passed']


def test_unsettled_transition_cannot_pass():
    report = deepcopy(passing_report())
    report['transitions'][-1]['settled_ms'] = None
    assert not evaluate_transport(report)['passed']


def test_campaign_requires_both_backends_same_corpus_and_listening(tmp_path):
    import json
    from stable_audio_wanderer.qualification import evaluate_campaign
    for name, backend in [('onnx','onnxruntime'),('native','pytorch')]:
        report = passing_report()
        report.update(backend=backend, device='cpu' if name == 'onnx' else 'mps', source={'sha256':'corpus-hash'})
        (tmp_path / f'{name}.json').write_text(json.dumps(report))
    numeric = {'numeric_passed':True, 'corpus_sha256':'corpus-hash',
               'comparisons':[{'window':w, 'passed':True, 'cross':{'rmse':.001}, 'rmse_limit':.003} for w in range(2,33,2)]}
    (tmp_path / 'numeric.json').write_text(json.dumps(numeric))
    manifest = {'transport_reports':{'onnx':'onnx.json','mps':'native.json'}, 'corpus_report':'numeric.json',
                'listening':{'reviewer':'test reviewer','date':'2026-10-02','observation':'similar', 'new_engine_regression':False},
                'unqualified_devices':['cuda']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    result = evaluate_campaign(path)
    assert result['passed'] and result['unqualified_devices'] == ['cuda']
    manifest['listening']['new_engine_regression'] = None
    path.write_text(json.dumps(manifest))
    assert not evaluate_campaign(path)['passed']
    manifest['listening']['new_engine_regression'] = False
    manifest['transport_reports'].pop('mps')
    path.write_text(json.dumps(manifest))
    assert not evaluate_campaign(path)['passed']
    manifest['transport_reports']['mps'] = 'native.json'
    numeric['corpus_sha256'] = 'different-corpus'
    (tmp_path / 'numeric.json').write_text(json.dumps(numeric))
    path.write_text(json.dumps(manifest))
    assert not evaluate_campaign(path)['passed']
