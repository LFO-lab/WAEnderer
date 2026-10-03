import sys
from types import SimpleNamespace

from stable_audio_wanderer.vae import decoder_availability as discovery
from stable_audio_wanderer.runtime.pipeline_server import PipelineManager


def test_discovery_reports_presence_separately_without_loading_models(tmp_path, monkeypatch):
    (tmp_path / 'decoder.json').write_text('{}')
    (tmp_path / 'decoder.onnx').write_bytes(b'not a validated graph')
    monkeypatch.setattr(discovery, 'find_spec', lambda _: object())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        try_to_load_from_cache=lambda *a, **kw: '/cached/weight'))
    choices = discovery.decoder_availability(tmp_path)
    assert [c['device'] for c in choices] == ['cpu', 'cpu', 'mps', 'cuda:0', 'cuda:1']
    assert all(c['selectable'] and not c['validated'] for c in choices)
    # Deliberately invalid ONNX is present, not falsely declared validated.
    assert choices[0]['weights'] is True


def test_discovery_handles_missing_dependencies_weights_and_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'find_spec', lambda _: None)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    choices = discovery.decoder_availability(tmp_path)
    assert not any(c['selectable'] or c['validated'] for c in choices)
    assert all(not c['hardware'] for c in choices if c['device'] != 'cpu')
    assert all(not c['weights'] for c in choices)


def test_pipeline_exposes_discovery_without_changing_phase(monkeypatch):
    result = [{'device': 'cpu', 'validated': False}]
    monkeypatch.setattr(discovery, 'decoder_availability', lambda _: result)
    messages = []
    pipeline = PipelineManager()
    pipeline.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=messages.append))
    pipeline.handle_message({'type': 'pipeline_list_decoders'})
    assert messages == [{'type': 'pipeline_decoder_list', 'decoders': result}]
    assert pipeline.phase == 'idle' and pipeline._app_decoder is None



def test_all_registered_vaes_keep_native_cpu_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'find_spec', lambda _: object())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        try_to_load_from_cache=lambda *a, **kw: '/cached/weight'))
    weight = tmp_path / 'ear.pyt'
    weight.write_bytes(b'presence-only')
    for vae_id in ('same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k'):
        choices = discovery.decoder_availability(tmp_path, corpus_spec={'vae_id':vae_id}, weight_path=str(weight))
        cpu = next(c for c in choices if (c['backend'],c['device']) == ('pytorch','cpu'))
        assert cpu['hardware'] and cpu['selectable'] and not cpu['validated']
        assert not any(c['selectable'] for c in choices if c['device'] != 'cpu')
        # SAME-S and Stable Audio Open expose ONNX; EAR preparation UI remains later work.
        assert any(c['backend']=='onnxruntime' for c in choices) == (vae_id in ('same_s', 'stable_audio_open'))


def test_ear_presence_is_not_checkpoint_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'find_spec', lambda _: object())
    monkeypatch.setitem(sys.modules, 'torch', None)
    for name in ('ear_vae_44k','ear_vae_48k'):
        missing = discovery.decoder_availability(corpus_spec={'vae_id':name})
        assert all(not c['weights'] and not c['validated'] for c in missing)
        path = tmp_path / (name+'.pyt')
        path.write_bytes(b'not a checkpoint')
        present = discovery.decoder_availability(corpus_spec={'vae_id':name},weight_path=str(path))
        assert all(c['weights'] and not c['validated'] for c in present)
