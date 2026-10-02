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
    assert [c['device'] for c in choices] == ['cpu', 'mps', 'cuda:0', 'cuda:1']
    assert all(c['selectable'] and not c['validated'] for c in choices)
    # Deliberately invalid ONNX is present, not falsely declared validated.
    assert choices[0]['weights'] is True


def test_discovery_handles_missing_dependencies_weights_and_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'find_spec', lambda _: None)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    choices = discovery.decoder_availability(tmp_path)
    assert not any(c['selectable'] or c['validated'] for c in choices)
    assert all(not c['hardware'] for c in choices[1:])
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
