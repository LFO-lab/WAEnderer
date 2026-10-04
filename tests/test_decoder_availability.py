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
    assert all(c['selectable'] and not c['validated'] for c in choices[1:])
    assert not choices[0]['selectable']
    assert 'failed_validation' in choices[0]['reason_codes']


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
    monkeypatch.setattr(PipelineManager, '_available_decoders', lambda self,spec,data: result)
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
    monkeypatch.setattr('stable_audio_wanderer.vae.ear_weights.file_identity', lambda *a,**k: {'weights_sha256':'custom', 'config_sha256':'custom'})
    for vae_id in ('same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k'):
        choices = discovery.decoder_availability(tmp_path, corpus_spec={'vae_id':vae_id}, weight_path=str(weight), config_path=str(weight))
        cpu = next(c for c in choices if (c['backend'],c['device']) == ('pytorch','cpu'))
        assert cpu['hardware'] and cpu['selectable'] and not cpu['validated']
        assert not any(c['selectable'] for c in choices if c['device'] != 'cpu')
        # All registered VAEs expose ONNX independently of native dependencies.
        assert any(c['backend']=='onnxruntime' for c in choices) is True


def test_ear_presence_is_not_checkpoint_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, 'find_spec', lambda _: object())
    monkeypatch.setitem(sys.modules, 'torch', None)
    for name in ('ear_vae_44k','ear_vae_48k'):
        missing = discovery.decoder_availability(corpus_spec={'vae_id':name})
        assert all(not c['weights'] and not c['validated'] for c in missing if c['backend']=='pytorch')
        path = tmp_path / (name+'.pyt')
        path.write_bytes(b'not a checkpoint')
        present = discovery.decoder_availability(corpus_spec={'vae_id':name},weight_path=str(path))
        assert all(c['weights'] and not c['validated'] for c in present if c['backend']=='pytorch')


def test_explicit_ear_checkpoint_conflict_cannot_leave_onnx_selectable(tmp_path,monkeypatch):
    weight=tmp_path/'selected.pyt';weight.write_bytes(b'another checkpoint')
    from stable_audio_wanderer.vae.onnx_artifacts import sha256
    monkeypatch.setattr(discovery,'find_spec',lambda _:object())
    monkeypatch.setattr('stable_audio_wanderer.vae.onnx_artifacts.resolve_artifact',lambda *a,**k:
        SimpleNamespace(source=(('weights_sha256','a'*64),),graphs=(),identity='old'))
    choices=discovery.decoder_availability(corpus_spec={'vae_id':'ear_vae_44k','weights_sha256':'a'*64},weight_path=str(weight))
    assert not choices[0]['selectable']
    assert 'conflicts' in choices[0]['detail']


def test_native_model_identity_survives_onnx_removal(monkeypatch):
    from stable_audio_wanderer.vae.stable_audio_open_weights import SOURCE_MODEL,SOURCE_REVISION,CONFIG_SHA256,WEIGHTS_SHA256
    artifact=SimpleNamespace(source=tuple(dict(model=SOURCE_MODEL,revision=SOURCE_REVISION,config_sha256=CONFIG_SHA256,weights_sha256=WEIGHTS_SHA256).items()),graphs=(),identity='export-a')
    monkeypatch.setattr('stable_audio_wanderer.vae.onnx_artifacts.resolve_artifact',lambda *a,**k:artifact)
    before=discovery.decoder_availability(corpus_spec={'vae_id':'stable_audio_open'})
    def missing(*a,**k):raise FileNotFoundError('removed ONNX')
    monkeypatch.setattr('stable_audio_wanderer.vae.onnx_artifacts.resolve_artifact',missing)
    after=discovery.decoder_availability(corpus_spec={'vae_id':'stable_audio_open'})
    assert before[1]['model_identity']==after[1]['model_identity']
    assert before[0]['artifact_verified'] and not after[0]['artifact_verified']
