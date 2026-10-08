from types import SimpleNamespace
import sys

from stable_audio_wanderer.vae import encoder_availability as availability


def test_missing_dependencies_and_weights_do_not_download(monkeypatch, tmp_path):
    monkeypatch.setattr(availability, 'installed', lambda name: False)
    assert not availability.encoder_availability('same_s')['ready']
    monkeypatch.setattr(availability, 'installed', lambda name: True)
    calls = []
    def cached(repo, filename, revision):
        calls.append((repo, filename, revision))
        return None
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(try_to_load_from_cache=cached))
    for vae in ('same_s', 'stable_audio_open'):
        result = availability.encoder_availability(vae)
        assert result['ready'] and result['download_required']
        assert ('No login required' if vae == 'same_s' else 'login and model access required') in result['detail']
    assert len(calls) == 4
    assert all(revision for _, _, revision in calls)
    # A stale cache path must not count as a downloaded model.
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'try_to_load_from_cache', lambda *a, **kw: str(tmp_path/'missing'))
    assert availability.encoder_availability('same_s')['download_required']
    file = tmp_path/'weights'
    file.write_bytes(b'cached')
    monkeypatch.setattr(sys.modules['huggingface_hub'], 'try_to_load_from_cache', lambda *a, **kw: str(file))
    assert availability.encoder_availability('same_s')['ready']
    assert availability.encoder_availability('stable_audio_open')['ready']


def test_preprocess_is_rejected_before_worker_starts(monkeypatch, tmp_path):
    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    monkeypatch.setattr(availability, 'encoder_availability', lambda _: dict(ready=False, detail='Weights missing'))
    manager = PipelineManager()
    messages = []
    monkeypatch.setattr(manager, '_emit', messages.append)
    try:
        manager.handle_message({'type':'pipeline_start_preprocess', 'config':{'audio_dir':str(tmp_path), 'vae_id':'same_s'}})
        assert manager.phase == 'idle'
        assert messages[-1]['error'] == 'Weights missing'
    finally:
        manager.close()


def test_same_s_encoder_uses_verified_local_files(monkeypatch, tmp_path):
    from stable_audio_wanderer.vae.adapters.same_s import SameSAdapter
    from stable_audio_wanderer.vae import same_s_weights
    calls = []
    class Model:
        def eval(self): return self
        def requires_grad_(self, value): return self
        def to(self, device): return self
    def load(config, weights, device):
        calls.append((config, weights))
        return Model()
    def wrapper(model, sr, device):
        assert sr == 44100
        return model
    monkeypatch.setitem(sys.modules, 'stable_audio_3', SimpleNamespace(AutoencoderModel=wrapper))
    monkeypatch.setitem(sys.modules, 'stable_audio_3.loading_utils', SimpleNamespace(load_autoencoder=load))
    def resolve(*, local_files_only):
        assert local_files_only
        return SimpleNamespace(config_path=tmp_path/'config.json', model_path=tmp_path/'model.safetensors')
    monkeypatch.setattr(same_s_weights, 'resolve_same_s_weights', resolve)
    assert SameSAdapter.load().info().vae_id == 'same_s'
    assert calls == [(str(tmp_path/'config.json'), str(tmp_path/'model.safetensors'))]


def test_download_reuses_cache_and_reports_missing_file(monkeypatch, tmp_path):
    import threading
    cached = tmp_path/'config'
    cached.write_bytes(b'config')
    calls, events = [], []
    def lookup(repo, name, revision):
        return str(cached) if name.endswith('json') else None
    def download(repo, name, revision):
        calls.append((repo, name, revision))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        try_to_load_from_cache=lookup, hf_hub_download=download))
    cancel = threading.Event()
    for vae in ('same_s', 'stable_audio_open'):
        availability.prepare_encoder_weights(vae, events.append, cancel)
    assert len(calls) == 2
    assert all(revision for _, _, revision in calls)
    assert [event['event'] for event in events] == ['model_download', 'model_download_done'] * 2
    cancel.set()
    availability.prepare_encoder_weights('same_s', events.append, cancel)
    assert len(calls) == 2


def test_download_errors_are_actionable(monkeypatch):
    import threading
    import pytest
    for status, expected in [(401, 'Authentication failed'), (403, 'Access denied'), (404, 'revision/file unavailable')]:
        error = RuntimeError('secret URL must not reach browser')
        error.response = SimpleNamespace(status_code=status)
        wrapped = RuntimeError('wrapper')
        wrapped.__cause__ = error
        assert expected in availability.download_error('model', wrapped)
        assert 'secret' not in availability.download_error('model', wrapped)
    assert 'Network/offline' in availability.download_error('model', ConnectionError())
    def fail(*a, **kw):
        raise ConnectionError('network down')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        try_to_load_from_cache=lambda *a, **kw: None, hf_hub_download=fail))
    events = []
    with pytest.raises(RuntimeError, match='Network/offline'):
        availability.prepare_encoder_weights('same_s', events.append, threading.Event())
    assert [event['event'] for event in events] == ['model_download']


def test_optional_encoder_setup_is_model_specific(monkeypatch):
    monkeypatch.setattr(availability, 'installed', lambda name: False)
    for vae, profile in [('stable_audio_open', 'requirements-stable-audio-open-native.txt'), ('ear_vae_44k', 'requirements-ear-native.txt')]:
        result = availability.encoder_availability(vae)
        assert not result['ready'] and result['install_required'] and result['optional']
        assert profile in result['setup']
