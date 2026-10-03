from types import SimpleNamespace
import numpy as np
import pytest
import torch
from stable_audio_wanderer.vae import adapter_decoder as native
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
from stable_audio_wanderer.vae.decoder_contract import DecoderRuntimeError
from stable_audio_wanderer.vae import decoder_factory as factory


def write_corpus(path, **overrides):
    values = dict(vae_id='stable_audio_open', sr=44100, latent_hz=21.5,
        Z_concat=np.zeros((16,64),np.float32), Z_mean=np.zeros(64,np.float32),
        Z_std=np.ones(64,np.float32))
    values.update(overrides)
    path.mkdir(exist_ok=True)
    np.savez(path/'corpus.npz', **values)
    return path


@pytest.mark.parametrize('vae_id,sr,ratio,hz', [('stable_audio_open',44100,2048,21.5),
    ('ear_vae_44k',44100,1024,44100/1024), ('ear_vae_48k',48000,1024,48000/1024)])
def test_registered_adapters_use_measured_pcm_timing_and_explicit_device(monkeypatch, vae_id, sr, ratio, hz):
    calls = []
    class Adapter:
        def info(self):
            return SimpleNamespace(latent_dim=64, sample_rate=sr, latent_hz=hz, channels=2)
        def decode(self, z):
            assert not torch.is_grad_enabled()
            return torch.zeros((1,2,z.shape[-1]*ratio),device=z.device)
    def load(name, **kwargs):
        calls.append((name, kwargs))
        return Adapter()
    monkeypatch.setattr(native, 'load_vae_adapter', load)
    decoder = native.AdapterTorchDecoder(vae_id=vae_id, device='cpu')
    assert calls[0][0] == vae_id and calls[0][1]['device'] == 'cpu'
    assert decoder.supported_windows == tuple(range(2,33,2))
    spec = dict(vae_id=vae_id, latent_dim=64, sample_rate=sr, latent_hz=hz)
    decoder.validate_corpus(spec)
    result = decoder.decode(np.zeros((4,64),np.float32))
    assert result.audio.shape == (4*ratio,2)
    assert result.metadata.audio_hop_samples == 2*ratio
    assert result.metadata.sample_rate == sr
    with pytest.raises(DecoderRuntimeError, match='dimensions'):
        decoder.validate_corpus({**spec,'latent_dim':256})
    with pytest.raises(DecoderRuntimeError, match='latent rate'):
        decoder.validate_corpus({**spec,'latent_hz':10})
    with pytest.raises(DecoderRuntimeError):
        decoder.decode(np.zeros((4,256),np.float32))
    with pytest.raises(DecoderRuntimeError):
        decoder.decode(np.full((4,64),np.nan,np.float32))
    decoder.close()
    decoder.close()
    with pytest.raises(DecoderRuntimeError,match='closed'):
        decoder.decode(np.zeros((2,64),np.float32))


def test_corpus_spec_accepts_legacy_sao_rate_without_same_s_assumptions(tmp_path):
    spec = corpus_decoder_spec(write_corpus(tmp_path/'rack'))
    assert spec == dict(vae_id='stable_audio_open',sample_rate=44100,latent_hz=21.5,latent_dim=64)
    with pytest.raises(ValueError,match='missing metadata'):
        factory.select_decoder({'decoder_backend':'onnxruntime','decoder_store_dir':str(tmp_path/'empty-store')},corpus_spec=spec)


@pytest.mark.parametrize('change', [dict(Z_std=np.zeros(64,np.float32)),
    dict(Z_mean=np.zeros(256,np.float32)),dict(Z_concat=np.zeros((2,64),np.float64)),
    dict(latent_hz=np.nan),dict(sr=48000),dict(vae_id='unknown')])
def test_corpus_spec_rejects_incompatible_data(tmp_path, change):
    with pytest.raises(ValueError):
        corpus_decoder_spec(write_corpus(tmp_path/'bad', **change))


def test_pipeline_selects_from_corpus_not_user_claimed_vae(tmp_path, monkeypatch):
    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    path = write_corpus(tmp_path/'rack')
    calls = []
    def select(config, **kwargs):
        calls.append(kwargs['corpus_spec'])
        return factory.DecoderSelection('pytorch','cpu',('sao',),vae_id='stable_audio_open')
    decoder = SimpleNamespace(supported_windows=(2,4,8),
        info=SimpleNamespace(backend='pytorch',provider='cpu',device='cpu',vae_id='stable_audio_open'),
        validate_corpus=lambda spec:calls.append(spec),close=lambda:None)
    monkeypatch.setattr(factory,'select_decoder',select)
    monkeypatch.setattr(factory,'create_decoder',lambda selection:decoder)
    manager = PipelineManager()
    manager.set_perform_setup_callback(lambda *args:SimpleNamespace(close=lambda:None))
    try:
        manager.handle_message({'type':'pipeline_start_perform','config':{
            'corpus_dir':str(path),'vae_id':'same_s','decoder_backend':'pytorch','decoder_device':'cpu'}})
        assert manager.phase == 'perform'
        assert len(calls) == 2 and all(c['vae_id']=='stable_audio_open' for c in calls)
        assert manager._active_decoder_state()['vae_id']=='stable_audio_open'
    finally:
        manager.close()


def test_ear_requires_weight_path_before_loading():
    with pytest.raises(ValueError,match='weight file'):
        factory.select_decoder({'decoder_backend':'pytorch','decoder_device':'cpu'},
            corpus_spec={'vae_id':'ear_vae_48k'})


def test_sao_selection_pins_cached_files_without_same_s_dependency(tmp_path, monkeypatch):
    import huggingface_hub
    root = tmp_path/'snapshot'/'vae'
    root.mkdir(parents=True)
    (root/'config.json').write_text('{}')
    (root/'diffusion_pytorch_model.safetensors').write_bytes(b'weights-a')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda repo,name,**kwargs:str(root/name.split('/')[-1]))
    from stable_audio_wanderer.vae import stable_audio_open_weights as weights
    monkeypatch.setattr(weights, 'CONFIG_SHA256', weights.sha256(root/'config.json'))
    monkeypatch.setattr(weights, 'WEIGHTS_SHA256', weights.sha256(root/'diffusion_pytorch_model.safetensors'))
    spec = {'vae_id':'stable_audio_open'}
    config = {'decoder_backend':'pytorch','decoder_device':'cpu'}
    first = factory.select_decoder(config,corpus_spec=spec)
    assert first.vae_id == 'stable_audio_open' and first.adapter_path == str(root.parent)
    (root/'diffusion_pytorch_model.safetensors').write_bytes(b'weights-b')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        factory.select_decoder(config,corpus_spec=spec)
