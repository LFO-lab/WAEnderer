import json
import numpy as np
import pytest
from stable_audio_wanderer.vae import stable_audio_open_export as export
from stable_audio_wanderer.vae import stable_audio_open_weights as weights


def test_pinned_source_validates_both_files_and_revision(tmp_path, monkeypatch):
    root = tmp_path/'vae'
    root.mkdir()
    (root/'config.json').write_text('{}')
    (root/'diffusion_pytorch_model.safetensors').write_bytes(b'weights')
    monkeypatch.setattr(weights, 'CONFIG_SHA256', weights.sha256(root/'config.json'))
    monkeypatch.setattr(weights, 'WEIGHTS_SHA256', weights.sha256(root/'diffusion_pytorch_model.safetensors'))
    calls = []
    def cached(repo, name, **kwargs):
        calls.append(kwargs)
        return str(tmp_path/name)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', cached)
    _, source = weights.resolve_source()
    assert all(c == {'revision':weights.SOURCE_REVISION, 'local_files_only':True} for c in calls)
    assert weights.resolve_source(tmp_path)[1] == source
    with pytest.raises(ValueError, match='immutable revision'):
        weights.resolve_source(revision='main')
    (root/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        weights.resolve_source(tmp_path)


@pytest.mark.parametrize('bad', [np.full((1,2,8), np.nan, np.float32),
    np.ones((1,2,8),np.float32), np.zeros((1,2,7),np.float32), np.zeros((1,2,8),np.float64)])
def test_parity_rejects_invalid_pcm_and_numerical_failure(bad):
    with pytest.raises(ValueError):
        export.parity(np.zeros((1,2,8),np.float32),bad)


def test_perfect_silent_parity_is_json_safe():
    x = np.zeros((1,2,8),np.float32)
    json.dumps(export.parity(x,x),allow_nan=False)


def test_vst_timing_uses_measured_ratio_instead_of_rounded_corpus(tmp_path, monkeypatch):
    from bin import export_vst_bundle as vst
    monkeypatch.setattr(export,'load_wrapper',lambda *a: (object(), {'revision':'pinned'},2048))
    def graph(wrapper,path,sample,**kw):path.write_bytes(b'graph')
    monkeypatch.setattr(export,'export_graph',graph)
    monkeypatch.setattr(export,'validate_graph',lambda *a:{'8':[{'max_abs_error':0, 'rmse':0}]})
    result=vst.export_stable_audio_open_decoder_onnx(models_dir=tmp_path/'models',
        reports_dir=tmp_path,latent_samples={8:np.zeros((1,64,8),np.float32)},
        samples_per_latent=2051,repo_or_path=weights.SOURCE_MODEL,opset=18)
    entry=result['windows']['8']
    assert entry['samples_per_latent']==2048
    assert entry['output_samples']==16384
    assert entry['audio_hop_samples']==8192


def test_dynamic_failure_falls_back_and_failed_fixed_parity_never_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec', lambda _: {'vae_id':'stable_audio_open','latent_dim':64,'sample_rate':44100,'latent_hz':21.5})
    (tmp_path/'corpus.npz').write_bytes(b'synthetic corpus')
    source={'model':'test'}
    monkeypatch.setattr(export,'resolve_source',lambda **kw:(tmp_path,source))
    monkeypatch.setattr(export,'load_wrapper',lambda **kw:(object(),source,2048))
    monkeypatch.setattr(export,'corpus_samples',lambda _: {w:[np.zeros((1,64,w),np.float32)] for w in export.WINDOWS})
    calls=[]
    def graph(wrapper,path,sample,**kw):
        calls.append(kw['dynamic'])
        path.write_bytes(b'partial')
        if kw['dynamic']:raise RuntimeError('unsupported dynamic op')
    monkeypatch.setattr(export,'export_graph',graph)
    monkeypatch.setattr(export,'validate_graph',lambda *a: (_ for _ in ()).throw(ValueError('parity failed')))
    monkeypatch.setattr(export,'publish_artifact',lambda *a,**kw:pytest.fail('must not publish'))
    with pytest.raises(ValueError,match='parity failed'):
        export.prepare(tmp_path,force=True)
    assert calls==[True,False]


@pytest.mark.parametrize('dynamic', [True, False])
def test_preparation_covers_all_windows_and_binds_report_before_publish(tmp_path, monkeypatch, dynamic):
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec', lambda _: {'vae_id':'stable_audio_open','latent_dim':64,'sample_rate':44100,'latent_hz':21.5})
    (tmp_path/'corpus.npz').write_bytes(b'synthetic corpus')
    source={'model':'synthetic'}
    monkeypatch.setattr(export,'resolve_source',lambda **kw:(tmp_path,source))
    monkeypatch.setattr(export,'load_wrapper',lambda **kw:(object(),source,2048))
    monkeypatch.setattr(export,'corpus_samples',lambda _: {w:[np.zeros((1,64,w),np.float32)] for w in export.WINDOWS})
    def graph(wrapper,path,sample,**kw):
        if kw['dynamic'] and not dynamic: raise RuntimeError('test dynamic blocker')
        path.write_bytes(b'synthetic graph')
        path.with_suffix('.data').write_bytes(b'synthetic external weights')
    monkeypatch.setattr(export,'export_graph',graph)
    seen=[]
    def validate(wrapper,path,samples):
        seen.extend(samples)
        return {str(w):[] for w in samples}
    monkeypatch.setattr(export,'validate_graph',validate)
    def publish(root,**kw):
        manifest=json.loads((root/'decoder.json').read_text())
        report=json.loads((root/'parity.json').read_text())
        assert manifest['supported_windows']==list(export.WINDOWS)
        assert sorted(w for g in manifest['graphs'] for w in g['windows'])==list(export.WINDOWS)
        assert len(manifest['graphs'])==(1 if dynamic else 16)
        assert all(g['dynamic']==dynamic for g in manifest['graphs'])
        assert report['passed'] and report['source']==source
        assert report['graphs']=={g['path']:export.sha256(root/g['path']) for g in manifest['graphs']}
        assert all(export.sha256(root/name)==digest for name,digest in manifest['files'].items())
        assert sum(name.endswith('.data') for name in manifest['files'])==(1 if dynamic else 16)
        return tmp_path/'published'
    monkeypatch.setattr(export,'publish_artifact',publish)
    assert export.prepare(tmp_path,force=True)==tmp_path/'published'
    assert sorted(seen)==list(export.WINDOWS)


def test_prepare_reuses_verified_export_without_exporting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec={'vae_id':'stable_audio_open','latent_dim':64,'sample_rate':44100,'latent_hz':21.5}
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec',lambda _:spec)
    monkeypatch.setattr(export,'resolve_source',lambda **kw:(tmp_path,{}))
    settings=dict(opset=18,tool_versions={name:export.version(name) for name in
        ('torch','diffusers','onnx','onnxruntime')},strategy='dynamic_then_fixed',tolerances=export.TOLERANCES)
    (tmp_path/'decoder.json').write_text(json.dumps({'export':settings}))
    artifact=SimpleNamespace(root=tmp_path,supported_windows=export.WINDOWS,validate_corpus=lambda s:None)
    monkeypatch.setattr(export,'resolve_artifact',lambda *a,**kw:artifact)
    seen=[]
    decoder=SimpleNamespace(decode=lambda z:seen.append(len(z)),close=lambda:seen.append('closed'))
    monkeypatch.setattr('stable_audio_wanderer.vae.artifact_decoder.load_artifact_decoder',lambda _:decoder)
    monkeypatch.setattr(export,'load_wrapper',lambda **kw:pytest.fail('must not export on reuse'))
    assert export.prepare(tmp_path)==tmp_path
    assert seen==[*export.WINDOWS,'closed']


def test_prepare_rejects_stale_corpus_source_before_export(tmp_path,monkeypatch):
    spec={'vae_id':'stable_audio_open','latent_dim':64,'sample_rate':44100,'latent_hz':21.5,'revision':'stale'}
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec',lambda _:spec)
    monkeypatch.setattr(export,'resolve_source',lambda **kw:(tmp_path,{'revision':weights.SOURCE_REVISION}))
    monkeypatch.setattr(export,'load_wrapper',lambda **kw:pytest.fail('must fail before loading'))
    with pytest.raises(ValueError,match='Corpus source revision'):
        export.prepare(tmp_path)
