"""EAR source, variant, export-wrapper and independent discovery contracts."""
import copy
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from stable_audio_wanderer.vae import ear_weights as sources
from stable_audio_wanderer.vae import ear_export as exporter
from stable_audio_wanderer.vae.adapters import ear_vae


def fixture(tmp_path, variant='ear_vae_44k', transformer=True):
    repo=tmp_path/'repo';(repo/'model').mkdir(parents=True);(repo/'config').mkdir()
    for name in ('ear_vae.py','autoencoders.py','transformer.py'):
        (repo/'model'/name).write_text('# synthetic test source\n')
    stride=sources.VARIANTS[variant]['ratio']
    config=dict(transformer={'depth':2},encoder={'config':{'strides':[stride]}},
                decoder={'config':{'strides':[stride],'latent_dim':64,'out_channels':2}})
    path=repo/'config'/sources.VARIANTS[variant]['config'];path.write_text(json.dumps(config))
    weights=tmp_path/'outside-repository.pyt'
    state={'decoder.weight':torch.ones(1)}
    if transformer:state['transformers.weight']=torch.ones(1)
    torch.save(state,weights)
    return repo,path,weights


@pytest.mark.parametrize('variant,transformer',[('ear_vae_44k',True),('ear_vae_48k',False)])
def test_effective_config_and_external_checkpoint_identity(tmp_path,variant,transformer):
    repo,config,weights=fixture(tmp_path,variant,transformer)
    source=sources.resolve_source(variant,weights,repo,config)
    assert bool(source.config['transformer'])==transformer
    assert source.identity['transformer']==('present' if transformer else 'absent')
    assert json.loads(config.read_text())['transformer']=={'depth':2}
    assert source.identity['code_sha256']==sources.code_digest(source.code_files)
    original=copy.deepcopy(source.identity)
    (repo/'model/autoencoders.py').write_text('# replacement\n')
    with pytest.raises(ValueError,match='Stale EAR'):
        sources.resolve_source(variant,weights,repo,config,original)


def test_unrecognized_checkpoint_needs_explicit_config_and_no_v2_fallback(tmp_path):
    repo,config,weights=fixture(tmp_path)
    with pytest.raises(ValueError,match='matching config explicitly'):
        sources.resolve_source('ear_vae_44k',weights,repo)
    with pytest.raises(ValueError,match='configuration missing'):
        sources.resolve_source('ear_vae_48k',weights,repo)


def test_supplied_checkpoint_wrong_variant_and_changed_config_rejected(tmp_path,monkeypatch):
    repo,config,weights=fixture(tmp_path)
    variants=copy.deepcopy(sources.VARIANTS)
    variants['ear_vae_44k'].update(weights_sha256=sources.sha256(weights),config_sha256=sources.sha256(config))
    monkeypatch.setattr(sources,'VARIANTS',variants)
    with pytest.raises(ValueError,match='different sample-rate'):
        sources.resolve_source('ear_vae_48k',weights,repo,config)
    config.write_text('{}')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        sources.resolve_source('ear_vae_44k',weights,repo,config)


def test_checkpoint_replacement_invalidates_expected_identity(tmp_path):
    repo,config,weights=fixture(tmp_path)
    first=sources.resolve_source('ear_vae_44k',weights,repo,config)
    torch.save({'decoder.weight':torch.zeros(1),'transformers.weight':torch.ones(1)},weights)
    with pytest.raises(ValueError,match='Stale EAR'):
        sources.resolve_source('ear_vae_44k',weights,repo,config,first.identity)


def test_no_arbitrary_checkpoint_container_or_transformer_removal(tmp_path):
    repo,config,weights=fixture(tmp_path)
    torch.save({'state_dict':{'weight':torch.ones(1)}},weights)
    with pytest.raises(ValueError,match='unwrapped tensor'):
        sources.resolve_source('ear_vae_44k',weights,repo,config)
    torch.save({'transformers.weight':torch.ones(1)},weights)
    raw=json.loads(config.read_text());raw['transformer']=None;config.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='transformer is missing'):
        sources.resolve_source('ear_vae_44k',weights,repo,config)


def test_wrong_geometry_and_missing_weights_fail_early(tmp_path):
    repo,config,weights=fixture(tmp_path)
    with pytest.raises(ValueError,match='geometry'):
        sources.resolve_source('ear_vae_48k',weights,repo,config)
    with pytest.raises(ValueError,match='weight file'):
        sources.resolve_source('ear_vae_44k','')


def test_import_origin_and_code_change_are_rejected(tmp_path,monkeypatch):
    repo,config,weights=fixture(tmp_path)
    source=sources.resolve_source('ear_vae_44k',weights,repo,config)
    monkeypatch.setitem(sys.modules,'model',SimpleNamespace(__path__=[str(tmp_path/'another-model')]))
    with pytest.raises(ValueError,match='Conflicting EAR'):
        sources.load_model_class(source)
    (repo/'model/ear_vae.py').write_text('# changed after selection\n')
    with pytest.raises(ValueError,match='code changed after selection'):
        sources.load_model_class(source)


def test_ear_wrapper_calls_full_decode_including_transformer(monkeypatch):
    class Model(torch.nn.Module):
        def decode(self,z):return z+7  # stands for transformer followed by decoder
        def decoder(self,z):pytest.fail('must not bypass transformer')
    adapter=SimpleNamespace(_model=Model())
    monkeypatch.setattr('stable_audio_wanderer.vae.registry.load_vae_adapter',lambda *a,**k:adapter)
    wrapper,_=exporter.load_wrapper('ear_vae_44k','test')
    assert torch.equal(wrapper(torch.zeros(1,64,2)),torch.full((1,64,2),7.))


def test_adapter_loads_checkpoint_strictly(tmp_path,monkeypatch):
    repo,config,weights=fixture(tmp_path)
    class WrongModel(torch.nn.Module):
        def __init__(self,model_config):super().__init__()
    monkeypatch.setattr(sources,'load_model_class',lambda s:WrongModel)
    with pytest.raises(RuntimeError,match='Unexpected key'):
        ear_vae.EarVAEAdapter.load(weight_path=str(weights),repo_path=str(repo),config_path=str(config),device='cpu')


def test_registry_enforces_sample_rate():
    with pytest.raises(ValueError,match='sample-rate mismatch'):
        ear_vae._load_ear_48k(sample_rate=44100)
    assert ear_vae._INFO_48K.latent_hz==50


def test_fixture_source_and_geometry_are_checked(tmp_path):
    path=tmp_path/'probes.npz'
    np.savez(path,vae_id='ear_vae_48k',source_identity='{}',raw_latents=np.zeros((3,64,128),np.float32))
    with pytest.raises(ValueError,match='source does not match'):
        exporter.probe_samples('ear_vae_44k',{},path)
    samples,evidence=exporter.probe_samples('ear_vae_48k',{},path)
    assert list(samples)==list(range(2,33,2))
    assert all(len(p)==5 for p in samples.values())
    assert not evidence['synthetic_only']
    with pytest.raises(ValueError,match='source does not match'):
        exporter.probe_samples('ear_vae_48k',{'weights_sha256':'new'},path)


def test_onnx_discovery_survives_missing_ear_dependencies(monkeypatch):
    from stable_audio_wanderer.vae import decoder_availability as discovery
    monkeypatch.setattr(discovery,'find_spec',lambda name: object() if name in ('onnx','onnxruntime') else None)
    artifact=SimpleNamespace(source=(),validate_corpus=lambda spec:None)
    monkeypatch.setattr('stable_audio_wanderer.vae.onnx_artifacts.resolve_artifact',lambda *a,**k:artifact)
    entries=discovery.decoder_availability(corpus_spec={'vae_id':'ear_vae_48k'})
    assert entries[0]['backend']=='onnxruntime' and entries[0]['selectable']
    assert all(not e['selectable'] for e in entries if e['backend']=='pytorch')


def test_dynamo_export_has_dynamic_time_and_fixed_fallback(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(torch.onnx,'export',lambda *a,**k:calls.append(k))
    sample=np.zeros((1,64,8),np.float32)
    exporter.export_graph(torch.nn.Identity(),tmp_path/'dynamic.onnx',sample,dynamic=True,opset=18)
    exporter.export_graph(torch.nn.Identity(),tmp_path/'fixed.onnx',sample,dynamic=False,opset=18)
    assert calls[0]['dynamo'] is True and 2 in calls[0]['dynamic_shapes'][0]
    assert calls[1]['dynamo'] is False and calls[1]['dynamic_axes'] is None


def test_onnx_selected_local_config_and_code_are_bound_without_native_imports(tmp_path,monkeypatch):
    repo,config,weights=fixture(tmp_path)
    inside=repo/'weight.pyt';inside.write_bytes(weights.read_bytes())
    identity=sources.selected_file_identity('ear_vae_44k',inside)
    assert identity['config_sha256']==sources.sha256(config)
    assert identity['code_sha256']==sources.code_digest(sources.code_files(repo))
    config.write_text('{}')
    assert sources.selected_file_identity('ear_vae_44k',inside)!=identity
    # Independently available ONNX needs no repository for a standalone weight.
    assert sources.selected_file_identity('ear_vae_44k',weights)=={'weights_sha256':sources.sha256(weights)}
