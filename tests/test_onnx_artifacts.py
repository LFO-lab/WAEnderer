"""Real tiny ONNX graphs exercise persistence/ORT without downloading VAE weights."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
from stable_audio_wanderer.vae.onnx_artifacts import (
    ArtifactError, FORMAT, PROVIDER, sha256, read_artifact, resolve_artifact, publish_artifact)
from stable_audio_wanderer.vae.artifact_decoder import ArtifactOnnxDecoder, load_artifact_decoder
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder
from stable_audio_wanderer.vae.decoder_contract import DecoderRuntimeError


def attest(root, manifest):
    graphs={entry['path']:sha256(root/entry['path']) for entry in manifest['graphs']}
    report={'passed':True,'source':manifest['source'],'graphs':graphs}
    (root/'parity.json').write_text(json.dumps(report))
    manifest['files']={p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*')
                       if p.is_file() and p.name!='decoder.json'}
    (root/'decoder.json').write_text(json.dumps(manifest))


def fixture(root, *, dynamic=True, external=False, input_layout='BDT', output_layout='BCT'):
    root.mkdir(parents=True)
    windows=[2,4,6]
    graphs=[]
    for window in [None] if dynamic else windows:
        t='T' if dynamic else window
        n='N' if dynamic else window*2
        inp=[1,3,t] if input_layout=='BDT' else [1,t,3]
        out=[1,2,n] if output_layout=='BCT' else [1,n,2]
        nodes=[]
        source='latents'
        if input_layout=='BTD':
            nodes.append(helper.make_node('Transpose',['latents'],['bdt'],perm=[0,2,1]))
            source='bdt'
        nodes += [helper.make_node('ReduceMean',[source],['mean'],axes=[1],keepdims=1),
                  helper.make_node('Mul',['mean','gain'],['scaled']),
                  helper.make_node('Tile',['scaled','repeat'],['bct' if output_layout=='BTC' else 'audio'])]
        if output_layout=='BTC': nodes.append(helper.make_node('Transpose',['bct'],['audio'],perm=[0,2,1]))
        graph=helper.make_graph(nodes,'test',
            [helper.make_tensor_value_info('latents',onnx.TensorProto.FLOAT,inp)],
            [helper.make_tensor_value_info('audio',onnx.TensorProto.FLOAT,out)],
            [numpy_helper.from_array(np.array([1],np.float32),'gain'),
             numpy_helper.from_array(np.array([1,2,2],np.int64),'repeat')])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=10)
        name=f'model{window or "dynamic"}.onnx'
        if external:
            # Keep shape-control constants embedded, as production exporters do;
            # ORT cannot infer Tile shapes from an external repeat tensor.
            onnx.external_data_helper.set_external_data(model.graph.initializer[0],location=f'{name}.data')
        onnx.save_model(model,str(root/name))
        graphs.append({'path':name,'dynamic':dynamic,'windows':windows if dynamic else [window]})
    manifest={'format_version':FORMAT,'backend':'onnxruntime','provider':PROVIDER,
        'vae_id':'stable_audio_open','source':{'model':'synthetic-test-only','revision':'test-revision',
            'weights_sha256':'1'*64,'config_sha256':'2'*64},
        'export':{'opset':13,'tool_versions':{'onnx':onnx.__version__}},
        'sample_rate':44100,'channels':2,'latent_dim':3,'samples_per_latent':2,
        'corpus_latent_hz':[22050], 'supported_windows':windows,'default_window':2,
        'input_name':'latents','output_name':'audio','input_layout':input_layout,'output_layout':output_layout,
        'ola_mode':'full_overlap_add','graphs':graphs,'validation_report':'parity.json'}
    attest(root,manifest)
    return manifest


@pytest.mark.parametrize('dynamic',[True,False])
@pytest.mark.parametrize('layouts',[('BDT','BCT'),('BTD','BTC'),('BDT','BTC'),('BTD','BCT')])
def test_real_ort_layouts_windows_cache_and_close(tmp_path,dynamic,layouts):
    root=tmp_path/'artifact'
    fixture(root,dynamic=dynamic,input_layout=layouts[0],output_layout=layouts[1])
    decoder=ArtifactOnnxDecoder(read_artifact(root))
    for window in (2,4,6,2):
        raw=np.ones((window,3),np.float32)
        out=decoder.decode(raw)
        assert out.audio.shape==(window*2,2) and np.all(out.audio==1)
        assert out.metadata.audio_hop_samples==window
        assert len(decoder._sessions)<=2
    with pytest.raises(DecoderRuntimeError):decoder.decode(np.zeros((3,3),np.float32))
    with pytest.raises(DecoderRuntimeError):decoder.decode(np.zeros((2,3),np.float64))
    with pytest.raises(DecoderRuntimeError):decoder.decode(np.full((2,3),np.nan,np.float32))
    decoder.close(); decoder.close()
    assert not decoder._sessions
    with pytest.raises(DecoderRuntimeError,match='closed'):decoder.decode(np.zeros((2,3),np.float32))


def test_publish_external_weights_resolve_and_offline_reload(tmp_path,monkeypatch):
    root=tmp_path/'prepared'
    fixture(root,external=True)
    store=tmp_path/'store'
    target=publish_artifact(root,store)
    assert target.is_dir()
    import socket
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('network access'))
    spec=dict(vae_id='stable_audio_open',latent_dim=3,sample_rate=44100,latent_hz=22050)
    selection=select_decoder({'decoder_backend':'onnxruntime','decoder_store_dir':str(store)},corpus_spec=spec)
    decoder=create_decoder(selection)
    assert np.all(decoder.decode(np.ones((2,3),np.float32)).audio==1)
    decoder.close()
    assert resolve_artifact('stable_audio_open',store_dir=store).root==target
    assert publish_artifact(root,store)==target


@pytest.mark.parametrize('kind',['graph','external','manifest','report'])
def test_corrupt_or_missing_file_fails_before_runtime(tmp_path,kind):
    root=tmp_path/'prepared'; fixture(root,external=True)
    name={'graph':'modeldynamic.onnx','external':'modeldynamic.onnx.data','manifest':'decoder.json','report':'parity.json'}[kind]
    (root/name).unlink()
    with pytest.raises(ArtifactError):read_artifact(root)


@pytest.mark.parametrize('change',[{'vae_id':'ear_vae_48k'}, {'sample_rate':48000},
    {'latent_dim':64},{'latent_hz':21.5},{'weights_sha256':'3'*64},{'config_sha256':'4'*64}])
def test_corpus_geometry_and_checkpoint_must_match(tmp_path,change):
    root=tmp_path/'prepared';fixture(root)
    spec=dict(vae_id='stable_audio_open',sample_rate=44100,latent_dim=3,latent_hz=22050)
    spec.update(change)
    with pytest.raises(ArtifactError):read_artifact(root).validate_corpus(spec)


def test_stale_source_selection_and_file_replacement(tmp_path):
    root=tmp_path/'prepared';fixture(root)
    with pytest.raises(ArtifactError,match='Stale'):
        read_artifact(root,expected_source={'weights_sha256':'0'*64})
    artifact=read_artifact(root)
    path=root/'modeldynamic.onnx'
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ArtifactError,match='SHA-256'):load_artifact_decoder(artifact)


def test_failed_publication_preserves_current_artifact(tmp_path):
    root=tmp_path/'good';fixture(root)
    store=tmp_path/'store';target=publish_artifact(root,store)
    pointer=(store/'stable_audio_open/current.json').read_bytes()
    bad=tmp_path/'bad';manifest=fixture(bad)
    manifest['samples_per_latent']=4
    manifest['corpus_latent_hz']=[11025]
    attest(bad,manifest)
    with pytest.raises(DecoderRuntimeError,match='PCM'):publish_artifact(bad,store)
    assert (store/'stable_audio_open/current.json').read_bytes()==pointer
    assert resolve_artifact('stable_audio_open',store_dir=store).root==target
    assert not list((store/'stable_audio_open').glob('.prepare-*'))


@pytest.mark.parametrize('location',['../outside.data','/tmp/outside.data','undeclared.data'])
def test_external_references_are_checked_before_ort(tmp_path,monkeypatch,location):
    root=tmp_path/'prepared';manifest=fixture(root,external=True)
    path=root/'modeldynamic.onnx'
    model=onnx.load(str(path),load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key=='location':entry.value=location
    path.write_bytes(model.SerializeToString());attest(root,manifest)
    import onnxruntime
    monkeypatch.setattr(onnxruntime,'InferenceSession',lambda *a,**k:pytest.fail('ORT opened unvalidated graph'))
    with pytest.raises(ArtifactError):ArtifactOnnxDecoder(read_artifact(root))


def test_manifest_paths_cannot_escape_via_symlinks(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root)
    path=root/'modeldynamic.onnx';outside=tmp_path/'outside.onnx'
    path.rename(outside);path.symlink_to(outside)
    with pytest.raises(ArtifactError,match='escapes'):read_artifact(root)


def test_parity_report_is_bound_to_graph_and_source(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root)
    manifest['source']['config_sha256']='a'*64
    (root/'decoder.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError,match='attest'):read_artifact(root)


def test_corrupted_external_weights_and_out_of_range_offsets(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root,external=True)
    weight=root/'modeldynamic.onnx.data'
    original=weight.read_bytes()
    weight.write_bytes(b'x'*len(original))
    with pytest.raises(ArtifactError,match='SHA-256'):read_artifact(root)
    weight.write_bytes(original)
    path=root/'modeldynamic.onnx'
    model=onnx.load(str(path),load_external_data=False)
    for entry in model.graph.initializer[0].external_data:
        if entry.key=='offset':entry.value=str(len(original)+1)
    path.write_bytes(model.SerializeToString());attest(root,manifest)
    with pytest.raises(ArtifactError,match='range'):ArtifactOnnxDecoder(read_artifact(root))


def test_layout_and_opset_mismatch_fail_at_load(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root)
    manifest['input_layout']='BTD';attest(root,manifest)
    with pytest.raises(ArtifactError,match='contract'):ArtifactOnnxDecoder(read_artifact(root))
    manifest['input_layout']='BDT';manifest['export']['opset']=14;attest(root,manifest)
    with pytest.raises(ArtifactError,match='opset'):ArtifactOnnxDecoder(read_artifact(root))


def test_selection_identity_changes_with_source_and_rejects_stale_selection(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root)
    first=read_artifact(root)
    manifest['source']['weights_sha256']='a'*64;attest(root,manifest)
    assert first.cache_key!=read_artifact(root).cache_key
    with pytest.raises(ArtifactError,match='changed'):load_artifact_decoder(first)


def test_new_export_replaces_pointer_without_overwriting_previous(tmp_path):
    root=tmp_path/'prepared';manifest=fixture(root)
    store=tmp_path/'store';old=publish_artifact(root,store)
    manifest['source']['weights_sha256']='a'*64;attest(root,manifest)
    new=publish_artifact(root,store)
    assert old!=new and old.is_dir() and new.is_dir()
    assert resolve_artifact('stable_audio_open',store_dir=store).root==new
    assert resolve_artifact('stable_audio_open',store_dir=store,artifact_id=old.name).root==old


def test_parallel_calls_and_close_use_one_bounded_session_pool(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    root=tmp_path/'prepared';fixture(root,dynamic=False)
    decoder=ArtifactOnnxDecoder(read_artifact(root))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs=list(pool.map(lambda w:decoder.decode(np.ones((w,3),np.float32)),[2,4,6]*4))
    assert all(np.all(result.audio==1) for result in outputs)
    assert len(decoder._sessions)<=2
    decoder.close()


def test_new_schema_load_does_not_import_native_model_libraries(tmp_path):
    import subprocess
    import sys
    root=tmp_path/'prepared';fixture(root,external=True)
    script='''
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('diffusers','stable_audio_3','dac','huggingface_hub','onnxscript'):
            raise AssertionError('Native/export-only import: '+fullname)
sys.meta_path.insert(0,NoNative())
from stable_audio_wanderer.vae.onnx_artifacts import read_artifact
from stable_audio_wanderer.vae.artifact_decoder import load_artifact_decoder
import numpy as np
model=load_artifact_decoder(read_artifact(sys.argv[1]))
assert model.decode(np.ones((2,3),np.float32)).audio.shape==(4,2)
model.close()
'''
    result=subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
