"""Offline inventory and SAME-S contract probe; never exports or loads EAR weights."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from huggingface_hub import try_to_load_from_cache
from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
from stable_audio_wanderer.vae.decoder_availability import decoder_availability
from stable_audio_wanderer.vae.onnx_decoder import load_same_s_app_decoder


def file_record(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file(): return {'path':str(path),'present':False}
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return {'path':str(path),'present':True,'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same-s-corpus',required=True)
    parser.add_argument('--sao-corpus',required=True)
    parser.add_argument('--ear-44k',required=True)
    parser.add_argument('--ear-48k',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--environment-python',action='append',default=[])
    args = parser.parse_args()
    report = {'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'environment':{},'native_sources':{},'discovery':{}}
    for package in ('torch','onnxruntime','diffusers','stable-audio-3','descript-audio-codec'):
        try: report['environment'][package] = metadata.version(package)
        except metadata.PackageNotFoundError: report['environment'][package] = None
    report['python_executable'] = sys.executable
    report['additional_environments'] = {}
    for executable in args.environment_python:
        code = """from importlib import metadata
import json
result={}
for name in ('torch','onnxruntime','diffusers','stable-audio-3','descript-audio-codec'):
    try: result[name]=metadata.version(name)
    except metadata.PackageNotFoundError: result[name]=None
print(json.dumps(result))
"""
        result=subprocess.run([executable,'-B','-c',code],capture_output=True,text=True,check=True,timeout=30)
        report['additional_environments'][str(Path(executable).absolute())]=json.loads(result.stdout)
    for vae_id,repo,revision,names in [
        ('same_s','stabilityai/SAME-S','fbeb3dcf53a326e5682f38e22e7f740202d44232',('model_config.json','model.safetensors')),
        ('stable_audio_open','stabilityai/stable-audio-open-1.0',None,('vae/config.json','vae/diffusion_pytorch_model.safetensors'))]:
        files=[]
        for name in names:
            cached=try_to_load_from_cache(repo,name,revision=revision)
            record=file_record(cached) if isinstance(cached,str) else {'present':False}
            record.update(name=name,cached_snapshot_path=cached if isinstance(cached,str) else None)
            files.append(record)
        report['native_sources'][vae_id]={'files':files,'loaded':False}
    for vae_id,weight in [('ear_vae_44k',args.ear_44k),('ear_vae_48k',args.ear_48k)]:
        path=Path(weight).expanduser().resolve()
        repo=next((p for p in list(path.parents)[:4] if (p/'model').is_dir()),None)
        config=repo/'config'/('ear_vae_v2.json' if 'v2' in path.name.lower() else 'model_config.json') if repo else None
        entry={'weight':file_record(path),'config':file_record(config) if config else None,'loaded':False}
        if config and config.is_file():
            data=json.loads(config.read_text())
            entry['configured_samples_per_latent']=math.prod(data['encoder']['config']['strides'])
            entry['configured_latent_dim']=data['decoder']['config']['latent_dim']
        report['native_sources'][vae_id]=entry
        report['discovery'][vae_id]=decoder_availability(corpus_spec={'vae_id':vae_id},weight_path=str(path))
    manager=PipelineManager()
    messages=[]
    manager.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=messages.append))
    try:
        for name,path in [('same_s',args.same_s_corpus),('stable_audio_open',args.sao_corpus)]:
            manager.handle_message({'type':'pipeline_list_decoders','corpus_dir':path})
            report['discovery'][name]=messages[-1]
            assert messages[-1].get('vae_id')==name,messages[-1]
        assert manager.phase=='idle' and manager._app_decoder is None
    finally:manager.close()
    root=Path('stable_audio_wanderer/resources/same_s')
    manifest=json.loads((root/'decoder.json').read_text())
    graph=file_record(root/manifest['model'])
    assert graph['sha256']==manifest['model_sha256']
    model=load_same_s_app_decoder()
    try:
        windows=[]
        for w in model.supported_windows:
            result=model.decode(np.zeros((w,256),np.float32))
            assert result.audio.shape==(w*4096,2) and np.isfinite(result.audio).all()
            windows.append({'window':w,'shape':list(result.audio.shape),'decode_ms':result.decode_time_ms})
        report['same_s_onnx']={'graph':graph,'manifest':file_record(root/'decoder.json'),
            'provider':model.info.provider,'offline':True,'windows':windows,'passed':True}
    finally:model.close()
    report['onnx_artifacts']=[str(p) for p in Path('stable_audio_wanderer/resources').rglob('*.onnx')]
    report['scope']='File identities, discovery and SAME-S CPU decode only; no EAR load/export or real-time qualification.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'same_s_onnx_passed':True,'ear_weights_present':all(report['native_sources'][k]['weight']['present'] for k in ('ear_vae_44k','ear_vae_48k'))}))


if __name__=='__main__':main()
