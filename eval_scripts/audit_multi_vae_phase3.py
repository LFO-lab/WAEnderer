"""Real EAR parity, offline load, GPU comparison and optional muted Web transport."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np


def offline(vae_id, fixture):
    import importlib.abc
    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'model','dac','audiotools','diffusers','huggingface_hub','onnxscript'}:
                raise ImportError('Audit blocked native/export library: '+fullname)
    sys.meta_path.insert(0,BlockNative())
    import socket
    def blocked(*a,**kw):raise RuntimeError('Audit blocked network access')
    socket.socket.connect=blocked
    from stable_audio_wanderer.vae.onnx_artifacts import resolve_artifact
    from stable_audio_wanderer.vae.decoder_factory import select_decoder,create_decoder
    artifact=resolve_artifact(vae_id)
    spec=dict(vae_id=vae_id,sample_rate=artifact.sample_rate,latent_hz=artifact.sample_rate/artifact.samples_per_latent,latent_dim=64)
    with np.load(fixture,allow_pickle=False) as data: raw=data['raw_latents'][0]
    rows=[]
    for explicit in (False,True):
        config=dict(decoder_backend='onnxruntime',decoder_device='cpu')
        if explicit:config['decoder_artifact_dir']=str(artifact.root)
        model=create_decoder(select_decoder(config,corpus_spec=spec))
        try:
            for w in model.supported_windows:
                result=model.decode(np.ascontiguousarray(raw[:,:w].T))
                assert result.audio.shape==(w*artifact.samples_per_latent,2)
            assert all(s.get_providers()==['CPUExecutionProvider'] for s in model._sessions.values())
            rows.append(dict(explicit=explicit,windows=list(model.supported_windows),provider=model.info.provider))
        finally:model.close()
    return dict(passed=True,loads=rows,working_directory=os.getcwd(),native_and_network_blocked=True)


def run(args):
    if args.offline_child:return offline(args.vae_id,args.fixture)
    import torch
    from stable_audio_wanderer.vae.ear_export import load_wrapper,probe_samples,TOLERANCES
    from stable_audio_wanderer.vae.ear_weights import VARIANTS
    from stable_audio_wanderer.vae.onnx_artifacts import resolve_artifact,sha256
    from stable_audio_wanderer.vae.decoder_factory import select_decoder,create_decoder
    from stable_audio_wanderer.vae.export_common import parity
    from stable_audio_wanderer.runtime.overlap_add import StreamingFullOverlapAdd
    torch.set_num_threads(1)
    artifact=resolve_artifact(args.vae_id)
    wrapper,adapter=load_wrapper(args.vae_id,args.weights,args.repo,expected_source=dict(artifact.source))
    samples,evidence=probe_samples(args.vae_id,adapter.source,args.fixture,args.corpus)
    v=VARIANTS[args.vae_id]
    spec=dict(vae_id=args.vae_id,sample_rate=v['sample_rate'],latent_hz=v['sample_rate']/v['ratio'],latent_dim=64)
    report=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),vae_id=args.vae_id,
        source=adapter.source,artifact_identity=artifact.identity,artifact_dir=str(artifact.root),
        files=dict(artifact.files),artifact_bytes=sum((artifact.root/n).stat().st_size for n,_ in artifact.files),
        graphs=[dict(path=g.path,dynamic=g.dynamic,windows=list(g.windows)) for g in artifact.graphs],
        export_parity=json.loads((artifact.root/'parity.json').read_text()),
        fixture_sha256=sha256(args.fixture),native_cpu_threads=torch.get_num_threads(),native_cpu=[],gpu=[],physical=[])
    model=create_decoder(select_decoder(dict(decoder_backend='onnxruntime',decoder_device='cpu'),corpus_spec=spec))
    references={}
    try:
        for w,probes in samples.items():
            references[w]=[]
            for sample in probes:
                with torch.inference_mode():
                    ref=wrapper(torch.from_numpy(sample)).numpy()
                    repeat=wrapper(torch.from_numpy(sample)).numpy()
                actual=model.decode(np.ascontiguousarray(sample[0].T)).audio.T[None]
                row=dict(window=w,shape=list(ref.shape),repeat=parity(ref,repeat,TOLERANCES[args.vae_id]),
                         cross=parity(ref,actual,TOLERANCES[args.vae_id]))
                report['native_cpu'].append(row);references[w].append(ref)
        with np.load(args.fixture,allow_pickle=False) as data:raw=data['raw_latents'][0]
        a,b=StreamingFullOverlapAdd(4*v['ratio']),StreamingFullOverlapAdd(4*v['ratio'])
        native,onnx=[],[]
        for start in range(0,32,4):
            sample=np.ascontiguousarray(raw[:,start:start+8][None])
            with torch.inference_mode():ref=wrapper(torch.from_numpy(sample)).numpy()[0]
            actual=model.decode(np.ascontiguousarray(sample[0].T)).audio.T
            native.append(a.push(ref));onnx.append(b.push(actual))
        report['ola']=dict(window=8,windows=8,hop_samples=4*v['ratio'],
            **parity(np.concatenate(native,axis=1),np.concatenate(onnx,axis=1),TOLERANCES[args.vae_id]))
    finally:model.close()
    # GPU policy frozen before the first GPU evaluation, separately from ONNX.
    gpu_limits=dict(max_abs_error=1e-3,rmse=1e-4)
    report['gpu_tolerances']=gpu_limits
    devices=[]
    if torch.backends.mps.is_available():devices.append('mps')
    if torch.cuda.is_available():devices.append('cuda:0')
    for device in devices:
        print(f'Comparing native CPU/{device}: {args.vae_id}',flush=True)
        wrapper.to(device)
        for w,probes in samples.items():
            for i,sample in enumerate(probes):
                with torch.inference_mode():actual=wrapper(torch.from_numpy(sample).to(device)).cpu().numpy()
                report['gpu'].append(dict(device=device,window=w,**parity(references[w][i],actual,gpu_limits)))
        wrapper.cpu()
        if device=='mps':torch.mps.empty_cache()
        else:torch.cuda.empty_cache()
    report['untested_hardware']=[d for d in ('mps','cuda:0') if d not in devices]
    del wrapper,adapter,references
    with tempfile.TemporaryDirectory() as temp:
        output=Path(temp)/'offline.json'
        command=[sys.executable,'-m','eval_scripts.audit_multi_vae_phase3','--offline-child',
                 '--vae-id',args.vae_id,'--fixture',str(args.fixture.resolve()),'--output',str(output)]
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'HF_HUB_OFFLINE':'1'}
        subprocess.run(command,cwd=temp,env=env,check=True)
        report['offline']=json.loads(output.read_text())
    if args.physical:
        from bin.serve import _setup_perform_phase
        from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
        from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
        from stable_audio_wanderer.vae.decoder_availability import decoder_availability
        import sounddevice
        if not args.corpus:raise ValueError('--physical requires a full navigation corpus')
        report['audio_device']=sounddevice.query_devices(kind='output')['name']
        report['discovery']=decoder_availability(corpus_spec=corpus_decoder_spec(args.corpus),weight_path=args.weights)
        pipeline=PipelineManager()
        broadcaster=SimpleNamespace(bind_nav_decoder=lambda **kw:None,broadcast_pipeline_message=lambda msg:None)
        pipeline.set_broadcaster(broadcaster)
        def setup(corpus,model,config):
            controller=_setup_perform_phase(corpus,model,config,broadcaster,8765)
            controller.decoder.set_gain(0)
            return controller
        pipeline.set_perform_setup_callback(setup)
        choices=[('onnxruntime','cpu'),('pytorch','cpu'),*(('pytorch',d) for d in devices),('onnxruntime','cpu')]
        try:
            for backend,device in choices:
                print(f'Web transport: {args.vae_id}/{backend}/{device}',flush=True)
                config=dict(corpus_dir=str(args.corpus.resolve()),decoder_backend=backend,decoder_device=device,
                            decoder_window=8,vae_weight_path=args.weights,vae_repo_path=args.repo)
                pipeline.handle_message(dict(type='pipeline_start_perform',config=config))
                if pipeline.phase!='perform':raise RuntimeError(pipeline._perform_error)
                controller=pipeline._perform_controller
                ok,message=controller.set_mode('manual');assert ok,message
                ok,message=controller.start();assert ok,message
                deadline=time.monotonic()+60
                while time.monotonic()<deadline:
                    state=controller._decoder_state()
                    if state['error']:raise RuntimeError(state['error'])
                    if state['decode_timing']['count']>=5 and not controller._prebuffering:break
                    time.sleep(.1)
                assert state['decode_timing']['count']>=5 and not controller._prebuffering,state
                controller.stop()
                row=dict(selection=pipeline._active_decoder_state(),transport=state)
                pipeline.handle_message(dict(type='pipeline_stop_perform'))
                assert pipeline.phase=='idle'
                row['stopped']=True;report['physical'].append(row)
        finally:pipeline.close()
    report.update(passed=True,scope='Functional CPU/ONNX/GPU parity, offline reload and short muted production transport. Not real-time qualification or listening approval.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-id',required=True)
    p.add_argument('--weights',default='')
    p.add_argument('--repo',default='')
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--corpus',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--physical',action='store_true')
    p.add_argument('--offline-child',action='store_true')
    args=p.parse_args()
    try:report=run(args)
    except Exception as exc:
        args.output.write_text(json.dumps(dict(passed=False,error=f'{type(exc).__name__}: {exc}'),indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Passed: {args.output}',flush=True)


if __name__=='__main__':main()
