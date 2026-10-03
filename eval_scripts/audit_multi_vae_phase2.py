"""Real Rack parity, offline restart and optional physical Web transport smoke."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from stable_audio_wanderer.vae.onnx_artifacts import resolve_artifact, sha256
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec


def offline_child(corpus, store):
    import importlib.abc
    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'diffusers','huggingface_hub','onnxscript','stable_audio_3','dac'}:
                raise ImportError('Offline audit blocked native/export dependency: '+fullname)
    sys.meta_path.insert(0, BlockNative())
    import socket
    def no_network(*args, **kwargs): raise RuntimeError('Offline audit blocked network')
    socket.socket.connect = no_network
    selection=select_decoder(dict(decoder_backend='onnxruntime',decoder_device='cpu',decoder_store_dir=store),
                             corpus_spec=corpus_decoder_spec(corpus))
    decoder=create_decoder(selection)
    try:
        for w in decoder.supported_windows:
            result=decoder.decode(np.zeros((w,64),np.float32))
            assert result.audio.shape==(w*2048,2)
        providers=[session.get_providers() for session in decoder._sessions.values()]
        assert all(p==['CPUExecutionProvider'] for p in providers)
        return dict(passed=True,windows=list(decoder.supported_windows),providers=providers,
                    artifact_identity=selection.artifact.identity,working_directory=os.getcwd())
    finally:decoder.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True,type=Path)
    parser.add_argument('--store-dir',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--physical',action='store_true')
    parser.add_argument('--offline-child',action='store_true')
    args=parser.parse_args()
    corpus=args.corpus.resolve()
    store=str(args.store_dir.resolve()) if args.store_dir else None
    if args.offline_child:
        args.output.write_text(json.dumps(offline_child(str(corpus),store),indent=2)+'\n')
        return
    import torch
    from stable_audio_wanderer.vae.stable_audio_open_export import load_wrapper, parity, TOLERANCES
    from stable_audio_wanderer.runtime.overlap_add import StreamingFullOverlapAdd
    torch.set_num_threads(1)
    artifact=resolve_artifact('stable_audio_open',store_dir=store)
    spec=corpus_decoder_spec(corpus)
    config=dict(decoder_backend='onnxruntime',decoder_device='cpu',decoder_store_dir=store)
    decoder=create_decoder(select_decoder(config,corpus_spec=spec))
    wrapper,source,_=load_wrapper()
    with np.load(corpus/'corpus.npz',allow_pickle=False) as data:
        raw=np.ascontiguousarray(data['Z_concat'][:64]*data['Z_std']+data['Z_mean'],dtype=np.float32)
    a,b=StreamingFullOverlapAdd(8192),StreamingFullOverlapAdd(8192)
    native,onnx=[],[]
    try:
        for start in range(0,32,4):
            sample=raw[start:start+8]
            with torch.inference_mode():ref=wrapper(torch.from_numpy(sample.T[None].copy())).numpy()[0]
            actual=decoder.decode(sample).audio.T
            native.append(a.push(ref));onnx.append(b.push(actual))
        assembled=parity(np.concatenate(native,axis=1),np.concatenate(onnx,axis=1))
    finally:decoder.close()
    del wrapper
    report=dict(artifact_identity=artifact.identity,artifact_dir=str(artifact.root),source=source,
        corpus_sha256=sha256(corpus/'corpus.npz'),graph_bytes=sum((artifact.root/g.path).stat().st_size for g in artifact.graphs),
        graph_count=len(artifact.graphs),dynamic=all(g.dynamic for g in artifact.graphs),
        tolerances=TOLERANCES,ola=dict(windows=8,window=8,hop_samples=8192,samples=65536,**assembled),
        export_parity=json.loads((artifact.root/'parity.json').read_text()),physical=[])
    with tempfile.TemporaryDirectory() as temp:
        output=Path(temp)/'offline.json'
        command=[sys.executable,'-m','eval_scripts.audit_multi_vae_phase2','--offline-child','--corpus',str(corpus),'--output',str(output)]
        if store:command+=['--store-dir',store]
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
             'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        subprocess.run(command,cwd=temp,env=env,check=True)
        report['offline_restart']=json.loads(output.read_text())
    if args.physical:
        from bin.serve import _setup_perform_phase
        from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
        from stable_audio_wanderer.vae.decoder_availability import decoder_availability
        import sounddevice
        report['audio_device']=sounddevice.query_devices(kind='output')['name']
        report['discovery']=decoder_availability(corpus_spec=spec)
        pipeline=PipelineManager()
        broadcaster=SimpleNamespace(bind_nav_decoder=lambda **kw:None,broadcast_pipeline_message=lambda msg:None)
        pipeline.set_broadcaster(broadcaster)
        def setup(corpus,model,cfg):
            controller=_setup_perform_phase(corpus,model,cfg,broadcaster,8765)
            controller.decoder.set_gain(0)
            return controller
        pipeline.set_perform_setup_callback(setup)
        choices=[('onnxruntime','cpu'),('pytorch','cpu')]
        if torch.backends.mps.is_available():choices.append(('pytorch','mps'))
        choices.append(('onnxruntime','cpu'))
        try:
            for backend,device in choices:
                print(f'Physical Web setup: {backend}/{device}',flush=True)
                pipeline.handle_message(dict(type='pipeline_start_perform',config=dict(
                    corpus_dir=str(corpus),decoder_backend=backend,decoder_device=device,decoder_window=8,decoder_store_dir=store)))
                if pipeline.phase!='perform':raise RuntimeError(pipeline._perform_error)
                controller=pipeline._perform_controller
                ok,message=controller.set_mode('manual')
                assert ok,message
                ok,message=controller.start()
                assert ok,message
                deadline=time.monotonic()+45
                while time.monotonic()<deadline:
                    state=controller._decoder_state()
                    if state['error']:raise RuntimeError(state['error'])
                    if state['decode_timing']['count']>=5 and not controller._prebuffering:break
                    time.sleep(.1)
                assert state['decode_timing']['count']>=5,state
                controller.stop()
                row=dict(selection=pipeline._active_decoder_state(),transport=state,
                         source_identity=list(pipeline._decoder_selection.identity))
                pipeline.handle_message({'type':'pipeline_stop_perform'})
                assert pipeline.phase=='idle'
                row['stopped']=True
                report['physical'].append(row)
        finally:pipeline.close()
    report['passed']=True
    report['scope']='Functional parity, offline restart, and optional brief muted physical Web setup/transport. No listening or real-time qualification.'
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Passed; report: {args.output}',flush=True)


if __name__=='__main__':main()
