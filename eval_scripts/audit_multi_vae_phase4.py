"""Real Phase 4 jobs, offline discovery/reconnect, and muted production playback."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import websockets
from stable_audio_wanderer.runtime.decoder_preparation_jobs import PreparationJobs
from .audit_multi_vae_phase2_web import exercise

CORPORA = {
    'same_s':('corpus/BurntMemory_20260915_211501',4096),
    'stable_audio_open':('corpus/Rack_20260428_181107',2048),
    'ear_vae_44k':('corpus/TC_Harpsichord_20261002_172959',1024),
    'ear_vae_48k':('corpus/phase3_ear_vae_48k_20261003_102119',960),
}


async def prepare_and_discover(vae,corpus,port):
    async with websockets.connect(f'ws://127.0.0.1:{port}',max_size=32*1024*1024) as ws:
        async def until(predicate):
            async def read():
                while True:
                    message=json.loads(await ws.recv())
                    if message.get('error'): raise RuntimeError(message['error'])
                    if predicate(message): return message
            return await asyncio.wait_for(read(),120)
        token='phase4-'+vae
        await ws.send(json.dumps(dict(type='pipeline_list_decoders',corpus_dir=corpus,request_id=token,fingerprint=token)))
        discovery=await until(lambda m:m.get('request_id')==token)
        assert discovery['preparation_supported'] and discovery['fingerprint']==token
        assert all(e['vae_id']==vae and vae in e['label'] for e in discovery['decoders'])
        assert any(e['backend']=='pytorch' and e['device']=='cpu' for e in discovery['decoders'])
        await ws.send(json.dumps(dict(type='pipeline_prepare_decoder',corpus_dir=corpus,request={'vae_id':vae})))
        result=await until(lambda m:m.get('type')=='pipeline_preparation' and m['job']['status']!='running')
        assert result['job']['status']=='already_valid',result
    # A new client recovers the job; no export or playback is launched by reconnect.
    async with websockets.connect(f'ws://127.0.0.1:{port}',max_size=32*1024*1024) as ws:
        await ws.send(json.dumps({'type':'pipeline_get_state'}))
        async def read_state():
            while True:
                message=json.loads(await ws.recv())
                if message.get('type')=='pipeline_state':return message
        recovered=await asyncio.wait_for(read_state(),10)
        assert recovered['phase']=='idle'
        assert recovered['corpus_dir']==corpus
        assert recovered['preparation']['job_id']==result['job']['job_id']
    return dict(discovery=discovery,job=result['job'],reconnect=recovered)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=18765)
    parser.add_argument('--http-port',type=int,default=18080)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=dict(scope='Offline real artifacts and production preparation/reconnect; muted short transport, not real-time qualification.',cli=[],runs=[])
    jobs=PreparationJobs(journal_dir=root/'build/phase4/cli-jobs')
    try:
        for vae in CORPORA:
            jobs.start({'vae_id':vae});result=jobs.wait()
            assert result['status']=='already_valid',result
            report['cli'].append(result)
    finally: jobs.close()
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        config=temp/'config.json'
        config.write_text(json.dumps({'journal_dir':str(temp/'jobs')}))
        for run in range(2):
            with (temp/f'server-{run}.log').open('w') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','bin.serve','--port',str(args.port),
                    '--http_port',str(args.http_port),'--decoder-preparation-config',str(config)],cwd=root,
                    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        if process.poll() is not None:raise RuntimeError((temp/f'server-{run}.log').read_text())
                        try:
                            with socket.create_connection(('127.0.0.1',args.port),timeout=.2):break
                        except OSError:time.sleep(.2)
                    rows=[]
                    for vae,(path,ratio) in CORPORA.items():
                        corpus=str((root/path).resolve())
                        row=dict(vae_id=vae,**asyncio.run(prepare_and_discover(vae,corpus,args.port)))
                        row['transport']=asyncio.run(exercise(corpus,args.port,vae_id=vae,samples_per_latent=ratio))
                        rows.append(row)
                        print(f'Run {run+1}: {vae} jobs/reconnect/transport passed',flush=True)
                    report['runs'].append(rows)
                finally:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                if process.returncode!=0:raise RuntimeError((temp/f'server-{run}.log').read_text())
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
