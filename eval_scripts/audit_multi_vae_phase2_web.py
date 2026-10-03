"""Start/restart isolated offline Web servers and exercise real WebSocket Perform."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import websockets


async def exercise(corpus, port):
    async with websockets.connect(f'ws://127.0.0.1:{port}',max_size=32*1024*1024) as ws:
        async def send(message):await ws.send(json.dumps(message))
        async def until(predicate,timeout=90):
            async def read():
                while True:
                    message=json.loads(await ws.recv())
                    if message.get('error'):raise RuntimeError(message['error'])
                    if predicate(message):return message
            return await asyncio.wait_for(read(),timeout)
        await send(dict(type='pipeline_list_decoders',corpus_dir=corpus))
        discovery=await until(lambda m:m.get('type')=='pipeline_decoder_list' and m.get('corpus_dir')==corpus)
        assert any(e['backend']=='onnxruntime' and e['selectable'] for e in discovery['decoders'])
        await send(dict(type='pipeline_start_perform',config=dict(corpus_dir=corpus,
            decoder_backend='onnxruntime',decoder_device='cpu',decoder_window=8)))
        ready=await until(lambda m:m.get('type')=='pipeline_phase_change' and m.get('phase')=='perform')
        assert ready['decoder']['provider']=='CPUExecutionProvider'
        await send(dict(type='decoder',params=dict(gain=0)))
        await send(dict(type='transport',action='set_mode',mode='manual'))
        await send(dict(type='transport',action='start'))
        playing=await until(lambda m:m.get('type')=='state' and m.get('decoder',{}).get('decode_timing',{}).get('count',0)>=3)
        assert playing['decoder']['backend']=='onnxruntime'
        assert playing['decoder']['audio_window_samples']==16384
        await send(dict(type='transport',action='stop'))
        await send(dict(type='pipeline_stop_perform'))
        stopped=await until(lambda m:m.get('type')=='pipeline_phase_change' and m.get('phase')=='idle')
        return dict(discovery=discovery,ready=ready,decoder=playing['decoder'],stopped=stopped)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=18765)
    parser.add_argument('--http-port',type=int,default=18080)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=dict(runs=[],scope='Isolated production HTTP/WebSocket server, offline restart, muted short functional playback; no real-time qualification.')
    with tempfile.TemporaryDirectory() as temp:
        for run in range(2):
            log=Path(temp)/f'server-{run}.log'
            with log.open('w') as stream:
                proc=subprocess.Popen([sys.executable,'-u',str(root/'bin/serve.py'),
                    '--port',str(args.port),'--http_port',str(args.http_port)],cwd=root,
                    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1'},
                    stdout=stream,stderr=subprocess.STDOUT)
                try:
                    import socket
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        if proc.poll() is not None:raise RuntimeError(log.read_text())
                        try:
                            with socket.create_connection(('127.0.0.1',args.port),timeout=.2):break
                        except OSError:time.sleep(.2)
                    report['runs'].append(asyncio.run(exercise(str(args.corpus.resolve()),args.port)))
                finally:
                    proc.send_signal(signal.SIGINT)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait()
                if proc.returncode!=0:raise RuntimeError(log.read_text())
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Passed server restart: {args.output}')


if __name__=='__main__':main()
