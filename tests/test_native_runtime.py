import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
from stable_audio_wanderer.vae.native_runtime import native_config
from stable_audio_wanderer.vae.process_decoder import ProcessNativeDecoder
from stable_audio_wanderer.vae.decoder_contract import DecoderRuntimeError


def test_registry_preserves_venv_symlink_and_rejects_web_interpreter(tmp_path,monkeypatch):
    python = tmp_path/'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    registry = tmp_path/'runtimes.json'
    registry.write_text(json.dumps(dict(schema_version=1,models={'same_s':{'python':str(python)}})))
    monkeypatch.setenv('WAENDERER_NATIVE_RUNTIMES',str(registry))
    assert native_config('same_s',{'decoder_python':'/untrusted/python'})['decoder_python'] == str(python)
    assert str(python) != str(python.resolve())


def test_worker_reused_and_abort_is_contained(tmp_path,monkeypatch):
    script = tmp_path/'worker.py'
    script.write_text('''import sys,json,base64
import numpy as np
json.loads(sys.stdin.readline())
meta=dict(latent_window=2,latent_dim=3,sample_rate=44100,channels=2,
 samples_per_latent=2,audio_window_samples=4,latent_hop=1,audio_hop_samples=2,ola_mode='full_overlap_add')
print(json.dumps(dict(info=dict(backend='pytorch',provider='cpu',device='cpu',vae_id='same_s',model_path='test'),default_window=2,windows={'2':meta})),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request['operation']=='close':break
 audio=np.ones((4,2),np.float32)
 print(json.dumps(dict(audio=base64.b64encode(audio.tobytes()).decode(),shape=[4,2])),flush=True)
''')
    popen = subprocess.Popen
    monkeypatch.setattr('stable_audio_wanderer.vae.process_decoder.subprocess.Popen',
        lambda *args,**kwargs:popen([sys.executable,str(script)],**kwargs))
    decoder = ProcessNativeDecoder(SimpleNamespace(worker_python=sys.executable,
        worker_request=json.dumps(dict(config={},corpus_spec={}))))
    pid = decoder._process.pid
    for _ in range(2):
        assert np.all(decoder.decode(np.ones((2,3),np.float32)).audio==1)
        assert decoder._process.pid == pid
    decoder._process.kill()
    decoder._process.wait()
    with pytest.raises(DecoderRuntimeError,match='exited'):
        decoder.decode(np.ones((2,3),np.float32))
    decoder.close()


def test_discovery_uses_registered_runtime_dependencies(monkeypatch):
    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    monkeypatch.setattr('stable_audio_wanderer.vae.native_runtime.native_config',
        lambda *args:dict(decoder_python='/native/python'))
    monkeypatch.setattr('stable_audio_wanderer.vae.decoder_availability.decoder_availability',
        lambda *args,**kwargs:[dict(backend='onnxruntime',selectable=True),dict(backend='pytorch',selectable=False)])
    monkeypatch.setattr('stable_audio_wanderer.vae.native_runtime.request_once',
        lambda *args:[dict(backend='pytorch',device='cpu',selectable=True,detail='available')])
    pipeline=PipelineManager()
    try:
        entries=pipeline._available_decoders({'vae_id':'same_s'},{})
        assert entries[1]['selectable'] is True
        assert entries[1]['runtime_python']=='/native/python'
    finally:
        pipeline.close()


def test_default_interpreter_and_relative_sources_are_checkout_independent(tmp_path,monkeypatch):
    folder=tmp_path/'settings'
    folder.mkdir()
    registry=folder/'runtimes.json'
    registry.write_text(json.dumps(dict(schema_version=1,models={'ear_vae_44k':dict(weights='checkpoint.pyt',repo='source')})))
    monkeypatch.setenv('WAENDERER_NATIVE_RUNTIMES',str(registry))
    monkeypatch.chdir(tmp_path)
    config=native_config('ear_vae_44k',{})
    assert config['decoder_python']==sys.executable
    assert config['vae_weight_path']==str(folder/'checkpoint.pyt')
    assert config['vae_repo_path']==str(folder/'source')
