"""Short stopped-engine changes and fresh offline ONNX reload for one corpus.

Uses PipelineManager ownership, without opening an audio device. Physical
transport and browser selection persistence are tested separately by the user.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from eval_scripts.multi_vae_validation_common import (
    add_selection_arguments, selection_config, evidence_context, digest, checked_decode, write_report,
    normalize_device, validate_native_source,
)
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder


def run(args):
    import torch
    torch.set_num_threads(1)
    spec = corpus_decoder_spec(args.corpus)
    corpus_path = args.corpus / 'corpus.npz' if args.corpus.is_dir() else args.corpus
    with np.load(corpus_path, allow_pickle=False) as data:
        raw = np.ascontiguousarray(data['Z_concat'][:8]*data['Z_std']+data['Z_mean'], dtype=np.float32)
    if len(raw) != 8:
        raise ValueError('Lifecycle probe requires eight corpus frames')
    selection = select_decoder(selection_config(args,'onnxruntime','cpu'),corpus_spec=spec)
    report = evidence_context(args,spec,selection)
    report.update(corpus_sha256=digest(corpus_path), torch_threads=torch.get_num_threads(),
        cycles=[], offline_reload=False, passed=False, error=None,
        scope='Pipeline model ownership and offline fresh-process reload; no physical audio or browser preference claim')
    if args.reload_child:
        import importlib.abc
        class BlockNative(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'model','dac','audiotools','diffusers','stable_audio_3','huggingface_hub','onnxscript'}:
                    raise ImportError('Offline audit blocked native library: '+fullname)
        sys.meta_path.insert(0,BlockNative())
        def blocked(*a,**kw):
            raise RuntimeError('Offline audit blocked network')
        socket.socket.connect = blocked
        decoder = create_decoder(selection)
        try:
            checked_decode(decoder,raw)
            report['offline_reload'] = report['passed'] = True
        finally:
            decoder.close()
        write_report(args.output,report)
        return
    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    pipeline = PipelineManager()
    decoded = []
    def setup(corpus, decoder, config):
        if decoder.info.backend == 'pytorch':
            validate_native_source(decoder,selection.artifact)
        result = checked_decode(decoder,raw)
        decoded.append(dict(shape=list(result.audio.shape), decode_ms=result.decode_time_ms))
        return SimpleNamespace(close=lambda: None)
    pipeline.set_perform_setup_callback(setup)
    try:
        for backend,device in [('onnxruntime','cpu'),('pytorch','cpu'),
                               *[('pytorch',d) for d in args.gpu],('onnxruntime','cpu')]:
            config = selection_config(args,backend,device)
            config.update(corpus_dir=str(corpus_path.parent.resolve()), decoder_window=8,
                          decoder_source_identity=dict(selection.artifact.source))
            start = time.perf_counter()
            pipeline.handle_message(dict(type='pipeline_start_perform',config=config))
            if pipeline.phase != 'perform':
                raise RuntimeError(pipeline._perform_error)
            pipeline.handle_message(dict(type='pipeline_stop_perform'))
            if pipeline.phase != 'idle':
                raise RuntimeError('Pipeline did not release Perform ownership')
            report['cycles'].append(dict(backend=backend,device=normalize_device(device), stopped=True,
                preparation_and_decode_ms=(time.perf_counter()-start)*1000, **decoded[-1]))
        pipeline.close()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'reload.json'
            command = [sys.executable,'-m','eval_scripts.validate_multi_vae_lifecycle',
                '--reload-child','--corpus',str(args.corpus.resolve()),'--output',str(output),
                '--artifact-dir',str(selection.artifact.root),'--protocol',str(args.protocol.resolve())]
            env = {**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
            subprocess.run(command,cwd=directory,env=env,check=True,timeout=120)
            child = json.loads(output.read_text())
            report['offline_reload'] = child['passed'] and child['artifact_identity'] == report['artifact_identity']
        report['passed'] = report['offline_reload']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            pipeline.close()
        except Exception as exc:
            report['passed'] = False
            report['error'] = f'Cleanup failed: {exc}'
        write_report(args.output,report)
    if not report['passed']:
        raise SystemExit('Lifecycle probe failed; inspect report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='append',default=[],help='Optional native device, e.g. mps; repeatable')
    parser.add_argument('--reload-child',action='store_true',help=argparse.SUPPRESS)
    add_selection_arguments(parser)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_report(args.output,dict(schema_version=1,passed=False,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
