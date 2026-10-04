"""Characterize native CPU/GPU before evaluating ONNX against that GPU.

The candidate is the existing CPU error limits, fixed before this run. Failure
keeps GPU qualification pending; this script never loosens limits or exports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
from stable_audio_wanderer.vae.decoder_factory import select_decoder,create_decoder
from eval_scripts.multi_vae_validation_common import (
    add_selection_arguments,selection_config,evidence_context,digest,write_report,
    checked_decode,compare,numerical_gate,normalize_device,validate_native_source,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--output',type=Path,required=True)
    add_selection_arguments(parser)
    args=parser.parse_args()
    torch.set_num_threads(1)
    spec=corpus_decoder_spec(args.corpus)
    protocol=json.loads(args.protocol.read_text())
    limits=protocol['models'][spec['vae_id']]
    if 'rmse' not in limits:
        parser.error('Use the established stochastic policy for SAME-S instead')
    path=args.corpus/'corpus.npz' if args.corpus.is_dir() else args.corpus
    with np.load(path,allow_pickle=False) as data:
        raw=np.ascontiguousarray(data['Z_concat'][:32]*data['Z_std']+data['Z_mean'],dtype=np.float32)
        if int(data['file_offsets'][1])<32:
            parser.error('First corpus file needs at least 32 frames')
    artifact_selection=select_decoder(selection_config(args,'onnxruntime','cpu'),corpus_spec=spec)
    report=evidence_context(args,spec,artifact_selection)
    report.update(device=normalize_device(args.device),corpus_sha256=digest(path),candidate_limits=limits,
        scope='Native CPU/GPU only; no ONNX decoding or realtime/listening qualification',rows=[],passed=False,error=None)
    references={}
    decoder=None
    try:
        for device in ('cpu',args.device):
            config=selection_config(args,'pytorch',device)
            config['decoder_source_identity']=dict(artifact_selection.artifact.source)
            selected=select_decoder(config,corpus_spec=spec)
            decoder=create_decoder(selected)
            validate_native_source(decoder,artifact_selection.artifact)
            report[device+'_identity']=list(selected.identity)
            for window in protocol['windows']:
                synthetic=np.random.default_rng(window*1000).normal(0,.05,(window,raw.shape[1])).astype(np.float32)
                for name,probe in (('real',raw[:window]),('synthetic',synthetic)):
                    a=checked_decode(decoder,probe).audio
                    b=checked_decode(decoder,probe).audio
                    within=compare(a,b)
                    if device=='cpu':
                        references[window,name]=(a,within)
                    else:
                        ref,within_cpu=references[window,name]
                        cross=compare(ref,a)
                        report['rows'].append(dict(window=window,kind=name,
                            input_sha256=hashlib.sha256(probe.tobytes()).hexdigest(),
                            cross=cross,within_cpu=within_cpu,within_gpu=within,
                            **numerical_gate(limits,cross,within_cpu,within)))
                print(device,f'T{window}: native reference complete',flush=True)
            decoder.close()
            decoder=None
        report['passed']=bool(report['rows']) and all(r['passed'] for r in report['rows'])
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if decoder is not None:
            try:decoder.close()
            except Exception as exc:
                report['error']=f'Cleanup failed: {exc}'
                report['passed']=False
        write_report(args.output,report)
    if not report['passed']:
        raise SystemExit('Native reference gate failed; GPU policy stays pending')


if __name__=='__main__':main()
