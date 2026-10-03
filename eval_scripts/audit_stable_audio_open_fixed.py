"""Exercise the fixed-window fallback on the real pinned Oobleck decoder."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from stable_audio_wanderer.vae.stable_audio_open_export import (
    load_wrapper, corpus_samples, export_graph, validate_graph, TOLERANCES)
from stable_audio_wanderer.vae.onnx_artifacts import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    wrapper,source,_=load_wrapper()
    report=dict(source=source,tolerances=TOLERANCES,windows={},
        scope='Real fixed-window fallback export/parity; temporary graphs discarded because the validated dynamic artifact is deployed.')
    for w,probes in corpus_samples(args.corpus).items():
        with tempfile.TemporaryDirectory(prefix='sao-fixed-audit-') as temp:
            path=Path(temp)/f'decoder_T{w}.onnx'
            export_graph(wrapper,path,probes[0],dynamic=False,opset=18)
            metrics=validate_graph(wrapper,path,{w:probes})
            report['windows'][str(w)]=dict(sha256=sha256(path),bytes=path.stat().st_size,parity=metrics[str(w)])
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
