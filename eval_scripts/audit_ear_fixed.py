"""Validate the real EAR fixed-window fallback without retaining duplicate models."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from stable_audio_wanderer.vae.ear_export import load_wrapper,probe_samples,TOLERANCES
from stable_audio_wanderer.vae.ear_weights import VARIANTS
from stable_audio_wanderer.vae.export_common import export_graph,validate_graph
from stable_audio_wanderer.vae.onnx_artifacts import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-id',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--fixture',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    wrapper,adapter=load_wrapper(args.vae_id,args.weights,args.repo)
    samples,evidence=probe_samples(args.vae_id,adapter.source,args.fixture)
    report=dict(vae_id=args.vae_id,source=adapter.source,evidence=evidence,tolerances=TOLERANCES[args.vae_id],windows={},
                scope='Real fixed-window parity, temporary duplicate graphs discarded; dynamic artifacts are deployed.')
    for w,probes in samples.items():
        with tempfile.TemporaryDirectory(prefix='ear-fixed-audit-') as temp:
            path=Path(temp)/f'decoder_T{w}.onnx'
            export_graph(wrapper,path,probes[0],dynamic=False,opset=18)
            result=validate_graph(wrapper,path,{w:probes},ratio=VARIANTS[args.vae_id]['ratio'],
                                  channels=2,tolerances=TOLERANCES[args.vae_id])
            report['windows'][str(w)]=dict(parity=result[str(w)],files={f.name:sha256(f) for f in Path(temp).iterdir()},
                                          bytes=sum(f.stat().st_size for f in Path(temp).iterdir()))
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
