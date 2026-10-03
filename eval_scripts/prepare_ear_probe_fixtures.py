"""Encode source-bound EAR test latents from three excerpts of local audio."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from stable_audio_wanderer.vae.ear_export import load_wrapper
from stable_audio_wanderer.vae.ear_weights import VARIANTS
from stable_audio_wanderer.vae.onnx_artifacts import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-id',choices=list(VARIANTS),required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--audio',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    _,adapter=load_wrapper(args.vae_id,args.weights,args.repo)
    v=VARIANTS[args.vae_id];sr,ratio=v['sample_rate'],v['ratio']
    info=sf.info(args.audio)
    target=128*ratio
    frames=math.ceil(target*info.samplerate/sr)
    if info.frames<frames:raise ValueError('Audio is too short for EAR probes')
    starts=sorted({0,(info.frames-frames)//2,info.frames-frames})
    probes=[]
    for i,start in enumerate(starts):
        audio,rate=sf.read(args.audio,start=start,frames=frames,dtype='float32',always_2d=True)
        if audio.shape[1]==1:audio=np.repeat(audio,2,axis=1)
        if audio.shape[1]!=2:raise ValueError('EAR probes require mono or stereo audio')
        divisor=math.gcd(sr,rate)
        audio=resample_poly(audio,sr//divisor,rate//divisor,axis=0)[:target]
        torch.manual_seed(20261003+i)
        z=adapter.encode(torch.from_numpy(np.ascontiguousarray(audio.T[None]))).numpy()[0]
        if z.shape!=(64,128) or not np.isfinite(z).all():raise ValueError(f'Unexpected encoded shape {z.shape}')
        probes.append(z)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,raw_latents=np.stack(probes),vae_id=args.vae_id,
        source_identity=json.dumps(adapter.source,sort_keys=True),sample_rate=sr,
        audio_path=str(args.audio.resolve()),audio_sha256=sha256(args.audio),source_sample_rate=info.samplerate,
        source_start_frames=np.array(starts),source_frames=frames,seed=20261003)
    print(args.output,flush=True)


if __name__=='__main__':main()
