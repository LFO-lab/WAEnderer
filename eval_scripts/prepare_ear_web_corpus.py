"""Build a small real EAR navigation corpus from local audio for Web auditing."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from stable_audio_wanderer.vae.ear_export import load_wrapper
from stable_audio_wanderer.vae.onnx_artifacts import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae-id',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--audio',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='JSON receipt; audio and corpus stay local')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261003)
    audio_dir=Path('build/phase3')/f'{args.vae_id}_web_audio'
    audio_dir.mkdir(parents=True,exist_ok=True)
    info=sf.info(args.audio)
    length=min(info.frames,8*info.samplerate)
    starts=sorted({0,(info.frames-length)//2,info.frames-length})
    for i,start in enumerate(starts):
        audio,sr=sf.read(args.audio,start=start,frames=length,dtype='float32',always_2d=True)
        sf.write(audio_dir/f'excerpt_{i}.wav',audio,sr,subtype='FLOAT')
    _,adapter=load_wrapper(args.vae_id,args.weights,args.repo)
    from bin.preprocess import run_preprocess
    result=run_preprocess(str(audio_dir),f'phase3_{args.vae_id}',vae=adapter,vae_id=args.vae_id,
                          trim_silence=False,encode_chunk_sec=10,encode_chunk_overlap_sec=0)
    path=Path(result['corpus_path'])
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    for name,key in [('weights_sha256','vae_weights_sha256'),('config_sha256','vae_config_sha256'),
                     ('effective_config_sha256','vae_effective_config_sha256'),('code_sha256','vae_code_sha256'),
                     ('revision','vae_source_revision')]:
        arrays[key]=np.array(adapter.source[name])
    np.savez_compressed(path,**arrays)
    from bin.train_policy import _save_manual_navigation_artifact
    _save_manual_navigation_artifact(arrays, str(path), str(path.parent/"manual_navigation.npz"), 32)
    receipt=dict(corpus_dir=str(path.parent.resolve()),corpus_sha256=sha256(path),source=adapter.source,
                 source_audio=str(args.audio.resolve()),source_audio_sha256=sha256(args.audio),
                 excerpt_starts=starts,excerpt_frames=length,source_sample_rate=info.samplerate,
                 encode_seed=20261003,scope='Real local audio encoded with the selected EAR checkpoint; full production preprocessing.')
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
