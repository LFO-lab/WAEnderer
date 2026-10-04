"""Prepare installed decoder artifacts explicitly, offline, one model at a time."""
import argparse
import json
from pathlib import Path
from stable_audio_wanderer.runtime.decoder_preparation_jobs import PreparationJobs
from stable_audio_wanderer.vae.onnx_artifacts import VAE_IDS


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--vae-id',choices=sorted(VAE_IDS))
    group.add_argument('--all-installed',action='store_true')
    parser.add_argument('--config-file',help='JSON object with interpreters and per-model requests')
    for flag in ('weights','repo','config','fixture','corpus','store-dir','revision'):
        parser.add_argument('--'+flag)
    parser.add_argument('--opset',type=int)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--fixed-only',action='store_true')
    parser.add_argument('--json',action='store_true',dest='as_json')
    args=vars(parser.parse_args(argv))
    config=json.loads(Path(args.pop('config_file')).read_text()) if args.get('config_file') else {}
    args.pop('config_file',None)
    ids=sorted(VAE_IDS) if args.pop('all_installed') else [args.pop('vae_id')]
    args.pop('vae_id',None)
    as_json=args.pop('as_json'); store=args.pop('store_dir')
    requests=config.get('models',{})
    results=[]
    last_progress=[None]
    def progress(state):
        key=(state['vae_id'],state.get('stage'),state.get('detail'))
        if not as_json and state['status']=='running' and key!=last_progress[0]:
            print(f"{key[0]} · {key[1]}: {key[2] or ''}",flush=True)
            last_progress[0]=key
    jobs=PreparationJobs(interpreters=config.get('interpreters'),store_dir=store or config.get('store_dir'),
                         journal_dir=config.get('journal_dir'),emit=progress)
    try:
        for vae in ids:
            request={**requests.get(vae,{}),**{k:v for k,v in args.items() if v is not None and v is not False},'vae_id':vae}
            try:
                jobs.start(request); result=jobs.wait()
            except (ValueError,TypeError) as exc:
                result={'vae_id':vae,'status':'failed','detail':str(exc)}
            results.append(result)
            if not as_json: print(f"{vae}: {result['status']} — {result.get('detail','')}",flush=True)
    finally: jobs.close()
    if as_json: print(json.dumps(results,indent=2))
    return 0 if all(r['status'] in ('prepared','already_valid') for r in results) else 1


if __name__=='__main__': raise SystemExit(main())
