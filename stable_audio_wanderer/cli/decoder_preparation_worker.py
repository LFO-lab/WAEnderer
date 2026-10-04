"""Internal offline worker protocol. Interpreter is configured by the server/CLI."""
import argparse
import json
import os
from pathlib import Path
from stable_audio_wanderer.vae.decoder_preparation import validate_request, stage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        result=stage(validate_request(json.loads(Path(args.request).read_text())),args.output)
    except Exception as exc:
        result=dict(status='failed',detail=f'{type(exc).__name__}: {exc}')
    print('WAENDERER_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__': main()
