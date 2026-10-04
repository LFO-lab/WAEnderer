"""Prepare and atomically publish a local EAR decoder without downloading weights."""
import argparse
from stable_audio_wanderer.cli.prepare_decoders import main as prepare_main


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vae-id',choices=['ear_vae_44k','ear_vae_48k'],required=True)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--repo',default='')
    parser.add_argument('--config',default='')
    parser.add_argument('--fixture')
    parser.add_argument('--corpus')
    parser.add_argument('--store-dir')
    parser.add_argument('--opset',type=int,default=18)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--fixed-only',action='store_true')
    args=vars(parser.parse_args())
    import sys
    return prepare_main(sys.argv[1:])


if __name__=='__main__':raise SystemExit(main())
