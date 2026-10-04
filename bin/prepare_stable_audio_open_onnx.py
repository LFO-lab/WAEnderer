"""Prepare the pinned Stable Audio Open decoder from installed weights, offline."""
import argparse
from bin.prepare_decoders import main as prepare_main
from stable_audio_wanderer.vae.stable_audio_open_weights import SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--store-dir')
    parser.add_argument('--revision', default=SOURCE_REVISION)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    import sys
    return prepare_main(['--vae-id','stable_audio_open',*sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
