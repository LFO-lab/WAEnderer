"""Prepare the pinned Stable Audio Open decoder from installed weights, offline."""
import argparse
from stable_audio_wanderer.vae.stable_audio_open_export import prepare
from stable_audio_wanderer.vae.stable_audio_open_weights import SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--store-dir')
    parser.add_argument('--revision', default=SOURCE_REVISION)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    print(prepare(args.corpus, store_dir=args.store_dir, revision=args.revision,
                  opset=args.opset, force=args.force))


if __name__ == '__main__':
    main()
