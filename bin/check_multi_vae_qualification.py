"""Print Phase 5 results without treating CPU deadline failures as decode failures."""
import argparse
import json
from pathlib import Path
from stable_audio_wanderer.qualification import evaluate_multi_vae_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate_multi_vae_campaign(args.manifest)
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    print(rendered, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
