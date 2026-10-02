"""Check measured scenarios without silently qualifying untested devices."""
import argparse
import json
from stable_audio_wanderer.qualification import evaluate_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='Campaign manifest JSON (report paths are relative to it)')
    args = parser.parse_args()
    result = evaluate_campaign(args.manifest)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__': main()
