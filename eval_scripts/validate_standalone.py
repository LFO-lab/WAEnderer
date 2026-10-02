"""Run the unchanged standalone entry point with muted real audio, then shut down.

Uses the default physical output, cached weights only, and an ephemeral OSC port.
The run is bounded and retains readiness, decoding and shutdown evidence.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

import sounddevice as sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 5: parser.error('At least five seconds of running audio are required')
    command = [sys.executable, '-m', 'bin.perform', '--corpus_dir', args.corpus,
               '--output_gain', '0', '--autostart', '--ws_port', '0', '--osc_port', '0',
               '--audio_stats', '--audio_stats_interval', '1', '--initial_navigation_mode', 'manual']
    audio_device = dict(sd.query_devices(kind='output'))
    env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / 'standalone.log'
        log = log_path.open('w')
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        ready_at = None
        deadline = time.monotonic()+180+args.seconds
        text = ''
        try:
            while time.monotonic() < deadline and process.poll() is None:
                text = log_path.read_text(errors='replace')
                if '[audio]' in text and ready_at is None:
                    ready_at = time.monotonic()
                if ready_at is not None and time.monotonic()-ready_at >= args.seconds:
                    break
                time.sleep(.2)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            text = log_path.read_text(errors='replace')
    stats = [line for line in text.splitlines() if line.startswith('[audio]')]
    underruns = [int(n) for n in re.findall(r'underruns_total=(\d+)', text)]
    decoded_stats = sum(float(value) > 0 for value in re.findall(r'decode_ms_last=([0-9.]+)', text))
    passed = (decoded_stats > 0 and ready_at is not None and process.returncode == 0 and '[info] Done.' in text
              and len(stats) >= int(args.seconds) and underruns and max(underruns) == 0)
    report = {'command':command, 'audio_device':audio_device, 'decoded_stats':decoded_stats, 'gain':0, 'offline':True, 'exit_code':process.returncode,
              'observed_audio_stats':len(stats), 'requested_seconds':args.seconds,
              'max_underruns':max(underruns) if underruns else None,
              'shutdown_complete':'[info] Done.' in text, 'passed':bool(passed), 'log':text}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'log'},indent=2))
    if not passed: raise SystemExit('Standalone qualification failed; inspect saved log')


if __name__ == '__main__': main()
