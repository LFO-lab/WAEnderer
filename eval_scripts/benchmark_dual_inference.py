"""Paced real decoder/transport benchmark with software or muted physical output.

Use --navigation production to exercise learned corpus policies and --audio-device
for Core Audio evidence. Run backends separately without other inference load.
Qualification combines these measurements with parity and listening evidence.
Optional --corpus accepts a supported corpus.npz (normalized Z_concat).
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import functools
import json
from pathlib import Path
import platform
import resource
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from stable_audio_wanderer.runtime.decoder_transport import DecoderTransportController
from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec


def percentiles(values):
    if not values:
        return {'count': 0, 'median': None, 'p95': None, 'p99': None}
    return dict(count=len(values), **dict(zip(('median', 'p95', 'p99'),
                                            map(float, np.percentile(values, [50, 95, 99])))))


class MeasuredDecoder:
    def __init__(self, decoder):
        self.decoder = decoder
        self.times = defaultdict(list)

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def decode(self, raw):
        result = self.decoder.decode(raw)
        self.times[len(raw)].append(result.decode_time_ms)
        return result


class ReplayNavigation:
    """Deterministic frame source, explicitly not a trained navigation policy."""
    control_dim = 3

    def __init__(self, z):
        self.z = z
        self.index = 0

    def has_variant(self, mode): return mode in ('random', 'reorganized')
    def set_policy_variant(self, mode): return self.has_variant(mode)
    def get_active_jump_rate(self, variant=None): return 0.0
    def get_random_controls(self): return {}
    def get_state(self): return {'policy_index': self.index, 'nearest_index': self.index}
    def set_faders(self, values): self.index = int(float(values[0]) * (len(self.z)-1))
    def step_with_faders(self, values):
        self.set_faders(values)
        return SimpleNamespace(nearest_index=self.index, distance=0.0)
    def step(self, fixed_retrieval_window=None):
        self.index = (self.index + 1) % len(self.z)
        return self.index
    def generate_batch(self, frames, exploration=0): return self.z[np.asarray(frames)]


class SilentStream:
    """Main thread supplies callbacks at wall-clock speed; no sounddevice output."""
    def __init__(self, **kwargs): self.active = False
    def start(self): self.active = True
    def stop(self): self.active = False
    def close(self): self.active = False


class ObservedPlayer(DecoderPlayer):
    """Observe the actual callback without replacing the production PCM logic."""
    def __init__(self, **kwargs):
        self.callback_times = []
        self.blocks = 0
        self.peak = 0.0
        self.sum_square = 0.0
        self.nonfinite = False
        self.underrun_events = []
        super().__init__(**kwargs)

    def _callback(self, output, frames, time_info, status):
        before = time.perf_counter()
        previous = self.underruns
        super()._callback(output, frames, time_info, status)
        if self.underruns != previous:
            self.underrun_events.append({'monotonic':before,
                'buffer_underruns':self.buffer_underruns, 'device_underruns':self.device_underruns,
                'state':self.get_state()})
        self.callback_times.append((time.perf_counter()-before)*1000)
        self.nonfinite |= not np.isfinite(output).all()
        self.blocks += 1
        self.peak = max(self.peak, float(np.max(np.abs(output))))
        self.sum_square += float(np.sum(output.astype(np.float64)**2))


def run(args):
    torch.set_num_threads(1)
    if args.corpus:
        spec = corpus_decoder_spec(args.corpus)
        path = args.corpus / "corpus.npz" if args.corpus.is_dir() else args.corpus
        with np.load(path, allow_pickle=False) as data:
            z, mean, std = (np.asarray(data[key], dtype=np.float32) for key in ('Z_concat','Z_mean','Z_std'))
            offsets = np.asarray(data['file_offsets'], dtype=np.int64)
        source = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        spec = None
        z = np.random.default_rng(5006).normal(0, .05, (256,256)).astype(np.float32)
        mean, std = np.zeros(256, np.float32), np.ones(256, np.float32)
        offsets = np.array([0,128,256])
        source = {'synthetic_seed': 5006, 'distribution': 'normal std=.05',
                  'sha256': hashlib.sha256(z.tobytes()).hexdigest()}
    file_ids = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    selection = select_decoder({'decoder_backend': args.backend, 'decoder_device': args.device}, corpus_spec=spec)
    started = time.perf_counter()
    decoder = MeasuredDecoder(create_decoder(selection))
    if hasattr(decoder, "validate_corpus"):
        decoder.validate_corpus(spec)
    preparation_ms = (time.perf_counter()-started)*1000
    sample_rate = decoder.metadata_for(decoder.default_window).sample_rate
    # Patch only stream creation; keep production PCM buffer, callback, OLA,
    # transport scheduler, transitions and decoder implementations.
    import sounddevice as sd
    stream_factory = (functools.partial(sd.OutputStream, device=args.audio_device)
                      if args.audio_device else SilentStream)
    with patch('stable_audio_wanderer.runtime.decoder_player.sd.OutputStream', stream_factory):
        if args.navigation == 'production':
            from bin.serve import _setup_perform_phase
            from stable_audio_wanderer.runtime.ws_server import WSBroadcaster
            def make_player(**kwargs):
                kwargs.update(blocksize=1024, gain=0.0 if args.audio_device else 1.0)
                return ObservedPlayer(**kwargs)
            with patch('stable_audio_wanderer.runtime.decoder_player.DecoderPlayer', make_player):
                controller = _setup_perform_phase(str(args.corpus if args.corpus.is_dir() else args.corpus.parent),
                    decoder, {'decoder_window':2}, WSBroadcaster(), 8765)
            player = controller.decoder
        else:
            player = ObservedPlayer(sr=sample_rate, blocksize=1024, gain=0.0 if args.audio_device else 1.0)
            replay = ReplayNavigation(z)
            controller = DecoderTransportController(nav=replay, manual_engine=replay, manifold=replay,
                player=player, latent_decoder=decoder, z_concat=z, file_offsets=offsets,
                frame_file_ids=file_ids, z_mean=mean, z_std=std, initial_window=2)
    output = np.empty((1024,2), np.float32)
    interval = 1024/sample_rate
    lag, samples, transitions = [], [], []
    pending = None
    cpu_started = time.process_time()
    start = time.perf_counter()
    deadline = start
    next_change, next_sample, next_progress = start, start, start
    index = ('random','manual','reorganized').index(args.initial_mode)*17
    failure = None
    try:
        while time.perf_counter()-start < args.seconds:
            now = time.perf_counter()
            if now >= next_change:
                # 16 fixed windows, then adaptive bounds, across all three modes.
                step = index % 17
                mode = ('random', 'manual', 'reorganized')[(index//17) % 3]
                if step == 0:
                    controller.stop()
                    assert controller.set_mode(mode)[0]
                    controller.set_window_controls({'mode':'fixed'})
                    controller.set_decoder_window(2)
                    assert controller.start()[0]
                elif step < 16:
                    assert controller.set_decoder_window(2*(step+1))[0]
                else:
                    assert controller.set_window_controls({'mode':'adaptive','minimum':2,'maximum':32})[0]
                if pending:
                    transitions.append({**pending, 'settled_ms': None})
                pending = {'mode':mode, 'step':step, 'requested_at':now-start,
                           'generation':controller._requested_generation}
                index += 1
                next_change = now + args.change_every
            if player._stream.active:
                if not args.audio_device:
                    player._callback(output, len(output), None, SimpleNamespace(output_underflow=False))
                if player.nonfinite: raise RuntimeError('Non-finite rendered PCM')
                if pending and player.current_generation == pending['generation']:
                    transitions.append({**pending, 'settled_ms':(now-start-pending['requested_at'])*1000})
                    pending = None
            if now >= next_sample:
                error = controller.get_extra_state()['transport']['error']
                if error: raise RuntimeError(error)
                samples.append({'elapsed':now-start, 'underruns':player.underruns,
                    'buffer_underruns':player.buffer_underruns, 'device_underruns':player.device_underruns,
                    'mps_allocated_bytes':torch.mps.current_allocated_memory() if args.device.startswith('mps') else None,
                    'cuda_allocated_bytes':torch.cuda.memory_allocated(args.device) if args.device.startswith('cuda') else None})
                next_sample = now + 5
            if now >= next_progress:
                print(f'{args.backend}/{args.device}: {now-start:.0f}s, underruns={player.underruns} (buffer={player.buffer_underruns}, device={player.device_underruns}), blocks={player.blocks}', flush=True)
                next_progress = now + 30
            deadline += interval
            delay = deadline-time.perf_counter()
            lag.append(max(0., -delay*1000))
            if delay > 0: time.sleep(delay)
            elif delay < -interval: deadline = time.perf_counter()
    except Exception as exc:
        failure = str(exc)
    finally:
        controller.close()
        decoder.close()
    duration = time.perf_counter()-start
    report = {'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(), 'python':platform.python_version(), 'torch':torch.__version__,
        'backend':args.backend, 'device':args.device, 'identity':selection.identity,
        'source':source, 'clock':('physical output (muted)' if args.audio_device else 'silent software callback') + f', 1024 samples at {sample_rate} Hz',
        'audio_device':args.audio_device,
        'navigation':args.navigation,
        'requested_seconds':args.seconds, 'duration_seconds':duration, 'started_monotonic':start,
        'preparation_ms':preparation_ms, 'underruns':player.underruns,
        'buffer_underruns':player.buffer_underruns, 'device_underruns':player.device_underruns,
        'callback_ms':percentiles(player.callback_times), 'clock_lateness_ms':percentiles(lag),
        'process_cpu_percent':100*(time.process_time()-cpu_started)/duration,
        'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'windows':{str(w):{'decode_ms':percentiles(v), 'hop_budget_ms':decoder.metadata_for(w).audio_hop_samples/decoder.metadata_for(w).sample_rate*1000}
                   for w,v in sorted(decoder.times.items())},
        'transitions':transitions, 'memory_samples':samples,
        'underrun_events':player.underrun_events, 'rendered_blocks':player.blocks, 'pcm_peak':player.peak,
        'pcm_rms':float(np.sqrt(player.sum_square/(player.blocks*1024*2))) if player.blocks else None,
        'error':failure, 'sustained_10_minutes':duration >= 600 and not failure,
        'software_buffer_gate_passed':failure is None and player.blocks > 0 and player.underruns == 0,
        'audio_device_qualified':False, 'listening_review':'pending'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if failure or not player.blocks: raise SystemExit(failure or 'No PCM rendered')
    if args.require_zero_underruns and player.underruns:
        raise SystemExit(f'Buffer gate failed: {player.underruns} underrun callbacks; see report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['onnxruntime','pytorch'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--change-every', type=float, default=10)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--initial-mode', choices=['random','manual','reorganized'], default='random')
    parser.add_argument('--navigation', choices=['replay','production'], default='replay')
    parser.add_argument('--require-zero-underruns', action='store_true', help='Exit nonzero if any buffer or device underrun occurs')
    parser.add_argument('--audio-device', help='Explicit output device name; muted hardware callback instead of software clock')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.change_every <= 0: parser.error('Durations must be positive')
    if args.navigation == 'production' and not args.corpus: parser.error('Production navigation requires --corpus')
    run(args)
