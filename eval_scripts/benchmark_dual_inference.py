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
from eval_scripts.multi_vae_validation_common import (
    add_selection_arguments, selection_config, evidence_context, write_report,
    validate_native_source,
)


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

    def has_variant(self, mode): return mode in ('wander', 'reorganized')
    def set_policy_variant(self, mode): return self.has_variant(mode)
    def get_active_jump_rate(self, variant=None): return 0.0
    def get_wander_controls(self): return {}
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
        self.rendered_samples = 0
        self.generation_scenarios = {}
        self.excluded_samples = 0
        self.scenario_samples = defaultdict(int)
        self.peak = 0.0
        self.sum_square = 0.0
        self.nonfinite = False
        self.underrun_events = []
        super().__init__(**kwargs)

    def _callback(self, output, frames, time_info, status):
        before = time.perf_counter()
        previous = self.underruns
        state_before = self.get_state()
        super()._callback(output, frames, time_info, status)
        if self.underruns != previous:
            self.underrun_events.append({'monotonic':before,
                'buffer_underruns':self.buffer_underruns, 'device_underruns':self.device_underruns,
                'state':self.get_state()})
        self.callback_times.append((time.perf_counter()-before)*1000)
        self.nonfinite |= not np.isfinite(output).all()
        self.blocks += 1
        self.rendered_samples += frames
        state_after = self.get_state()
        scenario = self.generation_scenarios.get(state_before['generation'])
        # Mixed transitions, empty queues and underrun callbacks cannot prove
        # steady playback for either the old or newly requested setting.
        if (scenario is not None and self.underruns == previous and
                state_before['generation'] == state_after['generation'] and
                state_before['transition_status'] != 'crossfading' and
                state_after['transition_status'] != 'crossfading'):
            self.scenario_samples[scenario] += frames
        else:
            self.excluded_samples += frames
        self.peak = max(self.peak, float(np.max(np.abs(output))))
        self.sum_square += float(np.sum(output.astype(np.float64)**2))


def run(args):
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(1)
    if args.corpus:
        spec = corpus_decoder_spec(args.corpus)
        path = args.corpus / "corpus.npz" if args.corpus.is_dir() else args.corpus
        with np.load(path, allow_pickle=False) as data:
            z, mean, std = (np.asarray(data[key], dtype=np.float32) for key in ('Z_concat','Z_mean','Z_std'))
            offsets = np.asarray(data['file_offsets'], dtype=np.int64)
        source = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        from stable_audio_wanderer.vae.onnx_artifacts import resolve_artifact
        artifact = resolve_artifact(args.vae_id, store_dir=args.store_dir or None,
                                    artifact_dir=args.artifact_dir or None)
        spec = dict(vae_id=args.vae_id, sample_rate=artifact.sample_rate,
                    latent_hz=artifact.sample_rate/artifact.samples_per_latent,
                    latent_dim=artifact.latent_dim)
        z = np.random.default_rng(5006).normal(0, .05, (256,artifact.latent_dim)).astype(np.float32)
        mean, std = np.zeros(artifact.latent_dim, np.float32), np.ones(artifact.latent_dim, np.float32)
        offsets = np.array([0,128,256])
        source = {'synthetic_seed': 5006, 'distribution': 'normal std=.05',
                  'sha256': hashlib.sha256(z.tobytes()).hexdigest()}
    file_ids = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    paired = select_decoder(selection_config(args, 'onnxruntime', 'cpu'), corpus_spec=spec)
    config = selection_config(args, args.backend, args.device)
    if args.backend == 'pytorch':
        from stable_audio_wanderer.vae.native_runtime import native_config
        config = native_config(spec['vae_id'], config)
        config['decoder_source_identity'] = dict(paired.artifact.source)
    selection = select_decoder(config, corpus_spec=spec)
    context = evidence_context(args, spec, selection)
    context['corpus_sha256'] = source['sha256'] if args.corpus else None
    if not selection.artifact:
        paired_context = evidence_context(args, spec, paired)
        for key in ('artifact_identity', 'artifact_files', 'model_source', 'geometry'):
            context[key] = paired_context[key]
    started = time.perf_counter()
    decoder = MeasuredDecoder(create_decoder(selection))
    if args.backend == 'pytorch':
        validate_native_source(decoder, paired.artifact)
    if hasattr(decoder, "validate_corpus"):
        decoder.validate_corpus(spec)
    preparation_ms = (time.perf_counter()-started)*1000
    meta = decoder.metadata_for(decoder.default_window)
    sample_rate = meta.sample_rate
    # Patch only stream creation; keep production PCM buffer, callback, OLA,
    # transport scheduler, transitions and decoder implementations.
    import sounddevice as sd
    device_info = None
    if args.audio_device:
        device_info = dict(sd.query_devices(args.audio_device, 'output'))
        device_info['hostapi_name'] = sd.query_hostapis(device_info['hostapi'])['name']
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
    profile = dict(windows=args.windows, modes=args.modes, adaptive=not args.no_adaptive)
    settings = [*[f'T{w}' for w in args.windows], *(['adaptive'] if profile['adaptive'] else [])]
    schedule = [(mode,step,setting) for mode in args.modes for step,setting in enumerate(settings)]
    index = args.modes.index(args.initial_mode)*len(settings) if args.initial_mode in args.modes else 0
    failure = None
    try:
        while time.perf_counter()-start < args.seconds:
            now = time.perf_counter()
            if now >= next_change:
                mode, step, setting = schedule[index % len(schedule)]
                if step == 0 or setting == 'adaptive':
                    controller.stop()
                    assert controller.set_mode(mode)[0]
                    assert controller.set_window_controls({'mode':'adaptive' if setting == 'adaptive' else 'fixed',
                        'minimum':min(args.windows),'maximum':max(args.windows)})[0]
                    assert controller.set_decoder_window(args.windows[0])[0]
                    assert controller.start()[0]
                elif setting != 'adaptive':
                    assert controller.set_decoder_window(int(setting[1:]))[0]
                scenario = f'{mode}:{setting}'
                player.generation_scenarios[controller._requested_generation] = scenario
                if pending:
                    transitions.append({**pending, 'settled_ms': None})
                pending = {'mode':mode, 'step':step, 'scenario':scenario, 'requested_at':now-start,
                           'generation':controller._requested_generation}
                index += 1
                next_change = now + args.change_every
            # Adaptive changes create generations inside the scheduler. Keep
            # previous generations' labels immutable until their PCM drains.
            with controller._lock:
                adaptive_generation = controller._adaptive_generation
            if setting == 'adaptive' and adaptive_generation is not None:
                player.generation_scenarios.setdefault(adaptive_generation, scenario)
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
                    'buffer_duration_seconds':player.get_state()['buffer_duration'],
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
        duration = time.perf_counter()-start
        cpu_seconds = time.process_time()-cpu_started
        stream_settings = {key:getattr(player._stream,key,None) for key in ('samplerate','blocksize','channels','latency')}
        if pending:
            transitions.append({**pending, 'settled_ms':None})
        try:
            controller.close()
        except Exception as exc:
            failure = failure or f'Transport cleanup failed: {exc}'
        try:
            decoder.close()
        except Exception as exc:
            failure = failure or f'Decoder cleanup failed: {exc}'
    report = {**context, 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(), 'python':platform.python_version(), 'torch':torch.__version__,
        'backend':args.backend, 'device':context['device'], 'identity':selection.identity,
        'source':source, 'clock':('physical output (muted)' if args.audio_device else 'silent software callback') + f', 1024 samples at {sample_rate} Hz',
        'audio_device':args.audio_device, 'audio_device_info':device_info,
        'stream_settings':stream_settings,
        'sample_rate':sample_rate, 'blocksize':1024, 'channels':meta.channels,
        'active_audio_seconds':sum(player.scenario_samples.values())/sample_rate,
        'callback_audio_seconds':player.rendered_samples/sample_rate,
        'excluded_samples':player.excluded_samples,
        'sample_attribution':'stable_pcm_generation_without_underrun',
        'realtime_profile':profile,
        'rendered_samples':player.rendered_samples,
        'scenario_samples':dict(player.scenario_samples),
        'declared_scenarios':[f'{mode}:{setting}' for mode,step,setting in schedule],
        'nonfinite_pcm':player.nonfinite, 'torch_threads':torch.get_num_threads(),
        'provider':decoder.info.provider,
        'decoder_settings':{key:getattr(decoder.info,key,None) for key in
            ('intra_op_num_threads','graph_optimization','runtime_python')},
        'navigation':args.navigation,
        'requested_seconds':args.seconds, 'duration_seconds':duration, 'started_monotonic':start,
        'preparation_ms':preparation_ms, 'underruns':player.underruns,
        'buffer_underruns':player.buffer_underruns, 'device_underruns':player.device_underruns,
        'callback_ms':percentiles(player.callback_times), 'clock_lateness_ms':percentiles(lag),
        'process_cpu_percent':100*cpu_seconds/duration,
        'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
        'windows':{str(w):{'decode_ms':percentiles(v), 'hop_budget_ms':decoder.metadata_for(w).audio_hop_samples/decoder.metadata_for(w).sample_rate*1000}
                   for w,v in sorted(decoder.times.items())},
        'transitions':transitions, 'memory_samples':samples,
        'underrun_events':player.underrun_events, 'rendered_blocks':player.blocks, 'pcm_peak':player.peak,
        'pcm_rms':float(np.sqrt(player.sum_square/(player.blocks*1024*2))) if player.blocks else None,
        'error':failure, 'sustained_10_minutes':sum(player.scenario_samples.values())/sample_rate >= 600 and not failure,
        'software_buffer_gate_passed':failure is None and player.blocks > 0 and player.underruns == 0,
        'audio_device_qualified':False, 'listening_review':'pending'}
    write_report(args.output, report)
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
    parser.add_argument('--vae-id', default='same_s', choices=['same_s','stable_audio_open','ear_vae_44k','ear_vae_48k'], help='Synthetic model; corpus selects its own VAE')
    add_selection_arguments(parser)
    parser.add_argument('--windows', nargs='+', type=int, default=list(range(2,33,2)), help='Fixed windows in the claimed combined profile')
    parser.add_argument('--modes', nargs='+', choices=['wander','manual','reorganized'], default=['wander','manual','reorganized'])
    parser.add_argument('--no-adaptive', action='store_true', help='Restrict profile to its fixed windows')
    parser.add_argument('--initial-mode', choices=['wander','manual','reorganized'], default='wander')
    parser.add_argument('--navigation', choices=['replay','production'], default='replay')
    parser.add_argument('--require-zero-underruns', action='store_true', help='Exit nonzero if any buffer or device underrun occurs')
    parser.add_argument('--audio-device', help='Explicit output device name; muted hardware callback instead of software clock')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.windows or len(set(args.windows)) != len(args.windows) or any(w not in range(2,33,2) for w in args.windows): parser.error('Windows must be distinct even T2..T32 values')
    if len(set(args.modes)) != len(args.modes): parser.error('Modes must be distinct')
    if args.seconds <= 0 or args.change_every <= 0: parser.error('Durations must be positive')
    if args.navigation == 'production' and not args.corpus: parser.error('Production navigation requires --corpus')
    try:
        run(args)
    except Exception as exc:
        write_report(args.output, dict(schema_version=1,backend=args.backend,device=args.device,
            error=f'{type(exc).__name__}: {exc}',audio_device_qualified=False))
        raise
