"""Experimental microphone queries over observed corpus frames.

Capture, each analyzer, and output decoding have independent lifecycles. Workers
consume only the latest buffer; no inference jobs are queued. An instance belongs
to exactly one corpus and is closed by its owning transport on corpus unload.
"""
from dataclasses import dataclass
import math
import threading
import time

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class InputSettings:
    window_seconds: float = 1.0
    update_seconds: float = 0.1
    freshness_seconds: float = 2.0
    dwell_seconds: float = 0.15
    silence_db: float = -45.0
    channels: int = 1
    encoder_tail_frames: int = 1

    def __post_init__(self):
        for name in ('window_seconds', 'update_seconds', 'freshness_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.dwell_seconds) or self.dwell_seconds < 0:
            raise ValueError('dwell_seconds must be finite and nonnegative')
        if (not math.isfinite(self.silence_db) or type(self.channels) is not int
                or self.channels not in (1, 2)):
            raise ValueError('invalid input channels or silence threshold')
        if type(self.encoder_tail_frames) is not int or self.encoder_tail_frames < 0:
            raise ValueError('encoder_tail_frames must be nonnegative')


@dataclass(frozen=True)
class QueryResult:
    path: str
    session: int
    input_time: float
    index: int
    distance: float
    analysis_ms: float


class CorpusMatcher:
    """Exact Euclidean searches in normalized latent and weighted descriptor spaces."""

    def __init__(self, corpus, manual):
        self.latents = np.asarray(corpus['Z_concat'], dtype=np.float32)
        self.mean = np.asarray(corpus['Z_mean'], dtype=np.float32).reshape(-1)
        self.std = np.asarray(corpus['Z_std'], dtype=np.float32).reshape(-1)
        self.descriptors = np.asarray(manual['manual_desc_weighted'], dtype=np.float32)
        self.center = np.asarray(manual['manual_desc_center'], dtype=np.float32).reshape(-1)
        self.scale = np.asarray(manual['manual_desc_scale'], dtype=np.float32).reshape(-1)
        self.weights = np.asarray(manual['manual_desc_scales'], dtype=np.float32).reshape(-1)
        self.offsets = np.asarray(corpus['file_offsets'], dtype=np.int64)
        self.paths = [str(path) for path in corpus.get('paths', [])]
        self.latent_hz = float(np.asarray(corpus['latent_hz']).item())
        if self.latents.ndim != 2 or not len(self.latents):
            raise ValueError('Audio Input requires nonempty corpus latents')
        if self.mean.shape != (self.latents.shape[1],) or self.std.shape != self.mean.shape:
            raise ValueError('latent normalization dimensions do not match corpus')
        if self.descriptors.shape != (len(self.latents), 35):
            raise ValueError('Audio Input requires 35 frame-aligned descriptors')
        if any(x.shape != (35,) for x in (self.center, self.scale, self.weights)):
            raise ValueError('descriptor normalization dimensions do not match corpus')
        for values in (self.latents, self.mean, self.std, self.descriptors,
                       self.center, self.scale, self.weights):
            if not np.all(np.isfinite(values)):
                raise ValueError('corpus query arrays must be finite')
        if np.any(self.std <= 0) or np.any(self.scale <= 0):
            raise ValueError('corpus normalization scales must be positive')
        self.trees = {'descriptors': cKDTree(self.descriptors),
                      'latents': cKDTree(self.latents)}

    def query(self, path, raw):
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        expected = 35 if path == 'descriptors' else len(self.mean)
        if path not in self.trees or raw.shape != (expected,) or not np.all(np.isfinite(raw)):
            raise ValueError('invalid audio query')
        if path == 'descriptors':
            from ..cli.preprocess import weight_descriptor_queries
            transformed = weight_descriptor_queries(raw, self.center, self.scale, self.weights)
        else:
            transformed = (raw - self.mean) / self.std
        distance, index = self.trees[path].query(transformed, k=1)
        return int(index), float(distance)


class AudioAnalyzers:
    """Shared by the recorded-input probe and microphone workers; lazy native loading."""

    def __init__(self, matcher, *, vae_id, sample_rate, latent_hz, settings,
                 encoder_loader=None):
        self.matcher = matcher
        self.vae_id = vae_id
        self.sample_rate = int(sample_rate)
        self.hop = int(round(self.sample_rate / float(latent_hz)))
        if self.sample_rate <= 0 or self.hop < 1:
            raise ValueError('invalid corpus sample or latent rate')
        if settings.window_seconds * self.sample_rate < max(2049, self.hop * (settings.encoder_tail_frames + 1)):
            raise ValueError('analysis window too short for this corpus and encoder tail setting')
        self.settings = settings
        self.encoder_loader = encoder_loader
        self.encoder = None
        self.mfcc = None
        self.encoder_device = None

    def analyze(self, path, audio, input_sr):
        import torch
        import torchaudio.functional as AF
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not np.all(np.isfinite(audio)):
            raise ValueError('input audio must be finite [samples, 1 or 2 channels]')
        x = torch.from_numpy(audio.T.copy())
        if int(input_sr) != self.sample_rate:
            x = AF.resample(x, int(input_sr), self.sample_rate)
        if path == 'descriptors':
            from ..cli.preprocess import (_build_mfcc_transform,
                compute_latent_aligned_descriptors, MANUAL_N_FFT)
            if self.mfcc is None:
                self.mfcc = _build_mfcc_transform(self.hop, self.sample_rate)
            # Query an actual STFT frame, not the repeated/padded alignment tail.
            count = max(1, 1 + (x.shape[1] - MANUAL_N_FFT) // self.hop)
            raw = compute_latent_aligned_descriptors(x.T.numpy(), count, self.mfcc,
                                                    self.hop, self.sample_rate)[-1]
        elif path == 'latents':
            if self.encoder is None:
                if self.encoder_loader is None:
                    from ..vae import load_vae_adapter
                    encoder = load_vae_adapter(self.vae_id)
                else:
                    encoder = self.encoder_loader()
                info = encoder.info()
                if (info.vae_id != self.vae_id or info.sample_rate != self.sample_rate
                        or info.latent_dim != len(self.matcher.mean)
                        or int(round(info.sample_rate / info.latent_hz)) != self.hop):
                    raise ValueError('encoder metadata does not match corpus')
                from ..config import DEVICE
                self.encoder_device = DEVICE
                self.encoder = encoder
            info = self.encoder.info()
            if x.shape[0] == 1 and info.channels == 2:
                x = x.repeat(2, 1)
            elif x.shape[0] == 2 and info.channels == 1:
                x = x.mean(dim=0, keepdim=True)
            # Whole, hop-aligned windows; the encoder is not assumed causal.
            usable = (x.shape[1] // self.hop) * self.hop
            if usable < self.hop * (self.settings.encoder_tail_frames + 1):
                raise ValueError('analysis window too short for encoder tail exclusion')
            with torch.inference_mode():
                z = self.encoder.encode(x[:, -usable:].unsqueeze(0).to(self.encoder_device))
            if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != len(self.matcher.mean):
                raise ValueError('encoder returned invalid latent dimensions')
            offset = self.settings.encoder_tail_frames + 1
            if z.shape[-1] < offset:
                raise ValueError('encoder returned too few frames')
            raw = z[0, :, -offset].detach().cpu().numpy()
        else:
            raise ValueError(f'unknown input path: {path}')
        return self.matcher.query(path, raw)


class AudioInputNavigation:
    """One capture stream, two latest-window workers, shared selection stabilization."""

    def __init__(self, analyzers, *, settings=None, clock=time.monotonic):
        self.analyzers = analyzers
        self.settings = settings or analyzers.settings
        self.clock = clock
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._stop = threading.Event()
        self._stop.set()
        self._threads = []
        self._stream = None
        self._session = 0
        self._closed = False
        self._buffer = None
        self._write = self._filled = self._sequence = 0
        self._timestamp = 0.0
        self._input_sr = 0
        self.path = 'descriptors'
        self.results = {}
        self.errors = {}
        self.worker_status = {}
        self.level_db = -120.0
        self.capture_error = ''
        self.device = None
        self.capture_status = 'stopped'
        self._index = None
        self._last_selection = 0.0

    @staticmethod
    def devices():
        import sounddevice as sd
        return [{'id': i, 'name': d['name'], 'channels': d['max_input_channels']}
                for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]

    def set_path(self, path):
        if path not in ('descriptors', 'latents'):
            raise ValueError('input path must be descriptors or latents')
        with self._lock:
            self.path = path
            self._last_selection = 0.0

    def start(self, device=None):
        with self._operation_lock:
            if self._closed:
                raise RuntimeError('Audio Input is closed')
            if self._stream is not None:
                raise RuntimeError('microphone already started')
            import sounddevice as sd
            info = sd.query_devices(device, 'input')
            rate = int(round(info['default_samplerate']))
            sd.check_input_settings(device=device, channels=self.settings.channels,
                                    samplerate=rate, dtype='float32')
            with self._lock:
                self._session += 1
                session = self._session
                self._input_sr = rate
                self.device = info['name']
                size = int(math.ceil(self.settings.window_seconds * rate))
                self._buffer = np.zeros((size, self.settings.channels), dtype=np.float32)
                self._write = self._filled = self._sequence = 0
                self.results.clear()
                self.errors.clear()
                self.worker_status.clear()
                self._index = None
                self.level_db = -120.0
                self.capture_error = ''
                self.capture_status = 'buffering'
                self._stop.clear()
            try:
                self._stream = sd.InputStream(device=device, samplerate=rate,
                    channels=self.settings.channels, dtype='float32',
                    callback=self._capture, blocksize=0)
                self._stream.start()
                self._threads = [threading.Thread(target=self._worker, args=(path, session),
                    name=f'waenderer-input-{path}', daemon=True)
                    for path in ('descriptors', 'latents')]
                for worker in self._threads:
                    worker.start()
            except Exception:
                self._stop_capture()
                raise

    def _capture(self, audio, frames, timing, status):
        with self._lock:
            if self._stop.is_set():
                return
            if status:
                self.capture_error = str(status)
            size = len(self._buffer)
            block = audio[-size:]
            n = len(block)
            end = min(n, size - self._write)
            self._buffer[self._write:self._write + end] = block[:end]
            self._buffer[:n - end] = block[end:]
            self._write = (self._write + n) % size
            self._filled = min(size, self._filled + n)
            self._timestamp = self.clock()
            self._sequence += 1
            # A lightweight block meter also gates silence while workers are busy.
            self.level_db = float(20 * np.log10(max(1e-6, np.sqrt(np.mean(audio ** 2)))))
            self.capture_status = 'running' if self._filled == size else 'buffering'

    def _snapshot(self):
        with self._lock:
            if self._buffer is None or self._filled != len(self._buffer):
                return None
            return (np.concatenate((self._buffer[self._write:], self._buffer[:self._write])),
                    self._timestamp, self._sequence, self.level_db)

    def _worker(self, path, session):
        sequence = -1
        while not self._stop.is_set():
            snapshot = self._snapshot()
            if snapshot is not None and snapshot[2] != sequence:
                audio, timestamp, sequence, level = snapshot
                if level >= self.settings.silence_db:
                    with self._lock:
                        self.worker_status[path] = 'analyzing / loading encoder' if path == 'latents' else 'analyzing'
                    started = self.clock()
                    try:
                        index, distance = self.analyzers.analyze(path, audio, self._input_sr)
                        result = QueryResult(path, session, timestamp, index, distance,
                                             (self.clock() - started) * 1000)
                        with self._lock:
                            if session == self._session and not self._stop.is_set():
                                self.results[path] = result
                                self.errors.pop(path, None)
                                self.worker_status[path] = 'ready'
                    except Exception as exc:
                        with self._lock:
                            self.errors[path] = str(exc)
                            self.worker_status[path] = 'error'
                        # Configuration/load failures require explicit stop/start.
                        return
            self._stop.wait(self.settings.update_seconds)

    def select(self, *, require_fresh=False):
        with self._lock:
            now = self.clock()
            result = self.results.get(self.path)
            eligible = (not self._stop.is_set() and not self.errors.get(self.path)
                and result is not None and result.session == self._session
                and 0 <= now - result.input_time <= self.settings.freshness_seconds
                and self.level_db >= self.settings.silence_db)
            if not eligible:
                if require_fresh:
                    raise RuntimeError(f'{self.path}: waiting for a fresh, non-silent input result')
                return self._index
            if self._index is None or now - self._last_selection >= self.settings.dwell_seconds:
                self._index = result.index
                self._last_selection = now
            return self._index

    def state(self):
        with self._lock:
            now = self.clock()
            paths = {}
            for path in ('descriptors', 'latents'):
                result = self.results.get(path)
                paths[path] = dict(status=self.worker_status.get(path, 'waiting'),
                    error=self.errors.get(path, ''),
                    index=result.index if result else None,
                    distance=result.distance if result else None,
                    analysis_ms=result.analysis_ms if result else None,
                    age_ms=max(0, now - result.input_time) * 1000 if result else None)
                if result is not None:
                    from pathlib import Path
                    matcher = self.analyzers.matcher
                    file_id = int(np.searchsorted(matcher.offsets, result.index, side='right') - 1)
                    paths[path]['source'] = (Path(matcher.paths[file_id]).name
                        if file_id < len(matcher.paths) else f'file {file_id}')
                    paths[path]['source_seconds'] = (result.index - matcher.offsets[file_id]) / matcher.latent_hz
            return dict(running=not self._stop.is_set(), status=self.capture_status,
                path=self.path, device=self.device, sample_rate=self._input_sr,
                level_db=self.level_db, capture_error=self.capture_error,
                selected_index=self._index, paths=paths,
                settings=vars(self.settings))

    def _stop_capture(self):
        self._stop.set()
        with self._lock:
            self._session += 1
            self.capture_status = 'stopped'
            self.results.clear()
            self._index = None
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        for worker in self._threads:
            worker.join()
        self._threads.clear()

    def stop(self):
        with self._operation_lock:
            self._stop_capture()

    def close(self):
        with self._operation_lock:
            self._closed = True
            self._stop_capture()
            self.analyzers.encoder = None


def create_audio_input(corpus, manual, config):
    from ..io.corpus_io import read_scalar
    from ..vae.native_runtime import native_config
    vae_id = str(read_scalar(corpus, 'vae_id', ''))
    settings = InputSettings(**config.get('audio_input_settings', {}))
    # Native model identity is currently pinned for these two adapters. EAR needs
    # explicit checkpoint provenance before live matching can be qualified.
    if vae_id not in ('same_s', 'stable_audio_open'):
        raise ValueError('Audio Input prototype supports SAME-S and Stable Audio Open corpora')
    if vae_id == 'same_s':
        from ..vae.same_s_weights import CONFIG_SHA256, WEIGHTS_SHA256, SOURCE_REVISION
    else:
        from ..vae.stable_audio_open_weights import CONFIG_SHA256, WEIGHTS_SHA256, SOURCE_REVISION
    for key, expected in (('vae_config_sha256', CONFIG_SHA256),
                          ('vae_weights_sha256', WEIGHTS_SHA256),
                          ('vae_source_revision', SOURCE_REVISION)):
        recorded = str(read_scalar(corpus, key, ''))
        if recorded and recorded != expected:
            raise ValueError(f'Corpus {key} conflicts with the pinned live encoder')

    def loader():
        import sys
        from pathlib import Path
        from ..vae import load_vae_adapter
        native = native_config(vae_id, config)
        if Path(native['decoder_python']).absolute() != Path(sys.executable).absolute():
            raise RuntimeError('Live encoder requires native dependencies in the server interpreter; '
                               'external native interpreters are not supported by this prototype')
        return load_vae_adapter(vae_id)

    matcher = CorpusMatcher(corpus, manual)
    analyzers = AudioAnalyzers(matcher, vae_id=vae_id,
        sample_rate=int(read_scalar(corpus, 'sr', 44100)),
        latent_hz=float(read_scalar(corpus, 'latent_hz', 21.5)),
        settings=settings, encoder_loader=loader)
    return AudioInputNavigation(analyzers)
