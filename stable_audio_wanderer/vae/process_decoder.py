"""One persistent native decoder process, with serialized CPU PCM exchanges."""
import base64
import json
from pathlib import Path
import subprocess
import queue
import threading
import time
from types import SimpleNamespace
import numpy as np
from .decoder_contract import DecodedAudioWindow, DecoderRuntimeError, DecoderWindowMetadata


class ProcessNativeDecoder:
    def __init__(self, selection):
        self._lock = threading.Lock()
        from .native_runtime import worker_environment, worker_directory
        self._process = subprocess.Popen([selection.worker_python, '-m',
            'stable_audio_wanderer.vae.native_worker', 'serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env=worker_environment(selection.worker_python), cwd=worker_directory(selection.worker_python))
        self._responses = queue.Queue()
        self._reader = threading.Thread(target=self._read_responses, daemon=True, name='native-decoder-pcm')
        self._reader.start()
        try:
            self._send(dict(config=json.loads(selection.worker_request)['config'],
                corpus_spec=json.loads(selection.worker_request)['corpus_spec']))
            ready = self._receive()
            self.info = SimpleNamespace(**ready['info'], runtime_python=selection.worker_python)
            self.info.model_path = Path(self.info.model_path)
            self.default_window = ready['default_window']
            self._windows = {int(w):DecoderWindowMetadata(**meta) for w,meta in ready['windows'].items()}
        except BaseException:
            self._process.kill()
            self._process.wait()
            self.close()
            raise

    @property
    def supported_windows(self):
        return tuple(self._windows)

    def metadata_for(self, window):
        if isinstance(window,(bool,np.bool_)) or not isinstance(window,(int,np.integer)) or window not in self._windows:
            raise DecoderRuntimeError(f'Unsupported native decoder window {window!r}')
        return self._windows[window]

    def validate_corpus(self, spec):
        meta = self.metadata_for(self.default_window)
        if spec['vae_id'] != self.info.vae_id or spec['sample_rate'] != meta.sample_rate or spec.get('latent_dim',meta.latent_dim) != meta.latent_dim:
            raise DecoderRuntimeError('Corpus does not match native worker')

    def _send(self, payload):
        try:
            self._process.stdin.write(json.dumps(payload)+'\n')
            self._process.stdin.flush()
        except (BrokenPipeError,OSError) as exc:
            raise DecoderRuntimeError('Native decoder process exited; Stop and reload the decoder') from exc

    def _read_responses(self):
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _receive(self):
        try:
            line = self._responses.get(timeout=120)
        except queue.Empty:
            self._process.kill()
            self._process.wait()
            raise DecoderRuntimeError('Native decoder timed out; worker terminated')
        if not line:
            raise DecoderRuntimeError(f'Native decoder process exited (status {self._process.poll()}); see server log')
        payload = json.loads(line)
        if payload.get('error'):
            raise DecoderRuntimeError(payload['error'])
        return payload

    def decode(self, raw_latents):
        raw = np.asarray(raw_latents)
        meta = self.metadata_for(len(raw))
        if raw.dtype != np.float32 or raw.shape != (meta.latent_window,meta.latent_dim) or not np.isfinite(raw).all():
            raise DecoderRuntimeError('Invalid native worker latent input')
        started = time.perf_counter()
        with self._lock:
            self._send(dict(operation='decode',shape=list(raw.shape),raw=base64.b64encode(raw.tobytes()).decode('ascii')))
            result = self._receive()
        audio = np.frombuffer(base64.b64decode(result['audio']),np.float32).reshape(result['shape']).copy()
        if audio.shape != (meta.audio_window_samples,meta.channels) or not np.isfinite(audio).all():
            raise DecoderRuntimeError('Invalid native worker PCM')
        return DecodedAudioWindow(audio,meta,(time.perf_counter()-started)*1000)

    def close(self):
        with self._lock:
            if self._process.poll() is None:
                try:
                    self._send(dict(operation='close'))
                    self._process.wait(timeout=10)
                except (DecoderRuntimeError,subprocess.TimeoutExpired):
                    self._process.kill()
                    self._process.wait()
            self._reader.join(timeout=1)
            try:
                self._process.stdin.close()
            except OSError:
                pass  # An aborted worker may leave an unwritable pipe buffer.
            self._process.stdout.close()
