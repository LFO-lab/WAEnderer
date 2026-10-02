"""Prepared decoder for the existing Stable Audio Open and EAR adapters."""
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
from .decoder_contract import DecodedAudioWindow, DecoderRuntimeError, DecoderWindowMetadata
from .registry import load_vae_adapter
from .torch_decoder import _resolve_device


class AdapterTorchDecoder:
    def __init__(self, *, vae_id, device, weight_path='', repo_or_path=None, local_files_only=True):
        import torch
        self._torch = torch
        self._device = _resolve_device(torch, device)
        self._lock = threading.Lock()
        self._adapter = None
        self._windows = {}
        try:
            options = {"repo_or_path": repo_or_path} if repo_or_path else {}
            self._adapter = load_vae_adapter(vae_id, **options, device=str(self._device),
                weight_path=weight_path, local_files_only=local_files_only)
            meta = self._adapter.info()
            self._latent_dim = meta.latent_dim
            self.info = SimpleNamespace(backend='pytorch', provider=str(self._device),
                device=str(self._device), vae_id=vae_id,
                model_path=Path(weight_path) if weight_path else Path(repo_or_path or 'stabilityai/stable-audio-open-1.0') / 'vae/diffusion_pytorch_model.safetensors',
                sample_rate=meta.sample_rate, latent_hz=meta.latent_hz)
            samples_per_latent = None
            self.warmup_decode_ms = {}
            for window in range(2, 33, 2):
                start = time.perf_counter()
                audio = self._decode_array(np.zeros((window, self._latent_dim), np.float32))
                ratio, remainder = divmod(len(audio), window)
                if remainder or ratio <= 0 or (samples_per_latent is not None and ratio != samples_per_latent):
                    raise DecoderRuntimeError('VAE output length is not proportional to latent window')
                samples_per_latent = ratio
                self._windows[window] = DecoderWindowMetadata(window, meta.latent_dim,
                    meta.sample_rate, meta.channels, ratio, window*ratio,
                    window//2, window//2*ratio, 'full_overlap_add')
                self.warmup_decode_ms[window] = (time.perf_counter()-start)*1000
        except Exception:
            self.close()
            raise

    @property
    def supported_windows(self):
        return tuple(self._windows)

    @property
    def default_window(self):
        return 8

    def metadata_for(self, window):
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window not in self._windows:
            raise DecoderRuntimeError(f'Unsupported decoder window {window!r}')
        return self._windows[window]

    def validate_corpus(self, spec):
        meta = self.metadata_for(self.default_window)
        if (spec['vae_id'] != self.info.vae_id or spec['latent_dim'] != meta.latent_dim
                or spec['sample_rate'] != meta.sample_rate):
            raise DecoderRuntimeError('Corpus dimensions/sample rate do not match the loaded VAE')
        # Old SAO corpora use the adapter\'s rounded 21.5 Hz metadata; PCM timing
        # always comes from the measured decoder ratio (2048 samples), not rounding.
        hz = spec['latent_hz']
        exact = meta.sample_rate/meta.samples_per_latent
        if not (np.isclose(hz, self.info.latent_hz, rtol=0, atol=1e-7)
                or np.isclose(hz, exact, rtol=0, atol=1e-7)):
            raise DecoderRuntimeError('Corpus latent rate does not match the loaded VAE')

    def _decode_array(self, raw):
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(np.array(raw.T[None], copy=True, order='C')).to(self._device)
            output = self._adapter.decode(tensor)
            if not isinstance(output, self._torch.Tensor) or output.ndim != 3 or output.shape[:2] != (1, 2):
                raise DecoderRuntimeError('VAE must return stereo PCM [1,2,samples]')
            if output.dtype != self._torch.float32 or output.device != self._device:
                raise DecoderRuntimeError('VAE output dtype/device does not match the selected decoder')
            audio = np.array(output.detach().cpu().numpy()[0].T, copy=True, order='C')
            if not np.isfinite(audio).all():
                raise DecoderRuntimeError('VAE returned non-finite PCM')
            return audio

    def decode(self, raw_latents):
        raw = np.asarray(raw_latents)
        if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != self._latent_dim or not np.isfinite(raw).all():
            raise DecoderRuntimeError(f'Expected finite float32 raw latents [T,{self._latent_dim}]')
        metadata = self.metadata_for(len(raw))
        start = time.perf_counter()
        with self._lock:
            if self._adapter is None:
                raise DecoderRuntimeError('Native decoder is closed')
            audio = self._decode_array(raw)
            if audio.shape != (metadata.audio_window_samples, metadata.channels):
                raise DecoderRuntimeError('VAE output shape changed after preparation')
        return DecodedAudioWindow(audio, metadata, (time.perf_counter()-start)*1000)

    def close(self):
        with self._lock:
            self._adapter = None
            if self._device.type == 'mps':
                self._torch.mps.empty_cache()
            elif self._device.type == 'cuda':
                with self._torch.cuda.device(self._device):
                    self._torch.cuda.empty_cache()
