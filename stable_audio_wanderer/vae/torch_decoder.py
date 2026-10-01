"""Prepared SAME-S PyTorch decoder; explicit devices and no backend fallback."""

from dataclasses import dataclass
import importlib.metadata as package_metadata
import json
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .decoder_contract import DecodedAudioWindow, DecoderRuntimeError, DecoderWindowMetadata
from .same_s_weights import NativeDecoderLoadError, load_same_s_model, resolve_same_s_weights


SUPPORTED_WINDOWS = tuple(range(2, 33, 2))


@dataclass(frozen=True)
class NativeDecoderInfo:
    model_path: Path
    config_path: Path
    model_sha256: str
    config_sha256: str
    source_model: str
    source_revision: str
    backend: str
    provider: str
    device: str
    vae_id: str
    supported_windows: tuple[int, ...]
    default_window: int
    windows: Mapping[int, DecoderWindowMetadata]
    warmup_decode_ms: Mapping[int, float]
    parameter_bytes: int
    torch_version: str
    library_version: str
    library_revision: str | None


def _resolve_device(torch, requested: str):
    if not isinstance(requested, str):
        raise NativeDecoderLoadError("Select an explicit device: cpu, mps or cuda[:index]")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise NativeDecoderLoadError(f"Invalid decoder device {requested!r}") from exc
    if device.type == "cpu":
        if device.index is not None:
            raise NativeDecoderLoadError("CPU decoder device must be 'cpu'")
    elif device.type == "mps":
        if device.index not in (None, 0):
            raise NativeDecoderLoadError("Only mps:0 is supported")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise NativeDecoderLoadError("Restart with PYTORCH_ENABLE_MPS_FALLBACK=0; CPU fallback is forbidden")
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise NativeDecoderLoadError("MPS is unavailable; no CPU fallback is permitted")
        device = torch.device("mps:0")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise NativeDecoderLoadError("CUDA is unavailable; no CPU fallback is permitted")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise NativeDecoderLoadError(f"CUDA device index {index} is unavailable")
        device = torch.device(f"cuda:{index}")
    else:
        raise NativeDecoderLoadError(f"Unsupported decoder device {requested!r}; use cpu, mps or cuda[:index]")
    return device


class SameSTorchDecoder:
    """Native LatentDecoder, usable independently of an ONNX installation.

    The constructor loads a dedicated model and validates all advertised
    windows. CPU is available explicitly for diagnostics, never as a fallback.
    Calls are serialized per instance, including warm-up. RNG state is not
    reset and inference remains stochastic like the original SAME-S model.
    The pipeline owns release; Web selection is implemented separately.
    """

    def __init__(self, *, device: str, local_files_only: bool = True):
        try:
            import torch
        except ImportError as exc:
            raise NativeDecoderLoadError("Native SAME-S requires PyTorch") from exc
        self._torch = torch
        self._device = _resolve_device(torch, device)
        self._lock = threading.Lock()
        self._model = None
        self._windows = MappingProxyType({
            window: DecoderWindowMetadata(
                latent_window=window, latent_dim=256, sample_rate=44100, channels=2,
                samples_per_latent=4096, audio_window_samples=window * 4096,
                latent_hop=window // 2, audio_hop_samples=window // 2 * 4096,
                ola_mode="full_overlap_add",
            ) for window in SUPPORTED_WINDOWS
        })
        weights = resolve_same_s_weights(local_files_only=local_files_only)
        try:
            model = load_same_s_model(weights)
            self._model = model.to(device=self._device, dtype=torch.float32).eval().requires_grad_(False)
            del model
            # Detect wrappers which silently retain tensors on the wrong device.
            for tensor in (*self._model.parameters(), *self._model.buffers()):
                if tensor.device != self._device:
                    raise NativeDecoderLoadError(f"SAME-S tensor is on {tensor.device}, expected {self._device}")
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise NativeDecoderLoadError("SAME-S floating tensors must be float32")
            warmup = {}
            for window in self.supported_windows:
                decoded = self.decode(np.zeros((window, 256), dtype=np.float32))
                warmup[window] = decoded.decode_time_ms
            distribution = package_metadata.distribution("stable-audio-3")
            origin = json.loads(distribution.read_text("direct_url.json") or "{}")
            self.info = NativeDecoderInfo(
                model_path=weights.model_path, config_path=weights.config_path,
                model_sha256=weights.model_sha256, config_sha256=weights.config_sha256,
                source_model=weights.source_model, source_revision=weights.source_revision,
                backend="pytorch", provider=str(self._device), device=str(self._device),
                vae_id="same_s", supported_windows=self.supported_windows, default_window=2,
                windows=self._windows, warmup_decode_ms=MappingProxyType(warmup),
                parameter_bytes=sum(p.numel() * p.element_size() for p in self._model.parameters()),
                torch_version=str(torch.__version__), library_version=distribution.version,
                library_revision=origin.get("vcs_info", {}).get("commit_id"),
            )
        except Exception as exc:
            self._model = None
            if isinstance(exc, NativeDecoderLoadError):
                raise
            raise NativeDecoderLoadError(f"SAME-S preparation failed on {self._device}: {exc}") from exc

    @property
    def supported_windows(self) -> tuple[int, ...]:
        return tuple(self._windows)

    @property
    def default_window(self) -> int:
        return 2

    def metadata_for(self, window: int) -> DecoderWindowMetadata:
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
            raise DecoderRuntimeError(f"Decoder window must be an integer, got {window!r}")
        try:
            return self._windows[window]
        except KeyError as exc:
            raise DecoderRuntimeError(f"Decoder window T{window} is unavailable; supported: {self.supported_windows}") from exc

    def decode(self, raw_latents: np.ndarray) -> DecodedAudioWindow:
        latents = np.asarray(raw_latents)
        if latents.dtype != np.float32:
            raise DecoderRuntimeError(f"Raw SAME-S latents must be float32, got {latents.dtype}")
        if latents.ndim != 2 or latents.shape[1] != 256:
            raise DecoderRuntimeError(f"Raw SAME-S latents must have shape [T,256], got {latents.shape}")
        metadata = self.metadata_for(latents.shape[0])
        if not np.isfinite(latents).all():
            raise DecoderRuntimeError("Raw SAME-S latents contain non-finite values")
        # Includes time queued behind another decode, transfers, synchronization,
        # output validation and production of an owned contiguous CPU PCM array.
        start = time.perf_counter()
        with self._lock, self._torch.inference_mode():
            if self._model is None:
                raise DecoderRuntimeError("Native SAME-S decoder is not prepared")
            try:
                tensor = self._torch.from_numpy(np.array(latents.T[None], copy=True, order="C"))
                output = self._model.decode_audio(tensor.to(self._device), chunked=False)
                expected = (1, metadata.channels, metadata.audio_window_samples)
                if not isinstance(output, self._torch.Tensor):
                    raise DecoderRuntimeError("Native SAME-S output must be a Tensor")
                if output.dtype != self._torch.float32 or tuple(output.shape) != expected:
                    raise DecoderRuntimeError(f"Native SAME-S output must be float32 {expected}, got {output.dtype} {tuple(output.shape)}")
                if output.device != self._device:
                    raise DecoderRuntimeError(f"Native SAME-S output is on {output.device}, expected {self._device}")
                # The blocking device-to-host copy completes inference before timing ends.
                audio = np.array(output.detach().cpu().numpy()[0].T, copy=True, order="C")
                if not np.isfinite(audio).all():
                    raise DecoderRuntimeError("Native SAME-S output contains non-finite values")
            except DecoderRuntimeError:
                raise
            except Exception as exc:
                raise DecoderRuntimeError(f"Native SAME-S decode failed on {self._device} for T{metadata.latent_window}: {exc}") from exc
        return DecodedAudioWindow(audio, metadata, (time.perf_counter() - start) * 1000)

    def close(self) -> None:
        """Wait for an in-flight native call, then drop the dedicated model."""
        with self._lock:
            if self._model is None:
                return
            self._model = None
            # Decode copies PCM back synchronously; no outstanding GPU work
            # belongs to this instance. Release only the selected device cache.
            if self._device.type == "mps":
                self._torch.mps.empty_cache()
            elif self._device.type == "cuda":
                with self._torch.cuda.device(self._device):
                    self._torch.cuda.empty_cache()
