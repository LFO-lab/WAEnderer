"""Backend-neutral boundary between latent decoding and audio transport.

This module describes already loaded decoders. Model loading, preflight and
ownership remain with backend factories/the pipeline; stopping a transport must
not release a decoder that the pipeline may reuse. No inference library is
imported here and no input coercion or validation is added by these types.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class DecoderRuntimeError(RuntimeError):
    """Invalid decode request or failure of a prepared decoder."""


@dataclass(frozen=True)
class DecoderWindowMetadata:
    """Validated timing and shape metadata for one latent window."""

    latent_window: int
    latent_dim: int
    sample_rate: int
    channels: int
    samples_per_latent: int
    audio_window_samples: int
    latent_hop: int
    audio_hop_samples: int
    ola_mode: str


@dataclass(frozen=True)
class DecodedAudioWindow:
    """Complete CPU PCM in [samples, channels] float32 layout, ready for OLA.

    decode_time_ms measures decoding until CPU PCM is available, including any
    device transfers. Audio is finite and matches the accompanying metadata.
    """

    audio: np.ndarray
    metadata: DecoderWindowMetadata
    decode_time_ms: float


@runtime_checkable
class DecoderInfo(Protocol):
    """Read-only identity shared by backend-specific metadata objects.

    Provider identifies the execution target (e.g. CPUExecutionProvider).
    Artifact provenance/validation stays backend-specific. In particular,
    resource_path, bundle_path and model I/O names are not transport requirements.
    """

    @property
    def backend(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def vae_id(self) -> str: ...

    @property
    def model_path(self) -> Path: ...


@runtime_checkable
class LatentDecoder(Protocol):
    """Prepared decoder consumed by the Web transport, without ONNX coupling.

    Inputs are finite, raw/denormalized float32 [T, latent_dim] arrays
    (256 dimensions for SAME-S, typically 64 for other adapters). The
    backend validates requests against its supported windows and returns one
    complete audio window, without transport-level padding or overlap-add.
    Decoding may be stochastic: identical inputs need not yield identical PCM.

    The current transport allows up to two overlapping decode calls for active
    and candidate generations. Implementations must safely support those calls
    or serialize them internally. Inputs and published metadata must not be
    mutated. This contract does not change scheduling or decoder ownership.
    """

    @property
    def info(self) -> DecoderInfo: ...

    @property
    def supported_windows(self) -> tuple[int, ...]: ...

    @property
    def default_window(self) -> int: ...

    def metadata_for(self, window: int) -> DecoderWindowMetadata: ...

    def decode(self, raw_latents: np.ndarray) -> DecodedAudioWindow: ...

    def close(self) -> None:
        """Release backend resources after producers are drained; idempotent.

        A closed decoder cannot be reused. The pipeline owns this operation;
        transport Stop only drains audio and keeps the decoder prepared.
        """
        ...


__all__ = [
    "DecodedAudioWindow",
    "DecoderInfo",
    "DecoderRuntimeError",
    "DecoderWindowMetadata",
    "LatentDecoder",
]
