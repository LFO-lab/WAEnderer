"""JUCE-parity full-output overlap-add helpers for decoded audio windows.

The latent decoder produces a complete audio window for every latent window.  A
stream is assembled by placing those decoded windows one audio hop apart,
applying the same sine synthesis envelope used by the JUCE renderer, and
normalizing by the accumulated envelope.  Exactly one hop becomes available
for every decoded window.

All public audio arrays are channel-first ``[channels, samples]`` float32.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


DEFAULT_MINIMUM_OLA_WEIGHT = 1.0e-6


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def sine_synthesis_window(length: int) -> np.ndarray:
    """Return ``sin(pi * (n + 0.5) / length)`` as a float32 vector."""

    length = _positive_int(length, "length")
    if length == 1:
        return np.ones((1,), dtype=np.float32)

    sample = np.arange(length, dtype=np.float32)
    phase = (sample + np.float32(0.5)) / np.float32(length)
    return np.sin(np.float32(np.pi) * phase).astype(np.float32, copy=False)


def _coerce_audio_window(decoded_audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(decoded_audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2:
        raise ValueError(
            f"decoded_audio must be [channels, samples], got {audio.shape}"
        )
    if audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError(
            f"decoded_audio must have at least one channel and sample, got {audio.shape}"
        )
    if not np.all(np.isfinite(audio)):
        raise ValueError("decoded_audio contains non-finite samples")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _map_output_channels(audio: np.ndarray, output_channels: int) -> np.ndarray:
    """Match JUCE's ``min(output_channel, source_channels - 1)`` mapping."""

    source_channels = int(audio.shape[0])
    channel_indices = np.minimum(
        np.arange(output_channels, dtype=np.intp), source_channels - 1
    )
    return audio[channel_indices]


class StreamingFullOverlapAdd:
    """Incrementally normalize complete decoded windows into fixed-size hops.

    A stream instance has one fixed hop, output channel count, and decoded
    window length.  The first :meth:`push` establishes the window length;
    changing decoder T therefore requires a separate staging instance or a
    reset, which prevents state from leaking across decoder generations.
    """

    def __init__(
        self,
        audio_hop_samples: int,
        channels: int = 2,
        minimum_weight: float = DEFAULT_MINIMUM_OLA_WEIGHT,
    ) -> None:
        self.audio_hop_samples = _positive_int(
            audio_hop_samples, "audio_hop_samples"
        )
        self.channels = _positive_int(channels, "channels")
        self.minimum_weight = float(minimum_weight)
        if not np.isfinite(self.minimum_weight) or self.minimum_weight < 0.0:
            raise ValueError(
                f"minimum_weight must be finite and non-negative, got {minimum_weight}"
            )
        self.reset()

    @property
    def window_samples(self) -> Optional[int]:
        """Decoded window length established by the first push, if any."""

        return self._window_samples

    @property
    def windows_pushed(self) -> int:
        return self._windows_pushed

    @property
    def buffered_samples(self) -> int:
        return int(self._weights.shape[0])

    def reset(self) -> None:
        """Discard all overlap state and allow a new decoded window length."""

        self._audio = np.zeros((self.channels, 0), dtype=np.float32)
        self._weights = np.zeros((0,), dtype=np.float32)
        self._buffer_start_position = 0
        self._read_position = 0
        self._next_window_position = 0
        self._window_samples: Optional[int] = None
        self._envelope = np.zeros((0,), dtype=np.float32)
        self._windows_pushed = 0

    def _ensure_capacity(self, required_samples: int) -> None:
        current_samples = int(self._weights.shape[0])
        if required_samples <= current_samples:
            return

        expanded_audio = np.zeros(
            (self.channels, required_samples), dtype=np.float32
        )
        expanded_weights = np.zeros((required_samples,), dtype=np.float32)
        if current_samples > 0:
            expanded_audio[:, :current_samples] = self._audio
            expanded_weights[:current_samples] = self._weights
        self._audio = expanded_audio
        self._weights = expanded_weights

    def push(self, decoded_audio: np.ndarray) -> np.ndarray:
        """Add one decoded window and return exactly one normalized audio hop."""

        source = _coerce_audio_window(decoded_audio)
        samples = int(source.shape[1])
        if samples < self.audio_hop_samples:
            raise ValueError(
                "decoded window must be at least one audio hop: "
                f"{samples} < {self.audio_hop_samples}"
            )

        if self._window_samples is None:
            self._window_samples = samples
            self._envelope = sine_synthesis_window(samples)
        elif samples != self._window_samples:
            raise ValueError(
                "decoded window length changed without reset: "
                f"{samples} != {self._window_samples}"
            )

        window_start = self._next_window_position - self._buffer_start_position
        if window_start < 0:
            raise RuntimeError("OLA window position moved behind the accumulator")

        window_end = window_start + samples
        self._ensure_capacity(window_end)
        mapped_source = _map_output_channels(source, self.channels)
        self._audio[:, window_start:window_end] += mapped_source * self._envelope
        self._weights[window_start:window_end] += self._envelope
        self._next_window_position += self.audio_hop_samples

        read_offset = self._read_position - self._buffer_start_position
        read_end = read_offset + self.audio_hop_samples
        if read_offset < 0 or read_end > self._weights.shape[0]:
            raise RuntimeError("decoded window did not make one complete OLA hop available")

        weights = self._weights[read_offset:read_end]
        hop = np.zeros(
            (self.channels, self.audio_hop_samples), dtype=np.float32
        )
        np.divide(
            self._audio[:, read_offset:read_end],
            weights[None, :],
            out=hop,
            where=(weights > self.minimum_weight)[None, :],
        )

        self._read_position += self.audio_hop_samples
        discard_samples = read_end
        self._audio = self._audio[:, discard_samples:].copy()
        self._weights = self._weights[discard_samples:].copy()
        self._buffer_start_position += discard_samples
        self._windows_pushed += 1
        return hop


def offline_full_overlap_add(
    decoded_windows: Iterable[np.ndarray],
    audio_hop_samples: int,
    *,
    output_channels: Optional[int] = None,
    minimum_weight: float = DEFAULT_MINIMUM_OLA_WEIGHT,
) -> np.ndarray:
    """Offline reference for the emitted portion of full-output OLA.

    The result contains exactly ``window_count * audio_hop_samples`` samples,
    matching the concatenation of :meth:`StreamingFullOverlapAdd.push` results.
    Any un-emitted tail after the final decoded window is intentionally omitted.
    """

    hop = _positive_int(audio_hop_samples, "audio_hop_samples")
    minimum_weight = float(minimum_weight)
    if not np.isfinite(minimum_weight) or minimum_weight < 0.0:
        raise ValueError(
            f"minimum_weight must be finite and non-negative, got {minimum_weight}"
        )

    windows = [_coerce_audio_window(window) for window in decoded_windows]
    if not windows:
        if output_channels is None:
            raise ValueError("output_channels is required when decoded_windows is empty")
        channels = _positive_int(output_channels, "output_channels")
        return np.zeros((channels, 0), dtype=np.float32)

    channels = (
        int(windows[0].shape[0])
        if output_channels is None
        else _positive_int(output_channels, "output_channels")
    )
    window_samples = int(windows[0].shape[1])
    if window_samples < hop:
        raise ValueError(
            f"decoded window must be at least one audio hop: {window_samples} < {hop}"
        )
    if any(int(window.shape[1]) != window_samples for window in windows[1:]):
        raise ValueError("all decoded windows must have the same sample length")

    total_samples = (len(windows) - 1) * hop + window_samples
    audio_accumulator = np.zeros((channels, total_samples), dtype=np.float32)
    weight_accumulator = np.zeros((total_samples,), dtype=np.float32)
    envelope = sine_synthesis_window(window_samples)

    for index, window in enumerate(windows):
        start = index * hop
        end = start + window_samples
        audio_accumulator[:, start:end] += (
            _map_output_channels(window, channels) * envelope
        )
        weight_accumulator[start:end] += envelope

    emitted_samples = len(windows) * hop
    emitted = np.zeros((channels, emitted_samples), dtype=np.float32)
    weights = weight_accumulator[:emitted_samples]
    np.divide(
        audio_accumulator[:, :emitted_samples],
        weights[None, :],
        out=emitted,
        where=(weights > minimum_weight)[None, :],
    )
    return emitted
