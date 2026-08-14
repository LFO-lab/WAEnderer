#!/usr/bin/env python3
"""Tests for JUCE-parity normalized full-output overlap-add."""

import numpy as np
import pytest

from stable_audio_wanderer.runtime.overlap_add import (
    StreamingFullOverlapAdd,
    offline_full_overlap_add,
    sine_synthesis_window,
)


def test_sine_window_and_amplitude_normalization():
    envelope = sine_synthesis_window(8)
    expected = np.sin(
        np.pi * (np.arange(8, dtype=np.float32) + 0.5) / 8.0
    ).astype(np.float32)
    assert envelope.dtype == np.float32
    assert np.allclose(envelope, expected, atol=1.0e-7)

    ola = StreamingFullOverlapAdd(audio_hop_samples=8, channels=2)
    emitted = [ola.push(np.ones((2, 16), dtype=np.float32)) for _ in range(4)]
    output = np.concatenate(emitted, axis=1)
    assert output.dtype == np.float32
    assert output.shape == (2, 32)
    assert np.allclose(output, 1.0, atol=1.0e-6)


def test_exact_hop_length_reset_and_window_length_guard():
    rng = np.random.default_rng(71)
    window = rng.normal(size=(2, 20)).astype(np.float32)
    ola = StreamingFullOverlapAdd(audio_hop_samples=7, channels=2)

    first = ola.push(window)
    second = ola.push(window)
    assert first.shape == second.shape == (2, 7)
    assert ola.windows_pushed == 2
    assert ola.window_samples == 20

    with pytest.raises(ValueError, match="changed without reset"):
        ola.push(np.zeros((2, 21), dtype=np.float32))

    ola.reset()
    assert ola.window_samples is None
    assert ola.windows_pushed == 0
    assert ola.buffered_samples == 0
    after_reset = ola.push(window)
    assert np.array_equal(after_reset, first)


def test_channel_mapping_matches_juce_and_rejects_invalid_audio():
    mono = np.linspace(-1.0, 1.0, 12, dtype=np.float32)[None, :]
    stereo_ola = StreamingFullOverlapAdd(audio_hop_samples=6, channels=2)
    stereo = stereo_ola.push(mono)
    assert np.array_equal(stereo[0], stereo[1])

    three_channel = np.stack(
        [
            np.full(12, 1.0, dtype=np.float32),
            np.full(12, 2.0, dtype=np.float32),
            np.full(12, 3.0, dtype=np.float32),
        ]
    )
    two_channel_ola = StreamingFullOverlapAdd(audio_hop_samples=6, channels=2)
    mapped = two_channel_ola.push(three_channel)
    assert np.allclose(mapped[0], 1.0)
    assert np.allclose(mapped[1], 2.0)

    with pytest.raises(ValueError, match="channels, samples"):
        stereo_ola.push(np.zeros((1, 2, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        stereo_ola.push(np.full((1, 12), np.nan, dtype=np.float32))


def test_streaming_ola_preserves_continuous_signal_across_seams():
    hop = 13
    window_samples = 26
    window_count = 6
    total = (window_count - 1) * hop + window_samples
    sample = np.arange(total, dtype=np.float32)
    continuous = np.stack(
        [
            np.sin(np.float32(0.071) * sample),
            np.cos(np.float32(0.043) * sample),
        ]
    ).astype(np.float32)
    windows = [
        continuous[:, start : start + window_samples]
        for start in range(0, window_count * hop, hop)
    ]

    ola = StreamingFullOverlapAdd(audio_hop_samples=hop, channels=2)
    streamed = np.concatenate([ola.push(window) for window in windows], axis=1)
    expected = continuous[:, : window_count * hop]

    assert streamed.shape == expected.shape
    assert np.allclose(streamed, expected, atol=1.0e-6)
    for seam in range(hop, streamed.shape[1], hop):
        assert np.allclose(
            streamed[:, seam - 1 : seam + 1],
            expected[:, seam - 1 : seam + 1],
            atol=1.0e-6,
        )


def test_streaming_matches_independent_offline_reference_within_1e5():
    rng = np.random.default_rng(904)
    windows = [rng.normal(size=(2, 32)).astype(np.float32) for _ in range(9)]
    hop = 16

    ola = StreamingFullOverlapAdd(audio_hop_samples=hop, channels=2)
    streamed = np.concatenate([ola.push(window) for window in windows], axis=1)
    offline = offline_full_overlap_add(
        windows,
        hop,
        output_channels=2,
    )

    assert streamed.shape == offline.shape == (2, len(windows) * hop)
    assert np.max(np.abs(streamed - offline)) <= 1.0e-5


if __name__ == "__main__":
    pytest.main([__file__])
