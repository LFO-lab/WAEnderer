#!/usr/bin/env python3
"""
Tests for preprocess silence trimming.
"""

import numpy as np

from stable_audio_wanderer.preprocess import SilenceTrimConfig, trim_silent_frames


def _make_wave(
    frame_levels,
    samples_per_frame: int = 64,
) -> np.ndarray:
    chunks = []
    for level in frame_levels:
        if isinstance(level, tuple):
            frame = np.tile(np.asarray(level, dtype=np.float32)[None, :], (samples_per_frame, 1))
        else:
            amp = float(level)
            frame = np.full((samples_per_frame, 2), amp, dtype=np.float32)
        chunks.append(frame)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def test_trim_silent_frames_removes_only_long_gaps_and_keeps_padding():
    cfg = SilenceTrimConfig(
        enabled=True,
        threshold_db=-20.0,
        min_silence_sec=0.30,
        keep_silence_sec=0.10,
    )
    wav = _make_wave([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    latents = np.arange(9 * 2, dtype=np.float32).reshape(9, 2)
    desc = np.arange(9 * 3, dtype=np.float32).reshape(9, 3)

    z_trimmed, desc_trimmed, result = trim_silent_frames(
        wav_stereo=wav,
        latents=latents,
        descriptors=desc,
        cfg=cfg,
        latent_hz=10.0,
    )

    expected_mask = np.array([True, True, True, False, False, False, True, True, True], dtype=bool)
    assert np.array_equal(result.keep_mask, expected_mask)
    assert result.original_frames == 9
    assert result.kept_frames == 6
    assert result.removed_frames == 3
    assert np.array_equal(z_trimmed, latents[expected_mask])
    assert np.array_equal(desc_trimmed, desc[expected_mask])


def test_trim_silent_frames_preserves_short_gaps():
    cfg = SilenceTrimConfig(
        enabled=True,
        threshold_db=-20.0,
        min_silence_sec=0.35,
        keep_silence_sec=0.0,
    )
    wav = _make_wave([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    latents = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)
    desc = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

    z_trimmed, desc_trimmed, result = trim_silent_frames(
        wav_stereo=wav,
        latents=latents,
        descriptors=desc,
        cfg=cfg,
        latent_hz=10.0,
    )

    assert np.all(result.keep_mask)
    assert result.removed_frames == 0
    assert np.array_equal(z_trimmed, latents)
    assert np.array_equal(desc_trimmed, desc)


def test_trim_silent_frames_keeps_fully_silent_files_intact():
    cfg = SilenceTrimConfig(
        enabled=True,
        threshold_db=-20.0,
        min_silence_sec=0.25,
        keep_silence_sec=0.10,
    )
    wav = _make_wave([(0.0, 0.0)] * 5)
    latents = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    desc = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)

    z_trimmed, desc_trimmed, result = trim_silent_frames(
        wav_stereo=wav,
        latents=latents,
        descriptors=desc,
        cfg=cfg,
        latent_hz=10.0,
    )

    assert np.all(result.keep_mask)
    assert result.raw_active_frames == 0
    assert result.removed_frames == 0
    assert np.array_equal(z_trimmed, latents)
    assert np.array_equal(desc_trimmed, desc)


def test_short_clips_keep_mask_aligned_when_padding_exceeds_clip_length():
    cfg = SilenceTrimConfig(threshold_db=-20.0, keep_silence_sec=0.1)
    for levels in ([1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0, 0.0]):
        frames = len(levels)
        latents = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2)
        desc = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
        z_trimmed, desc_trimmed, result = trim_silent_frames(
            _make_wave(levels), latents, desc, cfg, latent_hz=21.5,
        )
        assert result.keep_mask.shape == (frames,)
        assert result.keep_mask.all()
        assert result.removed_frames == 0
        np.testing.assert_array_equal(z_trimmed, latents)
        np.testing.assert_array_equal(desc_trimmed, desc)


def test_padding_short_mask_retains_active_frame_position():
    from stable_audio_wanderer.preprocess.silence import _pad_active_regions

    np.testing.assert_array_equal(
        _pad_active_regions(np.array([True, False, False, False]), keep_frames=2),
        [True, True, True, False],
    )
    np.testing.assert_array_equal(
        _pad_active_regions(np.array([False, False, False, True]), keep_frames=2),
        [False, True, True, True],
    )


def run_all():
    test_trim_silent_frames_removes_only_long_gaps_and_keeps_padding()
    test_trim_silent_frames_preserves_short_gaps()
    test_trim_silent_frames_keeps_fully_silent_files_intact()
    test_short_clips_keep_mask_aligned_when_padding_exceeds_clip_length()
    test_padding_short_mask_retains_active_frame_position()
    print("All preprocess silence tests passed.")


if __name__ == "__main__":
    run_all()
