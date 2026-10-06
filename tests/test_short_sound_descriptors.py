"""Regression coverage for short clips that cannot use 30-frame pitch smoothing."""
from unittest.mock import patch

import numpy as np
import pytest

from stable_audio_wanderer.cli.preprocess import (
    AF, _build_mfcc_transform, compute_latent_aligned_descriptors,
)


@pytest.mark.parametrize('hop,duration,expected_window', [
    (2048, 0.01, 3), (2048, 0.35, 7), (2048, 1.5, 30),
    (4096, 0.01, 3), (4096, 0.09, 3), (4096, 0.10, 3), (4096, 3.0, 30),
])
def test_pitch_smoothing_preserves_short_clips(hop, duration, expected_window):
    sr = 44100
    t = np.arange(int(sr * duration)) / sr
    mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = np.repeat(mono[:, None], 2, axis=1)
    target = max(1, int(np.ceil(len(mono) / hop)))
    transform = _build_mfcc_transform(hop, sr)
    detect = AF.detect_pitch_frequency
    with patch.object(AF, 'detect_pitch_frequency', wraps=detect) as spy:
        result = compute_latent_aligned_descriptors(wav, target, transform, hop, sr)
    assert spy.call_args.kwargs['win_length'] == expected_window
    assert result.shape == (target, 35)
    assert np.isfinite(result).all()
    if hop == 4096 and duration < 0.093:
        assert spy.call_args.kwargs['frame_time'] < hop / sr
    else:
        assert spy.call_args.kwargs['frame_time'] == hop / sr
    if expected_window == 30:
        def original_default(*args, **kwargs):
            kwargs.pop('win_length')
            return detect(*args, **kwargs)

        with patch.object(AF, 'detect_pitch_frequency', side_effect=original_default):
            baseline = compute_latent_aligned_descriptors(wav, target, transform, hop, sr)
        np.testing.assert_array_equal(result, baseline)
