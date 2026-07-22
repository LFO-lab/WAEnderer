#!/usr/bin/env python3
"""Tests for complete file-bounded manual latent windows."""

import numpy as np
import pytest

from stable_audio_wanderer.runtime.manual_windows import (
    assemble_manual_latent_window,
    build_file_bounded_latent_window,
    plan_file_bounded_latent_window,
)


def test_complete_chunks_clamp_at_each_file_start_and_end():
    offsets = np.array([0, 5, 11], dtype=np.int64)

    cases = {
        0: (0, 1, 2, 3),
        4: (1, 2, 3, 4),
        5: (5, 6, 7, 8),
        10: (7, 8, 9, 10),
    }
    for anchor, expected in cases.items():
        plan = plan_file_bounded_latent_window(anchor, 4, offsets)
        assert plan.frame_indices == expected
        assert all(plan.file_start <= index < plan.file_end for index in expected)
        assert len(set(expected)) == 4
        assert not plan.is_short_file


def test_short_files_repeat_only_their_final_frame():
    offsets = np.array([0, 2, 6], dtype=np.int64)

    short = plan_file_bounded_latent_window(1, 4, offsets)
    assert short.frame_indices == (0, 1, 1, 1)
    assert short.is_short_file
    assert short.repeated_final_frames == 2
    assert set(short.frame_indices).issubset({0, 1})

    complete = plan_file_bounded_latent_window(2, 4, offsets)
    assert complete.frame_indices == (2, 3, 4, 5)
    assert complete.repeated_final_frames == 0
    assert len(set(complete.frame_indices)) == 4


def test_static_anchor_repeats_identical_chunk_without_tail_state():
    latents = np.arange(9 * 3, dtype=np.float32).reshape(9, 3)
    offsets = np.array([0, 4, 9], dtype=np.int64)

    first_window, first_plan = build_file_bounded_latent_window(
        latents, offsets, anchor_frame=8, window_size=4
    )
    second_window, second_plan = build_file_bounded_latent_window(
        latents, offsets, anchor_frame=8, window_size=4
    )

    assert first_plan == second_plan
    assert first_plan.frame_indices == (5, 6, 7, 8)
    assert np.array_equal(first_window, second_window)
    assert np.array_equal(first_window, latents[[5, 6, 7, 8]])


def test_assembly_denormalizes_after_file_local_gather():
    normalized = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    offsets = np.array([0, 3, 6], dtype=np.int64)
    plan = plan_file_bounded_latent_window(2, 4, offsets)
    mean = np.array([10.0, -4.0], dtype=np.float32)
    std = np.array([2.0, 0.5], dtype=np.float32)

    raw = assemble_manual_latent_window(
        normalized,
        plan,
        mean=mean,
        std=std,
    )
    expected_normalized = normalized[[0, 1, 2, 2]]
    assert raw.dtype == np.float32
    assert np.array_equal(raw, expected_normalized * std + mean)


def test_anchor_clamp_and_invalid_boundaries_fail_closed():
    offsets = np.array([0, 3, 7], dtype=np.int64)
    assert plan_file_bounded_latent_window(-50, 2, offsets).anchor_frame == 0
    assert plan_file_bounded_latent_window(500, 2, offsets).anchor_frame == 6

    with pytest.raises(ValueError, match="start at 0"):
        plan_file_bounded_latent_window(1, 2, [1, 4])
    with pytest.raises(ValueError, match="nondecreasing"):
        plan_file_bounded_latent_window(1, 2, [0, 4, 3])
    with pytest.raises(ValueError, match="end at frame_count"):
        plan_file_bounded_latent_window(1, 2, [0, 4], frame_count=5)


if __name__ == "__main__":
    pytest.main([__file__])
