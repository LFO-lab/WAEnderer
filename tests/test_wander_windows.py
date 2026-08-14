import numpy as np
import pytest

from stable_audio_wanderer.runtime.wander_windows import (
    DEFAULT_WANDER_SEED,
    FRAME_SOURCE_CONTIGUOUS,
    FRAME_SOURCE_K_NEAREST,
    FRAME_SOURCE_MORPHOLOGY_GRAPH,
    WanderWindowPlanner,
    apply_frame_order_disorder,
    latent_colour_raw_noise,
    latent_colour_smoothed_noise,
    mix_frame_order_seed,
)


def _planner(
    *,
    frame_count=12,
    latent_dim=3,
    file_offsets=(0, 7, 12),
    manual_points=None,
    graph=None,
    seed=DEFAULT_WANDER_SEED,
):
    z_concat = (
        np.arange(frame_count * latent_dim, dtype=np.float32).reshape(frame_count, latent_dim)
        / np.float32(10.0)
    )
    if manual_points is None:
        manual_points = np.column_stack(
            (
                np.linspace(0.0, 1.0, frame_count, dtype=np.float32),
                np.linspace(1.0, 0.0, frame_count, dtype=np.float32),
            )
        )
    graph = graph or {}
    return WanderWindowPlanner(
        z_concat,
        np.asarray(file_offsets, dtype=np.int64),
        np.asarray(manual_points, dtype=np.float32),
        np.linspace(10.0, 20.0, latent_dim, dtype=np.float32),
        np.linspace(2.0, 4.0, latent_dim, dtype=np.float32),
        unit_start_idx=graph.get("starts"),
        unit_end_idx=graph.get("ends"),
        unit_graph_neighbors=graph.get("neighbors"),
        unit_graph_scores=graph.get("scores"),
        seed=seed,
    )


def test_k_nearest_uses_anchor_manual_map_position_and_can_cross_files():
    points = np.asarray(
        [
            [0.90, 0.90],
            [0.12, 0.10],
            [0.10, 0.10],  # anchor
            [0.80, 0.80],
            [0.30, 0.30],
            [0.11, 0.10],  # other source file, second closest
        ],
        dtype=np.float32,
    )
    planner = _planner(frame_count=6, latent_dim=2, file_offsets=(0, 3, 6), manual_points=points)

    result = planner.plan(2, 4)

    assert result.source_frames == (2, 5, 1, 4)
    assert result.input_frames == result.source_frames
    assert result.diagnostics.requested_frame_source == FRAME_SOURCE_K_NEAREST
    assert result.diagnostics.effective_frame_source == FRAME_SOURCE_K_NEAREST
    assert result.diagnostics.k_nearest_distances_nondecreasing
    normalized = np.arange(12, dtype=np.float32).reshape(6, 2) / np.float32(10.0)
    mean = np.linspace(10.0, 20.0, 2, dtype=np.float32)
    std = np.linspace(2.0, 4.0, 2, dtype=np.float32)
    np.testing.assert_array_equal(
        result.raw_latents,
        normalized[np.asarray(result.input_frames)] * std[None, :] + mean[None, :],
    )


@pytest.mark.parametrize(
    ("anchor", "window", "expected"),
    [
        (0, 4, (0, 1, 2, 3)),
        (4, 4, (1, 2, 3, 4)),
        (5, 4, (5, 6, 6, 6)),
        (6, 4, (5, 6, 6, 6)),
    ],
)
def test_contiguous_windows_shift_at_ends_and_only_pad_short_files(anchor, window, expected):
    planner = _planner(frame_count=7, file_offsets=(0, 5, 7))

    result = planner.plan(anchor, window, frame_source=FRAME_SOURCE_CONTIGUOUS)

    assert result.source_frames == expected
    assert result.input_frames == expected
    file_id = 0 if anchor < 5 else 1
    bounds = ((0, 5), (5, 7))[file_id]
    assert all(bounds[0] <= frame < bounds[1] for frame in result.source_frames)


@pytest.mark.parametrize(
    ("window", "expected", "mixed_seed"),
    [
        (2, [10, 11], 0x511641FC),
        (4, [10, 12, 11, 13], 0xE64BD99F),
        (8, [10, 12, 16, 14, 11, 15, 13, 17], 0xD4F79787),
        (
            16,
            [10, 20, 14, 21, 18, 16, 15, 23, 19, 22, 17, 12, 11, 24, 13, 25],
            0xB89F9FB7,
        ),
        (
            32,
            [
                10, 13, 31, 11, 29, 21, 17, 19, 39, 22, 23, 25, 27, 20, 12, 35,
                26, 14, 24, 16, 33, 15, 36, 32, 18, 28, 37, 30, 40, 38, 34, 41,
            ],
            0xFA1ADC82,
        ),
    ],
)
def test_frame_order_matches_juce_uint32_golden_and_preserves_endpoints(
    window, expected, mixed_seed
):
    source = np.arange(10, 10 + window, dtype=np.int64)

    ordered = apply_frame_order_disorder(
        source,
        1.0,
        DEFAULT_WANDER_SEED,
        chunk_start_frame=10,
        latent_window=window,
        decode_window_index=0,
    )

    assert mix_frame_order_seed(DEFAULT_WANDER_SEED, 10, window, 0) == mixed_seed
    assert ordered.tolist() == expected
    assert ordered[0] == source[0]
    assert ordered[-1] == source[-1]
    assert sorted(ordered.tolist()) == sorted(source.tolist())


def test_frame_order_preserves_repeated_frame_multiset_and_advances_decode_seed():
    source = np.asarray([5, 6, 6, 6, 6, 6, 6, 6], dtype=np.int64)
    first = apply_frame_order_disorder(source, 0.73, 123, 5, 8, 0)
    second = apply_frame_order_disorder(source, 0.73, 123, 5, 8, 1)

    assert first[0] == second[0] == 5
    assert first[-1] == second[-1] == 6
    assert sorted(first.tolist()) == sorted(source.tolist())
    assert sorted(second.tolist()) == sorted(source.tolist())
    assert mix_frame_order_seed(123, 5, 8, 0) != mix_frame_order_seed(123, 5, 8, 1)


@pytest.mark.parametrize(
    ("frame", "dimension", "raw", "smoothed"),
    [
        (0, 0, 0.6365024447441101, 0.3835291564464569),
        (0, 37, -0.6747004985809326, -0.29120099544525146),
        (8, 37, 0.1854221075773239, 0.33007773756980896),
        (39, 255, 0.29442986845970154, 0.2906986176967621),
    ],
)
def test_latent_colour_hash_and_five_tap_smoothing_match_juce_golden(
    frame, dimension, raw, smoothed
):
    assert float(latent_colour_raw_noise(DEFAULT_WANDER_SEED, frame, dimension)) == raw
    assert (
        float(
            latent_colour_smoothed_noise(
                DEFAULT_WANDER_SEED, frame, dimension, 0, 128
            )
        )
        == smoothed
    )


def test_latent_colour_smoothing_clamps_to_exclusive_source_file_bounds():
    seed = DEFAULT_WANDER_SEED
    end_of_first_file = latent_colour_smoothed_noise(seed, 2, 1, 0, 3)
    expected = np.float32(
        (
            latent_colour_raw_noise(seed, 0, 1)
            + np.float32(4.0) * latent_colour_raw_noise(seed, 1, 1)
            + np.float32(6.0) * latent_colour_raw_noise(seed, 2, 1)
            + np.float32(4.0) * latent_colour_raw_noise(seed, 2, 1)
            + latent_colour_raw_noise(seed, 2, 1)
        )
        * np.float32(1.0 / 16.0)
    )
    leaked = latent_colour_smoothed_noise(seed, 2, 1, 0, 6)

    assert end_of_first_file == expected
    assert end_of_first_file != leaked


def test_transform_order_is_selection_then_order_then_colour_then_denormalization():
    planner = _planner(frame_count=8, latent_dim=3, file_offsets=(0, 4, 8))
    result = planner.plan(
        0,
        4,
        frame_source=FRAME_SOURCE_CONTIGUOUS,
        frame_order=1.0,
        latent_colour=0.5,
    )

    assert result.source_frames == (0, 1, 2, 3)
    assert result.input_frames == (0, 2, 1, 3)
    normalized = np.arange(24, dtype=np.float32).reshape(8, 3) / np.float32(10.0)
    mean = np.linspace(10.0, 20.0, 3, dtype=np.float32)
    std = np.linspace(2.0, 4.0, 3, dtype=np.float32)
    expected = np.empty((4, 3), dtype=np.float32)
    sigma = np.float32(0.5)
    for row, frame in enumerate(result.input_frames):
        for dimension in range(3):
            noise = latent_colour_smoothed_noise(
                DEFAULT_WANDER_SEED, frame, dimension, 0, 4
            )
            coloured = np.float32(normalized[frame, dimension] + sigma * noise)
            expected[row, dimension] = np.float32(
                coloured * std[dimension] + mean[dimension]
            )
    np.testing.assert_allclose(result.raw_latents, expected, rtol=0.0, atol=2e-6)

    uncoloured = _planner(frame_count=8, latent_dim=3, file_offsets=(0, 4, 8)).plan(
        0,
        4,
        frame_source=FRAME_SOURCE_CONTIGUOUS,
        frame_order=1.0,
        latent_colour=0.0,
    )
    assert uncoloured.source_frames == result.source_frames
    assert uncoloured.input_frames == result.input_frames
    assert not np.array_equal(uncoloured.raw_latents, result.raw_latents)


def test_missing_or_invalid_morphology_graph_silently_falls_back_to_contiguous():
    missing = _planner(frame_count=8, file_offsets=(0, 4, 8))
    missing_result = missing.plan(3, 4, frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH)

    assert not missing.graph_available
    assert missing_result.source_frames == (0, 1, 2, 3)
    assert missing_result.diagnostics.requested_frame_source == FRAME_SOURCE_MORPHOLOGY_GRAPH
    assert missing_result.diagnostics.effective_frame_source == FRAME_SOURCE_CONTIGUOUS
    assert missing_result.diagnostics.graph_error

    invalid = _planner(
        frame_count=8,
        file_offsets=(0, 4, 8),
        graph={
            "starts": [0, 4],
            "ends": [4, 8],
            "neighbors": [[1], [2]],
        },
    )
    invalid_result = invalid.plan(4, 4, frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH)
    assert not invalid.graph_available
    assert invalid_result.source_frames == (4, 5, 6, 7)
    assert invalid_result.diagnostics.effective_frame_source == FRAME_SOURCE_CONTIGUOUS


def test_graph_walk_retains_state_and_twelve_planned_units_then_reset_replays():
    graph = {
        "starts": [0, 2, 4, 6, 8, 10],
        "ends": [2, 4, 6, 8, 10, 12],
        "neighbors": [[1], [2], [3], [4], [5], [0]],
        "scores": np.zeros((6, 1), dtype=np.float32),
    }
    planner = _planner(frame_count=12, file_offsets=(0, 6, 12), graph=graph)

    first = planner.plan(
        0,
        4,
        frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH,
        drift=0.5,
    )
    second = planner.plan(
        10,
        4,
        frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH,
        drift=0.5,
    )

    assert planner.graph_available
    assert first.source_frames == (0, 1, 2, 3)
    assert second.source_frames == (4, 5, 6, 7)
    assert first.diagnostics.effective_frame_source == FRAME_SOURCE_MORPHOLOGY_GRAPH
    assert first.diagnostics.planned_unit_count == 12
    assert second.diagnostics.planned_unit_count == 12
    assert len(second.diagnostics.recent_units) >= 3

    planner.reset()
    replay = planner.plan(
        0,
        4,
        frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH,
        drift=0.5,
    )
    assert replay.source_frames == first.source_frames
    assert replay.input_frames == first.input_frames
    assert replay.diagnostics.decode_window_index == 0


def test_graph_control_mapping_and_inverse_crossfile_penalty_affect_choice():
    points = np.zeros((3, 2), dtype=np.float32)
    graph = {
        "starts": [0, 1, 2],
        "ends": [1, 2, 3],
        "neighbors": [[1, 2], [1, 1], [2, 2]],
        "scores": [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    }
    planner = _planner(
        frame_count=3,
        latent_dim=2,
        file_offsets=(0, 2, 3),
        manual_points=points,
        graph=graph,
    )

    keep_file = planner.plan(
        0,
        2,
        frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH,
        phrase_scale=0.25,
        jump_rate=0.0,
        timbre_lock=0.75,
        drift=0.0,
        repeat_avoid=0.0,
        crossfile=0.0,
    )
    mapped = keep_file.diagnostics.graph_controls
    assert mapped.segment_length == 0.25
    assert mapped.continuity == 1.0
    assert mapped.radius == 0.25
    assert mapped.wander == mapped.motion_rate == 0.0
    assert mapped.novelty == 0.0
    assert mapped.crossfile_penalty == 1.0
    assert keep_file.source_frames == (0, 1)

    planner.reset()
    allow_crossfile = planner.plan(
        0,
        2,
        frame_source=FRAME_SOURCE_MORPHOLOGY_GRAPH,
        jump_rate=0.0,
        drift=0.0,
        repeat_avoid=0.0,
        crossfile=1.0,
    )
    assert allow_crossfile.diagnostics.graph_controls.crossfile_penalty == 0.0
    assert allow_crossfile.source_frames == (0, 2)


def test_invalid_source_fails_at_pure_api_boundary_without_changing_diagnostics():
    planner = _planner()
    before = planner.last_diagnostics

    with pytest.raises(ValueError, match="unsupported Wander frame source"):
        planner.plan(0, 4, frame_source="not-a-source")

    assert planner.last_diagnostics is before
