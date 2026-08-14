#!/usr/bin/env python3
"""
Tests for V2 unit-graph artifact construction.
"""

import numpy as np

from stable_audio_wanderer.policy.v2_units import UnitGraphConfig, build_v2_unit_artifact


def _synthetic_inputs(
    n_files: int = 2,
    frames_per_file: int = 120,
    desc_dim: int = 35,
):
    rng = np.random.default_rng(1234)
    n_frames = int(n_files * frames_per_file)
    file_offsets = np.arange(0, n_frames + 1, frames_per_file, dtype=np.int64)
    frame_file_ids = np.repeat(np.arange(n_files, dtype=np.int32), frames_per_file)
    frame_t = np.tile(np.arange(frames_per_file, dtype=np.int32), n_files)

    # Smooth random walk descriptors to emulate timbral trajectories.
    step = rng.normal(scale=0.05, size=(n_frames, desc_dim)).astype(np.float32)
    desc = np.cumsum(step, axis=0).astype(np.float32)
    return file_offsets, frame_file_ids, frame_t, desc


def test_v2_unit_artifact_shapes_and_ranges():
    file_offsets, frame_file_ids, frame_t, desc = _synthetic_inputs()
    cfg = UnitGraphConfig(
        min_sec=2.0,
        max_sec=6.0,
        target_sec=4.0,
        latent_hz=20.0,
        candidate_k=32,
        graph_k=12,
    )

    artifact = build_v2_unit_artifact(
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        desc_weighted=desc,
        cfg=cfg,
        source_corpus_path="/tmp/corpus.npz",
    )

    starts = artifact["unit_start_idx"]
    ends = artifact["unit_end_idx"]
    lens = artifact["unit_len"]
    frame_to_unit = artifact["frame_to_unit"]
    neighbors = artifact["unit_graph_neighbors"]
    scores = artifact["unit_graph_scores"]
    n_units = int(starts.shape[0])

    assert n_units > 0
    assert starts.shape == ends.shape == lens.shape
    assert frame_to_unit.shape[0] == desc.shape[0]
    assert neighbors.shape[0] == n_units
    assert scores.shape == neighbors.shape

    assert np.all(ends > starts)
    assert starts.min() >= 0
    assert ends.max() <= desc.shape[0]
    assert np.all(frame_to_unit >= 0)
    assert np.all(frame_to_unit < n_units)
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_units)
    assert np.all(np.isfinite(scores))

    # Every frame must be covered by exactly one unit id in mapping.
    assert frame_to_unit.min() >= 0
    assert frame_to_unit.max() < n_units

    # Basic monotonicity over unit boundaries.
    assert np.all(starts[1:] >= starts[:-1])
    assert np.all(ends[1:] >= ends[:-1])


def test_v2_graph_has_nontrivial_transitions():
    file_offsets, frame_file_ids, frame_t, desc = _synthetic_inputs(
        n_files=3,
        frames_per_file=90,
        desc_dim=16,
    )
    cfg = UnitGraphConfig(
        min_sec=0.8,
        max_sec=2.0,
        target_sec=1.2,
        latent_hz=20.0,
        candidate_k=24,
        graph_k=8,
    )
    artifact = build_v2_unit_artifact(
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        desc_weighted=desc,
        cfg=cfg,
        source_corpus_path="/tmp/corpus.npz",
    )

    neighbors = artifact["unit_graph_neighbors"]
    n_units = neighbors.shape[0]
    assert n_units > 1

    # Ensure graph is not fully degenerate: several rows should have >1 unique target.
    diverse_rows = 0
    for i in range(n_units):
        if len(set(int(x) for x in neighbors[i].tolist())) > 1:
            diverse_rows += 1
    assert diverse_rows >= max(1, n_units // 3)


def run_all():
    test_v2_unit_artifact_shapes_and_ranges()
    test_v2_graph_has_nontrivial_transitions()
    print("All V2 unit artifact tests passed.")


if __name__ == "__main__":
    run_all()
