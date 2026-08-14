#!/usr/bin/env python3
"""
Runtime tests for V2 unit-graph policy navigation.
"""

import numpy as np

from conftest import build_test_geometry
from stable_audio_wanderer.runtime.player import LatentNavigationEngine


def _build_v2_artifact(n_frames: int = 40, unit_size: int = 5, desc_dim: int = 12) -> dict:
    if n_frames % unit_size != 0:
        raise ValueError("n_frames must be divisible by unit_size for this synthetic setup.")

    n_files = 2
    file_len = n_frames // n_files
    units_per_file = file_len // unit_size
    n_units = n_frames // unit_size

    frame_file_ids = np.zeros((n_frames,), dtype=np.int32)
    frame_file_ids[file_len:] = 1
    frame_t = np.concatenate(
        [
            np.arange(file_len, dtype=np.int32),
            np.arange(file_len, dtype=np.int32),
        ],
        axis=0,
    )

    starts = np.arange(0, n_frames, unit_size, dtype=np.int32)
    ends = np.minimum(starts + unit_size, n_frames).astype(np.int32)
    unit_file = frame_file_ids[starts].astype(np.int32)
    unit_len = (ends - starts).astype(np.int32)

    frame_to_unit = np.zeros((n_frames,), dtype=np.int32)
    for u in range(n_units):
        frame_to_unit[starts[u] : ends[u]] = int(u)

    rng = np.random.default_rng(99)
    desc_frame = np.cumsum(
        rng.normal(scale=0.08, size=(n_frames, desc_dim)).astype(np.float32),
        axis=0,
    ).astype(np.float32)
    unit_entry = desc_frame[starts].astype(np.float32)
    unit_exit = desc_frame[ends - 1].astype(np.float32)
    unit_delta = (unit_exit - unit_entry).astype(np.float32)

    graph_k = 4
    neighbors = np.zeros((n_units, graph_k), dtype=np.int32)
    scores = np.zeros((n_units, graph_k), dtype=np.float32)
    for u in range(n_units):
        file_idx = u // units_per_file
        local = u % units_per_file
        same_next = file_idx * units_per_file + ((local + 1) % units_per_file)
        cross = ((file_idx + 1) % n_files) * units_per_file + local
        alt = file_idx * units_per_file + ((local + 2) % units_per_file)
        neighbors[u] = np.asarray([same_next, cross, alt, u], dtype=np.int32)
        scores[u] = np.asarray([0.10, 0.25, 0.40, 0.90], dtype=np.float32)

    return {
        "unit_start_idx": starts.astype(np.int32),
        "unit_end_idx": ends.astype(np.int32),
        "unit_file_id": unit_file.astype(np.int32),
        "unit_len": unit_len.astype(np.int32),
        "unit_entry_desc": unit_entry.astype(np.float32),
        "unit_exit_desc": unit_exit.astype(np.float32),
        "unit_delta_desc": unit_delta.astype(np.float32),
        "frame_to_unit": frame_to_unit.astype(np.int32),
        "unit_graph_neighbors": neighbors.astype(np.int32),
        "unit_graph_scores": scores.astype(np.float32),
        "unit_min_frames": np.array([unit_size], dtype=np.int32),
        "unit_max_frames": np.array([unit_size], dtype=np.int32),
        "unit_target_frames": np.array([unit_size], dtype=np.int32),
        "desc_frame": desc_frame.astype(np.float32),
    }


def _build_engine(policy_v2_enabled: bool, v2_artifact: dict) -> LatentNavigationEngine:
    n = 40
    rng = np.random.default_rng(77)
    latents = rng.normal(size=(n, 64)).astype(np.float32)
    geometry = build_test_geometry(latents)

    meta = np.zeros((n, 3), dtype=np.int32)
    meta[:, 0] = geometry.idx_to_file_id
    meta[:, 1] = geometry.idx_to_t
    file_offsets = np.array([0, n // 2, n], dtype=np.int64)

    desc = v2_artifact["desc_frame"].astype(np.float32)
    engine = LatentNavigationEngine(
        GG=latents,
        meta=meta,
        geometry=geometry,
        file_offsets=file_offsets,
        desc_weighted=desc,
        policy_v2_enabled=policy_v2_enabled,
        v2_artifact=v2_artifact,
        policy_recompose_enabled=True,
    )
    return engine


def test_policy_v2_runtime_emits_valid_indices():
    artifact = _build_v2_artifact()
    engine = _build_engine(policy_v2_enabled=True, v2_artifact=artifact)
    engine.set_random_controls(
        phrase_scale=0.4,
        jump_rate=0.9,
        timbre_lock=0.3,
        drift=0.8,
        repeat_avoid=0.8,
        crossfile=1.0,
    )

    emitted = []
    emitted_units = []
    for _ in range(96):
        frame = engine.step()
        idx = int(frame.nearest_idx)
        emitted.append(idx)
        emitted_units.append(int(artifact["frame_to_unit"][idx]))

    assert all(0 <= idx < engine.N for idx in emitted)
    assert len(set(emitted_units)) > 1

    state = engine.get_state()
    v2 = state["policy_v2"]
    assert v2["enabled"] is True
    assert v2["ready"] is True
    assert v2["active"] is True
    assert int(v2["unit_steps"]) >= 96
    assert int(v2["unit_transitions"]) > 0


def test_policy_v2_invalid_artifact_falls_back_to_v1():
    artifact = _build_v2_artifact()
    broken = dict(artifact)
    broken["frame_to_unit"] = artifact["frame_to_unit"][:-3]
    engine = _build_engine(policy_v2_enabled=True, v2_artifact=broken)

    assert engine._v2_ready is False
    for _ in range(8):
        frame = engine.step()
        assert 0 <= int(frame.nearest_idx) < engine.N

    state = engine.get_state()
    v2 = state["policy_v2"]
    assert v2["enabled"] is True
    assert v2["ready"] is False
    assert v2["active"] is False


def test_control_banks_and_variant_switching():
    artifact = _build_v2_artifact()
    engine = _build_engine(policy_v2_enabled=True, v2_artifact=artifact)

    engine.set_random_controls(jump_rate=0.31, timbre_lock=0.27, crossfile=0.73)
    engine.set_reorganized_controls(
        morph_len=0.82,
        jump_rate=0.64,
        timbre_lock=0.55,
        evolution=0.41,
        novelty=0.22,
        crossfile=0.66,
    )
    state = engine.get_state()
    random_ctrl = state["controls"]["random"]
    reorg_ctrl = state["controls"]["reorganized"]

    assert abs(float(random_ctrl["jump_rate"]) - 0.31) < 1e-6
    assert abs(float(random_ctrl["timbre_lock"]) - 0.27) < 1e-6
    assert abs(float(random_ctrl["crossfile"]) - 0.73) < 1e-6
    assert abs(float(reorg_ctrl["morph_len"]) - 0.82) < 1e-6
    assert abs(float(reorg_ctrl["jump_rate"]) - 0.64) < 1e-6
    assert abs(float(reorg_ctrl["timbre_lock"]) - 0.55) < 1e-6
    assert abs(float(reorg_ctrl["evolution"]) - 0.41) < 1e-6
    assert abs(float(reorg_ctrl["novelty"]) - 0.22) < 1e-6
    assert abs(float(reorg_ctrl["crossfile"]) - 0.66) < 1e-6

    assert engine.set_policy_variant("random") is True
    state_random = engine.get_state()
    assert state_random["active_variant"] == "random"
    assert state_random["policy_v2"]["active"] is False

    assert engine.set_policy_variant("reorganized") is True
    state_reorg = engine.get_state()
    assert state_reorg["active_variant"] == "reorganized"
    assert state_reorg["policy_v2"]["active"] is True


def run_all():
    test_policy_v2_runtime_emits_valid_indices()
    test_policy_v2_invalid_artifact_falls_back_to_v1()
    test_control_banks_and_variant_switching()
    print("All V2 runtime tests passed.")


if __name__ == "__main__":
    run_all()
