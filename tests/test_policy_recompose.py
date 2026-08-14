#!/usr/bin/env python3
"""
Tests for policy-mode short-unit recomposition (V1.1).
"""

import numpy as np

from conftest import build_test_geometry
from stable_audio_wanderer.runtime.player import LatentNavigationEngine


def _build_engine(
    n: int = 24,
    with_desc: bool = True,
    policy_recompose_enabled: bool = True,
) -> LatentNavigationEngine:
    rng = np.random.default_rng(29)
    latents = rng.normal(size=(n, 64)).astype(np.float32)
    geometry = build_test_geometry(latents)

    meta = np.zeros((n, 3), dtype=np.int32)
    meta[:, 0] = geometry.idx_to_file_id
    meta[:, 1] = geometry.idx_to_t
    file_offsets = np.array([0, n // 2, n], dtype=np.int64)

    desc = None
    if with_desc:
        desc = rng.normal(size=(n, 35)).astype(np.float32)

    engine = LatentNavigationEngine(
        GG=latents,
        meta=meta,
        geometry=geometry,
        file_offsets=file_offsets,
        desc_weighted=desc,
        policy_timbre_swap_enabled=True,
        policy_recompose_enabled=policy_recompose_enabled,
        control_jump_rate=0.35,
        control_timbre_lock=0.5,
        control_crossfile=0.6,
    )
    return engine


def test_recompose_path_validity():
    engine = _build_engine(with_desc=True, policy_recompose_enabled=True)
    engine.set_random_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=1.0)
    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        engine._fill_retrieval_buffer_recomposed(
            seed_idx=0,
            anchor_idx=0,
            chunk_len=12,
            direction=1,
        )
        path = list(engine._retrieval_buffer)
    finally:
        engine._lookahead_best_s1 = original_lookahead
    assert len(path) == 12
    assert all(0 <= int(idx) < engine.N for idx in path)


def test_recompose_creates_non_serial_transition():
    engine = _build_engine(with_desc=True, policy_recompose_enabled=True)
    engine.set_random_controls(jump_rate=1.0, timbre_lock=1.0, crossfile=0.0)

    # Force a strong same-file timbre attractor away from t+1.
    seed_idx = 0
    off_idx = 8
    seed_desc = np.zeros((engine.desc_weighted.shape[1],), dtype=np.float32)
    far_desc = np.ones((engine.desc_weighted.shape[1],), dtype=np.float32) * 10.0
    engine.desc_weighted[:] = far_desc
    engine.desc_weighted[seed_idx] = seed_desc
    engine.desc_weighted[off_idx] = seed_desc

    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        engine._fill_retrieval_buffer_recomposed(
            seed_idx=seed_idx,
            anchor_idx=seed_idx,
            chunk_len=8,
            direction=1,
        )
        path = [int(x) for x in list(engine._retrieval_buffer)]
    finally:
        engine._lookahead_best_s1 = original_lookahead

    non_serial = False
    for i in range(1, len(path)):
        prev_idx = path[i - 1]
        cur_idx = path[i]
        same_file = int(engine._idx_to_file_id[prev_idx]) == int(engine._idx_to_file_id[cur_idx])
        dt = int(engine._idx_to_t[cur_idx]) - int(engine._idx_to_t[prev_idx])
        if not (same_file and dt == 1):
            non_serial = True
            break

    assert non_serial, f"Expected at least one non-serial transition, got path={path}"


def test_navigation_uses_contiguous_fill_when_recompose_disabled():
    engine = _build_engine(with_desc=True, policy_recompose_enabled=False)
    engine.set_random_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=1.0)
    for _ in range(6):
        engine.step()
    state = engine.get_state()
    assert state["recompose"]["paths_built"] == 0


def test_recompose_avoids_self_stalls():
    engine = _build_engine(with_desc=True, policy_recompose_enabled=True)
    engine.set_random_controls(
        jump_rate=1.0,
        timbre_lock=0.75,
        repeat_avoid=1.0,
        crossfile=1.0,
    )
    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        engine._fill_retrieval_buffer_recomposed(
            seed_idx=0,
            anchor_idx=0,
            chunk_len=16,
            direction=0,
        )
        path = [int(x) for x in list(engine._retrieval_buffer)]
    finally:
        engine._lookahead_best_s1 = original_lookahead

    self_stalls = sum(int(path[i] == path[i - 1]) for i in range(1, len(path)))
    assert self_stalls == 0, f"Expected no immediate self stalls, got path={path}"


def test_jump_rate_increases_path_novelty():
    engine = _build_engine(with_desc=True, policy_recompose_enabled=True)
    assert engine.desc_weighted is not None

    # Neutralize timbre and embedding costs to isolate jump-rate effects.
    engine.desc_weighted[:] = 0.0
    base_emb = np.zeros((engine.embeddings_l2.shape[1],), dtype=np.float32)
    base_emb[0] = 1.0
    engine.embeddings_l2[:] = base_emb[None, :]

    def count_repeat_hits(path):
        recent = set()
        repeats = 0
        for idx in path:
            idx_i = int(idx)
            if idx_i in recent:
                repeats += 1
            recent.add(idx_i)
        return repeats

    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        engine.set_random_controls(
            jump_rate=0.0,
            timbre_lock=0.0,
            repeat_avoid=1.0,
            crossfile=0.0,
        )
        engine._retrieval_buffer.clear()
        engine._fill_retrieval_buffer_recomposed(
            seed_idx=0,
            anchor_idx=0,
            chunk_len=16,
            direction=1,
        )
        low_jump_path = [int(x) for x in list(engine._retrieval_buffer)]
        low_jump_repeats = count_repeat_hits(low_jump_path)
        low_jump_unique = len(set(int(i) for i in low_jump_path))

        engine._retrieval_buffer.clear()
        engine.set_random_controls(
            jump_rate=1.0,
            timbre_lock=0.0,
            repeat_avoid=1.0,
            crossfile=0.0,
        )
        engine._fill_retrieval_buffer_recomposed(
            seed_idx=0,
            anchor_idx=0,
            chunk_len=16,
            direction=1,
        )
        high_jump_path = [int(x) for x in list(engine._retrieval_buffer)]
        high_jump_repeats = count_repeat_hits(high_jump_path)
        high_jump_unique = len(set(int(i) for i in high_jump_path))
    finally:
        engine._lookahead_best_s1 = original_lookahead

    assert high_jump_unique > low_jump_unique, (
        "Expected higher jump_rate to visit more unique indices; "
        f"low_unique={low_jump_unique}, high_unique={high_jump_unique}, "
        f"low_repeats={low_jump_repeats}, high_repeats={high_jump_repeats}, "
        f"low_path={low_jump_path}, high_path={high_jump_path}"
    )


def run_all():
    test_recompose_path_validity()
    test_recompose_creates_non_serial_transition()
    test_navigation_uses_contiguous_fill_when_recompose_disabled()
    test_recompose_avoids_self_stalls()
    test_jump_rate_increases_path_novelty()
    print("All policy recomposition tests passed.")


if __name__ == "__main__":
    run_all()
