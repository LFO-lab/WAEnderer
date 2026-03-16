#!/usr/bin/env python3
"""
Tests for corpus-locked timbre-aware policy seed selection.
"""

import numpy as np

from stable_audio_wanderer.policy.latent_geometry import LatentGeometry
from stable_audio_wanderer.runtime.player import LatentNavigationEngine


def _build_geometry(latents: np.ndarray, n_embed: int = 8) -> LatentGeometry:
    n, d = latents.shape
    rng = np.random.default_rng(7)

    # Build synthetic per-file timeline metadata.
    file_ids = np.zeros(n, dtype=np.int32)
    file_ids[n // 2 :] = 1
    t_lat = np.zeros(n, dtype=np.int32)
    t_lat[: n // 2] = np.arange(n // 2, dtype=np.int32)
    t_lat[n // 2 :] = np.arange(n - (n // 2), dtype=np.int32)

    # Synthetic context embedding space.
    embeddings = rng.normal(size=(n, n_embed)).astype(np.float32)
    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = np.maximum(emb_norm, 1e-6)
    embeddings_l2 = embeddings / emb_norm

    # KNN in embedding space.
    k = min(8, n - 1)
    sims = embeddings_l2 @ embeddings_l2.T
    cos_dist = 1.0 - sims
    np.fill_diagonal(cos_dist, np.inf)
    knn_indices = np.argsort(cos_dist, axis=1)[:, :k].astype(np.int32)
    knn_distances = np.take_along_axis(cos_dist, knn_indices, axis=1).astype(np.float32)

    # Full-rank latent PCA placeholders.
    pca_mean = latents.mean(axis=0).astype(np.float32)
    pca_components = np.eye(d, dtype=np.float32)

    # Context projection placeholders used by embed_query().
    context_dim = d * 3
    ctx_pca_components = rng.normal(size=(n_embed, context_dim)).astype(np.float32)
    ctx_pca_mean = np.zeros((context_dim,), dtype=np.float32)

    return LatentGeometry(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        local_sigma=np.maximum(knn_distances.mean(axis=1), 1e-3).astype(np.float32),
        local_density=(1.0 / np.maximum(knn_distances.mean(axis=1), 1e-3)).astype(np.float32),
        time_gradients=np.zeros((n, d), dtype=np.float32),
        file_ids=file_ids.copy(),
        t_lat=t_lat.copy(),
        centroid=latents.mean(axis=0).astype(np.float32),
        pca_components=pca_components,
        pca_mean=pca_mean,
        embeddings=embeddings.astype(np.float32),
        idx_to_file_id=file_ids.copy(),
        idx_to_t=t_lat.copy(),
        ctx_pca_components=ctx_pca_components,
        ctx_pca_mean=ctx_pca_mean,
        k_short=min(4, k),
        ema_alpha_fast=0.60,
        ema_alpha_mid=0.80,
        ema_alpha_slow=0.95,
        use_ema_mid=False,
    )


def _build_engine(n: int = 24, with_desc: bool = True) -> LatentNavigationEngine:
    rng = np.random.default_rng(11)
    latents = rng.normal(size=(n, 64)).astype(np.float32)
    geometry = _build_geometry(latents)

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
        control_jump_rate=0.35,
        control_timbre_lock=0.5,
        control_crossfile=0.6,
    )
    return engine


def test_control_plumbing():
    engine = _build_engine(with_desc=True)
    engine.set_policy_controls(jump_rate=1.4, timbre_lock=-0.4, crossfile=0.25)
    state = engine.get_state()
    controls = state["controls"]
    assert abs(controls["jump_rate"] - 1.0) < 1e-6
    assert abs(controls["timbre_lock"] - 0.0) < 1e-6
    assert abs(controls["crossfile"] - 0.25) < 1e-6


def test_swap_disable_parity():
    engine = _build_engine(with_desc=True)
    engine.set_policy_controls(jump_rate=0.0, timbre_lock=0.5, crossfile=0.6)
    current_idx = int(engine._current_index)
    q_emb = engine._query_embedding(engine.z)
    candidates = engine.geometry.knn_indices[current_idx][:12]
    baseline = int(candidates[0])
    selected = engine._choose_timbre_swap_seed(
        baseline_idx=baseline,
        candidate_indices=candidates,
        current_idx=current_idx,
        expected_t=int(engine._current_t),
        query_embedding=q_emb,
    )
    assert selected == baseline


def test_no_descriptor_fallback():
    engine = _build_engine(with_desc=False)
    engine.set_policy_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=0.6)
    current_idx = int(engine._current_index)
    q_emb = engine._query_embedding(engine.z)
    candidates = engine.geometry.knn_indices[current_idx][:12]
    baseline = int(candidates[0])
    selected = engine._choose_timbre_swap_seed(
        baseline_idx=baseline,
        candidate_indices=candidates,
        current_idx=current_idx,
        expected_t=int(engine._current_t),
        query_embedding=q_emb,
    )
    assert selected == baseline


def test_crossfile_penalty_behavior():
    engine = _build_engine(with_desc=True)

    # Build a deterministic tiny candidate set where only file penalty should flip preference.
    current_idx = int(engine._current_index)
    baseline = 1
    cross = engine.N // 2 + 1
    engine._set_current_index(0)
    current_idx = int(engine._current_index)

    # Align descriptors so timbre/velocity are equal-ish.
    seed_desc = np.zeros((engine.desc_weighted.shape[1],), dtype=np.float32)
    engine.desc_weighted[baseline] = seed_desc
    engine.desc_weighted[cross] = seed_desc
    engine.desc_weighted[current_idx] = seed_desc

    # Force embedding preference toward cross-file candidate.
    q_emb = np.zeros((engine.embeddings_l2.shape[1],), dtype=np.float32)
    q_emb[0] = 1.0
    baseline_emb = np.zeros_like(q_emb)
    baseline_emb[1 if q_emb.shape[0] > 1 else 0] = 1.0
    engine.embeddings_l2[baseline] = baseline_emb
    engine.embeddings_l2[cross] = q_emb.copy()

    # Neutralize lookahead so the first-order score dominates.
    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        candidates = np.array([baseline, cross], dtype=np.int32)
        expected_t = int(engine._idx_to_t[baseline])
        engine.set_policy_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=0.0)
        low_cross = engine._choose_timbre_swap_seed(
            baseline_idx=baseline,
            candidate_indices=candidates,
            current_idx=current_idx,
            expected_t=expected_t,
            query_embedding=q_emb,
        )
        engine.set_policy_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=1.0)
        high_cross = engine._choose_timbre_swap_seed(
            baseline_idx=baseline,
            candidate_indices=candidates,
            current_idx=current_idx,
            expected_t=expected_t,
            query_embedding=q_emb,
        )
    finally:
        engine._lookahead_best_s1 = original_lookahead

    assert low_cross == baseline
    assert high_cross == cross


def test_swap_activity_and_index_validity():
    engine = _build_engine(with_desc=True)
    engine.set_policy_controls(jump_rate=1.0, timbre_lock=0.5, crossfile=1.0)

    # Force deterministic first-order scoring dominance.
    original_lookahead = engine._lookahead_best_s1
    engine._lookahead_best_s1 = lambda candidate_idx, direction, q_gate: 0.0
    try:
        applied = 0
        total = 64
        for i in range(total):
            baseline = i % engine.N
            alt = (baseline + 1) % engine.N
            engine._set_current_index(baseline)

            # Keep timbre-related costs neutral between candidates.
            engine.desc_weighted[alt] = engine.desc_weighted[baseline]
            q_emb = engine.embeddings_l2[alt]
            candidates = np.array([baseline, alt], dtype=np.int32)

            chosen = engine._choose_timbre_swap_seed(
                baseline_idx=baseline,
                candidate_indices=candidates,
                current_idx=baseline,
                expected_t=int(engine._idx_to_t[baseline]),
                query_embedding=q_emb,
            )
            assert 0 <= chosen < engine.N
            applied += int(chosen != baseline)
    finally:
        engine._lookahead_best_s1 = original_lookahead

    assert applied > 0, "Expected non-zero swap activity with forced swap=1.0."


def run_all():
    test_control_plumbing()
    test_swap_disable_parity()
    test_no_descriptor_fallback()
    test_crossfile_penalty_behavior()
    test_swap_activity_and_index_validity()
    print("All policy timbre swap tests passed.")


if __name__ == "__main__":
    run_all()
