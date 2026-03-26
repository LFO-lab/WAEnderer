"""Shared test fixtures and helpers."""

import numpy as np

from stable_audio_wanderer.policy.latent_geometry import LatentGeometry


def build_test_geometry(latents: np.ndarray, n_embed: int = 8, seed: int = 42) -> LatentGeometry:
    """Build a synthetic LatentGeometry for unit tests."""
    n, d = latents.shape
    rng = np.random.default_rng(seed)

    file_ids = np.zeros(n, dtype=np.int32)
    file_ids[n // 2 :] = 1
    t_lat = np.zeros(n, dtype=np.int32)
    t_lat[: n // 2] = np.arange(n // 2, dtype=np.int32)
    t_lat[n // 2 :] = np.arange(n - (n // 2), dtype=np.int32)

    embeddings = rng.normal(size=(n, n_embed)).astype(np.float32)
    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = np.maximum(emb_norm, 1e-6)
    embeddings_l2 = embeddings / emb_norm

    k = min(8, n - 1)
    sims = embeddings_l2 @ embeddings_l2.T
    cos_dist = 1.0 - sims
    np.fill_diagonal(cos_dist, np.inf)
    knn_indices = np.argsort(cos_dist, axis=1)[:, :k].astype(np.int32)
    knn_distances = np.take_along_axis(cos_dist, knn_indices, axis=1).astype(np.float32)

    pca_mean = latents.mean(axis=0).astype(np.float32)
    pca_components = np.eye(d, dtype=np.float32)

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
        ctx_pca_components=ctx_pca_components,
        ctx_pca_mean=ctx_pca_mean,
        k_short=min(4, k),
        ema_alpha_fast=0.60,
        ema_alpha_slow=0.95,
    )
