"""
Latent geometry/index utilities for causal context-aware navigation.

Primary index unit is (file_id, t) frame with embedding E_t = PCA(ctx_t), where
ctx_t is computed causally from observed latent history.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before importing faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from ..io.corpus_io import read_scalar as _read_scalar
from .sequence import group_meta_by_file


def compute_causal_ema_summaries(
    z_seq: np.ndarray,
    alpha_fast: float,
    alpha_slow: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute causal EMA summaries for one latent sequence [T, 64].

    Returns:
        m_fast: [T, 64]
        m_slow: [T, 64]
    """
    z_seq = np.asarray(z_seq, dtype=np.float32)
    if z_seq.ndim != 2:
        raise ValueError(f"Expected z_seq [T, D], got {z_seq.shape}")

    T, D = z_seq.shape
    m_fast = np.zeros((T, D), dtype=np.float32)
    m_slow = np.zeros((T, D), dtype=np.float32)

    if T == 0:
        return m_fast, m_slow

    fast_state = z_seq[0].copy()
    slow_state = z_seq[0].copy()

    for t in range(T):
        z_t = z_seq[t]
        if t == 0:
            fast_state = z_t.copy()
            slow_state = z_t.copy()
        else:
            fast_state = alpha_fast * fast_state + (1.0 - alpha_fast) * z_t
            slow_state = alpha_slow * slow_state + (1.0 - alpha_slow) * z_t

        m_fast[t] = fast_state
        m_slow[t] = slow_state

    return m_fast, m_slow


def build_context_features(
    latents: np.ndarray,
    meta: np.ndarray,
    alpha_fast: float,
    alpha_slow: float,
) -> np.ndarray:
    """
    Build causal context features for all frames.

    Context format: concat(z_t, m_fast_t, m_slow_t) -> 192 dims
    """
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N, D], got {latents.shape}")

    N, D = latents.shape
    sequences = group_meta_by_file(meta)

    m_fast = np.zeros((N, D), dtype=np.float32)
    m_slow = np.zeros((N, D), dtype=np.float32)

    for seq in sequences:
        if seq.size == 0:
            continue
        z_seq = latents[seq]
        seq_fast, seq_slow = compute_causal_ema_summaries(
            z_seq,
            alpha_fast=alpha_fast,
            alpha_slow=alpha_slow,
        )
        m_fast[seq] = seq_fast
        m_slow[seq] = seq_slow

    return np.concatenate([latents, m_fast, m_slow], axis=1).astype(np.float32)


@dataclass
class LatentGeometry:
    """
    Precomputed geometry and causal-context index over frame-level latents.

    All arrays are indexed by frame index i in [0, N-1], corresponding to
    mapping arrays (idx_to_file_id[i], idx_to_t[i]).
    """
    knn_indices: np.ndarray          # [N, K] neighbor frame indices in embedding space
    knn_distances: np.ndarray        # [N, K] cosine distances in embedding space
    local_sigma: np.ndarray          # [N] median kNN distance
    local_density: np.ndarray        # [N] 1 / local_sigma
    time_gradients: np.ndarray       # [N, 64] local forward-time direction in latent space
    file_ids: np.ndarray             # [N] per-frame file id
    t_lat: np.ndarray                # [N] per-frame latent time index
    centroid: np.ndarray             # [64] latent centroid
    pca_components: np.ndarray       # [64, 64] latent PCA basis (manifold/visualization)
    pca_mean: np.ndarray             # [64] latent PCA mean

    embeddings: np.ndarray           # [N, P] projected context embeddings E

    ctx_pca_components: np.ndarray   # [P, C] projection basis from context -> E
    ctx_pca_mean: np.ndarray         # [C]

    k_short: int
    ema_alpha_fast: float
    ema_alpha_slow: float

    @property
    def idx_to_file_id(self) -> np.ndarray:
        return self.file_ids

    @property
    def idx_to_t(self) -> np.ndarray:
        return self.t_lat

    @property
    def N(self) -> int:
        return self.knn_indices.shape[0]

    @property
    def K(self) -> int:
        return self.knn_indices.shape[1]

    @property
    def latent_dim(self) -> int:
        return self.time_gradients.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def context_dim(self) -> int:
        return self.ctx_pca_mean.shape[0]

    @property
    def pca_components_2d(self) -> np.ndarray:
        if self.pca_components.shape[0] < 2:
            raise ValueError("PCA components must have at least 2 rows for 2D projection.")
        return self.pca_components[:2]

    def project_to_2d(self, z: np.ndarray) -> np.ndarray:
        z_centered = z - self.pca_mean
        return z_centered @ self.pca_components_2d.T

    def build_context_vector(
        self,
        z_t: np.ndarray,
        m_fast_t: np.ndarray,
        m_slow_t: np.ndarray,
    ) -> np.ndarray:
        z_t = np.asarray(z_t, dtype=np.float32)
        m_fast_t = np.asarray(m_fast_t, dtype=np.float32)
        m_slow_t = np.asarray(m_slow_t, dtype=np.float32)
        return np.concatenate([z_t, m_fast_t, m_slow_t], axis=-1).astype(np.float32)

    def project_context(self, ctx: np.ndarray) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32)
        centered = ctx - self.ctx_pca_mean
        return centered @ self.ctx_pca_components.T

    def embed_query(
        self,
        z_t: np.ndarray,
        m_fast_t: np.ndarray,
        m_slow_t: np.ndarray,
    ) -> np.ndarray:
        ctx = self.build_context_vector(z_t, m_fast_t, m_slow_t)
        emb = self.project_context(ctx)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        return (emb / norm).astype(np.float32)


def compute_latent_geometry(
    latents: np.ndarray,
    meta: np.ndarray,
    k: int = 32,
    k_short: int = 8,
    ema_alpha_fast: float = 0.60,
    ema_alpha_slow: float = 0.95,
    pca_dim: int = 64,
) -> LatentGeometry:
    """
    Build geometry/index using frame-level latent trajectories and causal context.
    """
    disable_faiss = os.environ.get("STABLE_AUDIO_DISABLE_FAISS", "").lower() in ("1", "true", "yes")
    faiss = None
    if not disable_faiss:
        try:
            import faiss as _faiss
            _faiss.omp_set_num_threads(1)
            faiss = _faiss
        except Exception:
            faiss = None

    latents = np.ascontiguousarray(latents.astype(np.float32))
    meta = np.asarray(meta, dtype=np.int32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N, D], got {latents.shape}")
    if meta.ndim != 2 or meta.shape[1] < 2:
        raise ValueError(f"Expected meta [N, >=2], got {meta.shape}")

    N, D = latents.shape
    if meta.shape[0] != N:
        raise ValueError(f"Latents/meta length mismatch: {N} vs {meta.shape[0]}")

    # Causal context features and projection to retrieval embedding space.
    context = build_context_features(
        latents,
        meta,
        alpha_fast=float(ema_alpha_fast),
        alpha_slow=float(ema_alpha_slow),
    )

    pca_ctx_dim = int(max(2, min(int(pca_dim), context.shape[1], N)))
    ctx_mean = context.mean(axis=0).astype(np.float32)
    ctx_centered = context - ctx_mean[None, :]
    _, _, vt_ctx = np.linalg.svd(ctx_centered, full_matrices=False)
    ctx_components = vt_ctx[:pca_ctx_dim].astype(np.float32)
    embeddings = (ctx_centered @ ctx_components.T).astype(np.float32)

    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = np.maximum(emb_norm, 1e-6)
    embeddings_l2 = (embeddings / emb_norm).astype(np.float32)

    search_k = min(max(1, int(k)), max(1, N - 1))
    if faiss is not None:
        index = faiss.IndexFlatIP(pca_ctx_dim)
        index.add(embeddings_l2)
        distances, indices = index.search(embeddings_l2, search_k + 1)
    else:
        # cKDTree uses Euclidean distance over L2-normalized vectors.
        # For normalized vectors: ||a-b||^2 = 2 - 2*cos(a,b)  ->  cos = 1 - d^2/2.
        tree = cKDTree(embeddings_l2)
        dists, inds = tree.query(embeddings_l2, k=search_k + 1)
        dists = np.asarray(dists, dtype=np.float32)
        inds = np.asarray(inds, dtype=np.int32)
        if dists.ndim == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        distances = 1.0 - 0.5 * (dists ** 2)
        indices = inds

    knn_indices = indices[:, 1:].astype(np.int32)
    knn_distances = (1.0 - distances[:, 1:]).astype(np.float32)

    local_sigma = np.median(knn_distances, axis=1).astype(np.float32)
    local_sigma = np.maximum(local_sigma, 1e-6)
    local_density = (1.0 / local_sigma).astype(np.float32)

    file_ids = meta[:, 0].astype(np.int32)
    t_lat = meta[:, 1].astype(np.int32)

    # Time gradients are estimated in original latent space, but neighborhoods come from E.
    time_gradients = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        fid = file_ids[i]
        t_i = t_lat[i]

        neighbor_idx = knn_indices[i]
        same_file_mask = file_ids[neighbor_idx] == fid
        if same_file_mask.sum() < 2:
            continue

        same_file_neighbors = neighbor_idx[same_file_mask]
        t_neighbors = t_lat[same_file_neighbors]
        dt = t_neighbors - t_i
        future_mask = dt > 0
        if future_mask.sum() == 0:
            continue

        future_idx = same_file_neighbors[future_mask]
        future_dt = dt[future_mask]
        weights = 1.0 / (future_dt.astype(np.float32) + 1.0)
        weights /= weights.sum()

        direction = (latents[future_idx] - latents[i]) * weights[:, None]
        time_gradients[i] = direction.sum(axis=0)

        norm = np.linalg.norm(time_gradients[i])
        if norm > 1e-6:
            time_gradients[i] /= norm

    centroid = latents.mean(axis=0).astype(np.float32)

    # Full-rank latent PCA basis (D x D) for manifold projection and visualization.
    pca_mean = latents.mean(axis=0).astype(np.float32)
    centered = latents - pca_mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    D = centered.shape[1]
    if vt.shape[0] < D:
        # N < D: complete the basis with random orthonormal vectors via null space
        from scipy.linalg import null_space as _null_space
        ns = _null_space(vt).T
        vt = np.vstack([vt, ns[: D - vt.shape[0]]])
    pca_components = vt.astype(np.float32)

    return LatentGeometry(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        local_sigma=local_sigma,
        local_density=local_density,
        time_gradients=time_gradients,
        file_ids=file_ids,
        t_lat=t_lat,
        centroid=centroid,
        pca_components=pca_components,
        pca_mean=pca_mean,
        embeddings=embeddings,
        ctx_pca_components=ctx_components,
        ctx_pca_mean=ctx_mean,
        k_short=int(k_short),
        ema_alpha_fast=float(ema_alpha_fast),
        ema_alpha_slow=float(ema_alpha_slow),
    )


def save_geometry_to_dict(geometry: LatentGeometry) -> dict:
    return {
        "geom_knn_indices": geometry.knn_indices,
        "geom_knn_distances": geometry.knn_distances,
        "geom_local_sigma": geometry.local_sigma,
        "geom_local_density": geometry.local_density,
        "geom_time_gradients": geometry.time_gradients,
        "geom_file_ids": geometry.file_ids,
        "geom_t_lat": geometry.t_lat,
        "geom_centroid": geometry.centroid,
        "geom_pca_components": geometry.pca_components,
        "geom_pca_mean": geometry.pca_mean,
        "geom_embeddings": geometry.embeddings,
        "geom_idx_to_file_id": geometry.idx_to_file_id,
        "geom_idx_to_t": geometry.idx_to_t,
        "geom_ctx_pca_components": geometry.ctx_pca_components,
        "geom_ctx_pca_mean": geometry.ctx_pca_mean,
        "geom_k_short": np.array(int(geometry.k_short), dtype=np.int32),
        "geom_ema_alpha_fast": np.array(float(geometry.ema_alpha_fast), dtype=np.float32),
        "geom_ema_alpha_slow": np.array(float(geometry.ema_alpha_slow), dtype=np.float32),
    }


def load_geometry_from_dict(data: dict) -> Optional[LatentGeometry]:
    if "geom_knn_indices" not in data or "geom_embeddings" not in data:
        return None

    file_ids = data["geom_file_ids"] if "geom_file_ids" in data else data["geom_idx_to_file_id"]
    t_lat = data["geom_t_lat"] if "geom_t_lat" in data else data["geom_idx_to_t"]

    return LatentGeometry(
        knn_indices=data["geom_knn_indices"],
        knn_distances=data["geom_knn_distances"],
        local_sigma=data["geom_local_sigma"],
        local_density=data["geom_local_density"],
        time_gradients=data["geom_time_gradients"],
        file_ids=file_ids,
        t_lat=t_lat,
        centroid=data["geom_centroid"],
        pca_components=data["geom_pca_components"],
        pca_mean=data["geom_pca_mean"],
        embeddings=data["geom_embeddings"],
        ctx_pca_components=data["geom_ctx_pca_components"],
        ctx_pca_mean=data["geom_ctx_pca_mean"],
        k_short=int(_read_scalar(data, "geom_k_short", 8)),
        ema_alpha_fast=float(_read_scalar(data, "geom_ema_alpha_fast", 0.60)),
        ema_alpha_slow=float(_read_scalar(data, "geom_ema_alpha_slow", 0.95)),
    )
