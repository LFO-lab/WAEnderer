"""
Latent space geometry computations for 64D VAE latent navigation.
Provides precomputed kNN structures and local geometry features.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before importing faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LatentGeometry:
    """
    Precomputed geometric structure of the latent space for efficient navigation.

    All arrays are indexed by segment index [0, N-1].

    Attributes:
        knn_indices: [N, K] indices of K nearest neighbors per point
        knn_distances: [N, K] cosine distances to K nearest neighbors
        local_sigma: [N] local scale (median kNN distance) per point
        local_density: [N] local density estimate (1/sigma) per point
        time_gradients: [N, 64] estimated time direction in latent space per file
        file_ids: [N] source file ID for each segment
        t_lat: [N] latent time position (frame index within file)
        centroid: [64] global centroid of the latent space
        pca_components: [D, 64] PCA projection matrix (full rank)
        pca_mean: [64] mean used for PCA centering
    """
    knn_indices: np.ndarray       # [N, K] neighbor indices
    knn_distances: np.ndarray     # [N, K] cosine distances
    local_sigma: np.ndarray       # [N] median kNN distance
    local_density: np.ndarray     # [N] density estimate
    time_gradients: np.ndarray    # [N, 64] per-point time gradients
    file_ids: np.ndarray          # [N] source file ID
    t_lat: np.ndarray             # [N] latent time position
    centroid: np.ndarray          # [64] global centroid
    pca_components: np.ndarray    # [D, 64] PCA projection matrix (full-rank)
    pca_mean: np.ndarray          # [64] PCA mean for centering

    @property
    def N(self) -> int:
        """Number of segments in the corpus."""
        return self.knn_indices.shape[0]

    @property
    def K(self) -> int:
        """Number of nearest neighbors stored."""
        return self.knn_indices.shape[1]

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.time_gradients.shape[1]

    def get_knn_centroid(self, indices: np.ndarray, GG: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of the K nearest neighbors for given indices.

        Args:
            indices: [M] segment indices to compute centroids for
            GG: [N, D] full latent embedding matrix

        Returns:
            [M, D] centroids of the kNN neighborhoods
        """
        # Get all neighbor indices for the query points
        neighbor_idx = self.knn_indices[indices]  # [M, K]
        # Gather neighbor embeddings and average
        neighbor_embeddings = GG[neighbor_idx]  # [M, K, D]
        return neighbor_embeddings.mean(axis=1)  # [M, D]

    def project_to_2d(self, z: np.ndarray) -> np.ndarray:
        """
        Project 64D latent vector(s) to 2D for visualization.

        Args:
            z: [..., 64] latent vector(s)

        Returns:
            [..., 2] projected coordinates
        """
        # Center and project
        z_centered = z - self.pca_mean
        return z_centered @ self.pca_components_2d.T

    @property
    def pca_components_2d(self) -> np.ndarray:
        """Return the first two PCA components for 2D projection."""
        if self.pca_components.shape[0] < 2:
            raise ValueError("PCA components must have at least 2 rows for 2D projection.")
        return self.pca_components[:2]


def compute_latent_geometry(
    GG: np.ndarray,
    meta: np.ndarray,
    k: int = 32,
) -> LatentGeometry:
    """
    Compute latent space geometry from normalized embeddings.

    Args:
        GG: [N, 64] L2-normalized latent embeddings (z-scored then L2-normed)
        meta: [N, 3] segment metadata (file_id, t_lat, win_lat)
        k: Number of nearest neighbors to compute

    Returns:
        LatentGeometry with all precomputed structures
    """
    try:
        import faiss
        # Use single thread to avoid OpenMP issues
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for latent navigation. "
            "Install with: pip install faiss-cpu>=1.7.4"
        )

    N, D = GG.shape

    # Ensure GG is contiguous float32 for FAISS
    GG = np.ascontiguousarray(GG.astype(np.float32))

    # Build FAISS index for cosine similarity (inner product after L2 norm)
    index = faiss.IndexFlatIP(D)
    index.add(GG)

    # Query k+1 neighbors (first is self)
    distances, indices = index.search(GG, k + 1)

    # Remove self from results (first column)
    knn_indices = indices[:, 1:].astype(np.int32)
    # Convert inner product to distance: dist = 1 - similarity
    knn_distances = (1.0 - distances[:, 1:]).astype(np.float32)

    # Compute local sigma (median kNN distance)
    local_sigma = np.median(knn_distances, axis=1).astype(np.float32)
    local_sigma = np.maximum(local_sigma, 1e-6)  # Avoid division by zero

    # Compute density (inverse of sigma)
    local_density = (1.0 / local_sigma).astype(np.float32)

    # Extract file IDs and time positions from meta
    file_ids = meta[:, 0].astype(np.int32)
    t_lat = meta[:, 1].astype(np.int32)

    # Compute time gradients per point
    # For each point, estimate the local time direction from neighbors in same file
    time_gradients = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        fid = file_ids[i]
        t_i = t_lat[i]

        # Find neighbors in the same file
        neighbor_idx = knn_indices[i]
        same_file_mask = file_ids[neighbor_idx] == fid

        if same_file_mask.sum() >= 2:
            same_file_neighbors = neighbor_idx[same_file_mask]
            t_neighbors = t_lat[same_file_neighbors]

            # Compute weighted direction toward future
            dt = t_neighbors - t_i
            future_mask = dt > 0

            if future_mask.sum() > 0:
                future_idx = same_file_neighbors[future_mask]
                future_dt = dt[future_mask]
                # Weight by inverse time distance
                weights = 1.0 / (future_dt.astype(np.float32) + 1.0)
                weights /= weights.sum()

                # Compute weighted direction
                direction = (GG[future_idx] - GG[i]) * weights[:, None]
                time_gradients[i] = direction.sum(axis=0)

                # Normalize to unit length
                norm = np.linalg.norm(time_gradients[i])
                if norm > 1e-6:
                    time_gradients[i] /= norm

    # Compute global centroid
    centroid = GG.mean(axis=0).astype(np.float32)

    # Compute full PCA (for manifold projection + visualization)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=D)
    pca.fit(GG)
    pca_components = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)

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
    )


def save_geometry_to_dict(geometry: LatentGeometry) -> dict:
    """
    Convert LatentGeometry to a dict for saving to npz.

    Args:
        geometry: LatentGeometry instance

    Returns:
        Dict with all arrays prefixed with 'geom_'
    """
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
    }


def load_geometry_from_dict(data: dict) -> Optional[LatentGeometry]:
    """
    Load LatentGeometry from a corpus npz dict.

    Args:
        data: Dict loaded from npz file

    Returns:
        LatentGeometry if geometry arrays are present, None otherwise
    """
    if "geom_knn_indices" not in data:
        return None

    return LatentGeometry(
        knn_indices=data["geom_knn_indices"],
        knn_distances=data["geom_knn_distances"],
        local_sigma=data["geom_local_sigma"],
        local_density=data["geom_local_density"],
        time_gradients=data["geom_time_gradients"],
        file_ids=data["geom_file_ids"],
        t_lat=data["geom_t_lat"],
        centroid=data["geom_centroid"],
        pca_components=data["geom_pca_components"],
        pca_mean=data["geom_pca_mean"],
    )
