"""
Manifold-constrained latent generation utilities.
Anchors navigation outputs to the corpus manifold and projects perturbations.
"""
from dataclasses import dataclass
import numpy as np

from ..policy.latent_geometry import LatentGeometry
from .player import NavFrame


@dataclass
class ManifoldConfig:
    """Configuration for manifold-constrained generation."""
    k: int = 16
    n_local: int = 8
    n_global: int = 32
    sparse_quantile: float = 0.75


class ManifoldConstrainedGenerator:
    """
    Transforms navigation outputs into decodable latents constrained to the corpus manifold.
    """

    def __init__(
        self,
        corpus_latents: np.ndarray,
        geometry: LatentGeometry,
        config: ManifoldConfig = ManifoldConfig(),
    ):
        self.corpus = np.asarray(corpus_latents, dtype=np.float32)
        self.geometry = geometry
        self.config = config

        # Require full PCA for global projection
        if geometry.pca_components.shape[0] < geometry.latent_dim:
            raise ValueError(
                "Corpus PCA is not full-rank. Re-run preprocess.py to regenerate geometry."
            )

        # Sparse region threshold based on global local_sigma distribution
        self.sparse_threshold = float(
            np.quantile(self.geometry.local_sigma, self.config.sparse_quantile)
        )

    def _compute_anchor(self, frame: NavFrame):
        """Compute anchor point and neighbor set."""
        k = min(self.config.k, frame.indices.shape[0])
        indices = frame.indices[:k]
        distances = frame.distances[:k]
        neighbors = self.corpus[indices]  # [k, 64]

        sigma = max(frame.local_sigma, 1e-6)
        weights = np.exp(-distances / (2.0 * sigma * sigma))
        weights = weights / (weights.sum() + 1e-8)
        anchor = (neighbors * weights[:, None]).sum(axis=0)
        return anchor, neighbors

    def _project_local(self, delta: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Project delta onto local PCA basis from neighbors."""
        if neighbors.shape[0] < 2:
            raise ValueError("Not enough neighbors for local PCA.")

        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        # SVD for local PCA
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n_local = min(self.config.n_local, vt.shape[0])
        basis = vt[:n_local]  # [n_local, 64]
        coeffs = delta @ basis.T
        return coeffs @ basis

    def _project_global(self, delta: np.ndarray) -> np.ndarray:
        """Project delta onto global PCA basis."""
        n_global = min(self.config.n_global, self.geometry.pca_components.shape[0])
        basis = self.geometry.pca_components[:n_global]
        coeffs = delta @ basis.T
        return coeffs @ basis

    def _apply_magnitude_control(self, delta: np.ndarray, local_sigma: float, exploration: float) -> np.ndarray:
        """Clamp perturbation magnitude based on local density and exploration."""
        magnitude = float(np.linalg.norm(delta))
        exploration = float(np.clip(exploration, 0.0, 1.0))
        max_mult = 1.0 + exploration * 3.0
        max_magnitude = float(local_sigma) * max_mult
        if magnitude > max_magnitude and magnitude > 1e-8:
            delta = delta * (max_magnitude / magnitude)
        return delta

    def generate(self, frame: NavFrame, exploration: float = 0.5) -> np.ndarray:
        """
        Generate a manifold-constrained latent.

        Args:
            frame: NavFrame from the navigation engine
            exploration: [0,1] control mapping to perturbation magnitude
        """
        anchor, neighbors = self._compute_anchor(frame)
        delta = frame.z_nav - anchor

        use_global = frame.local_sigma >= self.sparse_threshold
        if not use_global:
            try:
                delta_valid = self._project_local(delta, neighbors)
            except Exception:
                delta_valid = self._project_global(delta)
        else:
            delta_valid = self._project_global(delta)

        delta_valid = self._apply_magnitude_control(delta_valid, frame.local_sigma, exploration)
        return anchor + delta_valid

    def generate_batch(self, frames: list, exploration: float = 0.5) -> np.ndarray:
        """
        Generate manifold-constrained latents for multiple frames.

        Args:
            frames: List of NavFrame objects from the navigation engine
            exploration: [0,1] control mapping to perturbation magnitude

        Returns:
            np.ndarray of shape [N, 64] containing latents for each frame
        """
        return np.stack([self.generate(f, exploration) for f in frames], axis=0)
