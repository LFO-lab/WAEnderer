"""
Manual navigation engine backed by PCA-reduced MFCC features and KD-tree lookup.
"""

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ManualFrame:
    """Manual navigation result for one query step."""

    nearest_index: int
    distance: float
    query_point: np.ndarray
    is_wandering: bool = False
    wander_progress: float = 0.0


class ManualNavigationEngine:
    """
    Enhanced manual navigation with timbral wandering.

    Faders define target timbre region, but engine can wander within similar timbres.
    When wandering is enabled (wander_k > 1), the engine smoothly transitions between
    nearby timbrally similar vectors instead of repeating the exact nearest neighbor.

    Wandering parameters:
    - wander_k: Number of nearest neighbors to consider (1 = disabled)
    - wander_speed: Transition speed between neighbors (0.0 = instant, 1.0 = slow)
    """

    def __init__(
        self,
        manual_points: np.ndarray,
        fader_p01: np.ndarray,
        fader_p99: np.ndarray,
        desc_weighted: Optional[np.ndarray] = None,
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
        coarse_k: int = 96,
        refine_k: int = 16,
        leafsize: int = 32,
        wander_k: int = 1,
        wander_speed: float = 0.0,
    ):
        points = np.asarray(manual_points, dtype=np.float32)
        p01 = np.asarray(fader_p01, dtype=np.float32).reshape(-1)
        p99 = np.asarray(fader_p99, dtype=np.float32).reshape(-1)
        if points.ndim != 2:
            raise ValueError(f"manual_points must be [N, D], got {points.shape}")
        if points.shape[1] != 8:
            raise ValueError(f"manual_points must have 8 dims, got {points.shape[1]}")
        if p01.shape[0] != 8 or p99.shape[0] != 8:
            raise ValueError("manual percentile ranges must have shape [8].")
        if points.shape[0] == 0:
            raise ValueError("manual_points is empty.")

        self.manual_points = points
        self.fader_p01 = p01
        self.fader_p99 = p99
        self.tree = cKDTree(self.manual_points, leafsize=max(1, int(leafsize)))
        self.search_mode = "legacy"
        self.coarse_k = int(max(1, min(self.N, coarse_k)))
        self.refine_k = int(max(1, min(self.N, refine_k)))
        self.desc_weighted = None
        self.pca_components = None
        self.pca_mean = None

        if (
            desc_weighted is not None
            and pca_components is not None
            and pca_mean is not None
        ):
            desc = np.asarray(desc_weighted, dtype=np.float32)
            components = np.asarray(pca_components, dtype=np.float32)
            mean = np.asarray(pca_mean, dtype=np.float32).reshape(-1)
            if desc.ndim != 2 or desc.shape[0] != points.shape[0]:
                raise ValueError(
                    "desc_weighted must be shape [N, D] aligned with manual_points."
                )
            if components.ndim != 2 or components.shape[0] != 8:
                raise ValueError("pca_components must be shape [8, D].")
            if mean.shape[0] != components.shape[1]:
                raise ValueError("pca_mean must be shape [D] matching pca_components.")
            if desc.shape[1] != components.shape[1]:
                raise ValueError(
                    "desc_weighted dim mismatch with pca_components second dimension."
                )
            self.desc_weighted = desc
            self.pca_components = components
            self.pca_mean = mean
            self.search_mode = "two_stage"

        # Wandering parameters
        self.wander_k = int(max(1, min(64, wander_k)))
        self.wander_speed = float(np.clip(wander_speed, 0.0, 1.0))

        # Wandering state
        self._faders = np.full(8, 0.5, dtype=np.float32)
        self._nearest_index = 0
        self._nearest_distance = float("inf")
        self._current_position = None  # Current position in 8D space
        self._source_position = None  # Interpolation origin for active wander segment
        self._target_position = None  # Target position in 8D space
        self._wander_progress = 0.0  # 0.0 to 1.0 interpolation
        self._current_target_idx = 0  # Current target index
        self._lock = threading.Lock()

    @property
    def N(self) -> int:
        return int(self.manual_points.shape[0])

    def set_faders(self, faders) -> None:
        arr = np.asarray(faders, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 8:
            raise ValueError(f"Expected 8 fader values, got {arr.shape[0]}")
        arr = np.clip(arr, 0.0, 1.0)
        with self._lock:
            self._faders = arr
            # Reset wandering state when faders change significantly
            self._current_position = None
            self._source_position = None
            self._target_position = None
            self._wander_progress = 0.0

    def set_wander_params(self, k: int = None, speed: float = None) -> None:
        """Set wandering parameters."""
        with self._lock:
            if k is not None:
                self.wander_k = int(max(1, min(64, k)))
            if speed is not None:
                self.wander_speed = float(np.clip(speed, 0.0, 1.0))

    def _faders_to_query(self, faders: np.ndarray) -> np.ndarray:
        span = np.maximum(self.fader_p99 - self.fader_p01, 1e-6)
        return self.fader_p01 + faders * span

    def _find_neighborhood(
        self, query_point: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k-nearest neighbors and their distances."""
        distances, indices = self.tree.query(query_point, k=k)
        # cKDTree returns scalars when k == 1; normalize to 1D arrays.
        indices = np.atleast_1d(indices).astype(np.int32)
        distances = np.atleast_1d(distances).astype(np.float32)
        return indices, distances

    def _query_to_descriptor(self, query_point: np.ndarray) -> np.ndarray:
        centered = query_point @ self.pca_components
        return (self.pca_mean + centered).astype(np.float32)

    def _two_stage_neighborhood(
        self, query_point: np.ndarray, k_needed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        coarse_size = int(min(self.N, max(k_needed, self.coarse_k)))
        coarse_indices, _ = self._find_neighborhood(query_point, coarse_size)
        coarse_indices = coarse_indices.reshape(-1)
        query_desc = self._query_to_descriptor(query_point)
        candidate_desc = self.desc_weighted[coarse_indices]
        distances = np.linalg.norm(candidate_desc - query_desc[None, :], axis=1).astype(
            np.float32
        )
        order = np.argsort(distances)
        refine_size = int(min(coarse_indices.shape[0], max(k_needed, self.refine_k)))
        keep = order[:refine_size]
        return coarse_indices[keep].astype(np.int32), distances[keep].astype(np.float32)

    def _select_wander_target(
        self,
        neighborhood: np.ndarray,
        distances: np.ndarray,
        current_target_idx: int = -1,
    ) -> int:
        """Select a new target from neighborhood based on wander parameters."""
        if self.wander_k <= 1 or neighborhood.shape[0] <= 1:
            return int(neighborhood[0])  # Stay at nearest

        candidate_mask = neighborhood != int(current_target_idx)
        if np.any(candidate_mask):
            candidates = neighborhood[candidate_mask]
            candidate_distances = distances[candidate_mask]
        else:
            candidates = neighborhood
            candidate_distances = distances

        # Create probability distribution favoring closer points
        max_dist = float(np.max(candidate_distances)) + 1e-8
        probs = np.exp(-candidate_distances / max_dist)
        probs = probs / (probs.sum() + 1e-8)

        choice = int(np.random.choice(len(candidates), p=probs))
        return int(candidates[choice])

    def _progress_step_size(self) -> float:
        """
        Convert [0,1] wander_speed into interpolation increment.
        0.0 means instant jumps; 1.0 means the slowest smooth transition.
        """
        if self.wander_speed <= 0.0:
            return 1.0
        return float(0.02 + (1.0 - self.wander_speed) * 0.28)

    def _step_from_faders(self, faders: np.ndarray, update_stored_faders: bool) -> ManualFrame:
        if update_stored_faders:
            with self._lock:
                self._faders = faders.copy()
        query = self._faders_to_query(faders)

        # Find neighborhood
        k = int(max(1, min(self.wander_k, self.N)))
        if self.search_mode == "two_stage":
            neighborhood, distances = self._two_stage_neighborhood(query, k_needed=k)
        else:
            neighborhood, distances = self._find_neighborhood(query, k)
        nearest_idx = int(neighborhood[0])

        # Update wandering state
        with self._lock:
            if self._current_position is None:
                # First call - initialize at nearest
                self._current_position = self.manual_points[nearest_idx].copy()
                self._source_position = self._current_position.copy()
                self._target_position = self._current_position.copy()
                self._wander_progress = 1.0
                self._current_target_idx = nearest_idx
            else:
                if self.wander_k <= 1:
                    # Wandering disabled: hard-lock to nearest query neighbor.
                    self._current_position = self.manual_points[nearest_idx].copy()
                    self._source_position = self._current_position.copy()
                    self._target_position = self._current_position.copy()
                    self._current_target_idx = nearest_idx
                    self._wander_progress = 1.0
                else:
                    # Keep the current target until the transition completes.
                    current_in_neighborhood = np.any(
                        neighborhood == int(self._current_target_idx)
                    )
                    need_new_target = (self._wander_progress >= 0.999) or (
                        not current_in_neighborhood
                    )
                    if need_new_target:
                        target_idx = self._select_wander_target(
                            neighborhood,
                            distances,
                            current_target_idx=self._current_target_idx,
                        )
                        self._current_target_idx = int(target_idx)
                        self._source_position = self._current_position.copy()
                        self._target_position = self.manual_points[target_idx].copy()
                        self._wander_progress = 0.0

                    step_size = self._progress_step_size()
                    self._wander_progress = min(1.0, self._wander_progress + step_size)
                    self._current_position = (
                        (1.0 - self._wander_progress) * self._source_position
                        + self._wander_progress * self._target_position
                    )

            current_position = self._current_position.copy()
            wander_progress = float(self._wander_progress)

        # Find actual nearest neighbor to current position for output
        final_k = 1
        if self.search_mode == "two_stage":
            final_indices, final_distances = self._two_stage_neighborhood(
                current_position, k_needed=final_k
            )
        else:
            final_indices, final_distances = self._find_neighborhood(
                current_position, final_k
            )
        final_idx = int(final_indices[0])
        final_dist = float(final_distances[0])

        with self._lock:
            self._nearest_index = final_idx
            self._nearest_distance = final_dist

        is_wandering = self.wander_k > 1 and wander_progress < 0.999

        return ManualFrame(
            nearest_index=final_idx,
            distance=final_dist,
            query_point=current_position if current_position is not None else query,
            is_wandering=is_wandering,
            wander_progress=wander_progress,
        )

    def step(self) -> ManualFrame:
        with self._lock:
            faders = self._faders.copy()
        return self._step_from_faders(faders, update_stored_faders=False)

    def step_with_faders(self, faders) -> ManualFrame:
        arr = np.asarray(faders, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 8:
            raise ValueError(f"Expected 8 fader values, got {arr.shape[0]}")
        arr = np.clip(arr, 0.0, 1.0)
        return self._step_from_faders(arr, update_stored_faders=True)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "faders": self._faders.tolist(),
                "nearest_index": int(self._nearest_index),
                "distance": float(self._nearest_distance),
                "wander_k": int(self.wander_k),
                "wander_speed": float(self.wander_speed),
                "search_mode": str(self.search_mode),
                "coarse_k": int(self.coarse_k),
                "refine_k": int(self.refine_k),
                "is_wandering": bool(
                    self.wander_k > 1 and self._wander_progress < 0.999
                ),
                "wander_progress": float(self._wander_progress),
            }

    def get_current_query(self) -> Tuple[np.ndarray, int, float]:
        with self._lock:
            faders = self._faders.copy()
            idx = int(self._nearest_index)
            dist = float(self._nearest_distance)
        query = self._faders_to_query(faders)
        return query.astype(np.float32), idx, dist
