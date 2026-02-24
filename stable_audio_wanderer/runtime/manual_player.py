"""
Manual navigation engine backed by PCA-reduced MFCC features and KD-tree lookup.
"""
from dataclasses import dataclass
import threading
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ManualFrame:
    """Manual navigation result for one query step."""
    nearest_index: int
    distance: float
    query_point: np.ndarray


class ManualNavigationEngine:
    """
    Nearest-neighbor manual navigation over an 8D PCA control space.

    Faders are normalized [0,1]. Each dimension is mapped to the corpus control
    range via robust percentiles, then queried against a KD-tree.
    """

    def __init__(
        self,
        manual_points: np.ndarray,
        fader_p01: np.ndarray,
        fader_p99: np.ndarray,
        leafsize: int = 32,
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

        self._faders = np.full(8, 0.5, dtype=np.float32)
        self._nearest_index = 0
        self._nearest_distance = float("inf")
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

    def _faders_to_query(self, faders: np.ndarray) -> np.ndarray:
        span = np.maximum(self.fader_p99 - self.fader_p01, 1e-6)
        return self.fader_p01 + faders * span

    def step(self) -> ManualFrame:
        with self._lock:
            faders = self._faders.copy()
        query = self._faders_to_query(faders)
        dist, idx = self.tree.query(query.astype(np.float32), k=1)
        idx_i = int(idx)
        dist_f = float(dist)
        with self._lock:
            self._nearest_index = idx_i
            self._nearest_distance = dist_f
        return ManualFrame(
            nearest_index=idx_i,
            distance=dist_f,
            query_point=query.astype(np.float32),
        )

    def get_state(self) -> dict:
        with self._lock:
            return {
                "faders": self._faders.tolist(),
                "nearest_index": int(self._nearest_index),
                "distance": float(self._nearest_distance),
            }

    def get_current_query(self) -> Tuple[np.ndarray, int, float]:
        with self._lock:
            faders = self._faders.copy()
            idx = int(self._nearest_index)
            dist = float(self._nearest_distance)
        query = self._faders_to_query(faders)
        return query.astype(np.float32), idx, dist
