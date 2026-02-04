"""
64D latent navigation engine for corpus exploration.
Produces continuous latent trajectories for downstream decoding.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before importing faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import threading
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple
from scipy.spatial import cKDTree
import torch

from ..config import DEVICE
from ..policy.latent_geometry import LatentGeometry
from ..policy.latent_policy import LatentPolicy, LatentPolicyConfig, build_local_features


@dataclass
class NavFrame:
    """Navigation output for a single step."""
    z_nav: np.ndarray           # [64] current latent position after step
    indices: np.ndarray         # [K] kNN indices around z_nav
    distances: np.ndarray       # [K] cosine distances to kNN indices
    nearest_idx: int            # nearest neighbor index
    local_sigma: float          # local sigma at nearest neighbor
    time_gradient: np.ndarray   # [64] time gradient at nearest neighbor
    t_lat: float                # latent time index for nearest neighbor
    file_id: int                # file id for nearest neighbor


class LatentNavigationEngine:
    """
    64D latent space navigation engine for corpus exploration.

    Maintains continuous position z in 64D embedding space.
    Uses FAISS for fast kNN search and Gaussian kernel weights for interpolation.

    Control parameters (6 total):
        - width (0-1): Temperature scaling for exploration breadth
        - energy (0-1): Scaling factor for displacement magnitude
        - gravity (0-1): Bias toward forward (>0.5) or backward (<0.5) movement
        - memory (0-1): Pull toward recently visited positions
        - coherence (0-1): Bias toward staying within same source file
        - exploration (0-1): Direct entropy injection for diversity
    """

    # Dynamics constants
    VELOCITY_DECAY = 0.95      # α: velocity decay factor
    MANIFOLD_ATTRACTION = 0.1  # β: pull toward kNN centroid

    def __init__(
        self,
        GG: np.ndarray,
        meta: np.ndarray,
        geometry: LatentGeometry,
        policy_path: str = None,
        policy_temperature: float = 1.0,
        policy_sample: bool = True,
        control_width: float = 0.5,
        control_energy: float = 0.5,
        control_gravity: float = 0.5,
        control_memory: float = 0.0,
        control_coherence: float = 0.0,
        control_exploration: float = 0.0,
    ):
        """
        Initialize latent navigation engine.

        Args:
            GG: Corpus embeddings [N_seg, 64] (L2-normalized VAE latents)
            meta: Segment metadata [N_seg, 3] with (file_id, t_lat, win_lat)
            geometry: LatentGeometry with precomputed kNN and local features
            policy_path: Path to trained LatentPolicy checkpoint (optional)
            policy_temperature: Base temperature for policy sampling
            policy_sample: Whether to sample from policy (True) or use mode (False)
            control_*: Initial control parameters (0-1)
        """
        self.GG = GG.astype(np.float32)
        self.meta = meta
        self.geometry = geometry
        self.N = self.GG.shape[0]
        self.latent_dim = self.GG.shape[1]

        # L2-normalize GG for cosine similarity
        norms = np.linalg.norm(self.GG, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        self.GG_l2 = (self.GG / norms).astype(np.float32)

        # Build FAISS index
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.latent_dim)
            self.faiss_index.add(self.GG_l2)
            self._has_faiss = True
        except ImportError:
            print("[warn] FAISS not available, falling back to scipy cKDTree")
            self.faiss_index = None
            self._kdt = cKDTree(self.GG_l2)
            self._has_faiss = False

        # Navigation state
        self.z = self.GG[0].copy()  # Start at first segment
        self.v = np.zeros(self.latent_dim, dtype=np.float32)  # Velocity

        # Policy state
        self.policy: Optional[LatentPolicy] = None
        self.policy_cfg: Optional[LatentPolicyConfig] = None
        self.policy_hidden = None
        self.policy_device = torch.device(DEVICE)
        self.policy_ready = False
        self.policy_temperature = max(1e-3, float(policy_temperature))
        self.policy_sample = bool(policy_sample)

        # Control parameters (6 dimensions)
        self.ctrl_width = float(np.clip(control_width, 0.0, 1.0))
        self.ctrl_energy = float(np.clip(control_energy, 0.0, 1.0))
        self.ctrl_gravity = float(np.clip(control_gravity, 0.0, 1.0))
        self.ctrl_memory = float(np.clip(control_memory, 0.0, 1.0))
        self.ctrl_coherence = float(np.clip(control_coherence, 0.0, 1.0))
        self.ctrl_exploration = float(np.clip(control_exploration, 0.0, 1.0))

        # Tracking state
        self._current_file_id = int(self.geometry.file_ids[0])
        self._recent_z = deque(maxlen=128)
        self._recent_indices = deque(maxlen=128)

        # Thread safety
        self._lock = threading.Lock()

        # Load policy if specified
        if policy_path is not None:
            try:
                self._setup_policy(policy_path)
            except Exception as e:
                print(f"[warn] Failed to load policy '{policy_path}': {e}")

    def _setup_policy(self, policy_path: str):
        """Load and initialize policy model."""
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg = LatentPolicyConfig(**cfg_dict) if isinstance(cfg_dict, dict) else LatentPolicyConfig()
        self.policy_cfg = cfg

        self.policy = LatentPolicy(cfg).to(self.policy_device)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()

        self.policy_ready = True
        self._reset_policy_state()

    def _reset_policy_state(self):
        """Reset policy state to initial conditions."""
        self.policy_hidden = None
        self._recent_z.clear()
        self._recent_indices.clear()

    # --- Cursor control (for 2D visualization interaction) ---
    def set_cursor_nd(self, coords):
        """Set navigation cursor position from 2D visualization."""
        with self._lock:
            # Project 2D coords back to 64D using stored PCA
            if len(coords) >= 2:
                coords_2d = np.array(coords[:2], dtype=np.float32)
                # Inverse project: 2D -> 64D (approximate via pseudo-inverse)
                pca_pinv = np.linalg.pinv(self.geometry.pca_components_2d)  # [64, 2]
                z_centered = coords_2d @ pca_pinv.T
                self.z = (z_centered + self.geometry.pca_mean).astype(np.float32)
                # Reset velocity when cursor is set
                self.v = np.zeros(self.latent_dim, dtype=np.float32)

    # --- Policy controls ---
    def set_policy_controls(
        self,
        width=None,
        energy=None,
        gravity=None,
        memory=None,
        coherence=None,
        exploration=None,
    ):
        """
        Set policy control parameters.

        Args:
            width: Temperature scaling (0-1), higher = more random
            energy: Displacement magnitude (0-1), higher = bigger jumps
            gravity: Forward/backward bias (0-1), 0.5=neutral, >0.5=forward
            memory: Pull toward recent positions (0-1)
            coherence: Stay within same file (0-1)
            exploration: Direct entropy injection (0-1)
        """
        with self._lock:
            if width is not None:
                self.ctrl_width = float(np.clip(width, 0.0, 1.0))
            if energy is not None:
                self.ctrl_energy = float(np.clip(energy, 0.0, 1.0))
            if gravity is not None:
                self.ctrl_gravity = float(np.clip(gravity, 0.0, 1.0))
            if memory is not None:
                self.ctrl_memory = float(np.clip(memory, 0.0, 1.0))
            if coherence is not None:
                self.ctrl_coherence = float(np.clip(coherence, 0.0, 1.0))
            if exploration is not None:
                self.ctrl_exploration = float(np.clip(exploration, 0.0, 1.0))

    def reset_policy(self, idx=None):
        """Reset policy state, optionally starting at a specific index."""
        with self._lock:
            if idx is not None:
                idx = int(np.clip(idx, 0, self.N - 1))
                self.z = self.GG[idx].copy()
            self.v = np.zeros(self.latent_dim, dtype=np.float32)
            self._reset_policy_state()

    # --- Control vector for policy ---
    def _control_vector(self) -> np.ndarray:
        """Build 6-dimensional control vector for policy."""
        return np.array([
            self.ctrl_width,
            self.ctrl_energy,
            self.ctrl_gravity,
            self.ctrl_memory,
            self.ctrl_coherence,
            self.ctrl_exploration,
        ], dtype=np.float32)

    # --- kNN search ---
    def _query_knn(self, z: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k nearest neighbors for a latent position.

        Returns:
            (indices, distances) arrays of shape [k]
        """
        # L2 normalize query
        z_norm = z / (np.linalg.norm(z) + 1e-6)
        z_norm = z_norm.astype(np.float32)

        if self._has_faiss:
            z_query = z_norm.reshape(1, -1)
            distances, indices = self.faiss_index.search(z_query, k)
            # Convert inner product to cosine distance
            return indices[0], (1.0 - distances[0])
        else:
            dists, inds = self._kdt.query(z_norm, k=k)
            return inds.astype(np.int32), dists.astype(np.float32)

    def _compute_gaussian_weights(
        self,
        distances: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Compute Gaussian kernel weights from distances."""
        weights = np.exp(-distances**2 / (2 * sigma**2))
        return weights / (weights.sum() + 1e-8)

    # --- Navigation step ---
    def _navigation_step(self) -> NavFrame:
        """
        Execute one navigation step using policy or stochastic dynamics.

        Returns:
            NavFrame for the updated latent position
        """
        # Get kNN around current position
        k = self.geometry.K
        indices, distances = self._query_knn(self.z, k=k)

        # Get local geometry features
        nearest_idx = indices[0]
        local_sigma = float(self.geometry.local_sigma[nearest_idx])
        local_density = float(self.geometry.local_density[nearest_idx])
        time_gradient = self.geometry.time_gradients[nearest_idx]
        t_lat = float(self.geometry.t_lat[nearest_idx])
        file_id = int(self.geometry.file_ids[nearest_idx])

        # Compute kNN centroid
        knn_centroid = self.GG[indices].mean(axis=0)

        # Compute displacement delta_z
        if self.policy_ready and self.policy is not None:
            # Build local features
            local_features = build_local_features(
                self.z, local_sigma, local_density, time_gradient,
                t_lat, file_id, knn_centroid
            )

            # Build tensors
            z_tensor = torch.tensor(self.z, device=self.policy_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            v_tensor = torch.tensor(self.v, device=self.policy_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ctrl_tensor = torch.tensor(self._control_vector(), device=self.policy_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            local_tensor = torch.tensor(local_features, device=self.policy_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Run policy
            delta_mean, delta_log_std, delta_weights, vel_delta, self.policy_hidden = self.policy.step(
                z_tensor, v_tensor, ctrl_tensor, local_tensor, self.policy_hidden
            )

            # Sample or use mode
            temperature = self.policy_temperature * (1.0 + 2.0 * self.ctrl_width + 2.0 * self.ctrl_exploration)
            if self.policy_sample or self.ctrl_exploration > 0:
                delta_z = self.policy.sample_delta(delta_mean, delta_log_std, delta_weights, temperature)
            else:
                delta_z = self.policy.get_mixture_mode(delta_mean, delta_weights)

            delta_z = delta_z.squeeze(0).cpu().numpy()
            dv = vel_delta.squeeze(0).cpu().numpy()
        else:
            # Stochastic dynamics without learned policy
            # Direction toward time gradient with gravity bias
            gravity = (self.ctrl_gravity - 0.5) * 2.0
            direction = time_gradient * gravity

            # Add exploration noise
            noise = np.random.randn(self.latent_dim).astype(np.float32)
            noise *= local_sigma * (0.1 + self.ctrl_exploration)

            delta_z = direction * local_sigma * (0.5 + self.ctrl_energy) + noise
            dv = np.zeros(self.latent_dim, dtype=np.float32)

        # Apply energy scaling
        energy_scale = 0.5 + self.ctrl_energy
        delta_z *= energy_scale

        # Update velocity with decay
        self.v = self.VELOCITY_DECAY * self.v + dv

        # Update position
        z_new = self.z + self.v + delta_z

        # Apply manifold attraction (pull toward kNN centroid)
        z_new += self.MANIFOLD_ATTRACTION * (knn_centroid - z_new)

        # Apply memory control (pull toward recent positions)
        if self.ctrl_memory > 0 and len(self._recent_z) > 0:
            recent_mean = np.mean(list(self._recent_z), axis=0)
            z_new = (1.0 - self.ctrl_memory) * z_new + self.ctrl_memory * recent_mean

        # Apply coherence control (bias toward same file)
        if self.ctrl_coherence > 0:
            same_file_mask = self.geometry.file_ids[indices] == self._current_file_id
            if same_file_mask.sum() > 0:
                same_file_centroid = self.GG[indices[same_file_mask]].mean(axis=0)
                z_new = (1.0 - self.ctrl_coherence) * z_new + self.ctrl_coherence * same_file_centroid

        # Update state
        self.z = z_new.astype(np.float32)
        self._recent_z.append(self.z.copy())

        # Re-query kNN at new position for updated neighborhood
        indices, distances = self._query_knn(self.z, k=k)

        nearest_idx = int(indices[0])
        local_sigma = float(self.geometry.local_sigma[nearest_idx])
        time_gradient = self.geometry.time_gradients[nearest_idx]
        t_lat = float(self.geometry.t_lat[nearest_idx])
        file_id = int(self.geometry.file_ids[nearest_idx])

        # Track state
        self._recent_indices.append(nearest_idx)
        self._current_file_id = file_id

        return NavFrame(
            z_nav=self.z.copy(),
            indices=indices.astype(np.int32),
            distances=distances.astype(np.float32),
            nearest_idx=nearest_idx,
            local_sigma=local_sigma,
            time_gradient=time_gradient.copy(),
            t_lat=t_lat,
            file_id=file_id,
        )

    # --- Main API ---
    def step(self) -> NavFrame:
        """Advance navigation by one step and return NavFrame."""
        with self._lock:
            return self._navigation_step()

    def get_segment_info(self, segment_idx: int) -> dict:
        """Get metadata for a segment."""
        if segment_idx < 0 or segment_idx >= len(self.meta):
            return {"file_id": 0, "t_lat": 0, "win_lat": 0}
        return {
            "file_id": int(self.meta[segment_idx, 0]),
            "t_lat": int(self.meta[segment_idx, 1]),
            "win_lat": int(self.meta[segment_idx, 2]),
        }

    def get_fractional_state(self) -> dict:
        """Get fractional interpolation state for smooth transitions."""
        with self._lock:
            # In latent mode, we don't have discrete indices, so return current state
            indices, distances = self._query_knn(self.z, k=2)
            idx_lower = int(indices[0])
            idx_upper = int(indices[1]) if len(indices) > 1 else idx_lower

            # Get file IDs for the nearest neighbors
            file_ids = self.geometry.file_ids[indices]

            # Compute weights from distances
            sigma = float(self.geometry.local_sigma[idx_lower])
            weights = self._compute_gaussian_weights(distances, sigma)
            frac = float(weights[1]) / (float(weights[0]) + float(weights[1]) + 1e-8) if len(weights) > 1 else 0.0

            return {
                "idx_lower": idx_lower,
                "idx_upper": idx_upper,
                "frac": frac,
                "same_file": int(file_ids[0]) == int(file_ids[1]) if len(file_ids) > 1 else True,
                "file_id_lower": int(file_ids[0]),
                "file_id_upper": int(file_ids[1]) if len(file_ids) > 1 else int(file_ids[0]),
            }

    def get_state(self) -> dict:
        """Get current navigation state for visualization."""
        with self._lock:
            # Project z to 2D for visualization (raw, unnormalized)
            # Let ws_server handle normalization for consistency with corpus display
            pos_2d = self.geometry.project_to_2d(self.z)

            # Query kNN for nearest index and fractional state
            indices, distances = self._query_knn(self.z, k=2)
            nearest_idx = int(indices[0])

            # Get file IDs for the nearest neighbors
            file_ids = self.geometry.file_ids[indices]

            # Compute weights from distances for fractional interpolation
            sigma = float(self.geometry.local_sigma[nearest_idx])
            weights = self._compute_gaussian_weights(distances, sigma)

            # Compute fractional state inline (avoid deadlock from calling get_fractional_state)
            idx_lower = nearest_idx
            idx_upper = int(indices[1]) if len(indices) > 1 else idx_lower
            frac = float(weights[1]) / (float(weights[0]) + float(weights[1]) + 1e-8) if len(weights) > 1 else 0.0
            fractional_state = {
                "idx_lower": idx_lower,
                "idx_upper": idx_upper,
                "frac": frac,
                "same_file": int(file_ids[0]) == int(file_ids[1]) if len(file_ids) > 1 else True,
                "file_id_lower": int(file_ids[0]),
                "file_id_upper": int(file_ids[1]) if len(file_ids) > 1 else int(file_ids[0]),
            }

            # Project recent latent positions to 2D for smooth trajectory visualization
            recent_z_list = list(self._recent_z)
            if len(recent_z_list) > 0:
                recent_z_array = np.array(recent_z_list)
                recent_2d = self.geometry.project_to_2d(recent_z_array)
                trajectory_2d_raw = recent_2d.tolist() if len(recent_2d.shape) > 1 else [recent_2d.tolist()]
            else:
                trajectory_2d_raw = []

            return {
                "cursor": pos_2d.tolist(),  # Raw 2D projection
                "policy_index": float(nearest_idx),
                "policy_velocity": float(np.linalg.norm(self.v)),
                "current_file_id": self._current_file_id,
                "recent_indices": list(self._recent_indices),
                "controls": {
                    "width": self.ctrl_width,
                    "energy": self.ctrl_energy,
                    "gravity": self.ctrl_gravity,
                    "memory": self.ctrl_memory,
                    "coherence": self.ctrl_coherence,
                    "exploration": self.ctrl_exploration,
                },
                "fractional": fractional_state,
                "latent": {
                    "z_norm": float(np.linalg.norm(self.z)),
                    "v_norm": float(np.linalg.norm(self.v)),
                    "position_2d": pos_2d.tolist(),  # Raw 2D projection
                    "trajectory_2d": trajectory_2d_raw,  # Raw 2D trajectory for visualization
                },
            }

    @property
    def _file_ids(self) -> np.ndarray:
        return self.geometry.file_ids
