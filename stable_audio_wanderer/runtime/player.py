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

from ..config import DEVICE, LAMBDA_DT, LAMBDA_FILE_SWITCH
from ..policy.latent_geometry import LatentGeometry
from ..policy.latent_policy import LatentPolicy, LatentPolicyConfig, build_local_features
from ..policy.sequence import compute_velocity_magnitudes


@dataclass
class NavFrame:
    """Navigation output for a single step."""
    z_nav: np.ndarray           # [64] selected decodable latent frame
    indices: np.ndarray         # [K] neighbor frame indices
    distances: np.ndarray       # [K] embedding-space distances for neighbors
    nearest_idx: int            # selected frame index
    local_sigma: float          # local sigma at selected frame
    time_gradient: np.ndarray   # [64] time gradient at selected frame
    t_lat: float                # latent time index within file
    file_id: int                # source file id
    predicted_window_size: int = 4  # Predicted window size in latent frames (2-64)


class LatentNavigationEngine:
    """
    Frame-level latent navigation engine.

    Retrieval index is built on causal context embeddings E(file_id, t).
    Controls and policy still act in latent space, but emitted frames are always
    observed corpus frames (no mean-pooled synthetic corpus points).
    """

    VELOCITY_DECAY = 0.95
    MANIFOLD_ATTRACTION = 0.1

    def __init__(
        self,
        GG: np.ndarray,
        meta: np.ndarray,
        geometry: LatentGeometry,
        file_offsets: Optional[np.ndarray] = None,
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
        self.GG = np.asarray(GG, dtype=np.float32)
        self.meta = np.asarray(meta, dtype=np.int32) if meta is not None else None
        self.geometry = geometry
        self.N = self.GG.shape[0]
        self.latent_dim = self.GG.shape[1]

        # Frame mapping and offsets for contiguous chunk retrieval.
        self._idx_to_file_id = np.asarray(self.geometry.idx_to_file_id, dtype=np.int32)
        self._idx_to_t = np.asarray(self.geometry.idx_to_t, dtype=np.int32)
        self.file_offsets = self._prepare_file_offsets(file_offsets)

        # Build ANN index on context embeddings E.
        self.embeddings = np.asarray(self.geometry.embeddings, dtype=np.float32)
        emb_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        emb_norm = np.maximum(emb_norm, 1e-6)
        self.embeddings_l2 = (self.embeddings / emb_norm).astype(np.float32)

        disable_faiss = os.environ.get("STABLE_AUDIO_DISABLE_FAISS", "").lower() in ("1", "true", "yes")
        if not disable_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.embeddings_l2.shape[1])
                self.faiss_index.add(self.embeddings_l2)
                self._has_faiss = True
            except Exception:
                self.faiss_index = None
                self._kdt = cKDTree(self.embeddings_l2)
                self._has_faiss = False
        else:
            self.faiss_index = None
            self._kdt = cKDTree(self.embeddings_l2)
            self._has_faiss = False

        if not self._has_faiss:
            print("[warn] FAISS disabled/unavailable, using scipy cKDTree for retrieval")

        # Navigation state.
        self._current_index = 0
        self.z = self.GG[self._current_index].copy()
        self.v = np.zeros(self.latent_dim, dtype=np.float32)

        self._ema_fast = self.z.copy()
        self._ema_slow = self.z.copy()
        self._ema_mid = self.z.copy() if self.geometry.use_ema_mid else None

        self._retrieval_buffer = deque()
        self._last_predicted_window = 4

        # Policy state.
        self.policy: Optional[LatentPolicy] = None
        self.policy_cfg: Optional[LatentPolicyConfig] = None
        self.policy_hidden = None
        self.policy_device = torch.device(DEVICE)
        self.policy_ready = False
        self.policy_temperature = max(1e-3, float(policy_temperature))
        self.policy_sample = bool(policy_sample)

        # Control parameters.
        self.ctrl_width = float(np.clip(control_width, 0.0, 1.0))
        self.ctrl_energy = float(np.clip(control_energy, 0.0, 1.0))
        self.ctrl_gravity = float(np.clip(control_gravity, 0.0, 1.0))
        self.ctrl_memory = float(np.clip(control_memory, 0.0, 1.0))
        self.ctrl_coherence = float(np.clip(control_coherence, 0.0, 1.0))
        self.ctrl_exploration = float(np.clip(control_exploration, 0.0, 1.0))

        # Tracking.
        self._current_file_id = int(self._idx_to_file_id[self._current_index])
        self._current_t = int(self._idx_to_t[self._current_index])
        self._recent_z = deque(maxlen=128)
        self._recent_indices = deque(maxlen=128)

        self._lock = threading.Lock()

        if policy_path is not None:
            try:
                self._setup_policy(policy_path)
            except Exception as e:
                print(f"[warn] Failed to load policy '{policy_path}': {e}")

    def _prepare_file_offsets(self, file_offsets: Optional[np.ndarray]) -> np.ndarray:
        if file_offsets is not None:
            offsets = np.asarray(file_offsets, dtype=np.int64)
            if offsets.ndim == 1 and offsets.size >= 2:
                return offsets

        # Fallback for legacy callers: infer offsets from idx_to_file_id ordering.
        file_ids = self._idx_to_file_id
        max_fid = int(file_ids.max()) if file_ids.size else -1
        offsets = [0]
        cursor = 0
        for fid in range(max_fid + 1):
            count = int((file_ids == fid).sum())
            cursor += count
            offsets.append(cursor)
        return np.asarray(offsets, dtype=np.int64)

    def _setup_policy(self, policy_path: str):
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
        self.policy_hidden = None
        self._retrieval_buffer.clear()
        self._recent_z.clear()
        self._recent_indices.clear()

    def set_cursor_nd(self, coords):
        """Set navigation cursor from 2D visualization coordinates."""
        with self._lock:
            if len(coords) < 2:
                return
            coords_2d = np.array(coords[:2], dtype=np.float32)
            pca_pinv = np.linalg.pinv(self.geometry.pca_components_2d)
            z_centered = coords_2d @ pca_pinv.T
            self.z = (z_centered + self.geometry.pca_mean).astype(np.float32)
            self.v = np.zeros(self.latent_dim, dtype=np.float32)

            self._ema_fast = self.z.copy()
            self._ema_slow = self.z.copy()
            if self.geometry.use_ema_mid:
                self._ema_mid = self.z.copy()

            # Snap cursor to nearest indexed frame in causal embedding space.
            indices, _ = self._query_knn(self.z, k=1)
            if indices.size > 0:
                self._set_current_index(int(indices[0]))
            self._retrieval_buffer.clear()

    def set_policy_controls(
        self,
        width=None,
        energy=None,
        gravity=None,
        memory=None,
        coherence=None,
        exploration=None,
    ):
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
        with self._lock:
            if idx is not None:
                idx = int(np.clip(idx, 0, self.N - 1))
                self._set_current_index(idx)
            self.v = np.zeros(self.latent_dim, dtype=np.float32)
            self._ema_fast = self.z.copy()
            self._ema_slow = self.z.copy()
            if self.geometry.use_ema_mid:
                self._ema_mid = self.z.copy()
            self._reset_policy_state()

    def _set_current_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.N - 1))
        self._current_index = idx
        self.z = self.GG[idx].copy().astype(np.float32)
        self._current_file_id = int(self._idx_to_file_id[idx])
        self._current_t = int(self._idx_to_t[idx])

    def _control_vector(self) -> np.ndarray:
        return np.array([
            self.ctrl_width,
            self.ctrl_energy,
            self.ctrl_gravity,
            self.ctrl_memory,
            self.ctrl_coherence,
            self.ctrl_exploration,
        ], dtype=np.float32)

    def _context_summaries(self) -> np.ndarray:
        if self.geometry.use_ema_mid:
            return np.concatenate([self._ema_fast, self._ema_mid, self._ema_slow], axis=0).astype(np.float32)
        return np.concatenate([self._ema_fast, self._ema_slow], axis=0).astype(np.float32)

    def _query_knn(self, z_query: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query nearest neighbors in context embedding space with a causal query context.
        """
        k = int(max(1, min(k, self.N)))
        q = self.geometry.embed_query(
            z_query,
            self._ema_fast,
            self._ema_slow,
            m_mid_t=self._ema_mid if self.geometry.use_ema_mid else None,
        ).astype(np.float32)

        if self._has_faiss:
            distances, indices = self.faiss_index.search(q.reshape(1, -1), k)
            return indices[0].astype(np.int32), (1.0 - distances[0]).astype(np.float32)

        dists, inds = self._kdt.query(q, k=k)
        inds = np.atleast_1d(inds).astype(np.int32)
        dists = np.atleast_1d(dists).astype(np.float32)
        return inds, dists

    def _continuity_rerank(self, indices: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if indices.size == 0:
            return indices, distances

        direction = 0
        if self.ctrl_gravity > 0.55:
            direction = 1
        elif self.ctrl_gravity < 0.45:
            direction = -1
        expected_t = self._current_t + direction

        lambda_file = float(LAMBDA_FILE_SWITCH) * (1.0 + 2.0 * self.ctrl_coherence)
        lambda_dt = float(LAMBDA_DT) * (1.0 + 2.0 * self.ctrl_memory)

        scores = np.zeros(indices.shape[0], dtype=np.float32)
        for i, idx in enumerate(indices):
            fid = int(self._idx_to_file_id[idx])
            t_lat = int(self._idx_to_t[idx])
            score = float(distances[i])
            if fid != self._current_file_id:
                score += lambda_file
            score += lambda_dt * abs(t_lat - expected_t)
            scores[i] = score

        order = np.argsort(scores)
        return indices[order], distances[order]

    def _update_ema(self, z_t: np.ndarray):
        z_t = np.asarray(z_t, dtype=np.float32)
        self._ema_fast = (
            self.geometry.ema_alpha_fast * self._ema_fast +
            (1.0 - self.geometry.ema_alpha_fast) * z_t
        ).astype(np.float32)
        self._ema_slow = (
            self.geometry.ema_alpha_slow * self._ema_slow +
            (1.0 - self.geometry.ema_alpha_slow) * z_t
        ).astype(np.float32)
        if self.geometry.use_ema_mid:
            self._ema_mid = (
                self.geometry.ema_alpha_mid * self._ema_mid +
                (1.0 - self.geometry.ema_alpha_mid) * z_t
            ).astype(np.float32)

    def _fill_retrieval_buffer(self, seed_idx: int, chunk_len: int):
        seed_idx = int(np.clip(seed_idx, 0, self.N - 1))
        chunk_len = int(max(1, chunk_len))

        file_id = int(self._idx_to_file_id[seed_idx])
        t_start = int(self._idx_to_t[seed_idx])

        file_start = int(self.file_offsets[file_id])
        file_end = int(self.file_offsets[file_id + 1])
        file_len = max(0, file_end - file_start)

        t_end = min(file_len, t_start + chunk_len)
        for t in range(t_start, t_end):
            self._retrieval_buffer.append(file_start + t)

        if not self._retrieval_buffer:
            self._retrieval_buffer.append(seed_idx)

    def _compute_gaussian_weights(self, distances: np.ndarray, sigma: float) -> np.ndarray:
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        return weights / (weights.sum() + 1e-8)

    def _emit_observed_frame(self, frame_idx: int, predicted_window: int) -> NavFrame:
        frame_idx = int(np.clip(frame_idx, 0, self.N - 1))

        z_prev = self.z.copy()
        z_obs = self.GG[frame_idx].astype(np.float32)

        jump = z_obs - z_prev
        self.v = self.VELOCITY_DECAY * self.v + (1.0 - self.VELOCITY_DECAY) * jump
        self.z = z_obs
        self._update_ema(z_obs)

        self._set_current_index(frame_idx)
        self._recent_z.append(self.z.copy())
        self._recent_indices.append(frame_idx)

        indices = self.geometry.knn_indices[frame_idx].astype(np.int32)
        distances = self.geometry.knn_distances[frame_idx].astype(np.float32)

        local_sigma = float(self.geometry.local_sigma[frame_idx])
        time_gradient = self.geometry.time_gradients[frame_idx]
        t_lat = float(self._idx_to_t[frame_idx])
        file_id = int(self._idx_to_file_id[frame_idx])

        return NavFrame(
            z_nav=self.z.copy(),
            indices=indices,
            distances=distances,
            nearest_idx=frame_idx,
            local_sigma=local_sigma,
            time_gradient=time_gradient.copy(),
            t_lat=t_lat,
            file_id=file_id,
            predicted_window_size=int(predicted_window),
        )

    def _navigation_step(self) -> NavFrame:
        """
        Execute one navigation step.

        1) Propose motion in latent space (policy or stochastic fallback).
        2) Query causal context index E and continuity-rerank candidates.
        3) Retrieve a real corpus chunk starting at (file_id, t).
        4) Emit one observed frame from that chunk.
        """
        if self._retrieval_buffer:
            idx = int(self._retrieval_buffer.popleft())
            return self._emit_observed_frame(idx, self._last_predicted_window)

        current_idx = int(self._current_index)
        local_sigma = float(self.geometry.local_sigma[current_idx])
        local_density = float(self.geometry.local_density[current_idx])
        time_gradient = self.geometry.time_gradients[current_idx]
        t_lat = float(self._idx_to_t[current_idx])
        file_id = int(self._idx_to_file_id[current_idx])

        neighbors = self.geometry.knn_indices[current_idx]
        if neighbors.size > 0:
            knn_centroid = self.GG[neighbors].mean(axis=0)
        else:
            knn_centroid = self.z.copy()

        if self.policy_ready and self.policy is not None:
            local_features = build_local_features(
                self.z,
                local_sigma,
                local_density,
                time_gradient,
                t_lat,
                file_id,
                knn_centroid,
            )

            z_tensor = torch.tensor(self.z, device=self.policy_device, dtype=torch.float32).view(1, 1, -1)
            v_tensor = torch.tensor(self.v, device=self.policy_device, dtype=torch.float32).view(1, 1, -1)
            ctrl_tensor = torch.tensor(self._control_vector(), device=self.policy_device, dtype=torch.float32).view(1, 1, -1)
            local_tensor = torch.tensor(local_features, device=self.policy_device, dtype=torch.float32).view(1, 1, -1)
            ctx_tensor = torch.tensor(self._context_summaries(), device=self.policy_device, dtype=torch.float32).view(1, 1, -1)

            delta_mean, delta_log_std, delta_weights, vel_delta, window_log2, self.policy_hidden = self.policy.step(
                z_tensor,
                v_tensor,
                ctrl_tensor,
                local_tensor,
                context_summaries=ctx_tensor,
                hidden=self.policy_hidden,
            )

            temperature = self.policy_temperature * (1.0 + 2.0 * self.ctrl_width + 2.0 * self.ctrl_exploration)
            if self.policy_sample or self.ctrl_exploration > 0:
                delta_z = self.policy.sample_delta(delta_mean, delta_log_std, delta_weights, temperature)
            else:
                delta_z = self.policy.get_mixture_mode(delta_mean, delta_weights)

            delta_z = delta_z.squeeze(0).cpu().numpy()
            dv = vel_delta.squeeze(0).cpu().numpy()
            predicted_window = self.policy.get_window_size(window_log2.squeeze(0))
        else:
            gravity = (self.ctrl_gravity - 0.5) * 2.0
            direction = time_gradient * gravity
            noise = np.random.randn(self.latent_dim).astype(np.float32)
            noise *= local_sigma * (0.1 + self.ctrl_exploration)
            delta_z = direction * local_sigma * (0.5 + self.ctrl_energy) + noise
            dv = np.zeros(self.latent_dim, dtype=np.float32)

            if not hasattr(self, "_velocity_magnitudes"):
                self._velocity_magnitudes = compute_velocity_magnitudes(self.GG, self.meta)
            local_velocity = float(self._velocity_magnitudes[current_idx])
            predicted_window = self._heuristic_window_size(local_velocity)

        energy_scale = 0.5 + self.ctrl_energy
        delta_z *= energy_scale

        self.v = self.VELOCITY_DECAY * self.v + dv

        z_new = self.z + self.v + delta_z
        z_new += self.MANIFOLD_ATTRACTION * (knn_centroid - z_new)

        if self.ctrl_memory > 0 and len(self._recent_z) > 0:
            recent_mean = np.mean(np.asarray(self._recent_z), axis=0)
            z_new = (1.0 - self.ctrl_memory) * z_new + self.ctrl_memory * recent_mean

        if self.ctrl_coherence > 0 and neighbors.size > 0:
            same_file_mask = self._idx_to_file_id[neighbors] == self._current_file_id
            if same_file_mask.sum() > 0:
                same_file_centroid = self.GG[neighbors[same_file_mask]].mean(axis=0)
                z_new = (1.0 - self.ctrl_coherence) * z_new + self.ctrl_coherence * same_file_centroid

        self.z = z_new.astype(np.float32)

        cand_indices, cand_distances = self._query_knn(self.z, k=self.geometry.K)
        cand_indices, cand_distances = self._continuity_rerank(cand_indices, cand_distances)

        seed_idx = int(cand_indices[0]) if cand_indices.size > 0 else current_idx
        self._last_predicted_window = int(max(2, min(64, predicted_window)))
        self._fill_retrieval_buffer(seed_idx, self._last_predicted_window)

        frame_idx = int(self._retrieval_buffer.popleft())
        return self._emit_observed_frame(frame_idx, self._last_predicted_window)

    def step(self) -> NavFrame:
        with self._lock:
            return self._navigation_step()

    def get_segment_info(self, segment_idx: int) -> dict:
        if self.meta is None or segment_idx < 0 or segment_idx >= len(self.meta):
            return {"file_id": 0, "t_lat": 0, "win_lat": 0}
        return {
            "file_id": int(self.meta[segment_idx, 0]),
            "t_lat": int(self.meta[segment_idx, 1]),
            "win_lat": int(self.meta[segment_idx, 2]),
        }

    def get_fractional_state(self) -> dict:
        with self._lock:
            indices, distances = self._query_knn(self.z, k=2)
            idx_lower = int(indices[0]) if indices.size > 0 else int(self._current_index)
            idx_upper = int(indices[1]) if indices.size > 1 else idx_lower

            file_ids = self._idx_to_file_id[indices] if indices.size > 0 else np.array([self._current_file_id], dtype=np.int32)

            sigma = float(self.geometry.local_sigma[idx_lower])
            weights = self._compute_gaussian_weights(distances, sigma) if distances.size > 0 else np.array([1.0], dtype=np.float32)
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
        with self._lock:
            pos_2d = self.geometry.project_to_2d(self.z)

            indices, distances = self._query_knn(self.z, k=2)
            nearest_idx = int(indices[0]) if indices.size > 0 else int(self._current_index)

            file_ids = self._idx_to_file_id[indices] if indices.size > 0 else np.array([self._current_file_id], dtype=np.int32)
            sigma = float(self.geometry.local_sigma[nearest_idx])
            weights = self._compute_gaussian_weights(distances, sigma) if distances.size > 0 else np.array([1.0], dtype=np.float32)

            idx_lower = nearest_idx
            idx_upper = int(indices[1]) if indices.size > 1 else idx_lower
            frac = float(weights[1]) / (float(weights[0]) + float(weights[1]) + 1e-8) if len(weights) > 1 else 0.0
            fractional_state = {
                "idx_lower": idx_lower,
                "idx_upper": idx_upper,
                "frac": frac,
                "same_file": int(file_ids[0]) == int(file_ids[1]) if len(file_ids) > 1 else True,
                "file_id_lower": int(file_ids[0]),
                "file_id_upper": int(file_ids[1]) if len(file_ids) > 1 else int(file_ids[0]),
            }

            recent_z_list = list(self._recent_z)
            if recent_z_list:
                recent_z_array = np.asarray(recent_z_list, dtype=np.float32)
                recent_2d = self.geometry.project_to_2d(recent_z_array)
                trajectory_2d_raw = recent_2d.tolist() if recent_2d.ndim > 1 else [recent_2d.tolist()]
            else:
                trajectory_2d_raw = []

            return {
                "cursor": pos_2d.tolist(),
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
                    "position_2d": pos_2d.tolist(),
                    "trajectory_2d": trajectory_2d_raw,
                },
            }

    def _heuristic_window_size(self, velocity: float) -> int:
        if not hasattr(self, "_velocity_percentiles"):
            if not hasattr(self, "_velocity_magnitudes"):
                self._velocity_magnitudes = compute_velocity_magnitudes(self.GG, self.meta)
            vels = self._velocity_magnitudes
            self._velocity_percentiles = (
                float(np.percentile(vels, 5)),
                float(np.percentile(vels, 95)),
            )
        v_low, v_high = self._velocity_percentiles
        v_norm = float(np.clip((velocity - v_low) / (v_high - v_low + 1e-8), 0, 1))
        log2_window = 6.0 - 5.0 * v_norm
        return max(2, min(64, round(2 ** log2_window)))

    @property
    def _file_ids(self) -> np.ndarray:
        return self._idx_to_file_id
