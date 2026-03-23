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
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree
import torch

from ..config import DEVICE, LAMBDA_DT, LAMBDA_FILE_SWITCH
from ..policy.latent_geometry import LatentGeometry
from ..policy.latent_policy import LatentPolicy, LatentPolicyConfig, build_local_features
from ..policy.sequence import compute_velocity_magnitudes
from ..policy.v2_transition_model import build_v2_pair_features, load_v2_transition_model


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
    TIMBRE_SWAP_N_LAT = 16
    TIMBRE_SWAP_N_DESC = 24
    TIMBRE_SWAP_LOOKAHEAD = 0.35
    TIMBRE_SWAP_NEXT_K = 8
    TIMBRE_SWAP_SERIAL_WEIGHT = 0.40
    TIMBRE_SWAP_REPEAT_WEIGHT = 0.30
    TIMBRE_REPEAT_WINDOW = 64
    RECOMPOSE_MIN_POOL = 3
    PHRASE_MIN_SECONDS = 2.0
    PHRASE_MAX_SECONDS = 10.0
    LATENT_FRAME_SECONDS = 0.0465
    V2_MIN_CANDIDATES = 4
    V2_MODEL_BLEND = 0.65
    V2_REPEAT_WINDOW = 32
    V2_ENTRY_NOVELTY_WEIGHT = 0.80
    POLICY_VARIANTS = ("random", "reorganized")

    def __init__(
        self,
        GG: np.ndarray,
        meta: np.ndarray,
        geometry: LatentGeometry,
        file_offsets: Optional[np.ndarray] = None,
        desc_weighted: Optional[np.ndarray] = None,
        policy_path: str = None,
        policy_temperature: float = 1.0,
        policy_sample: bool = True,
        control_phrase_scale: float = 0.40,
        control_jump_rate: float = 0.55,
        control_timbre_lock: float = 0.55,
        control_drift: float = 0.50,
        control_repeat_avoid: float = 0.75,
        control_crossfile: float = 0.70,
        control_morph_len: float = 0.50,
        control_reorg_jump_rate: float = 0.60,
        control_reorg_timbre_lock: float = 0.45,
        control_evolution: float = 0.60,
        control_novelty: float = 0.50,
        control_reorg_crossfile: float = 0.70,
        policy_timbre_swap_enabled: bool = True,
        policy_recompose_enabled: bool = True,
        policy_v2_enabled: bool = False,
        v2_artifact: Optional[dict] = None,
        policy_v2_model_path: Optional[str] = None,
        policy_v2_temperature: float = 1.0,
        policy_variant: Optional[str] = None,
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
        self.desc_weighted: Optional[np.ndarray] = None
        self._desc_tree = None
        if desc_weighted is not None:
            desc = np.asarray(desc_weighted, dtype=np.float32)
            if desc.ndim == 2 and desc.shape[0] == self.N and desc.shape[1] > 0:
                self.desc_weighted = desc
                self._desc_tree = cKDTree(self.desc_weighted)
            else:
                print(
                    "[warn] Invalid descriptor matrix for timbre swap; "
                    f"expected [N, D] with N={self.N}, got {desc.shape}. "
                    "Timbre swap will be disabled."
                )

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
        variant_default = "reorganized" if bool(policy_v2_enabled) else "random"
        if policy_variant is None:
            self.policy_variant = variant_default
        else:
            variant_name = str(policy_variant).strip().lower()
            self.policy_variant = (
                variant_name if variant_name in self.POLICY_VARIANTS else variant_default
            )
        self.ctrl_phrase_scale = float(np.clip(control_phrase_scale, 0.0, 1.0))
        self.ctrl_jump_rate = float(np.clip(control_jump_rate, 0.0, 1.0))
        self.ctrl_timbre_lock = float(np.clip(control_timbre_lock, 0.0, 1.0))
        self.ctrl_drift = float(np.clip(control_drift, 0.0, 1.0))
        self.ctrl_repeat_avoid = float(np.clip(control_repeat_avoid, 0.0, 1.0))
        self.ctrl_crossfile = float(np.clip(control_crossfile, 0.0, 1.0))
        self.ctrl_morph_len = float(np.clip(control_morph_len, 0.0, 1.0))
        self.ctrl_reorg_jump_rate = float(np.clip(control_reorg_jump_rate, 0.0, 1.0))
        self.ctrl_reorg_timbre_lock = float(
            np.clip(control_reorg_timbre_lock, 0.0, 1.0)
        )
        self.ctrl_evolution = float(np.clip(control_evolution, 0.0, 1.0))
        self.ctrl_novelty = float(np.clip(control_novelty, 0.0, 1.0))
        self.ctrl_reorg_crossfile = float(np.clip(control_reorg_crossfile, 0.0, 1.0))
        self.policy_timbre_swap_enabled = bool(policy_timbre_swap_enabled)
        self.policy_recompose_enabled = bool(policy_recompose_enabled)
        self.policy_v2_enabled = bool(policy_v2_enabled)
        self.policy_v2_temperature = max(1e-3, float(policy_v2_temperature))
        self._prev_random_control_vector = self._runtime_control_vector()
        self._prev_reorganized_control_vector = self._reorganized_control_vector()
        self._controls_dirty = False

        # Tracking.
        self._current_file_id = int(self._idx_to_file_id[self._current_index])
        self._current_t = int(self._idx_to_t[self._current_index])
        self._recent_z = deque(maxlen=128)
        self._recent_indices = deque(maxlen=128)
        self._timbre_swap_stats: Dict[str, int] = {
            "attempted": 0,
            "applied": 0,
            "skipped_no_descriptors": 0,
            "skipped_probability": 0,
            "skipped_gate_empty": 0,
        }
        self._recompose_stats: Dict[str, int] = {
            "paths_built": 0,
            "step_transitions": 0,
            "serial_forward": 0,
            "crossfile": 0,
            "repeat_hits": 0,
            "self_stall_steps": 0,
            "anti_stall_rescues": 0,
            "pool_fallbacks": 0,
        }
        self._phrase_frames_remaining = 0
        self._phrase_anchor_idx: Optional[int] = None
        self._phrase_direction = 0
        self._v2_ready = False
        self._v2_unit_start_idx: Optional[np.ndarray] = None
        self._v2_unit_end_idx: Optional[np.ndarray] = None
        self._v2_unit_file_id: Optional[np.ndarray] = None
        self._v2_unit_len: Optional[np.ndarray] = None
        self._v2_unit_entry_desc: Optional[np.ndarray] = None
        self._v2_unit_exit_desc: Optional[np.ndarray] = None
        self._v2_unit_delta_desc: Optional[np.ndarray] = None
        self._v2_frame_to_unit: Optional[np.ndarray] = None
        self._v2_graph_neighbors: Optional[np.ndarray] = None
        self._v2_graph_scores: Optional[np.ndarray] = None
        self._v2_recent_units = deque(maxlen=64)
        self._v2_current_unit: Optional[int] = None
        self._v2_bootstrapped = False
        self._v2_model = None
        self._v2_model_meta: Dict = {}
        self._v2_unit_min_frames = int(
            round(self.PHRASE_MIN_SECONDS / self.LATENT_FRAME_SECONDS)
        )
        self._v2_unit_max_frames = int(
            round(self.PHRASE_MAX_SECONDS / self.LATENT_FRAME_SECONDS)
        )
        self._v2_unit_target_frames = int(
            round(0.5 * (self._v2_unit_min_frames + self._v2_unit_max_frames))
        )
        self._v2_stats: Dict[str, int] = {
            "unit_steps": 0,
            "unit_transitions": 0,
            "unit_repeats": 0,
            "crossfile": 0,
            "model_scored": 0,
            "fallback_scored": 0,
        }

        self._lock = threading.Lock()

        if policy_path is not None:
            try:
                self._setup_policy(policy_path)
            except Exception as e:
                print(f"[warn] Failed to load policy '{policy_path}': {e}")
        if self.policy_v2_enabled:
            self._setup_policy_v2(v2_artifact=v2_artifact, model_path=policy_v2_model_path)

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

    def _setup_policy_v2(self, v2_artifact: Optional[dict], model_path: Optional[str]):
        self._v2_ready = False
        self._v2_model = None
        self._v2_model_meta = {}
        self._v2_current_unit = None
        self._v2_bootstrapped = False
        self._v2_recent_units.clear()

        if not self.policy_v2_enabled:
            return
        if v2_artifact is None:
            print(
                "[warn] Policy V2 enabled but no V2 artifact was provided. "
                "Falling back to V1.1 runtime."
            )
            return

        def _require(name: str, dtype, ndim: Optional[int] = None) -> np.ndarray:
            if name not in v2_artifact:
                raise KeyError(name)
            arr = np.asarray(v2_artifact[name], dtype=dtype)
            if ndim is not None and arr.ndim != int(ndim):
                raise ValueError(f"{name} expected ndim={ndim}, got {arr.ndim}")
            return arr

        try:
            unit_start = _require("unit_start_idx", np.int32, ndim=1).reshape(-1)
            unit_end = _require("unit_end_idx", np.int32, ndim=1).reshape(-1)
            unit_file = _require("unit_file_id", np.int32, ndim=1).reshape(-1)
            unit_len = _require("unit_len", np.int32, ndim=1).reshape(-1)
            unit_entry = _require("unit_entry_desc", np.float32, ndim=2)
            unit_exit = _require("unit_exit_desc", np.float32, ndim=2)
            unit_delta = _require("unit_delta_desc", np.float32, ndim=2)
            frame_to_unit = _require("frame_to_unit", np.int32, ndim=1).reshape(-1)
            graph_neighbors = _require("unit_graph_neighbors", np.int32, ndim=2)
            graph_scores = _require("unit_graph_scores", np.float32, ndim=2)
        except Exception as e:
            print(
                f"[warn] Invalid V2 artifact keys/shapes ({e}). "
                "Falling back to V1.1 runtime."
            )
            return

        n_units = int(unit_start.shape[0])
        if n_units <= 0:
            print("[warn] V2 artifact has zero units. Falling back to V1.1 runtime.")
            return

        if (
            unit_end.shape[0] != n_units
            or unit_file.shape[0] != n_units
            or unit_len.shape[0] != n_units
            or unit_entry.shape[0] != n_units
            or unit_exit.shape[0] != n_units
            or unit_delta.shape[0] != n_units
        ):
            print("[warn] V2 artifact unit arrays are misaligned. Falling back to V1.1 runtime.")
            return
        if not (
            unit_entry.shape[1] == unit_exit.shape[1] == unit_delta.shape[1]
            and unit_entry.shape[1] > 0
        ):
            print("[warn] V2 descriptor dimensions are invalid. Falling back to V1.1 runtime.")
            return
        if frame_to_unit.shape[0] != self.N:
            print(
                "[warn] V2 frame_to_unit length mismatch "
                f"({frame_to_unit.shape[0]} vs corpus {self.N}). Falling back to V1.1 runtime."
            )
            return
        if graph_neighbors.shape[0] != n_units or graph_scores.shape != graph_neighbors.shape:
            print("[warn] V2 graph arrays are invalid. Falling back to V1.1 runtime.")
            return
        if np.any(unit_start < 0) or np.any(unit_end <= unit_start) or np.any(unit_end > self.N):
            print("[warn] V2 unit boundaries are invalid. Falling back to V1.1 runtime.")
            return
        if np.any(frame_to_unit < 0) or np.any(frame_to_unit >= n_units):
            print("[warn] V2 frame_to_unit contains invalid unit ids. Falling back to V1.1 runtime.")
            return

        neighbors = graph_neighbors.copy().astype(np.int32)
        scores = graph_scores.copy().astype(np.float32)
        for i in range(n_units):
            row = neighbors[i]
            invalid = (row < 0) | (row >= n_units)
            if np.any(invalid):
                row[invalid] = int(i)
                scores[i, invalid] = np.inf

        def _scalar_int(name: str, default_value: int) -> int:
            if name not in v2_artifact:
                return int(default_value)
            try:
                return int(np.asarray(v2_artifact[name]).reshape(-1)[0])
            except Exception:
                return int(default_value)

        unit_min_frames = _scalar_int("unit_min_frames", int(np.min(unit_len)))
        unit_max_frames = _scalar_int("unit_max_frames", int(np.max(unit_len)))
        unit_target_frames = _scalar_int(
            "unit_target_frames",
            int(round(0.5 * (unit_min_frames + unit_max_frames))),
        )
        unit_min_frames = max(2, int(unit_min_frames))
        unit_max_frames = max(unit_min_frames, int(unit_max_frames))
        unit_target_frames = int(
            np.clip(unit_target_frames, unit_min_frames, unit_max_frames)
        )

        self._v2_unit_start_idx = unit_start
        self._v2_unit_end_idx = unit_end
        self._v2_unit_file_id = unit_file
        self._v2_unit_len = unit_len
        self._v2_unit_entry_desc = unit_entry
        self._v2_unit_exit_desc = unit_exit
        self._v2_unit_delta_desc = unit_delta
        self._v2_frame_to_unit = frame_to_unit
        self._v2_graph_neighbors = neighbors
        self._v2_graph_scores = scores
        self._v2_unit_min_frames = int(unit_min_frames)
        self._v2_unit_max_frames = int(unit_max_frames)
        self._v2_unit_target_frames = int(unit_target_frames)

        if model_path:
            try:
                model, meta = load_v2_transition_model(
                    checkpoint_path=str(model_path),
                    device=self.policy_device,
                )
                expected_dim = int(4 * self._v2_unit_entry_desc.shape[1] + 5)
                input_dim = int(meta.get("input_dim", expected_dim))
                if input_dim != expected_dim:
                    print(
                        "[warn] V2 model input dimension mismatch "
                        f"({input_dim} vs expected {expected_dim}). "
                        "Heuristic V2 scoring will be used."
                    )
                else:
                    self._v2_model = model
                    self._v2_model_meta = {
                        "path": str(model_path),
                        "input_dim": int(input_dim),
                    }
                    print(f"[info] Loaded V2 transition model: {model_path}")
            except Exception as e:
                print(
                    f"[warn] Failed to load V2 transition model '{model_path}': {e}. "
                    "Heuristic V2 scoring will be used."
                )

        self._v2_ready = True
        self._v2_current_unit = int(self._v2_frame_to_unit[int(self._current_index)])
        self._v2_recent_units.append(int(self._v2_current_unit))
        self._v2_bootstrapped = False
        print(
            "[info] Policy V2 ready: "
            f"units={n_units}, graph_k={int(self._v2_graph_neighbors.shape[1])}, "
            f"model={'yes' if self._v2_model is not None else 'no'}"
        )

    def _reset_policy_state(self):
        self.policy_hidden = None
        self._retrieval_buffer.clear()
        self._recent_z.clear()
        self._recent_indices.clear()
        self._phrase_frames_remaining = 0
        self._phrase_anchor_idx = None
        self._phrase_direction = 0
        self._v2_recent_units.clear()
        self._v2_current_unit = None
        self._v2_bootstrapped = False
        self._v2_stats = {
            "unit_steps": 0,
            "unit_transitions": 0,
            "unit_repeats": 0,
            "crossfile": 0,
            "model_scored": 0,
            "fallback_scored": 0,
        }

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

            # Snap cursor to nearest indexed frame in causal embedding space.
            indices, _ = self._query_knn(self.z, k=1)
            if indices.size > 0:
                self._set_current_index(int(indices[0]))
            self._retrieval_buffer.clear()
            self._phrase_frames_remaining = 0
            self._phrase_anchor_idx = None
            self._phrase_direction = 0
            if self._v2_ready and self._v2_frame_to_unit is not None:
                self._v2_current_unit = int(self._v2_frame_to_unit[int(self._current_index)])
                self._v2_recent_units.clear()
                self._v2_recent_units.append(int(self._v2_current_unit))
                self._v2_bootstrapped = False

    def set_cursor_index(self, idx):
        """Snap navigation directly to a corpus frame index."""
        with self._lock:
            idx_i = int(np.clip(int(idx), 0, self.N - 1))
            self._set_current_index(idx_i)
            self.v = np.zeros(self.latent_dim, dtype=np.float32)

            self._ema_fast = self.z.copy()
            self._ema_slow = self.z.copy()
            if self.geometry.use_ema_mid:
                self._ema_mid = self.z.copy()

            self._retrieval_buffer.clear()
            self._phrase_frames_remaining = 0
            self._phrase_anchor_idx = None
            self._phrase_direction = 0
            if self._v2_ready and self._v2_frame_to_unit is not None:
                self._v2_current_unit = int(self._v2_frame_to_unit[int(self._current_index)])
                self._v2_recent_units.clear()
                self._v2_recent_units.append(int(self._v2_current_unit))
                self._v2_bootstrapped = False

    _CONTROL_CHANGE_THRESHOLD = 0.05

    def set_policy_variant(self, variant: str) -> bool:
        variant_name = str(variant).strip().lower()
        if variant_name not in self.POLICY_VARIANTS:
            return False
        with self._lock:
            if variant_name == self.policy_variant:
                return True
            self.policy_variant = variant_name
            self._retrieval_buffer.clear()
            self._controls_dirty = True
            if self._v2_ready and self._v2_frame_to_unit is not None:
                self._v2_current_unit = int(
                    self._v2_frame_to_unit[int(self._current_index)]
                )
                self._v2_bootstrapped = False
                self._v2_recent_units.clear()
                self._v2_recent_units.append(int(self._v2_current_unit))
        return True

    def get_policy_variant(self) -> str:
        with self._lock:
            return str(self.policy_variant)

    def has_variant(self, variant: str) -> bool:
        variant_name = str(variant).strip().lower()
        if variant_name == "random":
            return True
        if variant_name == "reorganized":
            return bool(self.policy_v2_enabled and self._v2_ready)
        return False

    def get_active_jump_rate(self, variant: Optional[str] = None) -> float:
        variant_name = (
            str(variant).strip().lower() if variant is not None else self.policy_variant
        )
        if variant_name == "reorganized":
            return float(self.ctrl_reorg_jump_rate)
        return float(self.ctrl_jump_rate)

    def set_random_controls(
        self,
        phrase_scale=None,
        jump_rate=None,
        timbre_lock=None,
        drift=None,
        repeat_avoid=None,
        crossfile=None,
    ):
        with self._lock:
            if phrase_scale is not None:
                self.ctrl_phrase_scale = float(np.clip(phrase_scale, 0.0, 1.0))
            if jump_rate is not None:
                self.ctrl_jump_rate = float(np.clip(jump_rate, 0.0, 1.0))
            if timbre_lock is not None:
                self.ctrl_timbre_lock = float(np.clip(timbre_lock, 0.0, 1.0))
            if drift is not None:
                self.ctrl_drift = float(np.clip(drift, 0.0, 1.0))
            if repeat_avoid is not None:
                self.ctrl_repeat_avoid = float(np.clip(repeat_avoid, 0.0, 1.0))
            if crossfile is not None:
                self.ctrl_crossfile = float(np.clip(crossfile, 0.0, 1.0))

            new_vec = self._runtime_control_vector()
            if (
                np.max(np.abs(new_vec - self._prev_random_control_vector))
                > self._CONTROL_CHANGE_THRESHOLD
            ):
                self._controls_dirty = True
                self._prev_random_control_vector = new_vec

    def set_reorganized_controls(
        self,
        morph_len=None,
        jump_rate=None,
        timbre_lock=None,
        evolution=None,
        novelty=None,
        crossfile=None,
    ):
        with self._lock:
            if morph_len is not None:
                self.ctrl_morph_len = float(np.clip(morph_len, 0.0, 1.0))
            if jump_rate is not None:
                self.ctrl_reorg_jump_rate = float(np.clip(jump_rate, 0.0, 1.0))
            if timbre_lock is not None:
                self.ctrl_reorg_timbre_lock = float(np.clip(timbre_lock, 0.0, 1.0))
            if evolution is not None:
                self.ctrl_evolution = float(np.clip(evolution, 0.0, 1.0))
            if novelty is not None:
                self.ctrl_novelty = float(np.clip(novelty, 0.0, 1.0))
            if crossfile is not None:
                self.ctrl_reorg_crossfile = float(np.clip(crossfile, 0.0, 1.0))

            new_vec = self._reorganized_control_vector()
            if (
                np.max(np.abs(new_vec - self._prev_reorganized_control_vector))
                > self._CONTROL_CHANGE_THRESHOLD
            ):
                self._controls_dirty = True
                self._prev_reorganized_control_vector = new_vec

    def reset_policy(self, idx=None):
        with self._lock:
            if idx is not None:
                idx = int(np.clip(idx, 0, self.N - 1))
                self._set_current_index(idx)
            self.v = np.zeros(self.latent_dim, dtype=np.float32)
            self._ema_fast = self.z.copy()
            self._ema_slow = self.z.copy()
            self._reset_policy_state()
            self._recompose_stats = {
                "paths_built": 0,
                "step_transitions": 0,
                "serial_forward": 0,
                "crossfile": 0,
                "repeat_hits": 0,
                "self_stall_steps": 0,
                "anti_stall_rescues": 0,
                "pool_fallbacks": 0,
            }
            self._phrase_frames_remaining = 0
            self._phrase_anchor_idx = None
            self._phrase_direction = 0

    def _set_current_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.N - 1))
        self._current_index = idx
        self.z = self.GG[idx].copy().astype(np.float32)
        self._current_file_id = int(self._idx_to_file_id[idx])
        self._current_t = int(self._idx_to_t[idx])

    def _policy_model_controls(self) -> np.ndarray:
        # Policy checkpoint still expects 6 controls; map the new control surface.
        width = self.ctrl_jump_rate
        energy = 0.35 + 0.65 * self.ctrl_drift
        gravity = 0.5
        memory = 0.15 + 0.70 * self.ctrl_phrase_scale
        coherence = float(np.clip(1.0 - self.ctrl_crossfile, 0.0, 1.0))
        exploration = self.ctrl_jump_rate
        return np.array([
            width,
            energy,
            gravity,
            memory,
            coherence,
            exploration,
        ], dtype=np.float32)

    def _runtime_control_vector(self) -> np.ndarray:
        return np.array([
            self.ctrl_phrase_scale,
            self.ctrl_jump_rate,
            self.ctrl_timbre_lock,
            self.ctrl_drift,
            self.ctrl_repeat_avoid,
            self.ctrl_crossfile,
        ], dtype=np.float32)

    def _reorganized_control_vector(self) -> np.ndarray:
        return np.array(
            [
                self.ctrl_morph_len,
                self.ctrl_reorg_jump_rate,
                self.ctrl_reorg_timbre_lock,
                self.ctrl_evolution,
                self.ctrl_novelty,
                self.ctrl_reorg_crossfile,
            ],
            dtype=np.float32,
        )

    def _context_summaries(self) -> np.ndarray:
        return np.concatenate([self._ema_fast, self._ema_slow], axis=0).astype(np.float32)

    def _query_embedding(self, z_query: np.ndarray) -> np.ndarray:
        return self.geometry.embed_query(
            z_query,
            self._ema_fast,
            self._ema_slow,
        ).astype(np.float32)

    def _query_knn(
        self,
        z_query: np.ndarray,
        k: int = 32,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query nearest neighbors in context embedding space with a causal query context.
        """
        k = int(max(1, min(k, self.N)))
        q = query_embedding if query_embedding is not None else self._query_embedding(z_query)

        if self._has_faiss:
            distances, indices = self.faiss_index.search(q.reshape(1, -1), k)
            return indices[0].astype(np.int32), (1.0 - distances[0]).astype(np.float32)

        dists, inds = self._kdt.query(q, k=k)
        inds = np.atleast_1d(inds).astype(np.int32)
        dists = np.atleast_1d(dists).astype(np.float32)
        return inds, dists

    def _continuity_rerank(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        direction: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if indices.size == 0:
            return indices, distances

        direction = int(np.clip(direction, -1, 1))
        expected_t = self._current_t + direction

        lambda_file = float(LAMBDA_FILE_SWITCH) * float(
            np.clip(1.0 - self.ctrl_crossfile, 0.0, 1.0)
        )
        continuity_scale = float(np.clip(1.0 - 0.85 * self.ctrl_jump_rate, 0.10, 1.0))
        lambda_dt = (
            float(LAMBDA_DT)
            * float(0.5 + 0.5 * self.ctrl_phrase_scale)
            * continuity_scale
        )

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

    def _sample_phrase_length_frames(self) -> int:
        min_frames = int(max(2, round(self.PHRASE_MIN_SECONDS / self.LATENT_FRAME_SECONDS)))
        max_frames = int(max(min_frames + 1, round(self.PHRASE_MAX_SECONDS / self.LATENT_FRAME_SECONDS)))
        center = float(min_frames + self.ctrl_phrase_scale * (max_frames - min_frames))
        spread = float(0.25 * (max_frames - min_frames))
        sample = float(np.random.normal(loc=center, scale=max(1.0, spread)))
        return int(np.clip(round(sample), min_frames, max_frames))

    def _sample_phrase_direction(self) -> int:
        # Higher jump rates favor neutral/non-serial phrase drift.
        p_hold = float(0.35 * np.clip(self.ctrl_jump_rate, 0.0, 1.0))
        if np.random.rand() < p_hold:
            return 0
        return int(np.random.choice([-1, 1]))

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

    def _temporal_baseline_candidate(self, idx: int, direction: int) -> int:
        idx = int(np.clip(idx, 0, self.N - 1))
        file_id = int(self._idx_to_file_id[idx])
        t_lat = int(self._idx_to_t[idx])
        file_start = int(self.file_offsets[file_id])
        file_end = int(self.file_offsets[file_id + 1])
        file_len = max(0, file_end - file_start)
        if file_len <= 0:
            return idx

        if int(direction) == 0:
            return idx
        dt = 1 if direction >= 0 else -1
        t_next = t_lat + dt
        if 0 <= t_next < file_len:
            return int(file_start + t_next)
        return idx

    def _serial_costs(
        self,
        prev_idx: int,
        candidate_indices: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        if candidate_indices.size == 0:
            return np.empty((0,), dtype=np.float32)

        prev_idx = int(np.clip(prev_idx, 0, self.N - 1))
        prev_file = int(self._idx_to_file_id[prev_idx])
        prev_t = int(self._idx_to_t[prev_idx])

        cand_file = self._idx_to_file_id[candidate_indices]
        cand_t = self._idx_to_t[candidate_indices]
        dt = (cand_t - prev_t).astype(np.float32)
        same_file = (cand_file == prev_file).astype(np.float32)

        if direction > 0:
            target_delta = np.abs(dt - 1.0)
        elif direction < 0:
            target_delta = np.abs(dt + 1.0)
        else:
            target_delta = np.minimum(np.abs(dt - 1.0), np.abs(dt + 1.0))

        # Treat exact ±1 serial motion as the pattern to penalize.
        serial = (target_delta <= 0.5).astype(np.float32)
        return (serial * same_file).astype(np.float32)

    def _repeat_costs(
        self,
        candidate_indices: np.ndarray,
        path_prefix: list,
    ) -> np.ndarray:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        if candidate_indices.size == 0:
            return np.empty((0,), dtype=np.float32)

        recent = list(self._recent_indices)[-self.TIMBRE_REPEAT_WINDOW :]
        recent.extend(int(i) for i in path_prefix)
        if not recent:
            return np.zeros((candidate_indices.shape[0],), dtype=np.float32)

        recent_arr = np.asarray(recent, dtype=np.int32)
        recent_set = set(int(i) for i in recent_arr.tolist())
        recent_files = self._idx_to_file_id[recent_arr]
        recent_t = self._idx_to_t[recent_arr]

        costs = np.zeros((candidate_indices.shape[0],), dtype=np.float32)
        for i, idx in enumerate(candidate_indices):
            idx_i = int(idx)
            if idx_i in recent_set:
                costs[i] = 1.0
                continue

            same_file = recent_files == int(self._idx_to_file_id[idx_i])
            if not np.any(same_file):
                continue

            dt_min = int(
                np.min(np.abs(recent_t[same_file] - int(self._idx_to_t[idx_i])))
            )
            if dt_min <= 1:
                costs[i] = 0.7
            elif dt_min <= 2:
                costs[i] = 0.4

        return costs

    def _recompose_candidate_pool(
        self,
        prev_idx: int,
        anchor_idx: int,
        direction: int,
        q_gate: float,
    ) -> Tuple[np.ndarray, int]:
        baseline_idx = self._temporal_baseline_candidate(prev_idx, direction)

        lat_candidates = np.asarray(
            self.geometry.knn_indices[int(prev_idx)][: self.TIMBRE_SWAP_N_LAT],
            dtype=np.int32,
        ).reshape(-1)
        desc_candidates, _ = self._descriptor_candidates(
            int(prev_idx), k=self.TIMBRE_SWAP_N_DESC
        )

        union = np.concatenate(
            [
                np.array([int(prev_idx), int(anchor_idx), int(baseline_idx)], dtype=np.int32),
                lat_candidates,
                np.asarray(desc_candidates, dtype=np.int32).reshape(-1),
            ]
        )
        if union.size == 0:
            return np.empty((0,), dtype=np.int32), int(baseline_idx)

        _, order = np.unique(union, return_index=True)
        union = union[np.sort(order)]
        union = np.clip(union, 0, self.N - 1).astype(np.int32)

        if self.desc_weighted is None:
            return union, int(baseline_idx)

        d_anchor = np.linalg.norm(
            self.desc_weighted[union] - self.desc_weighted[int(anchor_idx)][None, :],
            axis=1,
        ).astype(np.float32)
        gate_thr = float(np.quantile(d_anchor, float(np.clip(q_gate, 0.0, 1.0))))
        keep = d_anchor <= gate_thr
        keep[union == int(anchor_idx)] = True
        keep[union == int(prev_idx)] = True
        keep[union == int(baseline_idx)] = True
        kept = union[keep]
        if kept.size == 0:
            kept = np.array([int(baseline_idx)], dtype=np.int32)

        if kept.size < self.RECOMPOSE_MIN_POOL:
            existing = set(int(i) for i in kept.tolist())
            extras = []
            for idx in lat_candidates:
                idx_i = int(idx)
                if idx_i not in existing:
                    extras.append(idx_i)
                    existing.add(idx_i)
                if len(extras) >= self.RECOMPOSE_MIN_POOL:
                    break
            if len(extras) < self.RECOMPOSE_MIN_POOL:
                for idx in np.asarray(desc_candidates, dtype=np.int32).reshape(-1):
                    idx_i = int(idx)
                    if idx_i not in existing:
                        extras.append(idx_i)
                        existing.add(idx_i)
                    if len(extras) >= self.RECOMPOSE_MIN_POOL:
                        break
            if extras:
                self._recompose_stats["pool_fallbacks"] += 1
                kept = np.asarray(
                    list(dict.fromkeys([int(i) for i in kept.tolist()] + extras)),
                    dtype=np.int32,
                )
        return kept.astype(np.int32), int(baseline_idx)

    def _fill_retrieval_buffer_recomposed(
        self,
        seed_idx: int,
        anchor_idx: int,
        chunk_len: int,
        direction: int,
    ):
        seed_idx = int(np.clip(seed_idx, 0, self.N - 1))
        anchor_idx = int(np.clip(anchor_idx, 0, self.N - 1))
        chunk_len = int(max(1, chunk_len))
        q_gate = 0.20 + 0.70 * float(np.clip(self.ctrl_jump_rate, 0.0, 1.0))

        path = [seed_idx]
        for _ in range(max(0, chunk_len - 1)):
            prev_idx = int(path[-1])
            expected_t = int(self._idx_to_t[prev_idx] + int(direction))
            pool, baseline_idx = self._recompose_candidate_pool(
                prev_idx=prev_idx,
                anchor_idx=anchor_idx,
                direction=direction,
                q_gate=q_gate,
            )
            if pool.size == 0:
                path.append(int(baseline_idx))
                continue

            emb_ref = self.embeddings_l2[prev_idx]
            emb_candidates = self.embeddings_l2[pool]
            c_lat = 1.0 - (emb_candidates @ emb_ref.reshape(-1, 1)).reshape(-1)
            c_time = np.abs(self._idx_to_t[pool].astype(np.float32) - float(expected_t))

            if self.desc_weighted is not None:
                desc_anchor = self.desc_weighted[anchor_idx]
                desc_prev = self.desc_weighted[prev_idx]
                desc_candidates = self.desc_weighted[pool]
                desc_target = (
                    (1.0 - self.ctrl_drift) * desc_anchor
                    + self.ctrl_drift * desc_prev
                ).astype(np.float32)
                c_timbre = np.linalg.norm(
                    desc_candidates - desc_target[None, :], axis=1
                ).astype(np.float32)
                if len(path) >= 2:
                    target_delta = (
                        self.desc_weighted[int(path[-1])]
                        - self.desc_weighted[int(path[-2])]
                    ).astype(np.float32)
                elif len(self._recent_indices) >= 2:
                    idx_prev = int(self._recent_indices[-2])
                    idx_last = int(self._recent_indices[-1])
                    target_delta = (
                        self.desc_weighted[idx_last] - self.desc_weighted[idx_prev]
                    ).astype(np.float32)
                else:
                    target_delta = np.zeros(
                        (self.desc_weighted.shape[1],), dtype=np.float32
                    )
                candidate_delta = desc_candidates - desc_prev[None, :]
                c_vel = np.linalg.norm(
                    candidate_delta - target_delta[None, :], axis=1
                ).astype(np.float32)
            else:
                c_timbre = np.zeros((pool.shape[0],), dtype=np.float32)
                c_vel = np.zeros((pool.shape[0],), dtype=np.float32)

            c_file = (
                self._idx_to_file_id[pool] != int(self._idx_to_file_id[prev_idx])
            ).astype(np.float32)
            c_serial = self._serial_costs(prev_idx, pool, direction=direction)
            c_repeat = self._repeat_costs(pool, path_prefix=path)

            c_lat_n = self._normalize_cost(c_lat)
            c_time_n = self._normalize_cost(c_time)
            c_timbre_n = self._normalize_cost(c_timbre)
            c_vel_n = self._normalize_cost(c_vel)

            repeat_strength = float(np.clip(self.ctrl_repeat_avoid, 0.0, 1.0))
            jump_strength = float(np.clip(self.ctrl_jump_rate, 0.0, 1.0))
            w_serial = (
                self.TIMBRE_SWAP_SERIAL_WEIGHT
                * repeat_strength
                * (0.5 + 0.5 * jump_strength)
            )
            w_repeat = (
                self.TIMBRE_SWAP_REPEAT_WEIGHT
                * repeat_strength
                * (0.5 + 0.5 * jump_strength)
            )

            scores_s1 = np.array(
                [
                    self._score_swap_candidate(
                        c_lat_n[i],
                        c_time_n[i],
                        c_timbre_n[i],
                        c_vel_n[i],
                        c_file[i],
                    )
                    + w_serial * float(c_serial[i])
                    + w_repeat * float(c_repeat[i])
                    for i in range(pool.shape[0])
                ],
                dtype=np.float32,
            )

            lookahead = np.array(
                [
                    self._lookahead_best_s1(int(idx), direction=direction, q_gate=q_gate)
                    for idx in pool
                ],
                dtype=np.float32,
            )
            total_scores = scores_s1 + self.TIMBRE_SWAP_LOOKAHEAD * lookahead
            order = np.lexsort((pool, c_file, c_time_n, total_scores))
            chosen_idx = int(pool[int(order[0])])
            if chosen_idx == prev_idx:
                self._recompose_stats["self_stall_steps"] += 1
                non_self = [int(pool[i]) for i in order if int(pool[i]) != prev_idx]
                if non_self:
                    chosen_idx = int(non_self[0])
                    self._recompose_stats["anti_stall_rescues"] += 1
            path.append(chosen_idx)

        self._recompose_stats["paths_built"] += 1
        for i in range(1, len(path)):
            prev_idx = int(path[i - 1])
            cur_idx = int(path[i])
            self._recompose_stats["step_transitions"] += 1
            if int(self._idx_to_file_id[cur_idx]) != int(self._idx_to_file_id[prev_idx]):
                self._recompose_stats["crossfile"] += 1
            if (
                int(self._idx_to_file_id[cur_idx]) == int(self._idx_to_file_id[prev_idx])
                and int(self._idx_to_t[cur_idx]) == int(self._idx_to_t[prev_idx]) + 1
            ):
                self._recompose_stats["serial_forward"] += 1
            if cur_idx in self._recent_indices:
                self._recompose_stats["repeat_hits"] += 1

        for idx in path:
            self._retrieval_buffer.append(int(idx))

        if not self._retrieval_buffer:
            self._retrieval_buffer.append(int(seed_idx))

    def _compute_gaussian_weights(self, distances: np.ndarray, sigma: float) -> np.ndarray:
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        return weights / (weights.sum() + 1e-8)

    def _descriptor_candidates(self, seed_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.desc_weighted is None or self._desc_tree is None:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
        seed_idx = int(np.clip(seed_idx, 0, self.N - 1))
        k = int(max(1, min(k, self.N)))
        q = self.desc_weighted[seed_idx]
        dists, inds = self._desc_tree.query(q, k=k)
        inds = np.atleast_1d(inds).astype(np.int32)
        dists = np.atleast_1d(dists).astype(np.float32)
        return inds, dists

    @staticmethod
    def _normalize_cost(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        vmax = float(arr.max())
        if vmax <= 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / vmax

    @staticmethod
    def _compute_continuity_weights(
        timbre_control: float,
        jump_rate: float,
    ) -> Tuple[float, float, float]:
        continuity_mass = float(np.clip(1.0 - timbre_control, 0.0, 1.0))
        continuity_mass *= float(np.clip(1.0 - 0.85 * jump_rate, 0.15, 1.0))
        shared = continuity_mass / 3.0
        return shared, shared, shared

    def _score_swap_candidate(
        self,
        c_lat: float,
        c_time: float,
        c_timbre: float,
        c_vel: float,
        c_file: float,
    ) -> float:
        w_lat, w_time, w_vel = self._compute_continuity_weights(
            self.ctrl_timbre_lock,
            self.ctrl_jump_rate,
        )
        w_timbre = float(np.clip(self.ctrl_timbre_lock, 0.0, 1.0))
        w_file = float(np.clip(1.0 - self.ctrl_crossfile, 0.0, 1.0))
        return (
            w_lat * float(c_lat)
            + w_time * float(c_time)
            + w_timbre * float(c_timbre)
            + w_vel * float(c_vel)
            + w_file * float(c_file)
        )

    def _score_candidates_s1(
        self,
        candidate_indices: np.ndarray,
        expected_t: int,
        ref_idx: int,
        query_embedding: np.ndarray,
        target_desc_delta: np.ndarray,
        ref_file_id: int,
        velocity_origin_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        if candidate_indices.size == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        emb_ref = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        emb_candidates = self.embeddings_l2[candidate_indices]
        lat_dist = 1.0 - (emb_candidates @ emb_ref.T).reshape(-1)

        t_vals = self._idx_to_t[candidate_indices].astype(np.float32)
        c_time = np.abs(t_vals - float(expected_t))

        desc_ref = self.desc_weighted[ref_idx]
        desc_candidates = self.desc_weighted[candidate_indices]
        c_timbre = np.linalg.norm(desc_candidates - desc_ref[None, :], axis=1).astype(np.float32)

        origin_idx = int(self._current_index) if velocity_origin_idx is None else int(velocity_origin_idx)
        current_desc = self.desc_weighted[origin_idx]
        candidate_delta = desc_candidates - current_desc[None, :]
        c_vel = np.linalg.norm(candidate_delta - target_desc_delta[None, :], axis=1).astype(np.float32)

        c_file = (self._idx_to_file_id[candidate_indices] != int(ref_file_id)).astype(np.float32)

        c_lat_n = self._normalize_cost(lat_dist)
        c_time_n = self._normalize_cost(c_time)
        c_timbre_n = self._normalize_cost(c_timbre)
        c_vel_n = self._normalize_cost(c_vel)

        scores = np.array([
            self._score_swap_candidate(c_lat_n[i], c_time_n[i], c_timbre_n[i], c_vel_n[i], c_file[i])
            for i in range(candidate_indices.shape[0])
        ], dtype=np.float32)
        return scores, c_time_n, c_file

    def _lookahead_best_s1(
        self,
        candidate_idx: int,
        direction: int,
        q_gate: float,
    ) -> float:
        if self.desc_weighted is None:
            return 0.0

        candidate_idx = int(candidate_idx)
        base_neighbors = self.geometry.knn_indices[candidate_idx][: self.TIMBRE_SWAP_NEXT_K]
        desc_next, _ = self._descriptor_candidates(candidate_idx, k=self.TIMBRE_SWAP_NEXT_K)
        next_indices = np.concatenate([
            np.array([candidate_idx], dtype=np.int32),
            np.asarray(base_neighbors, dtype=np.int32).reshape(-1),
            np.asarray(desc_next, dtype=np.int32).reshape(-1),
        ])
        if next_indices.size == 0:
            return 0.0

        _, order = np.unique(next_indices, return_index=True)
        next_indices = next_indices[np.sort(order)]
        next_indices = np.clip(next_indices, 0, self.N - 1).astype(np.int32)

        desc_ref = self.desc_weighted[candidate_idx]
        d_next = np.linalg.norm(
            self.desc_weighted[next_indices] - desc_ref[None, :], axis=1
        ).astype(np.float32)
        gate_thr = float(np.quantile(d_next, float(np.clip(q_gate, 0.0, 1.0))))
        keep_mask = d_next <= gate_thr
        keep_mask[next_indices == candidate_idx] = True
        kept = next_indices[keep_mask]
        if kept.size == 0:
            return 0.0

        expected_next_t = int(self._idx_to_t[candidate_idx]) + int(direction)
        emb_i = self.embeddings_l2[candidate_idx]
        target_delta = self.desc_weighted[candidate_idx] - self.desc_weighted[int(self._current_index)]
        scores, _, _ = self._score_candidates_s1(
            candidate_indices=kept,
            expected_t=expected_next_t,
            ref_idx=candidate_idx,
            query_embedding=emb_i,
            target_desc_delta=target_delta,
            ref_file_id=int(self._idx_to_file_id[candidate_idx]),
            velocity_origin_idx=candidate_idx,
        )
        if scores.size == 0:
            return 0.0
        return float(scores.min())

    def _choose_timbre_swap_seed(
        self,
        baseline_idx: int,
        candidate_indices: np.ndarray,
        current_idx: int,
        expected_t: int,
        query_embedding: np.ndarray,
    ) -> int:
        baseline_idx = int(np.clip(baseline_idx, 0, self.N - 1))
        self._timbre_swap_stats["attempted"] += 1

        if (
            not self.policy_timbre_swap_enabled
            or self.desc_weighted is None
            or self._desc_tree is None
        ):
            self._timbre_swap_stats["skipped_no_descriptors"] += 1
            return baseline_idx

        if self.ctrl_jump_rate <= 1e-6:
            self._timbre_swap_stats["skipped_probability"] += 1
            return baseline_idx

        lat_candidates = np.asarray(candidate_indices, dtype=np.int32)[: self.TIMBRE_SWAP_N_LAT]
        desc_candidates, _ = self._descriptor_candidates(
            baseline_idx, k=self.TIMBRE_SWAP_N_DESC
        )

        union = np.concatenate([
            np.array([baseline_idx], dtype=np.int32),
            lat_candidates.reshape(-1),
            desc_candidates.reshape(-1),
        ])
        if union.size == 0:
            self._timbre_swap_stats["skipped_gate_empty"] += 1
            return baseline_idx

        _, order = np.unique(union, return_index=True)
        union = union[np.sort(order)]
        union = np.clip(union, 0, self.N - 1).astype(np.int32)

        desc_seed = self.desc_weighted[baseline_idx]
        d_seed = np.linalg.norm(
            self.desc_weighted[union] - desc_seed[None, :], axis=1
        ).astype(np.float32)

        q_gate = 0.20 + 0.70 * float(np.clip(self.ctrl_jump_rate, 0.0, 1.0))
        gate_thr = float(np.quantile(d_seed, q_gate))
        keep_mask = d_seed <= gate_thr
        keep_mask[union == baseline_idx] = True
        kept = union[keep_mask]
        if kept.size == 0:
            self._timbre_swap_stats["skipped_gate_empty"] += 1
            return baseline_idx

        if len(self._recent_indices) >= 2:
            idx_prev = int(self._recent_indices[-2])
            idx_last = int(self._recent_indices[-1])
            target_desc_delta = self.desc_weighted[idx_last] - self.desc_weighted[idx_prev]
        else:
            target_desc_delta = np.zeros((self.desc_weighted.shape[1],), dtype=np.float32)

        scores_s1, c_time_n, c_file = self._score_candidates_s1(
            candidate_indices=kept,
            expected_t=int(expected_t),
            ref_idx=baseline_idx,
            query_embedding=query_embedding,
            target_desc_delta=target_desc_delta,
            ref_file_id=int(self._current_file_id),
            velocity_origin_idx=current_idx,
        )
        if scores_s1.size == 0:
            self._timbre_swap_stats["skipped_gate_empty"] += 1
            return baseline_idx

        direction = int(np.sign(int(expected_t) - int(self._current_t)))
        lookahead = np.array([
            self._lookahead_best_s1(int(idx), direction=direction, q_gate=q_gate)
            for idx in kept
        ], dtype=np.float32)
        total_scores = scores_s1 + self.TIMBRE_SWAP_LOOKAHEAD * lookahead

        best_order = np.lexsort((kept, c_file, c_time_n, total_scores))
        best_idx = int(kept[int(best_order[0])])
        if best_idx != baseline_idx:
            self._timbre_swap_stats["applied"] += 1
        return best_idx

    def _sample_v2_target_len_frames(self) -> int:
        min_frames = int(max(2, self._v2_unit_min_frames))
        max_frames = int(max(min_frames, self._v2_unit_max_frames))
        center = float(min_frames + self.ctrl_morph_len * (max_frames - min_frames))
        spread = float(0.25 * (max_frames - min_frames))
        sample = float(np.random.normal(loc=center, scale=max(1.0, spread)))
        return int(np.clip(round(sample), min_frames, max_frames))

    def _v2_repeat_costs(self, candidate_units: np.ndarray) -> np.ndarray:
        cand = np.asarray(candidate_units, dtype=np.int32).reshape(-1)
        if cand.size == 0:
            return np.empty((0,), dtype=np.float32)

        recent = list(self._v2_recent_units)[-self.V2_REPEAT_WINDOW :]
        if not recent:
            return np.zeros((cand.shape[0],), dtype=np.float32)

        costs = np.zeros((cand.shape[0],), dtype=np.float32)
        for i, unit_i in enumerate(cand.tolist()):
            unit_i = int(unit_i)
            best = 0.0
            for rank, prev_u in enumerate(reversed(recent)):
                if int(prev_u) != unit_i:
                    continue
                # Recent repeats are penalized strongly; older repeats decay.
                best = max(best, float(max(0.0, 1.0 - 0.08 * float(rank))))
            costs[i] = float(best)
        return costs

    def _choose_v2_next_unit(self, current_unit: int) -> int:
        n_units = int(self._v2_unit_start_idx.shape[0])
        cur = int(np.clip(current_unit, 0, n_units - 1))
        neighbors = self._v2_graph_neighbors[cur].astype(np.int32).reshape(-1)
        graph_scores = self._v2_graph_scores[cur].astype(np.float32).reshape(-1)
        valid = np.isfinite(graph_scores) & (neighbors >= 0) & (neighbors < n_units)
        if not np.any(valid):
            self._v2_stats["fallback_scored"] += 1
            return int(cur)

        cand_all = neighbors[valid]
        graph_all = graph_scores[valid]
        order = np.argsort(graph_all)
        cand_sorted = cand_all[order]
        graph_sorted = graph_all[order]

        dedup_units = []
        dedup_scores = []
        seen = set()
        for u, s in zip(cand_sorted.tolist(), graph_sorted.tolist()):
            u_i = int(u)
            if u_i in seen:
                continue
            seen.add(u_i)
            dedup_units.append(u_i)
            dedup_scores.append(float(s))
        if not dedup_units:
            self._v2_stats["fallback_scored"] += 1
            return int(cur)

        cand_sorted = np.asarray(dedup_units, dtype=np.int32)
        graph_sorted = np.asarray(dedup_scores, dtype=np.float32)

        min_k = int(min(self.V2_MIN_CANDIDATES, cand_sorted.shape[0]))
        max_k = int(cand_sorted.shape[0])
        if max_k <= 1:
            self._v2_stats["fallback_scored"] += 1
            return int(cand_sorted[0])
        pool_k = int(
            np.clip(
                round(min_k + self.ctrl_reorg_jump_rate * (max_k - min_k)),
                min_k,
                max_k,
            )
        )
        cand = cand_sorted[:pool_k]
        c_graph = self._normalize_cost(graph_sorted[:pool_k])

        cur_exit = self._v2_unit_exit_desc[cur]
        cur_delta = self._v2_unit_delta_desc[cur]
        cand_entry = self._v2_unit_entry_desc[cand]
        cand_delta = self._v2_unit_delta_desc[cand]
        c_entry = self._normalize_cost(
            np.linalg.norm(cand_entry - cur_exit[None, :], axis=1).astype(np.float32)
        )
        c_delta = self._normalize_cost(
            np.linalg.norm(cand_delta - cur_delta[None, :], axis=1).astype(np.float32)
        )

        target_len = float(self._sample_v2_target_len_frames())
        len_range = float(max(1, self._v2_unit_max_frames - self._v2_unit_min_frames))
        cand_len = self._v2_unit_len[cand].astype(np.float32)
        c_len = np.clip(np.abs(cand_len - target_len) / len_range, 0.0, 1.0)
        c_file = (
            self._v2_unit_file_id[cand] != int(self._v2_unit_file_id[cur])
        ).astype(np.float32)
        c_repeat = self._v2_repeat_costs(cand)

        timbre_strength = float(np.clip(self.ctrl_reorg_timbre_lock, 0.0, 1.0))
        evolution_strength = float(np.clip(self.ctrl_evolution, 0.0, 1.0))
        jump_strength = float(np.clip(self.ctrl_reorg_jump_rate, 0.0, 1.0))
        novelty_strength = float(np.clip(self.ctrl_novelty, 0.0, 1.0))
        cross_strength = float(np.clip(self.ctrl_reorg_crossfile, 0.0, 1.0))

        w_graph = 0.20 + 0.50 * (1.0 - jump_strength)
        w_entry = 0.25 + 0.75 * timbre_strength
        w_delta = 0.10 + 0.45 * timbre_strength
        w_len = 0.20 + 0.50 * (1.0 - jump_strength)
        w_repeat = 0.15 + 1.10 * novelty_strength
        w_file = 1.00 * (1.0 - cross_strength)
        w_novel = self.V2_ENTRY_NOVELTY_WEIGHT * evolution_strength

        heuristic = (
            w_graph * c_graph
            + (w_entry - w_novel) * c_entry
            + w_delta * c_delta
            + w_len * c_len
            + w_repeat * c_repeat
            + w_file * c_file
        ).astype(np.float32)

        total = heuristic.copy()
        if self._v2_model is not None:
            try:
                feat = build_v2_pair_features(
                    unit_entry_desc=self._v2_unit_entry_desc,
                    unit_exit_desc=self._v2_unit_exit_desc,
                    unit_delta_desc=self._v2_unit_delta_desc,
                    unit_len=self._v2_unit_len,
                    unit_file_id=self._v2_unit_file_id,
                    current_unit=cur,
                    candidate_units=cand,
                )
                feat_t = torch.from_numpy(feat).to(self.policy_device, dtype=torch.float32)
                with torch.no_grad():
                    logits = self._v2_model(feat_t).detach().cpu().numpy().astype(np.float32).reshape(-1)
                logits = logits - float(np.max(logits))
                probs = np.exp(logits / max(1e-3, self.policy_v2_temperature))
                probs = probs / (float(np.sum(probs)) + 1e-8)
                model_cost = self._normalize_cost(1.0 - probs.astype(np.float32))
                total = (
                    (1.0 - self.V2_MODEL_BLEND) * heuristic
                    + self.V2_MODEL_BLEND * model_cost
                ).astype(np.float32)
                self._v2_stats["model_scored"] += 1
            except Exception as e:
                print(
                    f"[warn] Disabling V2 model scoring due to runtime error: {e}. "
                    "Falling back to heuristic scoring."
                )
                self._v2_model = None
                self._v2_model_meta = {}
                self._v2_stats["fallback_scored"] += 1
        else:
            self._v2_stats["fallback_scored"] += 1

        choice = None
        if self.policy_sample and jump_strength > 1e-6:
            temp = self.policy_v2_temperature * (
                0.20 + 1.40 * jump_strength + 0.40 * novelty_strength
            )
            logits = -total / max(1e-3, temp)
            logits = logits - float(np.max(logits))
            probs = np.exp(logits).astype(np.float32)
            probs_sum = float(np.sum(probs))
            if probs_sum > 1e-8 and np.isfinite(probs_sum):
                probs = probs / probs_sum
                choice = int(np.random.choice(cand.shape[0], p=probs))
        if choice is None:
            order = np.lexsort((cand, c_file, c_repeat, total))
            choice = int(order[0])

        chosen = int(cand[choice])
        if chosen == cur and cand.shape[0] > 1:
            # Avoid immediate unit self-stalls when alternatives are available.
            order = np.lexsort((cand, c_file, c_repeat, total))
            for idx in order.tolist():
                cand_i = int(cand[int(idx)])
                if cand_i != cur:
                    chosen = cand_i
                    break
        return int(chosen)

    def _fill_retrieval_buffer_v2(self, unit_idx: int, start_frame_idx: Optional[int] = None):
        unit_idx = int(np.clip(unit_idx, 0, self._v2_unit_start_idx.shape[0] - 1))
        start = int(self._v2_unit_start_idx[unit_idx])
        end = int(self._v2_unit_end_idx[unit_idx])
        if end <= start:
            self._retrieval_buffer.append(start)
            self._last_predicted_window = 2
            return

        if start_frame_idx is not None:
            start = int(np.clip(int(start_frame_idx), start, end - 1))

        for idx in range(start, end):
            self._retrieval_buffer.append(int(idx))
        if not self._retrieval_buffer:
            self._retrieval_buffer.append(int(start))
        self._last_predicted_window = int(
            np.clip(int(self._v2_unit_len[unit_idx]), 2, 64)
        )

    def _navigation_step_v2(self) -> NavFrame:
        if self._controls_dirty and len(self._retrieval_buffer) > self._BUFFER_TRUNCATE_MAX:
            while len(self._retrieval_buffer) > self._BUFFER_TRUNCATE_MAX:
                self._retrieval_buffer.pop()
            self._controls_dirty = False

        if self._retrieval_buffer:
            idx = int(self._retrieval_buffer.popleft())
            out = self._emit_observed_frame(idx, self._last_predicted_window)
            self._v2_stats["unit_steps"] += 1
            return out

        if not self._v2_ready:
            return self._navigation_step()

        current_idx = int(self._current_index)
        current_unit = int(self._v2_frame_to_unit[current_idx])
        if not self._v2_recent_units or int(self._v2_recent_units[-1]) != current_unit:
            self._v2_recent_units.append(int(current_unit))

        # Bootstrap from the current unit once after reset/cursor moves.
        if not self._v2_bootstrapped:
            self._v2_current_unit = int(current_unit)
            self._fill_retrieval_buffer_v2(current_unit, start_frame_idx=current_idx)
            idx = int(self._retrieval_buffer.popleft())
            out = self._emit_observed_frame(idx, self._last_predicted_window)
            self._v2_stats["unit_steps"] += 1
            self._v2_bootstrapped = True
            return out

        next_unit = self._choose_v2_next_unit(current_unit)
        if next_unit in self._v2_recent_units:
            self._v2_stats["unit_repeats"] += 1
        if int(self._v2_unit_file_id[next_unit]) != int(self._v2_unit_file_id[current_unit]):
            self._v2_stats["crossfile"] += 1
        self._v2_stats["unit_transitions"] += 1
        self._v2_current_unit = int(next_unit)
        self._v2_recent_units.append(int(next_unit))

        if next_unit == current_unit:
            unit_end = int(self._v2_unit_end_idx[next_unit])
            if current_idx + 1 < unit_end:
                start_idx = int(current_idx + 1)
            else:
                start_idx = int(self._v2_unit_start_idx[next_unit])
        else:
            start_idx = int(self._v2_unit_start_idx[next_unit])

        self._fill_retrieval_buffer_v2(next_unit, start_frame_idx=start_idx)
        idx = int(self._retrieval_buffer.popleft())
        out = self._emit_observed_frame(idx, self._last_predicted_window)
        self._v2_stats["unit_steps"] += 1
        return out

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

    _BUFFER_TRUNCATE_MAX = 2

    def _navigation_step(self) -> NavFrame:
        """
        Execute one navigation step.

        1) Propose motion in latent space (policy or stochastic fallback).
        2) Query causal context index E and continuity-rerank candidates.
        3) Retrieve a real corpus chunk starting at (file_id, t).
        4) Emit one observed frame from that chunk.
        """
        if self._controls_dirty and len(self._retrieval_buffer) > self._BUFFER_TRUNCATE_MAX:
            while len(self._retrieval_buffer) > self._BUFFER_TRUNCATE_MAX:
                self._retrieval_buffer.pop()
            self._controls_dirty = False

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
            ctrl_tensor = torch.tensor(
                self._policy_model_controls(),
                device=self.policy_device,
                dtype=torch.float32,
            ).view(1, 1, -1)
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

            temperature = self.policy_temperature * (1.0 + 2.0 * self.ctrl_jump_rate)
            if self.policy_sample or self.ctrl_jump_rate > 0:
                delta_z = self.policy.sample_delta(delta_mean, delta_log_std, delta_weights, temperature)
            else:
                delta_z = self.policy.get_mixture_mode(delta_mean, delta_weights)

            delta_z = delta_z.squeeze(0).cpu().numpy()
            dv = vel_delta.squeeze(0).cpu().numpy()
            predicted_window = self.policy.get_window_size(window_log2.squeeze(0))
        else:
            drift = (self.ctrl_drift - 0.5) * 2.0
            direction = time_gradient * drift
            noise = np.random.randn(self.latent_dim).astype(np.float32)
            noise *= local_sigma * (0.15 + self.ctrl_jump_rate)
            delta_z = direction * local_sigma * (0.35 + 0.65 * self.ctrl_drift) + noise
            dv = np.zeros(self.latent_dim, dtype=np.float32)

            if not hasattr(self, "_velocity_magnitudes"):
                self._velocity_magnitudes = compute_velocity_magnitudes(self.GG, self.meta)
            local_velocity = float(self._velocity_magnitudes[current_idx])
            predicted_window = self._heuristic_window_size(local_velocity)

        phrase_window = int(np.clip(round(2 + 62 * self.ctrl_phrase_scale), 2, 64))
        predicted_window = int(
            np.clip(round(0.35 * float(predicted_window) + 0.65 * float(phrase_window)), 2, 64)
        )

        energy_scale = 0.5 + 0.5 * self.ctrl_jump_rate
        delta_z *= energy_scale

        self.v = self.VELOCITY_DECAY * self.v + dv

        z_new = self.z + self.v + delta_z
        z_new += self.MANIFOLD_ATTRACTION * (knn_centroid - z_new)

        memory_strength = float(0.15 + 0.70 * self.ctrl_phrase_scale)
        if memory_strength > 0 and len(self._recent_z) > 0:
            recent_mean = np.mean(np.asarray(self._recent_z), axis=0)
            z_new = (1.0 - memory_strength) * z_new + memory_strength * recent_mean

        coherence_strength = float(np.clip(1.0 - self.ctrl_crossfile, 0.0, 1.0))
        if coherence_strength > 0 and neighbors.size > 0:
            same_file_mask = self._idx_to_file_id[neighbors] == self._current_file_id
            if same_file_mask.sum() > 0:
                same_file_centroid = self.GG[neighbors[same_file_mask]].mean(axis=0)
                z_new = (1.0 - coherence_strength) * z_new + coherence_strength * same_file_centroid

        self.z = z_new.astype(np.float32)

        if self._phrase_frames_remaining <= 0:
            self._phrase_frames_remaining = self._sample_phrase_length_frames()
            self._phrase_direction = self._sample_phrase_direction()

        direction = int(self._phrase_direction)
        q_emb = self._query_embedding(self.z)
        cand_indices, cand_distances = self._query_knn(
            self.z, k=self.geometry.K, query_embedding=q_emb
        )
        cand_indices, cand_distances = self._continuity_rerank(
            cand_indices, cand_distances, direction=direction
        )

        seed_idx = int(cand_indices[0]) if cand_indices.size > 0 else current_idx
        expected_t = int(self._current_t + direction)
        seed_idx = self._choose_timbre_swap_seed(
            baseline_idx=seed_idx,
            candidate_indices=cand_indices,
            current_idx=current_idx,
            expected_t=expected_t,
            query_embedding=q_emb,
        )
        if self._phrase_anchor_idx is None:
            self._phrase_anchor_idx = int(seed_idx)
        chunk_len = int(max(2, min(64, min(predicted_window, self._phrase_frames_remaining))))
        self._last_predicted_window = chunk_len
        recompose_active = bool(
            self.policy_recompose_enabled
            and self.desc_weighted is not None
            and self._desc_tree is not None
        )
        if recompose_active:
            self._fill_retrieval_buffer_recomposed(
                seed_idx=seed_idx,
                anchor_idx=int(self._phrase_anchor_idx),
                chunk_len=chunk_len,
                direction=direction,
            )
        else:
            self._fill_retrieval_buffer(seed_idx, chunk_len)
        self._phrase_frames_remaining = max(0, int(self._phrase_frames_remaining - chunk_len))
        if self._phrase_frames_remaining <= 0:
            self._phrase_anchor_idx = None

        frame_idx = int(self._retrieval_buffer.popleft())
        return self._emit_observed_frame(frame_idx, self._last_predicted_window)

    def step(self) -> NavFrame:
        with self._lock:
            if (
                self.policy_variant == "reorganized"
                and self.policy_v2_enabled
                and self._v2_ready
            ):
                return self._navigation_step_v2()
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
                    "random": {
                        "phrase_scale": self.ctrl_phrase_scale,
                        "jump_rate": self.ctrl_jump_rate,
                        "timbre_lock": self.ctrl_timbre_lock,
                        "drift": self.ctrl_drift,
                        "repeat_avoid": self.ctrl_repeat_avoid,
                        "crossfile": self.ctrl_crossfile,
                    },
                    "reorganized": {
                        "morph_len": self.ctrl_morph_len,
                        "jump_rate": self.ctrl_reorg_jump_rate,
                        "timbre_lock": self.ctrl_reorg_timbre_lock,
                        "evolution": self.ctrl_evolution,
                        "novelty": self.ctrl_novelty,
                        "crossfile": self.ctrl_reorg_crossfile,
                    },
                    # Backward-compatible flat random controls.
                    "phrase_scale": self.ctrl_phrase_scale,
                    "jump_rate": self.ctrl_jump_rate,
                    "timbre_lock": self.ctrl_timbre_lock,
                    "drift": self.ctrl_drift,
                    "repeat_avoid": self.ctrl_repeat_avoid,
                    "crossfile": self.ctrl_crossfile,
                },
                "active_variant": str(self.policy_variant),
                "fractional": fractional_state,
                "timbre_swap": {
                    "enabled": bool(self.policy_timbre_swap_enabled),
                    "has_descriptors": bool(self.desc_weighted is not None),
                    "attempted": int(self._timbre_swap_stats["attempted"]),
                    "applied": int(self._timbre_swap_stats["applied"]),
                },
                "recompose": {
                    "enabled": bool(self.policy_recompose_enabled),
                    "active": bool(
                        self.policy_recompose_enabled
                        and self.desc_weighted is not None
                        and self._desc_tree is not None
                    ),
                    "paths_built": int(self._recompose_stats["paths_built"]),
                    "step_transitions": int(self._recompose_stats["step_transitions"]),
                    "serial_forward": int(self._recompose_stats["serial_forward"]),
                    "crossfile": int(self._recompose_stats["crossfile"]),
                    "repeat_hits": int(self._recompose_stats["repeat_hits"]),
                    "self_stall_steps": int(self._recompose_stats["self_stall_steps"]),
                    "anti_stall_rescues": int(self._recompose_stats["anti_stall_rescues"]),
                    "pool_fallbacks": int(self._recompose_stats["pool_fallbacks"]),
                    "serial_forward_ratio": float(
                        self._recompose_stats["serial_forward"]
                        / max(1, self._recompose_stats["step_transitions"])
                    ),
                    "crossfile_ratio": float(
                        self._recompose_stats["crossfile"]
                        / max(1, self._recompose_stats["step_transitions"])
                    ),
                    "repeat_ratio": float(
                        self._recompose_stats["repeat_hits"]
                        / max(1, self._recompose_stats["step_transitions"])
                    ),
                    "self_stall_ratio": float(
                        self._recompose_stats["self_stall_steps"]
                        / max(1, self._recompose_stats["step_transitions"])
                    ),
                    "phrase_frames_remaining": int(self._phrase_frames_remaining),
                    "phrase_direction": int(self._phrase_direction),
                },
                "policy_v2": {
                    "enabled": bool(self.policy_v2_enabled),
                    "ready": bool(self._v2_ready),
                    "active": bool(
                        self.policy_variant == "reorganized"
                        and self.policy_v2_enabled
                        and self._v2_ready
                    ),
                    "temperature": float(self.policy_v2_temperature),
                    "has_model": bool(self._v2_model is not None),
                    "unit_count": int(
                        0
                        if self._v2_unit_start_idx is None
                        else self._v2_unit_start_idx.shape[0]
                    ),
                    "current_unit": (
                        int(self._v2_frame_to_unit[int(self._current_index)])
                        if self._v2_ready and self._v2_frame_to_unit is not None
                        else None
                    ),
                    "unit_steps": int(self._v2_stats["unit_steps"]),
                    "unit_transitions": int(self._v2_stats["unit_transitions"]),
                    "unit_repeats": int(self._v2_stats["unit_repeats"]),
                    "crossfile": int(self._v2_stats["crossfile"]),
                    "model_scored": int(self._v2_stats["model_scored"]),
                    "fallback_scored": int(self._v2_stats["fallback_scored"]),
                    "crossfile_ratio": float(
                        self._v2_stats["crossfile"]
                        / max(1, self._v2_stats["unit_transitions"])
                    ),
                    "repeat_ratio": float(
                        self._v2_stats["unit_repeats"]
                        / max(1, self._v2_stats["unit_transitions"])
                    ),
                },
                "reorganized": {
                    "enabled": bool(self.policy_v2_enabled),
                    "ready": bool(self._v2_ready),
                    "active": bool(
                        self.policy_variant == "reorganized"
                        and self.policy_v2_enabled
                        and self._v2_ready
                    ),
                    "temperature": float(self.policy_v2_temperature),
                    "has_model": bool(self._v2_model is not None),
                    "unit_count": int(
                        0
                        if self._v2_unit_start_idx is None
                        else self._v2_unit_start_idx.shape[0]
                    ),
                    "current_unit": (
                        int(self._v2_frame_to_unit[int(self._current_index)])
                        if self._v2_ready and self._v2_frame_to_unit is not None
                        else None
                    ),
                    "unit_steps": int(self._v2_stats["unit_steps"]),
                    "unit_transitions": int(self._v2_stats["unit_transitions"]),
                    "unit_repeats": int(self._v2_stats["unit_repeats"]),
                    "crossfile": int(self._v2_stats["crossfile"]),
                    "model_scored": int(self._v2_stats["model_scored"]),
                    "fallback_scored": int(self._v2_stats["fallback_scored"]),
                    "crossfile_ratio": float(
                        self._v2_stats["crossfile"]
                        / max(1, self._v2_stats["unit_transitions"])
                    ),
                    "repeat_ratio": float(
                        self._v2_stats["unit_repeats"]
                        / max(1, self._v2_stats["unit_transitions"])
                    ),
                },
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
