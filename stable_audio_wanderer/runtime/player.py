"""
Navigation engine for corpus exploration.
Handles kNN search, policy-based navigation, and cursor tracking.
Audio playback is handled separately by GrainPlayer.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before importing faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import threading
from collections import deque
from typing import Optional, Tuple, Dict
from scipy.spatial import cKDTree
from ..config import DEVICE, LATENT_HZ
from ..policy import IndexPolicy, PolicyConfig, compute_annotations
from ..policy.latent_geometry import LatentGeometry, load_geometry_from_dict
from ..policy.latent_policy import LatentPolicy, LatentPolicyConfig, build_local_features
import torch


class NavigationEngine:
    """
    Navigation engine for corpus exploration.

    Manages cursor position, kNN search, and policy-based index selection.
    Does not handle audio playback - use GrainPlayer for that.

    Control parameters (6 total):
        - width (0-1): Temperature scaling for exploration breadth
        - energy (0-1): Scaling factor for displacement magnitude
        - gravity (0-1): Bias toward forward (>0.5) or backward (<0.5) movement
        - memory (0-1): Pull toward recently visited positions
        - coherence (0-1): Bias toward staying within same source file
        - exploration (0-1): Direct entropy injection for diversity
    """
    
    def __init__(
        self,
        ZZ: np.ndarray,
        meta: np.ndarray,
        policy_path: str = None,
        policy_temperature: float = 1.0,
        policy_sample: bool = True,
        control_width: float = 0.5,
        control_energy: float = 0.5,
        control_gravity: float = 0.5,
        control_memory: float = 0.0,
        control_coherence: float = 0.0,
        control_exploration: float = 0.0,
        grain_rate: float = 21.5,  # Default to LATENT_HZ
        grain_jitter: float = 0.0,
    ):
        """
        Initialize navigation engine.

        Args:
            ZZ: Corpus embeddings [N_seg, D]
            meta: Segment metadata [N_seg, 3] with (file_id, t_lat, win_lat)
            policy_path: Path to trained policy checkpoint (optional)
            policy_temperature: Base temperature for policy sampling
            policy_sample: Whether to sample from policy (True) or use argmax (False)
            control_width: Initial width control (0-1)
            control_energy: Initial energy control (0-1)
            control_gravity: Initial gravity control (0-1)
            control_memory: Initial memory control (0-1)
            control_coherence: Initial coherence control (0-1)
            control_exploration: Initial exploration control (0-1)
            grain_rate: Initial grain trigger rate (grains/sec)
            grain_jitter: Initial timing jitter (0-1)
        """
        self.ZZ = ZZ.astype(np.float32)
        self.nav_dim = int(self.ZZ.shape[1])
        print(f"[info] Navigation dimensions detected in corpus: D = {self.nav_dim}")
        self.meta = meta
        self.N = int(self.ZZ.shape[0])
        
        # Build file_id index for coherence control
        self._file_ids = meta[:, 0].astype(np.int32) if meta.shape[1] > 0 else np.zeros(self.N, dtype=np.int32)
        self._file_to_indices = {}
        for idx, fid in enumerate(self._file_ids):
            self._file_to_indices.setdefault(int(fid), []).append(idx)
        
        # kNN search tree
        self.kdt = cKDTree(self.ZZ)
        self.cursor = np.full((self.nav_dim,), 0.5, dtype=np.float32)
        
        # Policy state
        self.policy = None
        self.policy_cfg = None
        self.policy_hidden = None
        self.policy_device = torch.device(DEVICE)
        self.policy_ready = False
        self.policy_delta_max = None
        self.policy_desc_mean = None
        self.policy_desc_std = None
        self.policy_embed_min = None
        self.policy_embed_range = None
        self.policy_temperature = max(1e-3, float(policy_temperature))
        self.policy_sample = bool(policy_sample)
        self._policy_pending_idx = None
        self.policy_i = 0.0
        self.policy_v = 0.0
        self._recent_indices = deque(maxlen=128)

        # Policy annotation tables
        self.desc_table = None
        self.vel_table = None
        self.embed_norm = None
        self.policy_checkpoint_path = policy_path

        # Control parameters (6 dimensions)
        self.ctrl_width = float(np.clip(control_width, 0.0, 1.0))
        self.ctrl_energy = float(np.clip(control_energy, 0.0, 1.0))
        self.ctrl_gravity = float(np.clip(control_gravity, 0.0, 1.0))
        self.ctrl_memory = float(np.clip(control_memory, 0.0, 1.0))
        self.ctrl_coherence = float(np.clip(control_coherence, 0.0, 1.0))
        self.ctrl_exploration = float(np.clip(control_exploration, 0.0, 1.0))
        
        # Current file for coherence tracking
        self._current_file_id = 0
        
        # Grain rate control (minimum is LATENT_HZ to match encoder rate)
        self._grain_rate = float(max(LATENT_HZ, grain_rate))
        self._grain_jitter = float(np.clip(grain_jitter, 0.0, 1.0))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load policy if specified
        if policy_path is not None:
            try:
                self._setup_policy(policy_path)
            except Exception as e:
                print(f"[warn] Failed to load policy '{policy_path}': {e}")
    
    # --- Cursor control ---
    def set_cursor_nd(self, coords):
        """Set navigation cursor position (coordinates in [0, 1])."""
        with self._lock:
            n = min(len(coords), self.nav_dim)
            if n > 0:
                self.cursor[:n] = np.asarray(coords[:n], dtype=np.float32)
            if self.nav_dim > n:
                self.cursor[n:] = 0.5
            if self.policy_ready:
                idx = self._nearest_primary(self.cursor)
                self._policy_pending_idx = idx
    
    # --- Policy setup ---
    def _setup_policy(self, policy_path: str):
        """Load and initialize policy model."""
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg = PolicyConfig(**cfg_dict) if isinstance(cfg_dict, dict) else PolicyConfig()
        self.policy_cfg = cfg
        self.policy_delta_max = int(ckpt.get("delta_max", cfg.delta_max))
        
        def _safe_array(x):
            return None if x is None else np.asarray(x, dtype=np.float32)
        
        self.policy_desc_mean = _safe_array(ckpt.get("desc_mean"))
        self.policy_desc_std = _safe_array(ckpt.get("desc_std"))
        self.policy_embed_min = _safe_array(ckpt.get("embed_min"))
        self.policy_embed_range = _safe_array(ckpt.get("embed_range"))
        
        self.policy = IndexPolicy(cfg).to(self.policy_device)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()
        
        ann = compute_annotations(self.meta, self.ZZ)
        # Use all available descriptors from annotations
        desc_raw = ann.desc.copy()
        
        if self.policy_desc_mean is None or self.policy_desc_std is None:
            desc_norm = ann.desc
            self.policy_desc_mean = ann.desc_mean
            self.policy_desc_std = ann.desc_std
        else:
            # Normalize using checkpoint stats, handling dimension mismatch
            if desc_raw.shape[1] == self.policy_desc_mean.shape[0]:
                desc_norm = (desc_raw - self.policy_desc_mean[None, :]) / (self.policy_desc_std[None, :] + 1e-6)
            else:
                # Fallback: use raw annotations normalized
                desc_norm = ann.desc
                
        if desc_norm.shape[1] != cfg.desc_dim:
            if desc_norm.shape[1] > cfg.desc_dim:
                desc_norm = desc_norm[:, :cfg.desc_dim]
            else:
                pad = np.zeros((desc_norm.shape[0], cfg.desc_dim - desc_norm.shape[1]), dtype=np.float32)
                desc_norm = np.concatenate([desc_norm, pad], axis=1)
        
        embed_min = self.policy_embed_min if self.policy_embed_min is not None else ann.embed_min
        embed_range = self.policy_embed_range if self.policy_embed_range is not None else ann.embed_range
        
        self.desc_table = np.ascontiguousarray(desc_norm.astype(np.float32))
        self.vel_table = ann.velocities.astype(np.float32)
        self.embed_norm = (self.ZZ - embed_min[None, :]) / embed_range[None, :]
        self.embed_norm = np.clip(self.embed_norm, 0.0, 1.0).astype(np.float32)
        if self.embed_norm.shape[1] != cfg.embed_dim:
            if self.embed_norm.shape[1] > cfg.embed_dim:
                self.embed_norm = self.embed_norm[:, :cfg.embed_dim]
            else:
                pad = np.zeros((self.embed_norm.shape[0], cfg.embed_dim - self.embed_norm.shape[1]), dtype=np.float32)
                self.embed_norm = np.concatenate([self.embed_norm, pad], axis=1)
        
        self.policy_ready = True
        self._reset_policy_state(force_idx=None)
    
    def _reset_policy_state(self, force_idx=None):
        """Reset policy state to initial conditions."""
        if not self.policy_ready:
            return
        if force_idx is None:
            idx = self._nearest_primary(self.cursor)
        else:
            idx = int(np.clip(force_idx, 0, max(self.N - 1, 0)))
        self.policy_i = float(idx)
        self.policy_v = 0.0
        self.policy_hidden = None
        self._recent_indices.clear()
        self._recent_indices.append(idx)
        self._current_file_id = int(self._file_ids[idx])
    
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
        """Reset policy state."""
        with self._lock:
            self._reset_policy_state(force_idx=idx)
    
    # --- Grain rate control ---
    def set_grain_rate(self, rate: float):
        """Set grain trigger rate (minimum LATENT_HZ = 21.5 Hz)."""
        with self._lock:
            self._grain_rate = float(max(LATENT_HZ, rate))
    
    def set_grain_jitter(self, jitter: float):
        """Set grain timing jitter (0-1)."""
        with self._lock:
            self._grain_jitter = float(np.clip(jitter, 0.0, 1.0))
    
    @property
    def grain_rate(self) -> float:
        """Current grain trigger rate."""
        return self._grain_rate
    
    @property
    def grain_jitter(self) -> float:
        """Current grain timing jitter."""
        return self._grain_jitter
    
    def get_trigger_interval(self) -> float:
        """Get time between grain triggers, with optional jitter."""
        base_interval = 1.0 / self._grain_rate
        if self._grain_jitter > 0:
            jitter = (np.random.random() - 0.5) * 2 * self._grain_jitter * base_interval
            return max(0.01, base_interval + jitter)
        return base_interval
    
    # --- Policy inference ---
    def _control_vector(self) -> np.ndarray:
        """Build 6-dimensional control vector for policy."""
        return np.asarray([
            self.ctrl_width,
            self.ctrl_energy,
            self.ctrl_gravity,
            self.ctrl_memory,
            self.ctrl_coherence,
            self.ctrl_exploration,
        ], dtype=np.float32)
    
    def _policy_state_tensors(self, idx_int: int):
        idx_norm = float(idx_int) / max(float(self.N - 1), 1.0)
        idx_tensor = torch.tensor([[idx_norm]], device=self.policy_device, dtype=torch.float32)
        vel_tensor = torch.tensor([[self.policy_v]], device=self.policy_device, dtype=torch.float32)
        desc_tensor = torch.tensor(self.desc_table[idx_int][None, None, :], device=self.policy_device, dtype=torch.float32)
        emb_tensor = torch.tensor(self.embed_norm[idx_int][None, None, :], device=self.policy_device, dtype=torch.float32)
        return {
            "index_norm": idx_tensor,
            "velocity": vel_tensor,
            "descriptors": desc_tensor,
            "embedding": emb_tensor,
        }

    def _policy_state_tensors_interpolated(self, idx_lower: int, idx_upper: int, frac: float):
        """Build interpolated policy state tensors for smoother policy input.

        Args:
            idx_lower: Floor index
            idx_upper: Ceil index
            frac: Fractional interpolation weight (0-1)

        Returns:
            Dict of policy state tensors with interpolated values
        """
        # Interpolate normalized index
        idx_norm_lower = float(idx_lower) / max(float(self.N - 1), 1.0)
        idx_norm_upper = float(idx_upper) / max(float(self.N - 1), 1.0)
        idx_norm_interp = (1.0 - frac) * idx_norm_lower + frac * idx_norm_upper

        # Interpolate descriptors
        desc_lower = self.desc_table[idx_lower]
        desc_upper = self.desc_table[idx_upper]
        desc_interp = (1.0 - frac) * desc_lower + frac * desc_upper

        # Interpolate embeddings
        emb_lower = self.embed_norm[idx_lower]
        emb_upper = self.embed_norm[idx_upper]
        emb_interp = (1.0 - frac) * emb_lower + frac * emb_upper

        idx_tensor = torch.tensor([[idx_norm_interp]], device=self.policy_device, dtype=torch.float32)
        vel_tensor = torch.tensor([[self.policy_v]], device=self.policy_device, dtype=torch.float32)
        desc_tensor = torch.tensor(desc_interp[None, None, :], device=self.policy_device, dtype=torch.float32)
        emb_tensor = torch.tensor(emb_interp[None, None, :], device=self.policy_device, dtype=torch.float32)

        return {
            "index_norm": idx_tensor,
            "velocity": vel_tensor,
            "descriptors": desc_tensor,
            "embedding": emb_tensor,
        }
    
    def _delta_temperature(self) -> float:
        """Compute effective temperature for delta sampling."""
        base = self.policy_temperature
        # Width increases temperature (more exploration)
        temp = base + 2.0 * self.ctrl_width
        # Exploration adds additional randomness
        temp += self.ctrl_exploration * 2.0
        return float(np.clip(temp, 0.2, 6.0))
    
    def _apply_coherence_bias(self, idx_out: int) -> int:
        """Apply coherence control to bias toward same-file indices."""
        if self.ctrl_coherence <= 0.0:
            return idx_out

        # Get indices in current file
        current_file_indices = self._file_to_indices.get(self._current_file_id, [])
        if len(current_file_indices) <= 1:
            return idx_out

        # With probability proportional to coherence, constrain to same file
        if np.random.random() < self.ctrl_coherence:
            # Find nearest index in same file
            target_pos = self.ZZ[idx_out]
            same_file_positions = self.ZZ[current_file_indices]
            dists = np.linalg.norm(same_file_positions - target_pos, axis=1)
            best_local_idx = current_file_indices[np.argmin(dists)]
            return best_local_idx

        return idx_out
    
    def _policy_pick_index(self) -> int:
        """Pick next index using policy model with all control parameters."""
        if not self.policy_ready or self.policy is None:
            return self._nearest_primary(self.cursor)

        if self._policy_pending_idx is not None:
            self._reset_policy_state(force_idx=self._policy_pending_idx)
            self._policy_pending_idx = None

        # Use interpolated state tensors for smoother policy input
        idx_lower = int(np.floor(self.policy_i))
        idx_upper = int(np.ceil(self.policy_i))
        idx_lower = max(0, min(idx_lower, self.N - 1))
        idx_upper = max(0, min(idx_upper, self.N - 1))
        frac = self.policy_i - idx_lower if idx_lower != idx_upper else 0.0

        if frac > 0.0 and idx_lower != idx_upper:
            state = self._policy_state_tensors_interpolated(idx_lower, idx_upper, frac)
        else:
            idx_lookup = int(np.clip(round(self.policy_i), 0, max(self.N - 1, 0)))
            state = self._policy_state_tensors(idx_lookup)
        ctrl = torch.tensor(self._control_vector()[None, None, :], device=self.policy_device, dtype=torch.float32)
        
        # Handle control dimension mismatch
        if ctrl.shape[-1] != self.policy_cfg.control_dim:
            if ctrl.shape[-1] > self.policy_cfg.control_dim:
                ctrl = ctrl[:, :, :self.policy_cfg.control_dim]
            else:
                pad = torch.zeros(ctrl.size(0), ctrl.size(1), self.policy_cfg.control_dim - ctrl.shape[-1], device=ctrl.device)
                ctrl = torch.cat([ctrl, pad], dim=-1)
        
        delta_logits, dv_pred, self.policy_hidden = self.policy.step(
            state, controls=ctrl, hidden=self.policy_hidden
        )
        
        # Apply temperature and exploration
        logits = delta_logits[0]
        logits = logits / self._delta_temperature()
        
        # Apply gravity bias
        class_vals = torch.arange(-self.policy_delta_max, self.policy_delta_max + 1, device=logits.device, dtype=torch.float32)
        gravity = (self.ctrl_gravity - 0.5) * 2.0
        logits = logits + gravity * class_vals
        
        # Sample or argmax
        probs = torch.softmax(logits, dim=-1)
        if self.policy_sample or self.ctrl_exploration > 0.0:
            cls = torch.multinomial(probs, num_samples=1)
        else:
            cls = torch.argmax(probs, dim=-1, keepdim=True)
        
        delta_val = float(self.policy.delta_class_to_value(cls).item())
        
        # Apply energy scaling
        energy_scale = 0.5 + self.ctrl_energy
        delta_val *= energy_scale
        vel_delta = float(dv_pred[0].item()) * energy_scale
        
        # Update policy state
        self.policy_v = self.policy_v + vel_delta
        step = delta_val + self.policy_v
        self.policy_i = self.policy_i + step

        # Apply memory control
        if self.ctrl_memory > 0.0 and len(self._recent_indices) > 0:
            recent_mean = float(np.mean(self._recent_indices))
            self.policy_i = (1.0 - self.ctrl_memory) * self.policy_i + self.ctrl_memory * recent_mean
        
        # Clamp to valid range
        self.policy_i = float(np.clip(self.policy_i, 0.0, float(max(self.N - 1, 0))))
        idx_out = int(np.clip(round(self.policy_i), 0, max(self.N - 1, 0)))
        
        # Apply coherence control
        idx_out = self._apply_coherence_bias(idx_out)
        
        # Update state
        self._recent_indices.append(idx_out)
        self._current_file_id = int(self._file_ids[idx_out])
        
        return idx_out
    
    # --- Navigation ---
    def _nearest_primary(self, coords=None) -> int:
        """Find nearest segment index to cursor position."""
        query = self.cursor if coords is None else coords
        _, i = self.kdt.query(query, k=1)
        return int(i)
    
    def pick_next_index(self) -> int:
        """
        Pick next segment index based on cursor position and policy.
        
        Returns:
            Segment index (0 to N-1)
        """
        with self._lock:
            if self.policy_ready:
                return self._policy_pick_index()
            else:
                return self._nearest_primary()
    
    def get_segment_info(self, segment_idx: int) -> dict:
        """
        Get metadata for a segment.
        
        Returns:
            Dict with file_id, t_lat, win_lat
        """
        if segment_idx < 0 or segment_idx >= len(self.meta):
            return {"file_id": 0, "t_lat": 0, "win_lat": 0}
        return {
            "file_id": int(self.meta[segment_idx, 0]),
            "t_lat": int(self.meta[segment_idx, 1]),
            "win_lat": int(self.meta[segment_idx, 2]),
        }
    
    def get_fractional_state(self) -> dict:
        """Get fractional interpolation state for smooth transitions.

        Returns:
            Dict with:
                - idx_lower: Floor index
                - idx_upper: Ceil index
                - frac: Fractional part (0-1)
                - same_file: Whether both indices are in the same file
                - file_id_lower: File ID for lower index
                - file_id_upper: File ID for upper index
        """
        with self._lock:
            idx_lower = int(np.floor(self.policy_i))
            idx_upper = int(np.ceil(self.policy_i))
            idx_lower = max(0, min(idx_lower, self.N - 1))
            idx_upper = max(0, min(idx_upper, self.N - 1))
            frac = self.policy_i - idx_lower if idx_lower != idx_upper else 0.0

            file_id_lower = int(self._file_ids[idx_lower])
            file_id_upper = int(self._file_ids[idx_upper])

            return {
                "idx_lower": idx_lower,
                "idx_upper": idx_upper,
                "frac": float(frac),
                "same_file": file_id_lower == file_id_upper,
                "file_id_lower": file_id_lower,
                "file_id_upper": file_id_upper,
            }

    def get_state(self) -> dict:
        """Get current navigation state for visualization."""
        with self._lock:
            # Compute fractional state inline to avoid re-acquiring lock
            idx_lower = int(np.floor(self.policy_i))
            idx_upper = int(np.ceil(self.policy_i))
            idx_lower = max(0, min(idx_lower, self.N - 1))
            idx_upper = max(0, min(idx_upper, self.N - 1))
            frac = self.policy_i - idx_lower if idx_lower != idx_upper else 0.0
            file_id_lower = int(self._file_ids[idx_lower])
            file_id_upper = int(self._file_ids[idx_upper])

            return {
                "cursor": self.cursor.tolist(),
                "policy_index": self.policy_i,
                "policy_velocity": self.policy_v,
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
                "grain_rate": self._grain_rate,
                "grain_jitter": self._grain_jitter,
                "fractional": {
                    "idx_lower": idx_lower,
                    "idx_upper": idx_upper,
                    "frac": float(frac),
                    "same_file": file_id_lower == file_id_upper,
                    "file_id_lower": file_id_lower,
                    "file_id_upper": file_id_upper,
                },
            }


class LatentNavigationEngine:
    """
    64D latent space navigation engine for corpus exploration.

    Maintains continuous position z in 64D embedding space.
    Uses FAISS for fast kNN search and Gaussian kernel weights for sampling.

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
        grain_rate: float = 21.5,
        grain_jitter: float = 0.0,
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
            grain_rate: Initial grain trigger rate (grains/sec)
            grain_jitter: Initial timing jitter (0-1)
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

        # Grain rate control
        self._grain_rate = float(max(LATENT_HZ, grain_rate))
        self._grain_jitter = float(np.clip(grain_jitter, 0.0, 1.0))

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
                pca_pinv = np.linalg.pinv(self.geometry.pca_components)  # [64, 2]
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

    # --- Grain rate control ---
    def set_grain_rate(self, rate: float):
        """Set grain trigger rate (minimum LATENT_HZ = 21.5 Hz)."""
        with self._lock:
            self._grain_rate = float(max(LATENT_HZ, rate))

    def set_grain_jitter(self, jitter: float):
        """Set grain timing jitter (0-1)."""
        with self._lock:
            self._grain_jitter = float(np.clip(jitter, 0.0, 1.0))

    @property
    def grain_rate(self) -> float:
        return self._grain_rate

    @property
    def grain_jitter(self) -> float:
        return self._grain_jitter

    def get_trigger_interval(self) -> float:
        """Get time between grain triggers, with optional jitter."""
        base_interval = 1.0 / self._grain_rate
        if self._grain_jitter > 0:
            jitter = (np.random.random() - 0.5) * 2 * self._grain_jitter * base_interval
            return max(0.01, base_interval + jitter)
        return base_interval

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
    def _navigation_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute one navigation step using policy or stochastic dynamics.

        Returns:
            (indices, weights, times, file_ids) for K neighbors
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

        # Re-query kNN at new position for render weights
        indices, distances = self._query_knn(self.z, k=k)

        # Compute Gaussian weights
        weights = self._compute_gaussian_weights(distances, local_sigma)

        # Get time and file info
        times = self.geometry.t_lat[indices].astype(np.float32)
        file_ids = self.geometry.file_ids[indices].astype(np.int32)

        # Track state
        self._recent_indices.append(int(indices[0]))
        self._current_file_id = int(file_ids[0])

        return indices, weights, times, file_ids

    # --- Main API ---
    def pick_next_index(self) -> int:
        """
        Pick next segment index based on stochastic kNN sampling.

        Returns:
            Segment index (0 to N-1)
        """
        with self._lock:
            indices, weights, _, _ = self._navigation_step()
            # Stochastic sample from neighbors
            idx = np.random.choice(indices, p=weights)
            return int(idx)

    def get_render_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get kNN neighbors with sampling weights for render stage.

        Returns:
            (indices, weights, times, file_ids) arrays of shape [K]
        """
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
                "grain_rate": self._grain_rate,
                "grain_jitter": self._grain_jitter,
                "fractional": fractional_state,
                "latent": {
                    "z_norm": float(np.linalg.norm(self.z)),
                    "v_norm": float(np.linalg.norm(self.v)),
                    "position_2d": pos_2d.tolist(),  # Raw 2D projection
                    "trajectory_2d": trajectory_2d_raw,  # Raw 2D trajectory for visualization
                },
            }

    # Read-only properties for compatibility
    @property
    def ZZ(self) -> np.ndarray:
        """2D projection of latent positions for visualization compatibility."""
        return self.geometry.project_to_2d(self.GG)

    @property
    def nav_dim(self) -> int:
        return 2  # Visualization is always 2D

    @property
    def _file_ids(self) -> np.ndarray:
        return self.geometry.file_ids


# Backward compatibility alias
Player = NavigationEngine
