"""
Navigation engine for corpus exploration.
Handles kNN search, policy-based navigation, and cursor tracking.
Audio playback is handled separately by GrainPlayer.
"""
import numpy as np
import threading
from collections import deque
from scipy.spatial import cKDTree
from ..config import DEVICE, LATENT_HZ
from ..policy import IndexPolicy, PolicyConfig, compute_annotations
import torch


class NavigationEngine:
    """
    Navigation engine for corpus exploration.
    
    Manages cursor position, kNN search, and policy-based index selection.
    Does not handle audio playback - use GrainPlayer for that.
    
    Control parameters (7 total):
        - width (0-1): Temperature scaling for exploration breadth
        - energy (0-1): Scaling factor for displacement magnitude
        - gravity (0-1): Bias toward forward (>0.5) or backward (<0.5) movement
        - memory (0-1): Pull toward recently visited positions
        - coherence (0-1): Bias toward staying within same source file
        - exploration (0-1): Direct entropy injection for diversity
        - regime_bias (0-2): Force specific regime (0=drift, 1=turn, 2=linger)
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
        control_regime_bias: float = 0.0,
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
            control_regime_bias: Initial regime bias (0-2, 0=drift, 1=turn, 2=linger)
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
        self.policy_m = 0
        self._recent_indices = deque(maxlen=128)
        
        # Policy annotation tables
        self.desc_table = None
        self.vel_table = None
        self.regime_table = None
        self.embed_norm = None
        self.policy_checkpoint_path = policy_path
        
        # Control parameters (7 dimensions)
        self.ctrl_width = float(np.clip(control_width, 0.0, 1.0))
        self.ctrl_energy = float(np.clip(control_energy, 0.0, 1.0))
        self.ctrl_gravity = float(np.clip(control_gravity, 0.0, 1.0))
        self.ctrl_memory = float(np.clip(control_memory, 0.0, 1.0))
        self.ctrl_coherence = float(np.clip(control_coherence, 0.0, 1.0))
        self.ctrl_exploration = float(np.clip(control_exploration, 0.0, 1.0))
        self.ctrl_regime_bias = float(np.clip(control_regime_bias, 0.0, 2.0))
        
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
        self.regime_table = ann.regime.astype(np.int64)
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
        base_regime = int(self.regime_table[idx]) if self.regime_table is not None else 0
        self.policy_m = base_regime
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
        regime_bias=None,
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
            regime_bias: Force regime (0-2, 0=drift, 1=turn, 2=linger), fractional for blend
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
            if regime_bias is not None:
                self.ctrl_regime_bias = float(np.clip(regime_bias, 0.0, 2.0))
    
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
        """Build 7-dimensional control vector for policy."""
        return np.asarray([
            self.ctrl_width,
            self.ctrl_energy,
            self.ctrl_gravity,
            self.ctrl_memory,
            self.ctrl_coherence,
            self.ctrl_exploration,
            self.ctrl_regime_bias / 2.0,  # Normalize to 0-1 range
        ], dtype=np.float32)
    
    def _policy_state_tensors(self, idx_int: int):
        idx_norm = float(idx_int) / max(float(self.N - 1), 1.0)
        idx_tensor = torch.tensor([[idx_norm]], device=self.policy_device, dtype=torch.float32)
        vel_tensor = torch.tensor([[self.policy_v]], device=self.policy_device, dtype=torch.float32)
        desc_tensor = torch.tensor(self.desc_table[idx_int][None, None, :], device=self.policy_device, dtype=torch.float32)
        reg_tensor = torch.tensor([[self.policy_m]], device=self.policy_device, dtype=torch.long)
        emb_tensor = torch.tensor(self.embed_norm[idx_int][None, None, :], device=self.policy_device, dtype=torch.float32)
        return {
            "index_norm": idx_tensor,
            "velocity": vel_tensor,
            "descriptors": desc_tensor,
            "regime": reg_tensor,
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
    
    def _apply_regime_bias(self, regime_logits: torch.Tensor) -> int:
        """Apply regime bias control to influence regime selection."""
        if self.ctrl_regime_bias <= 0.0:
            # No bias, use argmax
            return int(torch.argmax(regime_logits, dim=-1).item())
        
        # Fractional regime bias allows blending between regimes
        target_regime = int(self.ctrl_regime_bias)
        blend_factor = self.ctrl_regime_bias - target_regime
        
        if blend_factor < 0.01:
            # Pure regime selection
            return min(target_regime, regime_logits.size(-1) - 1)
        
        # Blend: weighted average favoring target regime
        regime_probs = torch.softmax(regime_logits, dim=-1)
        num_regimes = regime_probs.size(-1)
        
        # Create bias toward target regime
        bias = torch.zeros_like(regime_probs)
        if target_regime < num_regimes:
            bias[target_regime] = 1.0 - blend_factor
        if target_regime + 1 < num_regimes:
            bias[target_regime + 1] = blend_factor
        
        # Blend original probs with bias
        blended_probs = 0.5 * regime_probs + 0.5 * bias
        return int(torch.argmax(blended_probs, dim=-1).item())
    
    def _policy_pick_index(self) -> int:
        """Pick next index using policy model with all control parameters."""
        if not self.policy_ready or self.policy is None:
            return self._nearest_primary(self.cursor)
        
        if self._policy_pending_idx is not None:
            self._reset_policy_state(force_idx=self._policy_pending_idx)
            self._policy_pending_idx = None
        
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
        
        delta_logits, dv_pred, regime_logits, self.policy_hidden = self.policy.step(
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
        
        # Apply regime bias
        self.policy_m = self._apply_regime_bias(regime_logits[0])
        
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
    
    def get_state(self) -> dict:
        """Get current navigation state for visualization."""
        with self._lock:
            return {
                "cursor": self.cursor.tolist(),
                "policy_index": self.policy_i,
                "policy_velocity": self.policy_v,
                "policy_regime": self.policy_m,
                "current_file_id": self._current_file_id,
                "recent_indices": list(self._recent_indices),
                "controls": {
                    "width": self.ctrl_width,
                    "energy": self.ctrl_energy,
                    "gravity": self.ctrl_gravity,
                    "memory": self.ctrl_memory,
                    "coherence": self.ctrl_coherence,
                    "exploration": self.ctrl_exploration,
                    "regime_bias": self.ctrl_regime_bias,
                },
                "grain_rate": self._grain_rate,
                "grain_jitter": self._grain_jitter,
            }


# Backward compatibility alias
Player = NavigationEngine
