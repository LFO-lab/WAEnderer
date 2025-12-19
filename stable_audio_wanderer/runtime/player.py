import numpy as np, sounddevice as sd
import threading
from collections import deque
from queue import Queue, Empty, Full
from scipy.spatial import cKDTree
from ..config import SR, LATENT_HZ, NORM_CLAMP, DEVICE
from ..vae.sae import decode_window
from ..policy import IndexPolicy, PolicyConfig, compute_annotations
import torch

class Player:
    def __init__(self, ae, ZZ, meta, paths, Z_mean, Z_std,
                 latent_bundle_loader,
                 beta_target=0.2, jump_thresh=64, micro_jitter=0,
                 win_sec=0.2, hop_sec=0.05,
                 mix_radius=1, mix_momentum=0.85, mix_sigma=0.6,
                 policy_path=None, policy_temperature=1.0, policy_sample=1,
                 control_width=0.5, control_energy=0.5, control_gravity=0.5, control_memory=0.0,
                 warp_speed=1.0, warp_inertia=0.85, warp_jitter=0.0, warp_max=8.0, warp_allow_reverse=False):
        self.ae = ae
        self.ZZ = ZZ.astype(np.float32)                  # [N_seg, D]
        self.nav_dim = int(self.ZZ.shape[1])
        print(f"[info] Navigation dimensions detected in corpus: D = {self.nav_dim}")
        self.meta = meta                                 # [N_seg, 3]
        self.paths = list(map(str, paths))
        self.Z_mean = Z_mean.astype(np.float32)
        self.Z_std  = Z_std.astype(np.float32)
        self._latent_bundle_loader = latent_bundle_loader

        self.kdt = cKDTree(self.ZZ)
        self.cursor = np.full((self.nav_dim,), 0.5, dtype=np.float32)
        self._stop = False
        self._stop_event = threading.Event()
        self._audio_queue = None
        self._playback_thread = None
        self._latent_cache = {}   # file_id -> z_full [T_lat, 64]

        self.play_file = None
        self.play_tlat = 0.0
        self.beta_target = float(beta_target)
        self.jump_thresh = int(jump_thresh)
        self.micro_jitter = int(micro_jitter)

        self.win_sec = float(win_sec)
        self.hop_sec = float(hop_sec)
        self.mix_radius = max(0, int(mix_radius))
        self.mix_momentum = float(np.clip(mix_momentum, 0.0, 0.999))
        self.mix_sigma = max(1e-3, float(mix_sigma))
        self._mix_offsets = np.arange(-self.mix_radius, self.mix_radius + 1, dtype=np.int32)
        self._mix_weights = None
        self._mix_fid = None

        self.N = int(self.ZZ.shape[0])
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

        self.ctrl_width = float(np.clip(control_width, 0.0, 1.0))
        self.ctrl_energy = float(np.clip(control_energy, 0.0, 1.0))
        self.ctrl_gravity = float(np.clip(control_gravity, 0.0, 1.0))
        self.ctrl_memory = float(np.clip(control_memory, 0.0, 1.0))

        self.warp_speed = float(warp_speed)
        self.warp_inertia = float(np.clip(warp_inertia, 0.0, 0.999))
        self.warp_jitter = float(max(0.0, warp_jitter))
        self.warp_max = float(max(warp_max, 0.1))
        self.warp_allow_reverse = bool(warp_allow_reverse)
        self._warp_velocity = None

        self.desc_table = None
        self.vel_table = None
        self.regime_table = None
        self.embed_norm = None
        self.policy_checkpoint_path = policy_path
        if policy_path is not None:
            try:
                self._setup_policy(policy_path)
            except Exception as e:
                print(f"[warn] Failed to load policy '{policy_path}': {e}")

    # --- OSC cursor ---
    def set_cursor_nd(self, coords):
        n = min(len(coords), self.nav_dim)
        if n > 0:
            self.cursor[:n] = np.asarray(coords[:n], dtype=np.float32)
        if self.nav_dim > n:
            self.cursor[n:] = 0.5
        if self.policy_ready:
            idx = self._nearest_primary(self.cursor)
            self._policy_pending_idx = idx

    # --- policy ---
    def _setup_policy(self, policy_path: str):
        # Explicitly disable weights_only to allow loading full checkpoints (PyTorch 2.6+ default changed).
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
        desc_raw = np.stack([ann.speed, ann.curvature, ann.recurrence], axis=1).astype(np.float32)
        if self.policy_desc_mean is None or self.policy_desc_std is None:
            desc_norm = ann.desc
            self.policy_desc_mean = ann.desc_mean
            self.policy_desc_std = ann.desc_std
        else:
            desc_norm = (desc_raw - self.policy_desc_mean[None, :]) / (self.policy_desc_std[None, :] + 1e-6)
        if desc_norm.shape[1] != cfg.desc_dim:
            if desc_norm.shape[1] > cfg.desc_dim:
                desc_norm = desc_norm[:, : cfg.desc_dim]
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
                self.embed_norm = self.embed_norm[:, : cfg.embed_dim]
            else:
                pad = np.zeros((self.embed_norm.shape[0], cfg.embed_dim - self.embed_norm.shape[1]), dtype=np.float32)
                self.embed_norm = np.concatenate([self.embed_norm, pad], axis=1)

        self.policy_ready = True
        self._reset_policy_state(force_idx=None)

    def _reset_policy_state(self, force_idx=None):
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

    def set_policy_controls(self, width=None, energy=None, gravity=None, memory=None):
        if width is not None:
            self.ctrl_width = float(np.clip(width, 0.0, 1.0))
        if energy is not None:
            self.ctrl_energy = float(np.clip(energy, 0.0, 1.0))
        if gravity is not None:
            self.ctrl_gravity = float(np.clip(gravity, 0.0, 1.0))
        if memory is not None:
            self.ctrl_memory = float(np.clip(memory, 0.0, 1.0))

    def set_time_warp(self, speed=None, inertia=None, jitter=None, max_speed=None, allow_reverse=None):
        if speed is not None:
            self.warp_speed = float(speed)
        if inertia is not None:
            self.warp_inertia = float(np.clip(inertia, 0.0, 0.999))
        if jitter is not None:
            self.warp_jitter = float(max(0.0, jitter))
        if max_speed is not None:
            self.warp_max = float(max(0.1, max_speed))
        if allow_reverse is not None:
            self.warp_allow_reverse = bool(allow_reverse)

    def reset_time_warp(self, speed=None):
        if speed is not None:
            self.warp_speed = float(speed)
        self._warp_velocity = None

    def _control_vector(self) -> np.ndarray:
        return np.asarray([self.ctrl_width, self.ctrl_energy, self.ctrl_gravity, self.ctrl_memory], dtype=np.float32)

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

    def reset_policy(self, idx=None):
        self._reset_policy_state(force_idx=idx)

    def _delta_temperature(self) -> float:
        base = self.policy_temperature
        return float(np.clip(base + 2.0 * self.ctrl_width, 0.2, 4.0))

    def _policy_pick_index(self):
        if not self.policy_ready or self.policy is None:
            return self._nearest_primary(self.cursor)

        if self._policy_pending_idx is not None:
            self._reset_policy_state(force_idx=self._policy_pending_idx)
            self._policy_pending_idx = None

        idx_lookup = int(np.clip(round(self.policy_i), 0, max(self.N - 1, 0)))
        state = self._policy_state_tensors(idx_lookup)
        ctrl = torch.tensor(self._control_vector()[None, None, :], device=self.policy_device, dtype=torch.float32)
        if ctrl.shape[-1] != self.policy_cfg.control_dim:
            if ctrl.shape[-1] > self.policy_cfg.control_dim:
                ctrl = ctrl[:, :, : self.policy_cfg.control_dim]
            else:
                pad = torch.zeros(ctrl.size(0), ctrl.size(1), self.policy_cfg.control_dim - ctrl.shape[-1], device=ctrl.device)
                ctrl = torch.cat([ctrl, pad], dim=-1)

        delta_logits, dv_pred, regime_logits, self.policy_hidden = self.policy.step(
            state, controls=ctrl, hidden=self.policy_hidden
        )
        logits = delta_logits[0]
        logits = logits / self._delta_temperature()
        class_vals = torch.arange(-self.policy_delta_max, self.policy_delta_max + 1, device=logits.device, dtype=torch.float32)
        gravity = (self.ctrl_gravity - 0.5) * 2.0
        logits = logits + gravity * class_vals
        probs = torch.softmax(logits, dim=-1)
        if self.policy_sample:
            cls = torch.multinomial(probs, num_samples=1)
        else:
            cls = torch.argmax(probs, dim=-1, keepdim=True)
        delta_val = float(self.policy.delta_class_to_value(cls).item())
        energy_scale = 0.5 + self.ctrl_energy
        delta_val *= energy_scale
        vel_delta = float(dv_pred[0].item()) * energy_scale

        self.policy_v = self.policy_v + vel_delta
        step = delta_val + self.policy_v
        self.policy_i = self.policy_i + step

        reg_logits = regime_logits[0]
        self.policy_m = int(torch.argmax(reg_logits, dim=-1).item())

        if self.ctrl_memory > 0.0 and len(self._recent_indices) > 0:
            recent_mean = float(np.mean(self._recent_indices))
            self.policy_i = (1.0 - self.ctrl_memory) * self.policy_i + self.ctrl_memory * recent_mean

        self.policy_i = float(np.clip(self.policy_i, 0.0, float(max(self.N - 1, 0))))
        idx_out = int(np.clip(round(self.policy_i), 0, max(self.N - 1, 0)))
        self._recent_indices.append(idx_out)
        return idx_out

    # --- latents ---
    def _load_full_latents(self, file_id: int):
        if file_id in self._latent_cache:
            return self._latent_cache[file_id]
        z = self._latent_bundle_loader(file_id, self.paths[file_id])
        self._latent_cache[file_id] = z
        return z

    def _decode_latent_window(self, z_win_np):
        z = (z_win_np - self.Z_mean) / self.Z_std
        z = np.clip(z, -NORM_CLAMP, NORM_CLAMP)
        z = z * self.Z_std + self.Z_mean
        return decode_window(self.ae, z)

    def _latent_slice(self, z_full: np.ndarray, start: int, win_lat: int) -> np.ndarray:
        max_start = max(0, z_full.shape[0] - 1)
        start = int(np.clip(start, 0, max_start))
        end = min(start + win_lat, z_full.shape[0])
        z_win = z_full[start:end]
        if z_win.shape[0] == 0:
            return np.zeros((win_lat, z_full.shape[1]), dtype=z_full.dtype)
        if z_win.shape[0] < win_lat:
            pad = np.repeat(z_win[-1:], win_lat - z_win.shape[0], axis=0)
            z_win = np.concatenate([z_win, pad], axis=0)
        return z_win

    def _target_mix_weights(self, delta_lat: float) -> np.ndarray:
        if self.mix_radius == 0:
            return np.ones((1,), dtype=np.float32)
        shift = float(np.clip(delta_lat, -self.mix_radius, self.mix_radius))
        offsets = self._mix_offsets.astype(np.float32)
        # Narrow Gaussian keeps the convex hull tight around the center.
        scale = max(self.mix_sigma * max(self.mix_radius, 1), 1e-3)
        dist = (offsets - shift) / scale
        weights = np.exp(-0.5 * dist * dist).astype(np.float32)
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum()
        return weights

    def _update_mix_weights(self, fid: int, target_w: np.ndarray, force_reset: bool) -> np.ndarray:
        reset = force_reset or self._mix_weights is None or self._mix_fid != fid or len(target_w) != len(self._mix_offsets)
        if reset:
            self._mix_weights = target_w
        else:
            mom = self.mix_momentum
            self._mix_weights = mom * self._mix_weights + (1.0 - mom) * target_w
            total = float(self._mix_weights.sum())
            if total > 1e-6:
                self._mix_weights = self._mix_weights / total
        self._mix_fid = fid
        return self._mix_weights

    def _barycentric_window(self, z_full: np.ndarray, base_start: int, win_lat: int, target_t: float, force_reset: bool):
        if self.mix_radius == 0:
            return self._latent_slice(z_full, base_start, win_lat)

        center = float(base_start) + 0.5 * float(win_lat)
        delta = float(target_t) - center
        target_weights = self._target_mix_weights(delta)
        weights = self._update_mix_weights(self.play_file, target_weights, force_reset=force_reset)

        z_mix = np.zeros((win_lat, z_full.shape[1]), dtype=z_full.dtype)
        max_start = max(0, z_full.shape[0] - win_lat)
        for offset, w in zip(self._mix_offsets, weights):
            start = int(np.clip(base_start + int(offset), 0, max_start))
            z_win = self._latent_slice(z_full, start, win_lat)
            z_mix += float(w) * z_win
        return z_mix

    def _time_warp_step(self, base_hop: float) -> float:
        """
        Smooth latent hop size with optional jitter to implement time warping
        (reparameterization) without inventing new latent states.
        """
        base_hop = max(float(base_hop), 1e-6)
        target = base_hop * self.warp_speed
        if not self.warp_allow_reverse and target < 0.0:
            target = 0.0
        if self._warp_velocity is None:
            self._warp_velocity = target
        alpha = self.warp_inertia
        self._warp_velocity = alpha * float(self._warp_velocity) + (1.0 - alpha) * target
        if self.warp_jitter > 0.0:
            self._warp_velocity += np.random.randn() * self.warp_jitter
        vmax = max(self.warp_max * base_hop, 1e-6)
        self._warp_velocity = float(np.clip(self._warp_velocity, -vmax, vmax))
        if not self.warp_allow_reverse:
            self._warp_velocity = max(self._warp_velocity, 0.0)
        return float(self._warp_velocity)

    # --- navigation ---
    def _nearest_primary(self, coords=None):
        query = self.cursor if coords is None else coords
        _, i = self.kdt.query(query, k=1)
        return int(i)

    def _ensure_playhead_initialized(self, idx_primary, win_lat):
        fid, t0, _ = int(self.meta[idx_primary, 0]), int(self.meta[idx_primary, 1]), int(self.meta[idx_primary, 2])
        if self.play_file is None:
            self.play_file = fid
            self.play_tlat = max(0.0, float(t0) - win_lat * 0.5)

    def _advance_playhead(self, hop_lat, idx_primary, win_lat):
        hop_lat = float(hop_lat)
        target_fid, target_t = int(self.meta[idx_primary, 0]), float(self.meta[idx_primary, 1])
        if self.play_file != target_fid:
            self.play_file = target_fid

        jumped = False
        if abs(self.play_tlat - target_t) > self.jump_thresh:
            self.play_tlat = target_t - win_lat * 0.5
            jumped = True
        else:
            self.play_tlat = (1.0 - self.beta_target) * self.play_tlat + self.beta_target * (target_t - win_lat * 0.5)

        self.play_tlat += hop_lat
        if self.micro_jitter:
            self.play_tlat += np.random.randint(-1, 2)

        z_full = self._load_full_latents(self.play_file)
        max_start = max(0, z_full.shape[0] - win_lat)
        self.play_tlat = float(np.clip(self.play_tlat, 0, max_start))
        return self.play_file, int(round(self.play_tlat)), jumped

    def _enqueue_audio_chunk(self, a0, a1):
        if self._audio_queue is None:
            return False
        while not self._stop_event.is_set():
            try:
                self._audio_queue.put((a0, a1), timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _request_playback_stop(self):
        if self._audio_queue is None:
            return
        while True:
            try:
                self._audio_queue.put(None, timeout=0.1)
                break
            except Full:
                if self._playback_thread is None or not self._playback_thread.is_alive():
                    break
                continue

    # --- audio loop ---
    def run(self):
        self._stop = False
        self._stop_event.clear()
        win_lat   = max(1, int(round(self.win_sec * LATENT_HZ)))
        hop_samps = int(round(self.hop_sec * SR))
        hop_lat_base = max(self.hop_sec * LATENT_HZ, 1e-6)

        fade = np.ascontiguousarray(np.hanning(2 * hop_samps).astype(np.float32))
        fade_in, fade_out = fade[:hop_samps], fade[hop_samps:]
        self._audio_queue = Queue(maxsize=8)

        def playback_loop():
            stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32', blocksize=hop_samps)
            stream.start()
            carry = np.ascontiguousarray(np.zeros((hop_samps, 2), np.float32))
            try:
                while True:
                    try:
                        item = self._audio_queue.get(timeout=0.1)
                    except Empty:
                        if self._stop_event.is_set():
                            break
                        continue
                    if item is None:
                        break
                    a0, a1 = item
                    out = carry * fade_out[:, None] + a0 * fade_in[:, None]
                    out = np.ascontiguousarray(out, dtype=np.float32)
                    stream.write(out)
                    carry = np.ascontiguousarray(a1, dtype=np.float32)
            finally:
                stream.stop()
                stream.close()

        self._playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self._playback_thread.start()

        try:
            while not self._stop_event.is_set():
                hop_lat = self._time_warp_step(hop_lat_base)
                idx_primary = self._policy_pick_index() if self.policy_ready else self._nearest_primary()
                self._ensure_playhead_initialized(idx_primary, win_lat)
                target_t = float(self.meta[idx_primary, 1])
                fid, s_lat, jumped = self._advance_playhead(hop_lat, idx_primary, win_lat)

                z_full = self._load_full_latents(fid)
                z_win = self._barycentric_window(z_full, s_lat, win_lat, target_t, force_reset=jumped)

                audio = self._decode_latent_window(z_win)
                a0 = audio[:hop_samps]
                a1 = audio[hop_samps:2 * hop_samps] if audio.shape[0] >= 2 * hop_samps else np.zeros_like(a0)
                a0 = np.ascontiguousarray(a0, dtype=np.float32)
                a1 = np.ascontiguousarray(a1, dtype=np.float32)
                if not self._enqueue_audio_chunk(a0, a1):
                    break
        finally:
            self._stop_event.set()
            self._request_playback_stop()
            if self._playback_thread is not None:
                self._playback_thread.join()
            self._audio_queue = None
            self._playback_thread = None

    def stop(self):
        self._stop = True
        self._stop_event.set()
        self._request_playback_stop()
