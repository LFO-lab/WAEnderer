import numpy as np, sounddevice as sd
import threading
from queue import Queue, Empty, Full
from scipy.spatial import cKDTree
from ..config import SR, LATENT_HZ, NORM_CLAMP
from ..vae.sae import decode_window
from ..models.latent_ar import InferenceBuffer, ManifoldProjector, spherical_projection, adaptive_clamping
import torch

class Player:
    def __init__(self, ae, ZZ, meta, paths, Z_mean, Z_std,
                 latent_bundle_loader,
                 beta_target=0.2, jump_thresh=64, micro_jitter=0,
                 win_sec=0.2, hop_sec=0.05,
                 kernel_blend=False, kernel_k=4, kernel_sigma=None,
                 kernel_sigma_scale=1.0, kernel_target_norm=None,
                 ar_model=None, ar_context=None, ar_drive=False, ar_noise_std=0.0,
                 # Inference stabilization options
                 ar_projector=None, ar_use_projection=True,
                 ar_clamp_std=3.0, ar_reanchor_interval=0, ar_target_norm=None):
        self.ae = ae
        self.ZZ = ZZ.astype(np.float32)                  # [N_seg, D]
        self.nav_dim = int(self.ZZ.shape[1])
        print(f"[info] Navigation dimensions detected in corpus: D = {self.nav_dim}")
        self.meta = meta                                 # [N_seg, 3]
        self.paths = list(map(str, paths))
        self.Z_mean = Z_mean.astype(np.float32)
        self.Z_std  = Z_std.astype(np.float32)
        self.latent_dim = int(self.Z_mean.shape[0])
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

        self.kernel_blend = bool(kernel_blend)
        self.kernel_k = max(1, min(int(kernel_k), self.ZZ.shape[0]))
        self.kernel_sigma = float(kernel_sigma) if (kernel_sigma is not None and float(kernel_sigma) > 0.0) else None
        self.kernel_sigma_scale = max(float(kernel_sigma_scale), 1e-6)
        target_norm = None
        if kernel_target_norm is not None and float(kernel_target_norm) > 0.0:
            target_norm = float(kernel_target_norm)
        self.kernel_target_norm = target_norm
        if self.kernel_blend:
            print(
                f"[info] Kernel blending enabled (k={self.kernel_k}, sigma="
                f"{self.kernel_sigma if self.kernel_sigma is not None else 'auto'}"
                f", target_norm={self.kernel_target_norm if self.kernel_target_norm is not None else 'none'})"
            )
        self.ar_model = ar_model
        self.ar_context = int(ar_context) if ar_context is not None else None
        self.ar_drive = bool(ar_drive and self.ar_model is not None and self.ar_context is not None)
        self.ar_noise_std = max(0.0, float(ar_noise_std))
        self.ar_batch_size = 4  # Generate multiple frames at once for efficiency
        self._ar_buffer = []
        self._ar_pos = 0
        self._ar_anchor_idx = None
        self._ar_device = None
        self._Z_mean_t = None
        self._Z_std_t = None
        self._inference_buffer: InferenceBuffer = None
        
        # Inference stabilization settings
        self.ar_projector = ar_projector
        self.ar_use_projection = bool(ar_use_projection and ar_projector is not None)
        self.ar_clamp_std = float(ar_clamp_std) if ar_clamp_std is not None and ar_clamp_std > 0 else None
        self.ar_reanchor_interval = max(0, int(ar_reanchor_interval)) if ar_reanchor_interval else 0
        self.ar_target_norm = float(ar_target_norm) if ar_target_norm is not None and ar_target_norm > 0 else None
        self._ar_step_counter = 0  # For re-anchoring
        
        # Running statistics for adaptive clamping
        self._running_mean = None
        self._running_var = None
        self._running_momentum = 0.99
        
        if self.ar_drive:
            self.ar_model.eval()
            self._ar_device = next(self.ar_model.parameters()).device
            # Pre-allocate normalization tensors on device (avoid repeated CPU->GPU transfers)
            self._Z_mean_t = torch.from_numpy(self.Z_mean).to(self._ar_device)
            self._Z_std_t = torch.from_numpy(self.Z_std).to(self._ar_device)
            # Pre-allocate inference buffer for efficient generation
            self._inference_buffer = InferenceBuffer(
                context_len=self.ar_context,
                latent_dim=self.latent_dim,
                device=self._ar_device,
                dtype=torch.float32,
            )
            # Pre-allocate context tensor buffer (avoids repeated allocations)
            self._ctx_tensor = torch.zeros(
                1, self.ar_context, self.latent_dim, 
                device=self._ar_device, dtype=torch.float32
            )
            
            # Initialize running stats to training stats (normalized to N(0,1))
            self._running_mean = torch.zeros(self.latent_dim, device=self._ar_device)
            self._running_var = torch.ones(self.latent_dim, device=self._ar_device)
            
            # Move projector to device if available
            if self.ar_projector is not None:
                self.ar_projector.to(self._ar_device)
                self.ar_projector.eval()
            
            stab_info = []
            if self.ar_use_projection:
                stab_info.append("projection")
            if self.ar_clamp_std is not None:
                stab_info.append(f"clamp±{self.ar_clamp_std}σ")
            if self.ar_target_norm is not None:
                stab_info.append(f"norm→{self.ar_target_norm}")
            if self.ar_reanchor_interval > 0:
                stab_info.append(f"reanchor@{self.ar_reanchor_interval}")
            stab_str = ", ".join(stab_info) if stab_info else "none"
            
            print(
                f"[info] Autoregressive drive enabled (context={self.ar_context}, "
                f"noise_std={self.ar_noise_std}, batch_size={self.ar_batch_size}, "
                f"stabilization=[{stab_str}])"
            )

    # --- OSC cursor ---
    def set_cursor_nd(self, coords):
        n = min(len(coords), self.nav_dim)
        if n > 0:
            self.cursor[:n] = np.asarray(coords[:n], dtype=np.float32)
        if self.nav_dim > n:
            self.cursor[n:] = 0.5

    # --- latents ---
    def _load_full_latents(self, file_id: int):
        if file_id in self._latent_cache:
            return self._latent_cache[file_id]
        z = self._latent_bundle_loader(file_id, self.paths[file_id])
        self._latent_cache[file_id] = z
        return z

    def _slice_latent_window(self, file_id: int, start_lat: int, win_lat: int):
        z_full = self._load_full_latents(file_id)
        start = int(start_lat)
        end = min(z_full.shape[0], start + win_lat)
        z_win = z_full[start:end]
        if z_win.shape[0] == 0:
            z_win = np.repeat(z_full[:1], win_lat, axis=0)
        elif z_win.shape[0] < win_lat:
            pad = np.repeat(z_win[-1:], win_lat - z_win.shape[0], axis=0)
            z_win = np.concatenate([z_win, pad], axis=0)
        elif z_win.shape[0] > win_lat:
            z_win = z_win[:win_lat]
        return np.ascontiguousarray(z_win.astype(np.float32))

    def _decode_latent_window(self, z_win_np):
        z = (z_win_np - self.Z_mean) / self.Z_std
        z = np.clip(z, -NORM_CLAMP, NORM_CLAMP)
        z = z * self.Z_std + self.Z_mean
        return decode_window(self.ae, z)

    def _renormalize_window(self, z_win_np: np.ndarray, target_norm: float):
        z_norm = (z_win_np - self.Z_mean) / (self.Z_std + 1e-8)
        norms = np.linalg.norm(z_norm, axis=1, keepdims=True)
        scales = target_norm / np.maximum(norms, 1e-6)
        z_scaled = z_norm * scales
        return z_scaled * self.Z_std + self.Z_mean

    # --- navigation ---
    def _nearest_primary(self):
        _, i = self.kdt.query(self.cursor, k=1)
        return int(i)

    def _blend_latent_window(self, win_lat: int):
        dist, idx = self.kdt.query(self.cursor, k=self.kernel_k)
        idx_arr = np.atleast_1d(idx).astype(int)
        dist_arr = np.atleast_1d(dist).astype(np.float32)

        mask = np.isfinite(dist_arr)
        idx_arr = idx_arr[mask]
        dist_arr = dist_arr[mask]
        if idx_arr.size == 0:
            idx_arr = np.array([self._nearest_primary()], dtype=int)
            dist_arr = np.array([0.0], dtype=np.float32)

        sigma = self.kernel_sigma
        if sigma is None:
            positive = dist_arr[dist_arr > 1e-6]
            if positive.size == 0:
                base = float(dist_arr.max()) if dist_arr.size else 1.0
            else:
                base = float(np.median(positive))
            if not np.isfinite(base) or base <= 0.0:
                base = 1e-3
            sigma = max(base * self.kernel_sigma_scale, 1e-4)

        weights = np.exp(-(dist_arr ** 2) / (2.0 * sigma * sigma))
        if not np.isfinite(weights).all() or weights.sum() <= 0.0:
            weights = np.ones_like(dist_arr)
        weights = weights / np.maximum(weights.sum(), 1e-12)

        acc = None
        for w, j in zip(weights.tolist(), idx_arr.tolist()):
            fid, t0, _ = self.meta[j]
            z_win = self._slice_latent_window(int(fid), int(t0), win_lat)
            acc = z_win * w if acc is None else acc + z_win * w

        if acc is None:
            acc = np.zeros((win_lat, self.latent_dim), dtype=np.float32)
        else:
            acc = np.ascontiguousarray(acc.astype(np.float32))

        if self.kernel_target_norm is not None and self.kernel_target_norm > 0.0:
            acc = self._renormalize_window(acc, target_norm=self.kernel_target_norm)
        return acc

    # --- autoregressive ---
    def _reseed_ar_buffer(self, idx_primary: int, win_lat: int):
        if not self.ar_drive:
            return
        seed_len = max(win_lat, self.ar_context)
        fid, t0, _ = self.meta[idx_primary]
        start = max(0, int(t0) - seed_len // 2)
        seed = self._slice_latent_window(int(fid), start, seed_len)
        self._ar_buffer = [row.astype(np.float32) for row in seed]
        self._ar_pos = 0
        self._ar_anchor_idx = idx_primary
        self.play_file = int(fid)
        self.play_tlat = float(start)

    def _ensure_ar_ready(self, win_lat: int):
        idx_primary = self._nearest_primary()
        if not self._ar_buffer or self._ar_anchor_idx != idx_primary:
            self._reseed_ar_buffer(idx_primary, win_lat)
        return idx_primary

    def _stabilize_prediction(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Apply stabilization techniques to a normalized prediction."""
        # 1. Apply manifold projection if available
        if self.ar_use_projection and self.ar_projector is not None:
            pred_norm = self.ar_projector(pred_norm)
        
        # 2. Apply adaptive clamping based on running statistics
        if self.ar_clamp_std is not None and self._running_var is not None:
            running_std = torch.sqrt(self._running_var + 1e-8)
            pred_norm = adaptive_clamping(
                pred_norm, self._running_mean, running_std, n_std=self.ar_clamp_std
            )
        
        # 3. Apply spherical projection (norm constraint)
        if self.ar_target_norm is not None:
            pred_norm = spherical_projection(pred_norm, self.ar_target_norm, soft=True)
        
        # 4. Update running statistics (EMA)
        if self._running_mean is not None:
            with torch.no_grad():
                batch_mean = pred_norm.mean(dim=0) if pred_norm.dim() > 1 else pred_norm
                batch_var = pred_norm.var(dim=0) if pred_norm.dim() > 1 and pred_norm.size(0) > 1 else self._running_var
                self._running_mean = self._running_momentum * self._running_mean + (1 - self._running_momentum) * batch_mean
                self._running_var = self._running_momentum * self._running_var + (1 - self._running_momentum) * batch_var
        
        return pred_norm
    
    def _generate_ar_latents(self, num_needed: int):
        """Generate AR latents using batched inference for efficiency."""
        if not self.ar_drive or num_needed <= 0:
            return
        
        # Check for re-anchoring
        if self.ar_reanchor_interval > 0:
            self._ar_step_counter += num_needed
            if self._ar_step_counter >= self.ar_reanchor_interval:
                idx_primary = self._nearest_primary()
                self._reseed_ar_buffer(idx_primary, max(1, int(round(self.win_sec * LATENT_HZ))))
                self._ar_step_counter = 0
                return
        
        # Generate in batches for better throughput
        remaining = num_needed
        while remaining > 0:
            batch_size = min(remaining, self.ar_batch_size)
            
            # Prepare context from buffer
            ctx = self._ar_buffer[-self.ar_context:]
            if len(ctx) == 0:
                break
            if len(ctx) < self.ar_context:
                ctx = ctx + [ctx[-1]] * (self.ar_context - len(ctx))
            
            # Use pre-allocated tensor buffer (avoids repeated allocations)
            ctx_arr = np.stack(ctx, axis=0).astype(np.float32)
            self._ctx_tensor.copy_(torch.from_numpy(ctx_arr).unsqueeze(0))
            ctx_norm = (self._ctx_tensor - self._Z_mean_t) / (self._Z_std_t + 1e-8)
            
            with torch.inference_mode():
                # Use batched generation if model supports it
                if hasattr(self.ar_model, 'generate_batch') and batch_size > 1:
                    # Generate multiple frames at once
                    noise = self.ar_noise_std / (self.Z_std.mean() + 1e-8) if self.ar_noise_std > 0 else 0.0
                    preds_norm = self.ar_model.generate_batch(
                        ctx_norm, num_frames=batch_size, noise_std=noise
                    ).squeeze(0)  # [batch_size, D]
                    
                    # Apply stabilization to each prediction
                    stabilized_preds = []
                    for i in range(preds_norm.size(0)):
                        stab_pred = self._stabilize_prediction(preds_norm[i:i+1])
                        stabilized_preds.append(stab_pred)
                    preds_norm = torch.cat(stabilized_preds, dim=0)
                    
                    # Convert all predictions to numpy at once (single transfer)
                    preds_np = preds_norm.cpu().numpy().astype(np.float32)
                    
                    # Denormalize and add to buffer
                    for i in range(batch_size):
                        pred = preds_np[i] * self.Z_std + self.Z_mean
                        self._ar_buffer.append(np.ascontiguousarray(pred.astype(np.float32)))
                else:
                    # Single-frame fallback
                    out = self.ar_model(ctx_norm, return_aux=False, return_delta=False)
                    pred_norm = out["pred"] if isinstance(out, dict) else out
                    
                    # Apply stabilization
                    pred_norm = self._stabilize_prediction(pred_norm)
                    
                    pred_norm = pred_norm.squeeze(0).cpu().numpy().astype(np.float32)
                    pred = pred_norm * self.Z_std + self.Z_mean
                    if self.ar_noise_std > 0.0:
                        pred = pred + np.random.randn(*pred.shape).astype(np.float32) * self.ar_noise_std
                    self._ar_buffer.append(np.ascontiguousarray(pred.astype(np.float32)))
                    batch_size = 1  # Only generated one
            
            remaining -= batch_size

    def _next_ar_window(self, win_lat: int, hop_lat: int):
        self._ensure_ar_ready(win_lat)
        needed = self._ar_pos + win_lat
        if len(self._ar_buffer) < needed:
            self._generate_ar_latents(needed - len(self._ar_buffer))
        if len(self._ar_buffer) < needed:
            return self._blend_latent_window(win_lat) if self.kernel_blend else self._slice_latent_window(
                self.play_file if self.play_file is not None else 0, max(int(self.play_tlat), 0), win_lat
            )
        z_win = np.stack(self._ar_buffer[self._ar_pos : self._ar_pos + win_lat], axis=0).astype(np.float32)
        self._ar_pos += hop_lat
        if self._ar_pos > self.ar_context * 4 and len(self._ar_buffer) > (self.ar_context * 6):
            drop = min(self._ar_pos - self.ar_context * 2, len(self._ar_buffer) - win_lat)
            if drop > 0:
                self._ar_buffer = self._ar_buffer[drop:]
                self._ar_pos -= drop
        return z_win

    def _ensure_playhead_initialized(self, idx_primary, win_lat):
        fid, t0, _ = int(self.meta[idx_primary, 0]), int(self.meta[idx_primary, 1]), int(self.meta[idx_primary, 2])
        if self.play_file is None:
            self.play_file = fid
            self.play_tlat = max(0.0, float(t0) - win_lat * 0.5)

    def _advance_playhead(self, hop_lat, idx_primary, win_lat):
        target_fid, target_t = int(self.meta[idx_primary, 0]), float(self.meta[idx_primary, 1])
        if self.play_file != target_fid:
            self.play_file = target_fid

        if abs(self.play_tlat - target_t) > self.jump_thresh:
            self.play_tlat = target_t - win_lat * 0.5
        else:
            self.play_tlat = (1.0 - self.beta_target) * self.play_tlat + self.beta_target * (target_t - win_lat * 0.5)

        self.play_tlat += hop_lat
        if self.micro_jitter:
            self.play_tlat += np.random.randint(-1, 2)

        z_full = self._load_full_latents(self.play_file)
        max_start = max(0, z_full.shape[0] - win_lat)
        self.play_tlat = float(np.clip(self.play_tlat, 0, max_start))
        return self.play_file, int(round(self.play_tlat))

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
        hop_lat   = max(1, int(round(self.hop_sec * LATENT_HZ)))

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
                if self.ar_drive:
                    z_win = self._next_ar_window(win_lat, hop_lat)
                elif self.kernel_blend:
                    z_win = self._blend_latent_window(win_lat)
                else:
                    idx_primary = self._nearest_primary()
                    self._ensure_playhead_initialized(idx_primary, win_lat)
                    fid, s_lat = self._advance_playhead(hop_lat, idx_primary, win_lat)
                    z_win = self._slice_latent_window(fid, s_lat, win_lat)

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
