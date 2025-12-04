import numpy as np, sounddevice as sd
import threading
from queue import Queue, Empty, Full
from scipy.spatial import cKDTree
from ..config import SR, LATENT_HZ, NORM_CLAMP
from ..vae.sae import decode_window
import torch

class Player:
    def __init__(self, ae, ZZ, meta, paths, Z_mean, Z_std,
                 latent_bundle_loader,
                 beta_target=0.2, jump_thresh=64, micro_jitter=0,
                 win_sec=0.2, hop_sec=0.05,
                 kernel_blend=False, kernel_k=4, kernel_sigma=None,
                 kernel_sigma_scale=1.0, kernel_target_norm=None):
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
                if self.kernel_blend:
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
