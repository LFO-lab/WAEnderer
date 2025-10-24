import numpy as np, sounddevice as sd
from scipy.spatial import cKDTree
from ..config import SR, LATENT_HZ, NORM_CLAMP
from ..vae.sae import decode_window
import torch

class Player:
    def __init__(self, ae, ZZ, meta, paths, Z_mean, Z_std,
                 latent_bundle_loader,
                 beta_target=0.2, jump_thresh=64, micro_jitter=0,
                 win_sec=0.2, hop_sec=0.05):
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
        self._latent_cache = {}   # file_id -> z_full [T_lat, 64]

        self.play_file = None
        self.play_tlat = 0.0
        self.beta_target = float(beta_target)
        self.jump_thresh = int(jump_thresh)
        self.micro_jitter = int(micro_jitter)

        self.win_sec = float(win_sec)
        self.hop_sec = float(hop_sec)

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

    def _decode_latent_window(self, z_win_np):
        z = (z_win_np - self.Z_mean) / self.Z_std
        z = np.clip(z, -NORM_CLAMP, NORM_CLAMP)
        z = z * self.Z_std + self.Z_mean
        return decode_window(self.ae, z)

    # --- navigation ---
    def _nearest_primary(self):
        _, i = self.kdt.query(self.cursor, k=1)
        return int(i)

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

    # --- audio loop ---
    def run(self):
        win_lat   = max(1, int(round(self.win_sec * LATENT_HZ)))
        hop_samps = int(round(self.hop_sec * SR))
        hop_lat   = max(1, int(round(self.hop_sec * LATENT_HZ)))

        fade = np.ascontiguousarray(np.hanning(2 * hop_samps).astype(np.float32))
        fade_in, fade_out = fade[:hop_samps], fade[hop_samps:]

        stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32', blocksize=hop_samps)
        stream.start()
        carry = np.ascontiguousarray(np.zeros((hop_samps, 2), np.float32))

        try:
            while not self._stop:
                idx_primary = self._nearest_primary()
                self._ensure_playhead_initialized(idx_primary, win_lat)
                fid, s_lat = self._advance_playhead(hop_lat, idx_primary, win_lat)

                z_full = self._load_full_latents(fid)
                e_lat = min(z_full.shape[0], s_lat + win_lat)
                z_win = z_full[s_lat:e_lat]
                if z_win.shape[0] < win_lat:
                    pad = np.repeat(z_win[-1:], win_lat - z_win.shape[0], axis=0)
                    z_win = np.concatenate([z_win, pad], axis=0)

                audio = self._decode_latent_window(z_win)
                a0 = audio[:hop_samps]
                a1 = audio[hop_samps:2 * hop_samps] if audio.shape[0] >= 2 * hop_samps else np.zeros_like(a0)

                out = carry * fade_out[:, None] + a0 * fade_in[:, None]
                out = np.ascontiguousarray(out, dtype=np.float32)
                stream.write(out)
                carry = np.ascontiguousarray(a1, dtype=np.float32)
        finally:
            stream.stop(); stream.close()

    def stop(self):
        self._stop = True
