# perform_vae_nav.py  — continuous-playhead, KDTree-based, MPS-friendly
# OSC-controlled 2D cursor → continuous latent playhead → decode short windows with overlap-add.

import argparse, threading, numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.spatial import cKDTree

import torch
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from diffusers import AutoencoderOobleck

# -----------------------
# Tunables (also override via CLI if needed)
# -----------------------
SR = 44100
LATENT_HZ = 21.5      # ~ frames per second in Stable Audio VAE latent stream
WIN_SEC = 0.2         # decoded window length (s), L should be ≈ 2*HOP_SEC for COLA with Hann
HOP_SEC = 0.05         # audio hop (s) → output cadence / latency
NORM_CLAMP = 3.0      # z-score clamp before decode (stay near prior)
# Playhead behavior
BETA_TARGET = 0.2     # 0..1: how fast we drift toward nearest neighbor (timeline gravity)
JUMP_THRESH = 64      # if distance to target center > this many latent frames, hard-jump
MICRO_JITTER = 0      # 0/1: add ±1 latent-frame jitter to decorrelate repeated boundaries

def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = pick_device()
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")


class Player:
    def __init__(self, ae, corpus_npz,
                 beta_target=BETA_TARGET, jump_thresh=JUMP_THRESH, micro_jitter=MICRO_JITTER):
        self.ae = ae.eval().to(DEVICE)

        data = np.load(corpus_npz, allow_pickle=True)
        self.ZZ   = data["ZZ"].astype(np.float32)       # [N,2] (already normalized 0..1 if using latest preprocess)
        self.meta = data["meta"]                        # [N,2] (file_id, t_lat)
        self.paths= list(map(str, data["paths"]))
        self.Z_mean = data["Z_mean"].astype(np.float32)
        self.Z_std  = data["Z_std"].astype(np.float32)

        # 2D KD-tree for nearest neighbor queries (OpenMP-free)
        self.kdt = cKDTree(self.ZZ)

        # Cursor & streaming state
        self.cursor = np.array([0.5, 0.5], dtype=np.float32)  # default center in [0,1]^2
        self._stop = False
        self._latent_cache = {}   # file_id -> [T_lat, C] latents

        # Continuous latent playhead (sticky timeline)
        self.play_file = None     # current file_id
        self.play_tlat = 0.0      # current latent index (float)
        self.beta_target = float(beta_target)
        self.jump_thresh = int(jump_thresh)
        self.micro_jitter = int(micro_jitter)

    # -----------------------
    # Cursor control
    # -----------------------
    def set_cursor(self, x, y):
        self.cursor[:] = (float(x), float(y))

    # -----------------------
    # Latent utilities
    # -----------------------
    def _encode_wav_to_latents(self, path):
        # Load + (if needed) resample on CPU
        x, sr = sf.read(path, always_2d=True)
        if sr != SR:
            import torchaudio
            wt = torch.from_numpy(x.T).unsqueeze(0).float()
            res = torchaudio.functional.resample(wt, sr, SR)
            x = res.squeeze(0).numpy().T
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x.astype(np.float32)

        with torch.inference_mode():
            xt = torch.from_numpy(x.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)
            out = self.ae.encode(xt)
            z = out.latent_dist.sample().squeeze(0).permute(1,0).cpu().numpy()
        return z  # [T_lat, C]

    def _load_full_latents(self, file_id: int):
        if file_id in self._latent_cache:
            return self._latent_cache[file_id]
        z = self._encode_wav_to_latents(self.paths[file_id])
        self._latent_cache[file_id] = z
        return z

    def _decode_latent_window(self, z_win_np):
        # z-score clamp to remain near prior, then de-standardize
        z = (z_win_np - self.Z_mean) / self.Z_std
        z = np.clip(z, -NORM_CLAMP, NORM_CLAMP)
        z = z * self.Z_std + self.Z_mean
        with torch.inference_mode():
            zt = torch.from_numpy(z.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,C,T_lat]
            x = self.ae.decode(zt).sample                                     # [1,2,T_audio]
            x = x.squeeze(0).permute(1,0).cpu().numpy().astype(np.float32)    # [T_audio,2]
        return x

    # -----------------------
    # Continuous playhead logic
    # -----------------------
    def _nearest_primary(self):
        """Return the single nearest frame index to the cursor."""
        d, i = self.kdt.query(self.cursor, k=1)
        return int(i)

    def _ensure_playhead_initialized(self, idx_primary, win_lat):
        fid, t0 = int(self.meta[idx_primary,0]), int(self.meta[idx_primary,1])
        if self.play_file is None:
            self.play_file = fid
            # start a bit before target center so we have room for a full window
            self.play_tlat = max(0.0, float(t0) - win_lat * 0.5)

    def _advance_playhead(self, hop_lat, idx_primary, win_lat):
        """Advance by hop_lat; gently steer toward the nearest target neighbor."""
        target_fid, target_t = int(self.meta[idx_primary,0]), float(self.meta[idx_primary,1])

        # If switching file, adopt the new file immediately (we'll set a start near target)
        if self.play_file != target_fid:
            self.play_file = target_fid

        # Steer latent position toward target (timeline gravity)
        if abs(self.play_tlat - target_t) > self.jump_thresh:
            # far → jump to target vicinity
            self.play_tlat = target_t - win_lat * 0.5
        else:
            # near → small drift toward target
            self.play_tlat = (1.0 - self.beta_target) * self.play_tlat + self.beta_target * (target_t - win_lat * 0.5)

        # Advance by fixed hop in latent frames
        self.play_tlat += hop_lat

        # micro jitter (±1 latent frame) to decorrelate repeated boundaries
        if self.micro_jitter:
            self.play_tlat += np.random.randint(-1, 2)

        # bounds check vs. file length
        z_full = self._load_full_latents(self.play_file)
        max_start = max(0, z_full.shape[0] - win_lat)
        self.play_tlat = float(np.clip(self.play_tlat, 0, max_start))

        return self.play_file, int(round(self.play_tlat))

    # -----------------------
    # Audio thread
    # -----------------------
    def audio_loop(self):
        win_lat   = max(1, int(round(WIN_SEC * LATENT_HZ)))
        hop_samps = int(round(HOP_SEC * SR))

        # Hann synthesis window (COLA with L=2*hop). sqrt-Hann is also an option.
        fade = np.ascontiguousarray(np.hanning(2 * hop_samps).astype(np.float32))
        fade_in, fade_out = fade[:hop_samps], fade[hop_samps:]

        stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32', blocksize=hop_samps)
        stream.start()

        carry = np.ascontiguousarray(np.zeros((hop_samps, 2), np.float32))

        try:
            while not self._stop:
                # 1) nearest neighbor (target) in 2D + lazy init
                idx_primary = self._nearest_primary()
                self._ensure_playhead_initialized(idx_primary, win_lat)

                # 2) advance sticky playhead along latent timeline
                hop_lat = max(1, int(round(HOP_SEC * LATENT_HZ)))
                fid, s_lat = self._advance_playhead(hop_lat, idx_primary, win_lat)

                # 3) slice a consecutive latent window from current file
                z_full = self._load_full_latents(fid)
                e_lat = min(z_full.shape[0], s_lat + win_lat)
                z_win = z_full[s_lat:e_lat]
                if z_win.shape[0] < win_lat:
                    pad = np.repeat(z_win[-1:], win_lat - z_win.shape[0], axis=0)
                    z_win = np.concatenate([z_win, pad], axis=0)

                # 4) decode and overlap-add (output exactly hop_samps now; keep next hop as carry)
                audio = self._decode_latent_window(z_win)  # ≈ WIN_SEC audio
                a0 = audio[:hop_samps]
                if audio.shape[0] >= 2*hop_samps:
                    a1 = audio[hop_samps:2*hop_samps]
                else:
                    a1 = np.zeros_like(a0)

                out = carry * fade_out[:, None] + a0 * fade_in[:, None]
                out = np.ascontiguousarray(out, dtype=np.float32)  # required by sounddevice
                stream.write(out)
                carry = np.ascontiguousarray(a1, dtype=np.float32)
        finally:
            stream.stop(); stream.close()

    def stop(self):
        self._stop = True


# -----------------------
# OSC glue
# -----------------------
def run_server(player, ip="127.0.0.1", port=9000):
    disp = Dispatcher()
    def on_cursor(addr, x, y):
        # Expect x,y in [0,1]; if you send raw PCA coords, scale before calling this.
        player.set_cursor(x, y)
    disp.map("/cursor", on_cursor)
    server = BlockingOSCUDPServer((ip, port), disp)
    print(f"OSC listening on {ip}:{port} — send /cursor x y (0..1)")
    server.serve_forever()


# -----------------------
# CLI
# -----------------------
def main():
    global WIN_SEC, HOP_SEC  # ← moved here

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_npz", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--osc_ip", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--win_sec", type=float, default=WIN_SEC)
    ap.add_argument("--hop_sec", type=float, default=HOP_SEC)
    ap.add_argument("--beta_target", type=float, default=BETA_TARGET)
    ap.add_argument("--jump_thresh", type=int, default=JUMP_THRESH)
    ap.add_argument("--micro_jitter", type=int, default=MICRO_JITTER)
    args = ap.parse_args()

    WIN_SEC = args.win_sec
    HOP_SEC = args.hop_sec

    ae = AutoencoderOobleck.from_pretrained(args.pretrained, subfolder="vae")
    player = Player(
        ae, args.corpus_npz,
        beta_target=args.beta_target,
        jump_thresh=args.jump_thresh,
        micro_jitter=args.micro_jitter
    )

    t = threading.Thread(target=player.audio_loop, daemon=True)
    t.start()
    try:
        run_server(player, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        t.join()


if __name__ == "__main__":
    main()
