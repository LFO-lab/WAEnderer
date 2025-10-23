# perform_vae_nav.py — continuous-playhead, KDTree-based, MPS-friendly
# OSC-controlled 2D cursor → continuous latent playhead → decode short windows with overlap-add.
#
# Usage:
#   python perform_vae_nav.py --corpus_dir ./corpus/<prefix>_<timestamp> \
#       --win_sec 0.2 --hop_sec 0.05
#
# Folder structure expected:
#   ./corpus/<prefix>_<timestamp>/
#       <prefix>_corpus_<timestamp>.npz
#       <prefix>_latents_<timestamp>.npz

import argparse, threading, os, glob, numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.spatial import cKDTree

import torch
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from diffusers import AutoencoderOobleck

# -----------------------
# Tunables
# -----------------------
SR = 44100
LATENT_HZ = 21.5      # ~ frames/s in Stable Audio VAE latent stream
WIN_SEC = 0.2         # decoded window length (s), L ≈ 2*HOP_SEC for COLA with Hann
HOP_SEC = 0.05        # audio hop (s) → output cadence / latency
NORM_CLAMP = 3.0      # z-score clamp before decode (stay near prior)

# Playhead behavior
BETA_TARGET = 0.2     # 0..1: drift toward nearest neighbor (timeline gravity)
JUMP_THRESH = 64      # if |play_tlat - target_t| > this many frames → hard-jump
MICRO_JITTER = 0      # 0/1: add ±1 latent-frame jitter to decorrelate boundaries


def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")


def find_corpus_file(corpus_dir: str) -> str:
    """Find a '*_corpus_*.npz' in corpus_dir. If multiple, pick the newest by mtime."""
    pattern = os.path.join(corpus_dir, "*_corpus_*.npz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No corpus file matching '*_corpus_*.npz' in {corpus_dir}")
    if len(matches) == 1:
        return matches[0]
    # pick newest
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def find_latents_file(corpus_dir: str) -> str:
    """Find a '*_latents_*.npz' in corpus_dir. If multiple, pick the newest by mtime."""
    pattern = os.path.join(corpus_dir, "*_latents_*.npz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No latents file matching '*_latents_*.npz' in {corpus_dir}")
    if len(matches) == 1:
        return matches[0]
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


class Player:
    def __init__(self, ae, corpus_dir,
                 beta_target=BETA_TARGET, jump_thresh=JUMP_THRESH, micro_jitter=MICRO_JITTER):
        self.ae = ae.eval().to(DEVICE)

        # -------- resolve files from folder --------
        corpus_npz = find_corpus_file(corpus_dir)
        print(f"[info] Using corpus: {corpus_npz}")

        data = np.load(corpus_npz, allow_pickle=True)

        # 2D map (MFCC→RobustScaler→PCA→[0,1])
        self.ZZ   = data["ZZ"].astype(np.float32)            # [N_seg, 2]
        self.meta = data["meta"]                              # [N_seg, 3] (file_id, t_lat, win_lat)
        self.paths= list(map(str, data["paths"]))
        # stats latentes globales pour clamp au décodage
        self.Z_mean = data["Z_mean"].astype(np.float32)
        self.Z_std  = data["Z_std"].astype(np.float32)

        # Try to read bundle path from corpus; if missing/invalid, fallback to local latents file
        self.latent_bundle = None
        self.latent_bundle_path = None
        if "latent_bundle_path" in data.files:
            candidate = str(data["latent_bundle_path"])
            if os.path.isfile(candidate):
                self.latent_bundle_path = candidate
            else:
                # try same folder
                self.latent_bundle_path = find_latents_file(os.path.dirname(corpus_npz))
        else:
            self.latent_bundle_path = find_latents_file(os.path.dirname(corpus_npz))

        print(f"[info] Using latents bundle: {self.latent_bundle_path}")

        # KD-tree pour kNN 2D
        self.kdt = cKDTree(self.ZZ)

        # État navigation / audio
        self.cursor = np.array([0.5, 0.5], dtype=np.float32)  # centre par défaut
        self._stop = False
        self._latent_cache = {}   # file_id -> [T_lat, 64] latents

        # Playhead latent continu
        self.play_file = None     # file_id courant
        self.play_tlat = 0.0      # index latent courant (float)
        self.beta_target = float(beta_target)
        self.jump_thresh = int(jump_thresh)
        self.micro_jitter = int(micro_jitter)

    # -----------------------
    # Cursor control
    # -----------------------
    def set_cursor(self, x, y):
        # attendu: x,y ∈ [0,1]
        self.cursor[:] = (float(x), float(y))

    # -----------------------
    # Latent utilities
    # -----------------------
    def _ensure_bundle_loaded(self):
        if (self.latent_bundle is None) and (self.latent_bundle_path is not None):
            self.latent_bundle = np.load(self.latent_bundle_path, allow_pickle=True)

    def _encode_wav_to_latents(self, path):
        """Fallback si pas de bundle: encoder un WAV → latents [T_lat, 64]."""
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
            z = out.latent_dist.sample().squeeze(0).permute(1, 0).cpu().numpy()
        return z  # [T_lat, 64]

    def _load_full_latents(self, file_id: int):
        """Charge les latents complets pour un fichier (depuis le bundle si dispo)."""
        if file_id in self._latent_cache:
            return self._latent_cache[file_id]

        z = None
        if self.latent_bundle_path is not None:
            self._ensure_bundle_loaded()
            key = f"z_{file_id}"
            if isinstance(self.latent_bundle, np.lib.npyio.NpzFile):
                if key in self.latent_bundle.files:
                    z = self.latent_bundle[key]
        if z is None:
            # fallback: encode from audio path
            z = self._encode_wav_to_latents(self.paths[file_id])

        self._latent_cache[file_id] = z
        return z

    def _decode_latent_window(self, z_win_np):
        """Clamp + decode une fenêtre latente → audio stéréo float32."""
        z = (z_win_np - self.Z_mean) / self.Z_std
        z = np.clip(z, -NORM_CLAMP, NORM_CLAMP)
        z = z * self.Z_std + self.Z_mean
        with torch.inference_mode():
            zt = torch.from_numpy(z.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,64,T_lat]
            x = self.ae.decode(zt).sample                                     # [1,2,T_audio]
            x = x.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)   # [T_audio,2]
        return x

    # -----------------------
    # Continuous playhead logic
    # -----------------------
    def _nearest_primary(self):
        """Renvoie l'indice du voisin le plus proche du curseur dans ZZ."""
        d, i = self.kdt.query(self.cursor, k=1)
        return int(i)

    def _ensure_playhead_initialized(self, idx_primary, win_lat):
        fid, t0, _ = int(self.meta[idx_primary, 0]), int(self.meta[idx_primary, 1]), int(self.meta[idx_primary, 2])
        if self.play_file is None:
            self.play_file = fid
            self.play_tlat = max(0.0, float(t0) - win_lat * 0.5)

    def _advance_playhead(self, hop_lat, idx_primary, win_lat):
        """Avance de hop_lat et attire la tête de lecture vers la cible la plus proche."""
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

    # -----------------------
    # Audio thread
    # -----------------------
    def audio_loop(self):
        win_lat   = max(1, int(round(WIN_SEC * LATENT_HZ)))
        hop_samps = int(round(HOP_SEC * SR))
        hop_lat   = max(1, int(round(HOP_SEC * LATENT_HZ)))

        fade = np.ascontiguousarray(np.hanning(2 * hop_samps).astype(np.float32))
        fade_in, fade_out = fade[:hop_samps], fade[hop_samps:]

        stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32', blocksize=hop_samps)
        stream.start()

        carry = np.ascontiguousarray(np.zeros((hop_samps, 2), np.float32))

        try:
            while not self._stop:
                idx_primary = self._nearest_primary()
                # Option: use per-segment window length from meta
                                # win_lat = int(self.meta[idx_primary, 2])

                self._ensure_playhead_initialized(idx_primary, win_lat)

                fid, s_lat = self._advance_playhead(hop_lat, idx_primary, win_lat)

                z_full = self._load_full_latents(fid)
                e_lat = min(z_full.shape[0], s_lat + win_lat)
                z_win = z_full[s_lat:e_lat]
                if z_win.shape[0] < win_lat:
                    pad = np.repeat(z_win[-1:], win_lat - z_win.shape[0], axis=0)
                    z_win = np.concatenate([z_win, pad], axis=0)

                audio = self._decode_latent_window(z_win)  # ≈ WIN_SEC audio
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


# -----------------------
# OSC glue
# -----------------------
def run_server(player, ip="127.0.0.1", port=9000):
    dispatcher = Dispatcher()
    def on_cursor(addr, x, y):
        player.set_cursor(x, y)
    dispatcher.map("/cursor", on_cursor)
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC listening on {ip}:{port} — send /cursor x y (0..1)")
    server.serve_forever()


# -----------------------
# CLI
# -----------------------
def main():
    global WIN_SEC, HOP_SEC

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True,
                    help="Directory containing <prefix>_corpus_<ts>.npz and <prefix>_latents_<ts>.npz")
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
        ae, args.corpus_dir,
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
