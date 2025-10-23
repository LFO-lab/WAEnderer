# preprocess_latents.py
# Builds a timbre-oriented 2D map and bundles all latents in a single file.
# - MFCC (drop c0) → mean/var → RobustScaler → PCA(2D) → min–max [0,1]
# - Encodes Stable Audio VAE latents per file
# - Saves:
#   ./corpus/<prefix>_<timestamp>/<prefix>_corpus_<timestamp>.npz
#   ./corpus/<prefix>_<timestamp>/<prefix>_latents_<timestamp>.npz
#
# The corpus file contains a pointer to the latents bundle path ("latent_bundle_path").

import os, glob, argparse, datetime, numpy as np
import soundfile as sf

import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from diffusers import AutoencoderOobleck  # Stable Audio VAE (encode/decode)

SR = 44100
LATENT_HZ = 21.5   # ~ frames/sec for Stable Audio VAE latent stream
N_MFCC = 15        # we'll drop c0 → keep 1..N_MFCC-1
N_MELS = 64
N_FFT  = 2048
HOP_STFT = 512     # for MFCC

def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = pick_device()
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")

# ---------------------------
# Audio I/O
# ---------------------------
def load_wav(path, target_sr=SR):
    x, sr = sf.read(path, always_2d=True)
    if sr != target_sr:
        wt = torch.from_numpy(x.T).unsqueeze(0).float()
        res = torchaudio.functional.resample(wt, sr, target_sr)
        x = res.squeeze(0).numpy().T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32)

# ---------------------------
# VAE encode
# ---------------------------
def wav_to_latents_full(ae, wav_np):
    with torch.inference_mode():
        x = torch.from_numpy(wav_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,2,T]
        out = ae.encode(x)
        z = out.latent_dist.sample().squeeze(0).permute(1,0).cpu().numpy()   # [T_lat, 64]
    return z

# ---------------------------
# MFCC transforms (center=False to avoid reflection padding issues)
# ---------------------------
_mfcc = torchaudio.transforms.MFCC(
    sample_rate=SR, n_mfcc=N_MFCC,
    melkwargs={"n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_STFT, "center": False}
)

def segment_mfcc_features(x_np, seg_start_samp, seg_len_samp, add_rms=False):
    """
    MFCC mean/var (mono) over [seg_start_samp : seg_start_samp+seg_len_samp]
    - zero-pad to seg_len_samp, then to N_FFT if needed
    - drop c0 (energy)
    returns 1D vector (2*(N_MFCC-1)) or (+1 if add_rms)
    """
    T = x_np.shape[0]
    s = int(np.clip(seg_start_samp, 0, max(0, T - 1)))
    e = int(np.clip(s + seg_len_samp, 0, T))

    seg = x_np[s:e]
    if seg.shape[0] < seg_len_samp:
        pad = np.zeros((seg_len_samp - seg.shape[0], seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg.astype(np.float32), pad], axis=0)
    if seg.shape[0] < N_FFT:
        pad2 = np.zeros((N_FFT - seg.shape[0], seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg, pad2], axis=0)

    x = torch.from_numpy(seg.T).unsqueeze(0)   # [1,2,L]
    x_mono = x.mean(dim=1, keepdim=True)       # [1,1,L]

    mfcc = _mfcc(x_mono).squeeze(0)            # [N_MFCC, frames]
    mfcc_no0 = mfcc[1:, :] if mfcc.shape[0] > 1 else mfcc
    m_mean = mfcc_no0.mean(dim=-1).numpy().astype(np.float32)  # [N_MFCC-1]
    m_var  = mfcc_no0.var(dim=-1).numpy().astype(np.float32)   # [N_MFCC-1]
    feats = np.concatenate([m_mean, m_var], axis=0)            # [2*(N_MFCC-1)]
    if add_rms:
        rms = np.sqrt((seg.astype(np.float32) ** 2).mean()).astype(np.float32)
        feats = np.concatenate([feats, [rms]], axis=0)
    return feats.reshape(-1).astype(np.float32)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Directory with .wav files")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output artifacts (used in names)")
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0",
                    help="HF repo or local path (subfolder='vae')")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--add_rms", type=int, default=0, help="Append RMS feature (0/1)")
    args = ap.parse_args()

    # Canonical paths
    prefix = os.path.basename(args.out_prefix)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.getcwd(), "corpus")
    out_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= N_FFT, "seg_sec must be >= N_FFT/SR (~0.0464 s)"

    # Load VAE (encoder only)
    ae = AutoencoderOobleck.from_pretrained(args.pretrained, subfolder="vae").to(DEVICE).eval()

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files found in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    feat_list = []        # per-segment feature vectors
    meta_list = []        # (file_id, t_lat, win_lat)
    all_latent_means = [] # for global Z_mean/Z_std
    all_latent_vars  = []

    # We'll collect latents in-memory then write a single bundle .npz
    latents_dict = {}

    print("Encoding files to latents & building segments…")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        z_full = wav_to_latents_full(ae, wav)  # [T_lat, 64]
        latents_dict[f"z_{fid}"] = z_full

        all_latent_means.append(z_full.mean(axis=0))
        all_latent_vars.append(z_full.var(axis=0))

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * SR))
            feats = segment_mfcc_features(wav, samp_start, seg_len_samp, add_rms=bool(args.add_rms))
            feat_list.append(feats)
            meta_list.append((fid, t_lat, win_lat))

    # Build feature matrix 2D
    F = np.vstack([np.asarray(f, dtype=np.float32).reshape(1, -1) for f in feat_list]).astype(np.float32)
    if not np.isfinite(F).all():
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    # Global latent stats (approx)
    Z_mean = np.mean(np.stack(all_latent_means, axis=0), axis=0).astype(np.float32)
    file_means = np.stack(all_latent_means, axis=0)
    file_vars  = np.stack(all_latent_vars,  axis=0)
    Z_var = file_vars.mean(axis=0) + file_means.var(axis=0)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    # Robust scaling -> PCA 2D -> min-max [0,1]
    print("Scaling (RobustScaler)…")
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    F_scaled = scaler.fit_transform(F).astype(np.float32)

    print("PCA(2D)…")
    pca2d = PCA(n_components=2, whiten=True, random_state=0).fit(F_scaled)
    ZZ = pca2d.transform(F_scaled).astype(np.float32)

    print("Min–max to [0,1]…")
    min_vals = ZZ.min(axis=0, keepdims=True)
    max_vals = ZZ.max(axis=0, keepdims=True)
    ZZ = (ZZ - min_vals) / (max_vals - min_vals + 1e-8)
    pca_min = min_vals.squeeze().astype(np.float32)
    pca_max = max_vals.squeeze().astype(np.float32)

    # ---------- Write outputs ----------
    # 1) Latents bundle
    latents_name = f"{prefix}_latents_{timestamp}.npz"
    latents_path = os.path.join(out_dir, latents_name)
    # add paths array too, for easy reference
    latents_dict["paths"] = paths_arr
    np.savez_compressed(latents_path, **latents_dict)

    # 2) Corpus file (points to the latents bundle)
    corpus_name = f"{prefix}_corpus_{timestamp}.npz"
    corpus_path = os.path.join(out_dir, corpus_name)
    np.savez_compressed(
        corpus_path,
        ZZ=ZZ,
        meta=meta,                   # (file_id, t_lat, win_lat)
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        # pointer to the single latents bundle
        latent_bundle_path=np.array(latents_path),
        # RobustScaler params
        scaler_center=np.asarray(scaler.center_, dtype=np.float32) if hasattr(scaler, "center_") else None,
        scaler_scale =np.asarray(scaler.scale_,  dtype=np.float32)  if hasattr(scaler, "scale_")  else None,
        scaler_quantile_range=np.asarray(scaler.quantile_range, dtype=np.float32),
        # PCA params
        pca_components_=pca2d.components_.astype(np.float32),
        pca_mean_=pca2d.mean_.astype(np.float32),
        pca_whiten=bool(pca2d.whiten),
        # 2D min-max bounds
        pca_min=pca_min,
        pca_max=pca_max
    )

    print("\nSaved:")
    print("  Latents :", latents_path)
    print("  Corpus  :", corpus_path)
    print("  Folder  :", out_dir)

if __name__ == "__main__":
    main()
