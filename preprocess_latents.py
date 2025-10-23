# preprocess_latents.py
# Encode WAVs with Stable Audio VAE → latent frames → PCA(2D) → min–max [0,1].
# Saves: <out_prefix>_corpus.npz  (perform_vae_nav.py lit ZZ directement)

import os, glob, argparse, numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
from sklearn.decomposition import PCA
from diffusers import AutoencoderOobleck  # Stable Audio VAE (encode/decode)

SR = 44100                 # Stable Audio VAE expects 44.1 kHz
LATENT_HZ = 21.5           # ~frames per second in latent stream
PCA_2D_WHITEN = True       # True → variance équilibrée sur PC1/PC2 (nuage plus "rond")

def pick_device():
    # Prefer MPS on Apple Silicon; else CUDA; else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = pick_device()
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")

def load_wav(path, target_sr=SR):
    """Load WAV as float32 stereo @ target_sr; resample on CPU if needed."""
    x, sr = sf.read(path, always_2d=True)
    if sr != target_sr:
        import torchaudio
        wt = torch.from_numpy(x.T).unsqueeze(0).float()  # CPU tensor
        res = torchaudio.functional.resample(wt, sr, target_sr)
        x = res.squeeze(0).numpy().T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32)

def wav_to_latents(ae, wav_np):
    """Return latents [T_lat, C] (C≈64) time-dense at ~21.5 Hz."""
    with torch.inference_mode():
        x = torch.from_numpy(wav_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,2,T]
        out = ae.encode(x)                          # AutoencoderOobleckOutput
        z = out.latent_dist.sample()                # [1, C, T_lat]
        z = z.squeeze(0).permute(1, 0).cpu().numpy()# -> [T_lat, C]
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Directory of input .wav files")
    ap.add_argument("--out_prefix", required=True, help="Output prefix for artifacts")
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0",
                    help="HF repo or local path with subfolder='vae'")
    args = ap.parse_args()

    # ---- load VAE only
    ae = AutoencoderOobleck.from_pretrained(args.pretrained, subfolder="vae").to(DEVICE).eval()

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files found in --audio_dir")

    all_frames = []   # [N_frames, C]
    frame_meta = []   # (file_id, t_lat)
    for fid, p in enumerate(tqdm(paths, desc="Encoding → latents")):
        wav = load_wav(p)
        z_full = wav_to_latents(ae, wav)            # [T_lat, C]
        for t_idx in range(z_full.shape[0]):
            all_frames.append(z_full[t_idx])
            frame_meta.append((fid, t_idx))

    Z = np.stack(all_frames, axis=0).astype(np.float32)   # [N, C]
    meta = np.array(frame_meta, dtype=np.int32)

    # ---- PCA to 2D
    print("PCA(2D)…")
    pca2d = PCA(n_components=2, whiten=PCA_2D_WHITEN, random_state=0).fit(Z)
    ZZ = pca2d.transform(Z).astype(np.float32)  # [N, 2]

    # ---- Min–Max normalize each axis to [0, 1]
    min_vals = ZZ.min(axis=0, keepdims=True)
    max_vals = ZZ.max(axis=0, keepdims=True)
    ZZ = (ZZ - min_vals) / (max_vals - min_vals + 1e-8)
    pca_min = min_vals.squeeze().astype(np.float32)  # for future projections
    pca_max = max_vals.squeeze().astype(np.float32)

    # ---- save artifacts
    np.savez_compressed(
        f"{args.out_prefix}_corpus.npz",
        ZZ=ZZ,                                  # 2D coords (PC1/PC2) normalisées [0,1]
        Z_mean=Z.mean(0),                       # pour clamp / interpolation
        Z_std=Z.std(0) + 1e-6,
        meta=meta,                              # (file_id, t_lat)
        paths=np.array(paths),
        # Persist PCA params to project new points consistently later
        pca2d_components_=pca2d.components_.astype(np.float32),
        pca2d_mean_=pca2d.mean_.astype(np.float32),
        pca2d_whiten=PCA_2D_WHITEN,
        pca2d_min=pca_min,
        pca2d_max=pca_max
    )
    print("Done. Saved:", f"{args.out_prefix}_corpus.npz")

if __name__ == "__main__":
    main()
