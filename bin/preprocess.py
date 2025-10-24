#!/usr/bin/env python3
import os, argparse, datetime, glob, numpy as np
from tqdm import tqdm
from stable_audio_wanderer.config import SR, LATENT_HZ, DEVICE
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.features.mfcc import segment_mfcc_no_c0, ensure_2d_stack
from stable_audio_wanderer.dr.pca import fit_transform
from stable_audio_wanderer.io.corpus_io import save_latents_bundle, save_corpus

def main():
    ap = argparse.ArgumentParser(description="Préprocess: MFCC→PCA + bundle latents.")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--pca_dim", type=int, default=2)
    ap.add_argument("--add_rms", type=int, default=0)
    args = ap.parse_args()

    prefix = os.path.basename(args.out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= 2048, "seg_sec must be >= 2048/44100"

    ae = load_vae(args.pretrained)

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    feat_list, meta_list = [], []
    all_latent_means, all_latent_vars = [], []
    latents_dict = {}

    print("Encoding + feature extraction…")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        z_full = encode_full(ae, wav)             # [T_lat, 64]
        latents_dict[f"z_{fid}"] = z_full
        all_latent_means.append(z_full.mean(axis=0))
        all_latent_vars.append(z_full.var(axis=0))

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * SR))
            feats = segment_mfcc_no_c0(wav, samp_start, seg_len_samp, add_rms=bool(args.add_rms))
            feat_list.append(feats)
            meta_list.append((fid, t_lat, win_lat))

    F = ensure_2d_stack(feat_list)
    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    Z_mean = np.mean(np.stack(all_latent_means, axis=0), axis=0).astype(np.float32)
    file_means = np.stack(all_latent_means, axis=0)
    file_vars  = np.stack(all_latent_vars,  axis=0)
    Z_var = file_vars.mean(axis=0) + file_means.var(axis=0)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    ZZ, dr_meta = fit_transform(F, args.pca_dim)

    latents_path = os.path.join(out_dir, f"{prefix}_latents_{ts}.npz")
    save_latents_bundle(latents_path, latents_dict, paths_arr)

    corpus_path = os.path.join(out_dir, f"{prefix}_corpus_{ts}.npz")
    save_corpus(
        corpus_path,
        ZZ=ZZ, pca_dim=np.array(int(args.pca_dim), dtype=np.int32),
        meta=meta, paths=paths_arr,
        Z_mean=Z_mean, Z_std=Z_std,
        latent_bundle_path=np.array(latents_path),
        **{k: (v if v is None else np.asarray(v, dtype=np.float32)) for k, v in dr_meta.items()}
    )

    print("\nSaved:")
    print("  Latents :", latents_path)
    print("  Corpus  :", corpus_path)
    print("  Folder  :", out_dir)

if __name__ == "__main__":
    main()
