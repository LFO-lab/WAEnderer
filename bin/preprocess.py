#!/usr/bin/env python3
"""
Preprocess audio files into a 64D latent corpus with geometry.
Pipeline:
  - Encode audio with Stable Audio Open VAE
  - Mean-pool latents per segment -> GG (64D)
  - Normalize GG with global Z_mean/Z_std
  - Compute latent geometry (kNN + PCA)
"""
import os
import argparse
import datetime
import glob
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from stable_audio_wanderer.config import SR, LATENT_HZ
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.io.corpus_io import save_corpus
from stable_audio_wanderer.policy import compute_latent_geometry, save_geometry_to_dict


def compute_window_targets(local_sigma: np.ndarray) -> np.ndarray:
    """
    Compute window size targets based on local geometry (sigma).

    Uses quantile-based classification:
    - sigma < q25 (dense regions) → class 3 (8 frames, ~372ms) - long window
    - sigma < q50 → class 2 (4 frames, ~186ms) - default
    - sigma < q75 → class 1 (2 frames, ~93ms) - percussive
    - sigma >= q75 (sparse regions) → class 0 (1 frame, ~47ms) - sharp transients

    The intuition: dense regions have similar neighbors, so longer windows
    maintain coherence. Sparse regions have unique content, so shorter
    windows preserve transients.

    Args:
        local_sigma: [N] array of local sigma values from geometry

    Returns:
        window_targets: [N] int32 array of window class indices (0-4)
    """
    # Compute quantiles
    q25 = np.percentile(local_sigma, 25)
    q50 = np.percentile(local_sigma, 50)
    q75 = np.percentile(local_sigma, 75)

    # Initialize with default class 2
    targets = np.full(len(local_sigma), 2, dtype=np.int32)

    # Dense regions (low sigma) -> longer windows
    targets[local_sigma < q25] = 3  # 8 frames

    # Medium-dense -> default
    # Already set to 2 (4 frames)

    # Medium-sparse -> shorter windows
    mask_q50_q75 = (local_sigma >= q50) & (local_sigma < q75)
    targets[mask_q50_q75] = 1  # 2 frames

    # Sparse regions (high sigma) -> shortest windows
    targets[local_sigma >= q75] = 0  # 1 frame

    # Optional: add class 4 (16 frames) for very dense regions
    q10 = np.percentile(local_sigma, 10)
    targets[local_sigma < q10] = 4  # 16 frames for very static regions

    return targets




def compute_segment_latents(latents_dict: Dict[str, np.ndarray], meta: np.ndarray, win_lat: int) -> np.ndarray:
    """Mean-pool latent windows into per-segment embeddings (GG)."""
    seg_latents = []
    for fid, start, seg_len in meta:
        key = f"z_{int(fid)}"
        z_full = latents_dict[key]
        start = int(start)
        seg_len = int(seg_len) if int(seg_len) > 0 else int(win_lat)
        end = min(z_full.shape[0], start + seg_len)
        z_slice = z_full[start:end]
        if z_slice.shape[0] == 0:
            z_slice = np.repeat(z_full[:1], seg_len, axis=0)
        if z_slice.shape[0] < seg_len:
            pad = np.repeat(z_slice[-1:], seg_len - z_slice.shape[0], axis=0)
            z_slice = np.concatenate([z_slice, pad], axis=0)
        seg_latents.append(z_slice.mean(axis=0))
    return np.stack(seg_latents, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess: VAE latents -> 64D corpus + geometry."
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--latent_nav_k", type=int, default=32, help="k for latent kNN geometry.")

    args = ap.parse_args()

    prefix = os.path.basename(args.out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= 2048, "seg_sec must be >= 2048/44100"

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    ae = load_vae(args.pretrained)

    meta_list = []
    latent_sequences: List[np.ndarray] = []
    latents_dict = {}
    wav_dict: Dict[int, np.ndarray] = {}

    print("Encoding audio with VAE...")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        wav_dict[fid] = wav
        z_full = encode_full(ae, wav).astype(np.float32)
        z_full = np.ascontiguousarray(z_full)
        latents_dict[f"z_{fid}"] = z_full
        latent_sequences.append(z_full)

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            meta_list.append((fid, t_lat, win_lat))

    if not latent_sequences:
        raise RuntimeError("No latent sequences were extracted.")

    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    # Compute normalization stats from all VAE latents
    latent_stack = np.concatenate(latent_sequences, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    # Compute GG (segment latents) from VAE embeddings - mean pooling per segment
    GG = compute_segment_latents(latents_dict, meta, win_lat)

    # Normalize GG using VAE latent statistics
    GG_norm = (GG - Z_mean[None, :]) / Z_std[None, :]
    GG = np.clip(GG_norm, -5.0, 5.0).astype(np.float32)

    # Compute latent geometry (kNN + PCA projection)
    GG_norms = np.linalg.norm(GG, axis=1, keepdims=True)
    GG_norms = np.maximum(GG_norms, 1e-6)
    GG_l2 = (GG / GG_norms).astype(np.float32)

    print(f"Computing latent geometry (k={args.latent_nav_k})...")
    geometry = compute_latent_geometry(GG_l2, meta, k=int(args.latent_nav_k))
    geometry_arrays = save_geometry_to_dict(geometry)

    # Compute window targets for adaptive decoding
    print("Computing window targets from local geometry...")
    window_targets = compute_window_targets(geometry.local_sigma)
    class_counts = np.bincount(window_targets, minlength=5)
    print(f"  Window class distribution: {dict(enumerate(class_counts.tolist()))}")

    # Save corpus
    corpus_path = os.path.join(out_dir, "corpus.npz")
    save_corpus(
        corpus_path,
        GG=GG,
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        sr=np.array(int(SR), dtype=np.int32),
        latent_hz=np.array(float(LATENT_HZ), dtype=np.float32),
        segment_dur=np.array(float(args.seg_sec), dtype=np.float32),
        hop_dur=np.array(float(args.hop_sec), dtype=np.float32),
        latent_nav_k=np.array(int(args.latent_nav_k), dtype=np.int32),
        window_targets=window_targets,
        **geometry_arrays,
    )

    # Clear wav_dict to free memory
    wav_dict.clear()

    print("\nSaved:")
    print("  Corpus         :", corpus_path)


if __name__ == "__main__":
    main()
