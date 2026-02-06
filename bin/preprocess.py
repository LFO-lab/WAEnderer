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
from stable_audio_wanderer.policy.sequence import compute_velocity_magnitudes


def compute_window_targets(GG: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """
    Map velocity to continuous log2(window_size) targets.

    Low velocity → high log2 (long windows), high velocity → low log2 (short windows).
    Uses robust percentile scaling to normalize velocity to [0, 1].

    Args:
        GG: [N, 64] normalized segment latents
        meta: [N, 3] segment metadata (file_id, t_lat, win_lat)

    Returns:
        targets_log2: [N] float32 array of log2(window_size) in [1.0, 6.0]
    """
    velocity = compute_velocity_magnitudes(GG, meta)
    # Normalize velocity to [0, 1] using robust percentile scaling
    v_low, v_high = np.percentile(velocity, [5, 95])
    v_norm = np.clip((velocity - v_low) / (v_high - v_low + 1e-8), 0, 1)
    # Invert and map to log2 range [1.0, 6.0]
    # Low velocity (v_norm~0) → log2=6.0 (64 frames)
    # High velocity (v_norm~1) → log2=1.0 (2 frames)
    targets_log2 = 6.0 - 5.0 * v_norm  # [6.0 → 1.0]
    return targets_log2.astype(np.float32)




def _compute_decoder_quality_targets(
    latents_dict: Dict[str, np.ndarray],
    meta: np.ndarray,
    win_lat: int,
    vae,
    Z_mean: np.ndarray,
    Z_std: np.ndarray,
) -> np.ndarray:
    """
    Compute decoder quality targets via offline spectral coherence analysis.

    For each segment, decode at multiple window sizes and find the elbow where
    marginal quality improvement drops below threshold.

    Args:
        latents_dict: Dict of per-file latent sequences
        meta: [N, 3] segment metadata (file_id, t_lat, win_lat)
        win_lat: Window size in latent frames
        vae: Loaded VAE model
        Z_mean, Z_std: Normalization stats

    Returns:
        targets_log2: [N] float32 array of optimal log2(window_size) per segment
    """
    import math
    from stable_audio_wanderer.vae.decoder import decode_latents

    test_sizes = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    N = meta.shape[0]
    targets_log2 = np.full(N, 3.0, dtype=np.float32)  # Default: log2(8)

    for seg_idx in tqdm(range(N), desc="Decoder quality analysis"):
        fid, t_start, seg_len = int(meta[seg_idx, 0]), int(meta[seg_idx, 1]), int(meta[seg_idx, 2])
        key = f"z_{fid}"
        z_full = latents_dict[key]
        T_full = z_full.shape[0]

        coherences = []
        for ws in test_sizes:
            # Extract window centered on segment
            center = t_start + seg_len // 2
            half = ws // 2
            start = max(0, center - half)
            end = min(T_full, start + ws)
            z_window = z_full[start:end]
            if z_window.shape[0] < 2:
                coherences.append(0.0)
                continue

            # Denormalize and decode
            z_raw = z_window * Z_std + Z_mean
            try:
                audio = decode_latents(vae, z_raw)
                # Measure temporal correlation as proxy for spectral coherence
                if len(audio) > 2048:
                    # Simple: autocorrelation at 1-frame lag
                    frame_len = len(audio) // z_window.shape[0]
                    if frame_len > 0 and z_window.shape[0] > 1:
                        frames = [audio[i*frame_len:(i+1)*frame_len] for i in range(z_window.shape[0])]
                        corrs = []
                        for i in range(len(frames) - 1):
                            min_len = min(len(frames[i]), len(frames[i+1]))
                            if min_len > 0:
                                c = np.corrcoef(frames[i][:min_len], frames[i+1][:min_len])[0, 1]
                                if not np.isnan(c):
                                    corrs.append(c)
                        coherences.append(float(np.mean(corrs)) if corrs else 0.0)
                    else:
                        coherences.append(0.0)
                else:
                    coherences.append(0.0)
            except Exception:
                coherences.append(0.0)

        # Find elbow: window size where marginal improvement < threshold
        if len(coherences) >= 2:
            threshold = 0.02
            best_idx = 0
            for i in range(1, len(coherences)):
                improvement = coherences[i] - coherences[i - 1]
                if improvement < threshold:
                    best_idx = i
                    break
            else:
                best_idx = len(coherences) - 1
            targets_log2[seg_idx] = np.clip(math.log2(test_sizes[best_idx]), 1.0, 6.0)

    return targets_log2


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
    ap.add_argument("--encode_chunk_sec", type=float, default=60.0,
                    help="Chunk size (seconds) for VAE encoding. Set 0 to disable chunking.")
    ap.add_argument("--encode_chunk_overlap_sec", type=float, default=1.0,
                    help="Chunk overlap (seconds) for VAE encoding.")
    ap.add_argument("--compute_decoder_targets", action="store_true",
                    help="Compute decoder quality targets (expensive: 11 VAE decodes per segment).")

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
    encode_chunk_sec = float(args.encode_chunk_sec)
    encode_overlap_sec = float(args.encode_chunk_overlap_sec)
    chunk_sec = encode_chunk_sec if encode_chunk_sec > 0.0 else None

    print("Encoding audio with VAE...")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        z_full = encode_full(ae, wav, chunk_sec=chunk_sec, overlap_sec=encode_overlap_sec).astype(np.float32)
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

    # Compute continuous window targets for adaptive decoding
    print("Computing window targets from latent velocity...")
    window_targets_log2 = compute_window_targets(GG, meta)
    print(f"  Window targets (log2): min={window_targets_log2.min():.2f}, "
          f"max={window_targets_log2.max():.2f}, mean={window_targets_log2.mean():.2f}")

    # Optional decoder quality targets
    extra_arrays = {}
    if args.compute_decoder_targets:
        print("Computing decoder quality targets (this may take a while)...")
        try:
            decoder_targets = _compute_decoder_quality_targets(
                latents_dict, meta, win_lat, ae, Z_mean, Z_std
            )
            extra_arrays["decoder_quality_targets_log2"] = decoder_targets
            print(f"  Decoder quality targets: min={decoder_targets.min():.2f}, "
                  f"max={decoder_targets.max():.2f}, mean={decoder_targets.mean():.2f}")
        except Exception as e:
            print(f"  [warn] Failed to compute decoder quality targets: {e}")

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
        window_targets_log2=window_targets_log2,
        **geometry_arrays,
        **extra_arrays,
    )

    print("\nSaved:")
    print("  Corpus         :", corpus_path)


if __name__ == "__main__":
    main()
