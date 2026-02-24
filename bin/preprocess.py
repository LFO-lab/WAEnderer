#!/usr/bin/env python3
"""
Preprocess audio files into a 64D latent corpus with geometry.
Pipeline:
  - Encode audio with Stable Audio Open VAE
  - Extract latent-rate MFCC features for manual navigation
  - Keep full latent trajectories per file (no mean pooling)
  - Normalize latents with global Z_mean/Z_std
  - Compute causal-context geometry/index over all frames
  - Compute manual 8D PCA control points + robust fader ranges
"""
import os
import argparse
import datetime
import glob
from typing import List, Dict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from stable_audio_wanderer.config import (
    SR,
    LATENT_HZ,
    K_SHORT,
    EMA_ALPHA_FAST,
    EMA_ALPHA_MID,
    EMA_ALPHA_SLOW,
    USE_EMA_MID,
    CONTEXT_PCA_DIM,
)
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.io.corpus_io import save_corpus
from stable_audio_wanderer.policy import compute_latent_geometry, save_geometry_to_dict
from stable_audio_wanderer.policy.sequence import compute_velocity_magnitudes

MANUAL_N_MFCC = 20
MANUAL_PCA_DIM = 8
MANUAL_PERCENTILE_LOW = 1.0
MANUAL_PERCENTILE_HIGH = 99.0


def _align_feature_frames(feature_seq: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Trim/pad a [T, D] feature sequence so its time axis matches target_frames.
    """
    feat = np.asarray(feature_seq, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(f"Expected feature_seq [T, D], got {feat.shape}")
    target = int(max(1, target_frames))
    if feat.shape[0] > target:
        return feat[:target]
    if feat.shape[0] < target:
        if feat.shape[0] == 0:
            return np.zeros((target, feat.shape[1]), dtype=np.float32)
        pad = np.repeat(feat[-1:], target - feat.shape[0], axis=0)
        return np.concatenate([feat, pad], axis=0)
    return feat


def _build_mfcc_transform(hop_length: int):
    """
    Build MFCC extractor using latent-aligned hop length.
    """
    return torchaudio.transforms.MFCC(
        sample_rate=int(SR),
        n_mfcc=int(MANUAL_N_MFCC),
        melkwargs={
            "n_fft": 2048,
            "win_length": 2048,
            "hop_length": int(hop_length),
            "n_mels": 64,
            "center": False,
            "power": 2.0,
        },
    )


def compute_latent_aligned_mfcc(
    wav_stereo: np.ndarray,
    target_frames: int,
    mfcc_transform,
) -> np.ndarray:
    """
    Compute mono MFCC features aligned to latent frame rate.
    Returns [target_frames, MANUAL_N_MFCC].
    """
    wav = np.asarray(wav_stereo, dtype=np.float32)
    if wav.ndim != 2:
        raise ValueError(f"Expected stereo wav [T, C], got {wav.shape}")
    mono = wav.mean(axis=1, dtype=np.float32)
    min_len = 2048 + 1  # must exceed n_fft used by MelSpectrogram
    if mono.shape[0] < min_len:
        mono = np.pad(mono, (0, min_len - mono.shape[0]))
    x = torch.from_numpy(mono[None, :])
    with torch.inference_mode():
        mfcc = mfcc_transform(x).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    return _align_feature_frames(mfcc, int(target_frames))


def compute_manual_navigation_features(mfcc_concat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert [N, n_mfcc] MFCC stack to 8D PCA control space with robust ranges.
    """
    mfcc = np.asarray(mfcc_concat, dtype=np.float32)
    if mfcc.ndim != 2 or mfcc.shape[1] != MANUAL_N_MFCC:
        raise ValueError(f"Expected MFCC stack [N, {MANUAL_N_MFCC}], got {mfcc.shape}")
    if mfcc.shape[0] == 0:
        raise ValueError("No MFCC frames available for manual feature extraction.")

    mfcc_mean = mfcc.mean(axis=0, keepdims=True).astype(np.float32)
    mfcc_std = np.sqrt(mfcc.var(axis=0, keepdims=True).astype(np.float32) + 1e-6).astype(np.float32)
    mfcc_z = (mfcc - mfcc_mean) / mfcc_std

    pca_mean = mfcc_z.mean(axis=0).astype(np.float32)
    centered = mfcc_z - pca_mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    components = np.zeros((MANUAL_PCA_DIM, MANUAL_N_MFCC), dtype=np.float32)
    available = int(min(MANUAL_PCA_DIM, vt.shape[0], vt.shape[1]))
    if available > 0:
        components[:available] = vt[:available].astype(np.float32)

    points = (centered @ components.T).astype(np.float32)
    p01 = np.percentile(points, MANUAL_PERCENTILE_LOW, axis=0).astype(np.float32)
    p99 = np.percentile(points, MANUAL_PERCENTILE_HIGH, axis=0).astype(np.float32)
    p99 = np.maximum(p99, p01 + 1e-6).astype(np.float32)

    return {
        "manual_pca_points": points,
        "manual_pca_components": components,
        "manual_pca_mean": pca_mean,
        "manual_fader_p01": p01,
        "manual_fader_p99": p99,
        "manual_n_mfcc": np.array(int(MANUAL_N_MFCC), dtype=np.int32),
        "manual_pca_dim": np.array(int(MANUAL_PCA_DIM), dtype=np.int32),
        "manual_percentile_low": np.array(float(MANUAL_PERCENTILE_LOW), dtype=np.float32),
        "manual_percentile_high": np.array(float(MANUAL_PERCENTILE_HIGH), dtype=np.float32),
    }


def compute_window_targets(Z_concat: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """
    Map velocity to continuous log2(window_size) targets.

    Low velocity → high log2 (long windows), high velocity → low log2 (short windows).
    Uses robust percentile scaling to normalize velocity to [0, 1].

    Args:
        Z_concat: [N, 64] normalized frame latents
        meta: [N, 3] frame metadata (file_id, t_lat, 1)

    Returns:
        targets_log2: [N] float32 array of log2(window_size) in [1.0, 6.0]
    """
    velocity = compute_velocity_magnitudes(Z_concat, meta)
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
    vae,
    Z_mean: np.ndarray,
    Z_std: np.ndarray,
) -> np.ndarray:
    """
    Compute decoder quality targets via offline spectral coherence analysis.

    For each segment, decode at multiple window sizes and find the elbow where
    marginal quality improvement drops below threshold.

    Args:
        latents_dict: Dict of per-file normalized latent sequences
        meta: [N, 3] frame metadata (file_id, t_lat, 1)
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
            # Extract window centered on frame
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


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess: VAE latents -> 64D corpus + geometry."
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2,
                    help="Deprecated in frame-mode corpus; kept for CLI compatibility.")
    ap.add_argument("--hop_sec", type=float, default=0.05,
                    help="Deprecated in frame-mode corpus; kept for CLI compatibility.")
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

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    ae = load_vae(args.pretrained)

    latent_sequences_raw: List[np.ndarray] = []
    manual_mfcc_sequences: List[np.ndarray] = []
    encode_chunk_sec = float(args.encode_chunk_sec)
    encode_overlap_sec = float(args.encode_chunk_overlap_sec)
    chunk_sec = encode_chunk_sec if encode_chunk_sec > 0.0 else None
    latent_hop = max(1, int(round(float(SR) / float(LATENT_HZ))))
    mfcc_transform = _build_mfcc_transform(hop_length=latent_hop)

    print("Encoding audio with VAE + extracting latent-aligned MFCC features...")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        z_full = encode_full(ae, wav, chunk_sec=chunk_sec, overlap_sec=encode_overlap_sec).astype(np.float32)
        latent_sequences_raw.append(np.ascontiguousarray(z_full))
        mfcc_seq = compute_latent_aligned_mfcc(
            wav_stereo=wav,
            target_frames=z_full.shape[0],
            mfcc_transform=mfcc_transform,
        )
        manual_mfcc_sequences.append(np.ascontiguousarray(mfcc_seq))

    if not latent_sequences_raw:
        raise RuntimeError("No latent sequences were extracted.")
    if len(manual_mfcc_sequences) != len(latent_sequences_raw):
        raise RuntimeError("MFCC extraction count mismatch.")

    paths_arr = np.array(paths)

    # Compute normalization stats from all VAE latents
    latent_stack = np.concatenate(latent_sequences_raw, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    # Canonical corpus representation: all observed frame latents (normalized), grouped by file offsets.
    file_offsets = [0]
    meta_list = []
    normalized_by_file: Dict[str, np.ndarray] = {}
    normalized_sequences = []

    for fid, z_full_raw in enumerate(latent_sequences_raw):
        z_norm = np.clip((z_full_raw - Z_mean[None, :]) / Z_std[None, :], -5.0, 5.0).astype(np.float32)
        z_norm = np.ascontiguousarray(z_norm)
        normalized_sequences.append(z_norm)
        normalized_by_file[f"z_{fid}"] = z_norm

        T_lat = z_norm.shape[0]
        for t_lat in range(T_lat):
            meta_list.append((fid, t_lat, 1))
        file_offsets.append(file_offsets[-1] + T_lat)

    Z_concat = np.concatenate(normalized_sequences, axis=0).astype(np.float32)
    mfcc_concat = np.concatenate(manual_mfcc_sequences, axis=0).astype(np.float32)
    file_offsets = np.asarray(file_offsets, dtype=np.int64)
    meta = np.asarray(meta_list, dtype=np.int32)
    frame_file_ids = meta[:, 0].astype(np.int32)
    frame_t = meta[:, 1].astype(np.int32)

    if mfcc_concat.shape[0] != Z_concat.shape[0]:
        raise RuntimeError(
            f"Manual MFCC frame count mismatch: {mfcc_concat.shape[0]} vs latent frames {Z_concat.shape[0]}"
        )

    print("Computing manual navigation PCA features...")
    manual_arrays = compute_manual_navigation_features(mfcc_concat)
    p01 = manual_arrays["manual_fader_p01"]
    p99 = manual_arrays["manual_fader_p99"]
    print(
        "  Manual PCA points:",
        manual_arrays["manual_pca_points"].shape,
        f"(fader p01 mean={p01.mean():.3f}, p99 mean={p99.mean():.3f})",
    )

    print(f"Computing latent geometry (k={args.latent_nav_k})...")
    geometry = compute_latent_geometry(
        Z_concat,
        meta,
        k=int(args.latent_nav_k),
        k_short=int(K_SHORT),
        ema_alpha_fast=float(EMA_ALPHA_FAST),
        ema_alpha_mid=float(EMA_ALPHA_MID),
        ema_alpha_slow=float(EMA_ALPHA_SLOW),
        use_ema_mid=bool(USE_EMA_MID),
        pca_dim=int(CONTEXT_PCA_DIM),
    )
    geometry_arrays = save_geometry_to_dict(geometry)

    # Compute continuous window targets for adaptive decoding
    print("Computing window targets from latent velocity...")
    window_targets_log2 = compute_window_targets(Z_concat, meta)
    print(f"  Window targets (log2): min={window_targets_log2.min():.2f}, "
          f"max={window_targets_log2.max():.2f}, mean={window_targets_log2.mean():.2f}")

    # Optional decoder quality targets
    extra_arrays = {}
    if args.compute_decoder_targets:
        print("Computing decoder quality targets (this may take a while)...")
        try:
            decoder_targets = _compute_decoder_quality_targets(
                normalized_by_file, meta, ae, Z_mean, Z_std
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
        Z_concat=Z_concat,
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        sr=np.array(int(SR), dtype=np.int32),
        latent_hz=np.array(float(LATENT_HZ), dtype=np.float32),
        segment_dur=np.array(float(1.0 / LATENT_HZ), dtype=np.float32),
        hop_dur=np.array(float(1.0 / LATENT_HZ), dtype=np.float32),
        latent_nav_k=np.array(int(args.latent_nav_k), dtype=np.int32),
        k_short=np.array(int(K_SHORT), dtype=np.int32),
        ema_alpha_fast=np.array(float(EMA_ALPHA_FAST), dtype=np.float32),
        ema_alpha_mid=np.array(float(EMA_ALPHA_MID), dtype=np.float32),
        ema_alpha_slow=np.array(float(EMA_ALPHA_SLOW), dtype=np.float32),
        use_ema_mid=np.array(int(USE_EMA_MID), dtype=np.int32),
        context_pca_dim=np.array(int(CONTEXT_PCA_DIM), dtype=np.int32),
        window_targets_log2=window_targets_log2,
        **geometry_arrays,
        **manual_arrays,
        **extra_arrays,
    )

    print("\nSaved:")
    print("  Corpus         :", corpus_path)


if __name__ == "__main__":
    main()
