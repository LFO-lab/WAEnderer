#!/usr/bin/env python3
"""
Preprocess audio files into a model-specific latent corpus with geometry.
Pipeline:
  - Encode audio with SAME-S by default, or an optional VAE
  - Extract latent-rate timbre descriptors for manual navigation
  - Keep full latent trajectories per file (no mean pooling)
  - Normalize latents with global Z_mean/Z_std
  - Compute causal-context geometry/index over all frames
  - Compute manual embedding control points (PCA or UMAP) + robust ranges
"""
import os
import argparse
import datetime
import math
import threading
from typing import List, Dict

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from stable_audio_wanderer.config import (
    SR,
    LATENT_HZ,
    K_SHORT,
    EMA_ALPHA_FAST,
    EMA_ALPHA_SLOW,
    CONTEXT_PCA_DIM,
)
from stable_audio_wanderer.vae import load_vae_adapter
from stable_audio_wanderer.vae.sae import load_vae, encode_full
from stable_audio_wanderer.io.audio_io import load_wav
from stable_audio_wanderer.io.corpus_io import save_corpus
from stable_audio_wanderer.policy import (
    UnitGraphConfig,
    build_v2_unit_artifact,
    compute_latent_geometry,
    save_geometry_to_dict,
)
from stable_audio_wanderer.policy.sequence import compute_velocity_magnitudes
from stable_audio_wanderer.preprocess import SilenceTrimConfig, trim_silent_frames

MANUAL_EMBED_DIM = 4
MANUAL_PERCENTILE_LOW = 1.0
MANUAL_PERCENTILE_HIGH = 99.0
MANUAL_MFCC_TOTAL = 20
MANUAL_MFCC_SLICE = slice(1, 14)  # Use MFCC 1..13 (exclude c0)
MANUAL_N_FFT = 2048
MANUAL_ROLLOFF = 0.85
MANUAL_REDUCER_PCA = "pca"
MANUAL_REDUCER_UMAP = "umap"
SILENCE_THRESHOLD_DB_DEFAULT = -45.0
SILENCE_MIN_DURATION_SEC_DEFAULT = 0.25
SILENCE_KEEP_SEC_DEFAULT = 0.10

MANUAL_DESCRIPTOR_NAMES = [
    "mfcc_01",
    "mfcc_02",
    "mfcc_03",
    "mfcc_04",
    "mfcc_05",
    "mfcc_06",
    "mfcc_07",
    "mfcc_08",
    "mfcc_09",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
    "spec_centroid",
    "spec_spread",
    "spec_skew",
    "spec_kurtosis",
    "spec_rolloff",
    "spec_flatness",
    "spec_crest",
    "chroma_00",
    "chroma_01",
    "chroma_02",
    "chroma_03",
    "chroma_04",
    "chroma_05",
    "chroma_06",
    "chroma_07",
    "chroma_08",
    "chroma_09",
    "chroma_10",
    "chroma_11",
    "pitch_log_hz",
    "pitch_conf",
    "loudness_log_rms",
]

GROUP_SLICES = {
    "mfcc": slice(0, 13),
    "spectral_shape": slice(13, 18),  # centroid/spread/skew/kurtosis/rolloff
    "texture": slice(18, 20),  # flatness/crest
    "chroma": slice(20, 32),
    "pitch": slice(32, 34),
    "loudness": slice(34, 35),
}

GROUP_WEIGHTS = {
    "mfcc": 0.50,
    "spectral_shape": 0.22,
    "texture": 0.18,
    "chroma": 0.02,
    "pitch": 0.01,
    "loudness": 0.07,
}

PITCH_CONF_INDEX = 33
PITCH_CHROMA_SLICE = slice(20, 34)


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


def _build_mfcc_transform(hop_length: int, sample_rate: int = SR):
    """
    Build MFCC extractor using latent-aligned hop length.
    """
    return torchaudio.transforms.MFCC(
        sample_rate=int(sample_rate),
        n_mfcc=int(MANUAL_MFCC_TOTAL),
        melkwargs={
            "n_fft": int(MANUAL_N_FFT),
            "win_length": int(MANUAL_N_FFT),
            "hop_length": int(hop_length),
            "n_mels": 64,
            "center": False,
            "power": 2.0,
        },
    )


def _build_chroma_matrix(n_fft: int, sr: int) -> np.ndarray:
    n_bins = int(n_fft // 2 + 1)
    freqs = np.linspace(0.0, float(sr) * 0.5, n_bins, dtype=np.float32)
    chroma = np.zeros((12, n_bins), dtype=np.float32)
    valid = freqs > 1.0
    midi = np.zeros_like(freqs, dtype=np.float32)
    midi[valid] = 69.0 + 12.0 * np.log2(freqs[valid] / 440.0)
    pitch_class = np.round(midi).astype(np.int32) % 12
    for idx in range(n_bins):
        if valid[idx]:
            chroma[pitch_class[idx], idx] = 1.0
    return chroma


def _robust_standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(features, axis=0, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(features - center), axis=0, keepdims=True).astype(np.float32)
    scale = np.maximum(1e-6, mad * 1.4826).astype(np.float32)
    normalized = (features - center) / scale
    normalized = np.clip(normalized, -8.0, 8.0).astype(np.float32)
    return normalized, center.reshape(-1), scale.reshape(-1)


def _descriptor_scales() -> np.ndarray:
    scales = np.zeros((len(MANUAL_DESCRIPTOR_NAMES),), dtype=np.float32)
    for group_name, slc in GROUP_SLICES.items():
        n_dims = max(1, slc.stop - slc.start)
        group_weight = float(GROUP_WEIGHTS[group_name])
        scales[slc] = np.sqrt(group_weight / float(n_dims))
    return scales


def weight_descriptor_queries(desc, center, scale, weights):
    """Apply corpus-fitted transforms to raw descriptor frames, never refit them."""
    desc = np.asarray(desc, dtype=np.float32)
    normalized = np.clip((desc - center) / scale, -8.0, 8.0)
    weighted = (normalized * weights).astype(np.float32)
    weighted[..., PITCH_CHROMA_SLICE] *= np.clip(
        desc[..., PITCH_CONF_INDEX:PITCH_CONF_INDEX + 1], 0.0, 1.0
    ) ** 2
    return weighted


def compute_latent_aligned_descriptors(
    wav_stereo: np.ndarray,
    target_frames: int,
    mfcc_transform,
    hop_length: int,
    sample_rate: int = SR,
) -> np.ndarray:
    """
    Compute descriptor stack aligned to latent frame rate.
    Returns [target_frames, D].
    """
    wav = np.asarray(wav_stereo, dtype=np.float32)
    if wav.ndim != 2:
        raise ValueError(f"Expected stereo wav [T, C], got {wav.shape}")
    mono = wav.mean(axis=1, dtype=np.float32)
    min_len = int(MANUAL_N_FFT + 1)
    if mono.shape[0] < min_len:
        mono = np.pad(mono, (0, min_len - mono.shape[0]))
    x = torch.from_numpy(mono[None, :]).to(dtype=torch.float32)
    window = torch.hann_window(int(MANUAL_N_FFT), dtype=torch.float32)

    with torch.inference_mode():
        mfcc_full = (
            mfcc_transform(x).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
        )
        stft = torch.stft(
            x.squeeze(0),
            n_fft=int(MANUAL_N_FFT),
            hop_length=int(hop_length),
            win_length=int(MANUAL_N_FFT),
            window=window,
            center=False,
            return_complex=True,
        )
        power = (stft.abs().pow(2.0) + 1e-8).cpu().numpy().astype(np.float32)
        pitch_frame_time = float(hop_length) / float(sample_rate)
        pitch_frame_size = math.ceil(sample_rate * pitch_frame_time)
        pitch_frames = math.ceil(x.shape[-1] / pitch_frame_size)
        if pitch_frames == 1:
            # Torchaudio's median smoother cannot handle a single frame,
            # even with win_length=1 (its padding code concatenates no tensors).
            # Use two shorter analysis intervals, without extending the audio;
            # pitch is aligned back to target_frames below, as usual.
            pitch_frame_time = float(math.ceil(x.shape[-1] / 2)) / float(sample_rate)
            pitch_frame_size = math.ceil(sample_rate * pitch_frame_time)
            pitch_frames = math.ceil(x.shape[-1] / pitch_frame_size)
        # Preserve Torchaudio's default smoothing on clips with >=30 frames.
        # Short clips use an odd window no larger than their frame count,
        # except two-frame clips need 3: Torchaudio cannot smooth with 1.
        pitch_window = 30 if pitch_frames >= 30 else max(3, pitch_frames - (pitch_frames % 2 == 0))
        pitch_hz = AF.detect_pitch_frequency(
            x,
            sample_rate=int(sample_rate),
            frame_time=pitch_frame_time,
            win_length=pitch_window,
            freq_low=50,
            freq_high=2000,
        )
        pitch_hz = pitch_hz.squeeze(0).cpu().numpy().astype(np.float32)

    mfcc = _align_feature_frames(mfcc_full[:, MANUAL_MFCC_SLICE], int(target_frames))
    n_bins, n_frames_stft = power.shape
    freqs = np.linspace(0.0, float(sample_rate) * 0.5, n_bins, dtype=np.float32)[:, None]
    p_sum = power.sum(axis=0, keepdims=True) + 1e-8
    centroid = (freqs * power).sum(axis=0, keepdims=True) / p_sum
    diff = freqs - centroid
    spread = np.sqrt((power * (diff**2)).sum(axis=0, keepdims=True) / p_sum + 1e-8)
    skew = (power * (diff**3)).sum(axis=0, keepdims=True) / (p_sum * (spread**3 + 1e-8))
    kurtosis = (power * (diff**4)).sum(axis=0, keepdims=True) / (
        p_sum * (spread**4 + 1e-8)
    )
    cumulative = np.cumsum(power, axis=0)
    roll_threshold = float(MANUAL_ROLLOFF) * p_sum
    roll_idx = np.argmax(cumulative >= roll_threshold, axis=0)
    rolloff = freqs.reshape(-1)[roll_idx][None, :]
    flatness = np.exp(np.mean(np.log(power + 1e-8), axis=0, keepdims=True)) / (
        np.mean(power, axis=0, keepdims=True) + 1e-8
    )
    crest = np.max(power, axis=0, keepdims=True) / (np.mean(power, axis=0, keepdims=True) + 1e-8)

    chroma_map = _build_chroma_matrix(int(MANUAL_N_FFT), int(sample_rate))
    chroma = chroma_map @ power
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)

    loudness = np.log1p(np.sqrt(np.mean(power, axis=0, keepdims=True) + 1e-8))
    pitch_log = np.log1p(np.clip(pitch_hz, 0.0, None))[:, None]
    pitch_conf = np.clip((crest.reshape(-1) - 1.0) / 20.0, 0.0, 1.0)[:, None]

    spectral_block = np.concatenate(
        [
            centroid.T,
            spread.T,
            skew.T,
            kurtosis.T,
            rolloff.T,
            flatness.T,
            crest.T,
        ],
        axis=1,
    ).astype(np.float32)
    chroma_block = chroma.T.astype(np.float32)
    loudness_block = loudness.T.astype(np.float32)

    spectral_block = _align_feature_frames(spectral_block, int(target_frames))
    chroma_block = _align_feature_frames(chroma_block, int(target_frames))
    loudness_block = _align_feature_frames(loudness_block, int(target_frames))
    pitch_log = _align_feature_frames(pitch_log.astype(np.float32), int(target_frames))
    pitch_conf = _align_feature_frames(pitch_conf.astype(np.float32), int(target_frames))

    descriptor = np.concatenate(
        [mfcc, spectral_block, chroma_block, pitch_log, pitch_conf, loudness_block],
        axis=1,
    ).astype(np.float32)
    expected_dim = len(MANUAL_DESCRIPTOR_NAMES)
    if descriptor.shape[1] != expected_dim:
        raise RuntimeError(
            f"Descriptor dimension mismatch: {descriptor.shape[1]} vs expected {expected_dim}"
        )
    return descriptor


def _compute_pca_embedding(
    desc_weighted: np.ndarray,
    embed_dim: int,
) -> Dict[str, np.ndarray]:
    pca_mean = desc_weighted.mean(axis=0).astype(np.float32)
    centered = desc_weighted - pca_mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    desc_dim = int(desc_weighted.shape[1])
    components = np.zeros((embed_dim, desc_dim), dtype=np.float32)
    available = int(min(embed_dim, vt.shape[0], vt.shape[1]))
    if available > 0:
        components[:available] = vt[:available].astype(np.float32)

    points = (centered @ components.T).astype(np.float32)
    return {
        "manual_embed_points": points,
        "manual_pca_components": components,
        "manual_pca_mean": pca_mean,
    }


def _compute_umap_embedding(
    desc_weighted: np.ndarray,
    embed_dim: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
) -> Dict[str, np.ndarray]:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP reducer requested but umap-learn is not installed. "
            "Install with `pip install umap-learn`."
        ) from exc

    reducer = umap.UMAP(
        n_components=int(embed_dim),
        n_neighbors=int(max(2, umap_n_neighbors)),
        min_dist=float(max(0.0, umap_min_dist)),
        metric=str(umap_metric),
        random_state=int(umap_random_state),
        low_memory=True,
    )
    points = reducer.fit_transform(desc_weighted).astype(np.float32)
    return {
        "manual_embed_points": points,
        "manual_umap_n_neighbors": np.array(int(max(2, umap_n_neighbors)), dtype=np.int32),
        "manual_umap_min_dist": np.array(float(max(0.0, umap_min_dist)), dtype=np.float32),
        "manual_umap_metric": np.array([str(umap_metric)], dtype=np.str_),
        "manual_umap_random_state": np.array(int(umap_random_state), dtype=np.int32),
    }


def compute_manual_navigation_features(
    descriptor_concat: np.ndarray,
    reducer: str = MANUAL_REDUCER_PCA,
    embed_dim: int = MANUAL_EMBED_DIM,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.05,
    umap_metric: str = "euclidean",
    umap_random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Convert descriptor stack [N, D] to weighted timbre space + manual embedding controls.
    """
    desc = np.asarray(descriptor_concat, dtype=np.float32)
    expected_dim = len(MANUAL_DESCRIPTOR_NAMES)
    if desc.ndim != 2 or desc.shape[1] != expected_dim:
        raise ValueError(f"Expected descriptor stack [N, {expected_dim}], got {desc.shape}")
    if desc.shape[0] == 0:
        raise ValueError("No descriptor frames available for manual feature extraction.")
    if int(embed_dim) < 3 or int(embed_dim) > 4:
        raise ValueError(
            f"manual embedding dimension must be 3 or 4 for current controls, got {embed_dim}"
        )

    desc_norm, desc_center, desc_scale = _robust_standardize(desc)
    scales = _descriptor_scales()
    desc_weighted = weight_descriptor_queries(desc, desc_center, desc_scale, scales)

    reducer_name = str(reducer).lower().strip()
    if reducer_name == MANUAL_REDUCER_PCA:
        embedding_arrays = _compute_pca_embedding(desc_weighted, int(embed_dim))
    elif reducer_name == MANUAL_REDUCER_UMAP:
        embedding_arrays = _compute_umap_embedding(
            desc_weighted=desc_weighted,
            embed_dim=int(embed_dim),
            umap_n_neighbors=int(umap_n_neighbors),
            umap_min_dist=float(umap_min_dist),
            umap_metric=str(umap_metric),
            umap_random_state=int(umap_random_state),
        )
    else:
        raise ValueError(f"Unsupported manual reducer: {reducer_name}")

    points = np.asarray(embedding_arrays["manual_embed_points"], dtype=np.float32)
    p01 = np.percentile(points, MANUAL_PERCENTILE_LOW, axis=0).astype(np.float32)
    p99 = np.percentile(points, MANUAL_PERCENTILE_HIGH, axis=0).astype(np.float32)
    p99 = np.maximum(p99, p01 + 1e-6).astype(np.float32)

    out = {
        "manual_embed_points": points,
        "manual_embed_dim": np.array(int(embed_dim), dtype=np.int32),
        "manual_embed_reducer": np.array([reducer_name], dtype=np.str_),
        "manual_desc_weighted": desc_weighted.astype(np.float32),
        "manual_desc_center": desc_center.astype(np.float32),
        "manual_desc_scale": desc_scale.astype(np.float32),
        "manual_desc_scales": scales.astype(np.float32),
        "manual_desc_names": np.array(MANUAL_DESCRIPTOR_NAMES, dtype=np.str_),
        "manual_desc_dim": np.array(int(expected_dim), dtype=np.int32),
        "manual_pitch_confidence_index": np.array(int(PITCH_CONF_INDEX), dtype=np.int32),
        "manual_fader_p01": p01,
        "manual_fader_p99": p99,
        "manual_n_mfcc": np.array(int(MANUAL_MFCC_TOTAL), dtype=np.int32),
        "manual_n_mfcc_used": np.array(
            int(MANUAL_MFCC_SLICE.stop - MANUAL_MFCC_SLICE.start), dtype=np.int32
        ),
        "manual_percentile_low": np.array(float(MANUAL_PERCENTILE_LOW), dtype=np.float32),
        "manual_percentile_high": np.array(float(MANUAL_PERCENTILE_HIGH), dtype=np.float32),
    }
    out.update(embedding_arrays)
    return out


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
    vae,  # VAEAdapter
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


def run_preprocess(
    audio_dir: str,
    out_prefix: str,
    pretrained: str = "stabilityai/stable-audio-open-1.0",
    vae=None,
    vae_id: str = "",
    vae_weight_path: str = "",
    progress_callback=None,
    cancel_event=None,
    latent_nav_k: int = 32,
    encode_chunk_sec: float = 60.0,
    encode_chunk_overlap_sec: float = 1.0,
    trim_silence: bool = True,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB_DEFAULT,
    silence_min_duration_sec: float = SILENCE_MIN_DURATION_SEC_DEFAULT,
    silence_keep_sec: float = SILENCE_KEEP_SEC_DEFAULT,
    compute_decoder_targets: bool = False,
    manual_reducer: str = MANUAL_REDUCER_PCA,
    manual_embed_dim: int = MANUAL_EMBED_DIM,
    manual_umap_n_neighbors: int = 30,
    manual_umap_min_dist: float = 0.05,
    manual_umap_metric: str = "euclidean",
    manual_umap_random_state: int = 42,
    reorg_min_sec: float = 2.0,
    reorg_max_sec: float = 10.0,
    reorg_target_sec: float = 5.0,
    reorg_candidate_k: int = 64,
    reorg_graph_k: int = 24,
    reorg_weight_entry: float = 0.70,
    reorg_weight_delta: float = 0.30,
    reorg_crossfile_penalty: float = 0.10,
    reorg_boundary_smoothness_weight: float = 0.35,
) -> dict:
    """
    Run the full preprocessing pipeline.

    Args:
        audio_dir: Path to directory containing .wav files.
        out_prefix: Name prefix for the output corpus directory.
        pretrained: HuggingFace model ID for the VAE.
        vae: Pre-loaded VAE model (avoids double load when shared with perform).
        progress_callback: Optional callable(dict) for progress events.
        cancel_event: Optional threading.Event for cancellation.
        **remaining kwargs: All parameter defaults match CLI argparse defaults.

    Returns:
        dict with keys: corpus_dir, corpus_path, reorg_sidecar_path,
        total_files, total_frames, silence_removed_pct, vae (the loaded model).
    """

    def _emit(event_type, **data):
        if progress_callback is not None:
            progress_callback({"event": event_type, **data})

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    prefix = os.path.basename(out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    _emit("scan_start", audio_dir=audio_dir)
    from stable_audio_wanderer.io.audio_discovery import find_wav_files
    paths = find_wav_files(audio_dir)
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir or its subfolders")
    _emit("scan_done", file_count=len(paths), files=[os.path.relpath(p, audio_dir) for p in paths])

    if _cancelled():
        return {"corpus_dir": out_dir, "cancelled": True}

    _emit("vae_load_start")
    # Load VAE adapter: prefer vae_id (new path), fall back to pretrained (compat)
    from stable_audio_wanderer.vae.base import VAEAdapter
    if isinstance(vae, VAEAdapter):
        adapter = vae
    elif vae_id:
        from stable_audio_wanderer.vae.encoder_availability import prepare_encoder_weights
        def model_progress(event):
            if progress_callback is not None:
                progress_callback(event)
            else:
                print(event['detail'], flush=True)
        prepare_encoder_weights(vae_id, model_progress, cancel_event or threading.Event())
        if _cancelled():
            return {"corpus_dir": out_dir, "cancelled": True}
        adapter = load_vae_adapter(vae_id, weight_path=vae_weight_path)
    elif vae is not None:
        # Legacy: raw model passed in — wrap in Stable Audio adapter
        from stable_audio_wanderer.vae.adapters.stable_audio_open import StableAudioOpenAdapter
        adapter = StableAudioOpenAdapter(vae)
    else:
        adapter = load_vae(pretrained)
    ae = adapter  # alias for backward compat in this file
    vae_info = adapter.info()
    corpus_sr = vae_info.sample_rate
    corpus_latent_hz = vae_info.latent_hz
    _emit("vae_load_done")

    latent_sequences_raw: List[np.ndarray] = []
    manual_descriptor_sequences: List[np.ndarray] = []
    encode_chunk = float(encode_chunk_sec)
    encode_overlap = float(encode_chunk_overlap_sec)
    chunk_sec_val = encode_chunk if encode_chunk > 0.0 else None
    latent_hop = max(1, int(round(float(corpus_sr) / float(corpus_latent_hz))))
    mfcc_transform = _build_mfcc_transform(hop_length=latent_hop, sample_rate=corpus_sr)
    silence_cfg = SilenceTrimConfig(
        enabled=bool(trim_silence),
        threshold_db=float(silence_threshold_db),
        min_silence_sec=float(max(0.0, silence_min_duration_sec)),
        keep_silence_sec=float(max(0.0, silence_keep_sec)),
    )
    silence_original_frames = 0
    silence_removed_frames = 0

    _emit("encode_start", total_files=len(paths))
    print("Encoding audio with VAE + extracting latent-aligned timbre descriptors...")
    for fid, p in enumerate(tqdm(paths)):
        if _cancelled():
            return {"corpus_dir": out_dir, "cancelled": True}
        _emit("encode_file_start", file_index=fid, file_name=os.path.basename(p), total_files=len(paths))
        wav = load_wav(p, target_sr=corpus_sr)
        z_full = encode_full(adapter, wav, chunk_sec=chunk_sec_val, overlap_sec=encode_overlap).astype(np.float32)
        descriptor_seq = compute_latent_aligned_descriptors(
            wav_stereo=wav,
            target_frames=z_full.shape[0],
            mfcc_transform=mfcc_transform,
            hop_length=latent_hop,
            sample_rate=corpus_sr,
        )
        z_full, descriptor_seq, trim_result = trim_silent_frames(
            wav_stereo=wav,
            latents=z_full,
            descriptors=descriptor_seq,
            cfg=silence_cfg,
            latent_hz=float(corpus_latent_hz),
        )
        silence_original_frames += int(trim_result.original_frames)
        silence_removed_frames += int(trim_result.removed_frames)
        latent_sequences_raw.append(np.ascontiguousarray(z_full))
        manual_descriptor_sequences.append(np.ascontiguousarray(descriptor_seq))
        _emit(
            "encode_file_done",
            file_index=fid,
            file_name=os.path.basename(p),
            total_files=len(paths),
            frames=z_full.shape[0],
            original_frames=int(trim_result.original_frames),
            removed_frames=int(trim_result.removed_frames),
        )

    if not latent_sequences_raw:
        raise RuntimeError("No latent sequences were extracted.")
    if len(manual_descriptor_sequences) != len(latent_sequences_raw):
        raise RuntimeError("Manual descriptor extraction count mismatch.")

    if _cancelled():
        return {"corpus_dir": out_dir, "cancelled": True}

    paths_arr = np.array(paths)
    silence_removed_pct = 0.0
    if silence_cfg.enabled:
        silence_kept_frames = int(silence_original_frames - silence_removed_frames)
        silence_removed_pct = 100.0 * float(silence_removed_frames) / float(max(1, silence_original_frames))
        print(
            "Silence trim summary:",
            f"kept={silence_kept_frames}/{silence_original_frames} frames",
            f"removed={silence_removed_frames} ({silence_removed_pct:.1f}%)",
            f"threshold={silence_cfg.threshold_db:.1f} dB",
            f"min_silence={silence_cfg.min_silence_sec:.2f}s",
            f"keep={silence_cfg.keep_silence_sec:.2f}s",
        )

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
    descriptor_concat = np.concatenate(manual_descriptor_sequences, axis=0).astype(np.float32)
    file_offsets = np.asarray(file_offsets, dtype=np.int64)
    meta = np.asarray(meta_list, dtype=np.int32)
    frame_file_ids = meta[:, 0].astype(np.int32)
    frame_t = meta[:, 1].astype(np.int32)

    if descriptor_concat.shape[0] != Z_concat.shape[0]:
        raise RuntimeError(
            f"Manual descriptor frame count mismatch: {descriptor_concat.shape[0]} vs latent frames {Z_concat.shape[0]}"
        )

    if _cancelled():
        return {"corpus_dir": out_dir, "cancelled": True}

    _emit("embedding_start", reducer=manual_reducer, dim=int(manual_embed_dim))
    print(
        "Computing manual navigation embedding features "
        f"(reducer={manual_reducer}, dim={int(manual_embed_dim)})..."
    )
    manual_arrays = compute_manual_navigation_features(
        descriptor_concat,
        reducer=str(manual_reducer),
        embed_dim=int(manual_embed_dim),
        umap_n_neighbors=int(manual_umap_n_neighbors),
        umap_min_dist=float(manual_umap_min_dist),
        umap_metric=str(manual_umap_metric),
        umap_random_state=int(manual_umap_random_state),
    )
    p01 = manual_arrays["manual_fader_p01"]
    p99 = manual_arrays["manual_fader_p99"]
    desc_dim = int(manual_arrays["manual_desc_dim"])
    reducer_name = str(np.asarray(manual_arrays["manual_embed_reducer"]).reshape(-1)[0])
    print(
        "  Manual embedding points:",
        manual_arrays["manual_embed_points"].shape,
        f"reducer={reducer_name}",
        f"desc_dim={desc_dim}",
        f"(fader p01 mean={p01.mean():.3f}, p99 mean={p99.mean():.3f})",
    )
    _emit("embedding_done")

    if _cancelled():
        return {"corpus_dir": out_dir, "cancelled": True}

    _emit("geometry_start", k=int(latent_nav_k))
    print(f"Computing latent geometry (k={latent_nav_k})...")
    geometry = compute_latent_geometry(
        Z_concat,
        meta,
        k=int(latent_nav_k),
        k_short=int(K_SHORT),
        ema_alpha_fast=float(EMA_ALPHA_FAST),
        ema_alpha_slow=float(EMA_ALPHA_SLOW),
        pca_dim=int(CONTEXT_PCA_DIM),
    )
    geometry_arrays = save_geometry_to_dict(geometry)
    _emit("geometry_done")

    # Compute continuous window targets for adaptive decoding
    print("Computing window targets from latent velocity...")
    window_targets_log2 = compute_window_targets(Z_concat, meta)
    print(f"  Window targets (log2): min={window_targets_log2.min():.2f}, "
          f"max={window_targets_log2.max():.2f}, mean={window_targets_log2.mean():.2f}")

    # Optional decoder quality targets
    extra_arrays = {}
    if compute_decoder_targets:
        print("Computing decoder quality targets (this may take a while)...")
        try:
            decoder_targets_arr = _compute_decoder_quality_targets(
                normalized_by_file, meta, ae, Z_mean, Z_std
            )
            extra_arrays["decoder_quality_targets_log2"] = decoder_targets_arr
            print(f"  Decoder quality targets: min={decoder_targets_arr.min():.2f}, "
                  f"max={decoder_targets_arr.max():.2f}, mean={decoder_targets_arr.mean():.2f}")
        except Exception as e:
            print(f"  [warn] Failed to compute decoder quality targets: {e}")

    if _cancelled():
        return {"corpus_dir": out_dir, "cancelled": True}

    corpus_path = os.path.join(out_dir, "corpus.npz")
    _emit("units_start")
    print("Building reorganized unit artifact...")
    reorg_cfg = UnitGraphConfig(
        min_sec=float(reorg_min_sec),
        max_sec=float(reorg_max_sec),
        target_sec=float(reorg_target_sec),
        latent_hz=float(corpus_latent_hz),
        candidate_k=int(reorg_candidate_k),
        graph_k=int(reorg_graph_k),
        weight_entry=float(reorg_weight_entry),
        weight_delta=float(reorg_weight_delta),
        crossfile_penalty=float(reorg_crossfile_penalty),
        boundary_smoothness_weight=float(reorg_boundary_smoothness_weight),
    )
    reorg_artifact = build_v2_unit_artifact(
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        desc_weighted=manual_arrays["manual_desc_weighted"],
        cfg=reorg_cfg,
        source_corpus_path=os.path.abspath(corpus_path),
    )
    reorg_sidecar_path = os.path.join(out_dir, "policy_v2_units.npz")
    np.savez_compressed(reorg_sidecar_path, **reorg_artifact)
    reorg_embed_arrays = {
        key: value
        for key, value in reorg_artifact.items()
        if key.startswith("unit_") or key == "frame_to_unit"
    }
    print(
        "  Reorganized units:",
        int(np.asarray(reorg_artifact["unit_start_idx"], dtype=np.int32).shape[0]),
        f"(graph_k={int(np.asarray(reorg_artifact['unit_graph_neighbors']).shape[1])})",
    )
    _emit("units_done")

    # Save corpus
    _emit("save_start")
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
        vae_id=np.array(vae_info.vae_id),
        sr=np.array(int(corpus_sr), dtype=np.int32),
        latent_hz=np.array(float(corpus_latent_hz), dtype=np.float32),
        segment_dur=np.array(float(1.0 / corpus_latent_hz), dtype=np.float32),
        hop_dur=np.array(float(1.0 / corpus_latent_hz), dtype=np.float32),
        latent_nav_k=np.array(int(latent_nav_k), dtype=np.int32),
        k_short=np.array(int(K_SHORT), dtype=np.int32),
        ema_alpha_fast=np.array(float(EMA_ALPHA_FAST), dtype=np.float32),
        ema_alpha_slow=np.array(float(EMA_ALPHA_SLOW), dtype=np.float32),
        context_pca_dim=np.array(int(CONTEXT_PCA_DIM), dtype=np.int32),
        trim_silence=np.array(int(silence_cfg.enabled), dtype=np.int32),
        silence_threshold_db=np.array(float(silence_cfg.threshold_db), dtype=np.float32),
        silence_min_duration_sec=np.array(float(silence_cfg.min_silence_sec), dtype=np.float32),
        silence_keep_sec=np.array(float(silence_cfg.keep_silence_sec), dtype=np.float32),
        silence_original_frames=np.array(int(silence_original_frames), dtype=np.int64),
        silence_removed_frames=np.array(int(silence_removed_frames), dtype=np.int64),
        window_targets_log2=window_targets_log2,
        **geometry_arrays,
        **manual_arrays,
        **reorg_embed_arrays,
        **extra_arrays,
    )

    print("\nSaved:")
    print("  Corpus         :", corpus_path)
    print("  Reorg units    :", reorg_sidecar_path)

    result = {
        "corpus_dir": out_dir,
        "corpus_path": corpus_path,
        "reorg_sidecar_path": reorg_sidecar_path,
        "total_files": len(paths),
        "total_frames": int(Z_concat.shape[0]),
        "silence_removed_pct": silence_removed_pct,
        "vae": adapter,
        "cancelled": False,
    }
    _emit("complete", **{k: v for k, v in result.items() if k != "vae"})
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess: VAE latents -> corpus + geometry."
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0",
                    help="HuggingFace model ID (legacy, use --vae_id instead).")
    ap.add_argument("--vae_id", default="same_s", help="VAE adapter ID (e.g. stable_audio_open, same_s, ear_vae_44k).")
    ap.add_argument("--vae_weight_path", default="", help="Path to local weight file (for VAEs that require it).")
    ap.add_argument("--latent_nav_k", type=int, default=32, help="k for latent kNN geometry.")
    ap.add_argument("--encode_chunk_sec", type=float, default=60.0,
                    help="Chunk size (seconds) for VAE encoding. Set 0 to disable chunking.")
    ap.add_argument("--encode_chunk_overlap_sec", type=float, default=1.0,
                    help="Chunk overlap (seconds) for VAE encoding.")
    ap.add_argument(
        "--trim_silence",
        dest="trim_silence",
        action="store_true",
        default=True,
        help="Trim long silent runs from the stored corpus sequences (default: enabled).",
    )
    ap.add_argument(
        "--no_trim_silence",
        dest="trim_silence",
        action="store_false",
        help="Disable silence trimming during preprocessing.",
    )
    ap.add_argument(
        "--silence_threshold_db",
        type=float,
        default=SILENCE_THRESHOLD_DB_DEFAULT,
        help="RMS threshold in dBFS below which frames are considered silent.",
    )
    ap.add_argument(
        "--silence_min_duration_sec",
        type=float,
        default=SILENCE_MIN_DURATION_SEC_DEFAULT,
        help="Only silent runs at least this long are removed from the corpus.",
    )
    ap.add_argument(
        "--silence_keep_sec",
        type=float,
        default=SILENCE_KEEP_SEC_DEFAULT,
        help="Silence padding kept around active regions after trimming.",
    )
    ap.add_argument("--compute_decoder_targets", action="store_true",
                    help="Compute decoder quality targets (expensive: 11 VAE decodes per segment).")
    ap.add_argument(
        "--manual_reducer",
        choices=[MANUAL_REDUCER_PCA, MANUAL_REDUCER_UMAP],
        default=MANUAL_REDUCER_PCA,
        help="Reducer used for manual timbre embedding.",
    )
    ap.add_argument(
        "--manual_embed_dim",
        type=int,
        default=MANUAL_EMBED_DIM,
        help="Manual embedding dimensionality (3 or 4; dim4 maps to W/color axis).",
    )
    ap.add_argument(
        "--manual_umap_n_neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors (used when --manual_reducer=umap).",
    )
    ap.add_argument(
        "--manual_umap_min_dist",
        type=float,
        default=0.05,
        help="UMAP min_dist (used when --manual_reducer=umap).",
    )
    ap.add_argument(
        "--manual_umap_metric",
        type=str,
        default="euclidean",
        help="UMAP metric (used when --manual_reducer=umap).",
    )
    ap.add_argument(
        "--manual_umap_random_state",
        type=int,
        default=42,
        help="UMAP random state for deterministic embeddings.",
    )
    ap.add_argument(
        "--reorg_min_sec",
        type=float,
        default=2.0,
        help="Reorganized mode: minimum morphology unit duration (seconds).",
    )
    ap.add_argument(
        "--reorg_max_sec",
        type=float,
        default=10.0,
        help="Reorganized mode: maximum morphology unit duration (seconds).",
    )
    ap.add_argument(
        "--reorg_target_sec",
        type=float,
        default=5.0,
        help="Reorganized mode: target morphology unit duration (seconds).",
    )
    ap.add_argument(
        "--reorg_candidate_k",
        type=int,
        default=64,
        help="Reorganized mode: candidate transition pool size per unit.",
    )
    ap.add_argument(
        "--reorg_graph_k",
        type=int,
        default=24,
        help="Reorganized mode: outgoing transitions saved per unit.",
    )
    ap.add_argument(
        "--reorg_weight_entry",
        type=float,
        default=0.70,
        help="Reorganized mode: exit->entry timbre continuity weight.",
    )
    ap.add_argument(
        "--reorg_weight_delta",
        type=float,
        default=0.30,
        help="Reorganized mode: descriptor-delta compatibility weight.",
    )
    ap.add_argument(
        "--reorg_crossfile_penalty",
        type=float,
        default=0.10,
        help="Reorganized mode: additive penalty for cross-file transitions.",
    )
    ap.add_argument(
        "--reorg_boundary_smoothness_weight",
        type=float,
        default=0.35,
        help="Reorganized mode: boundary preference for low descriptor-velocity cuts.",
    )

    args = ap.parse_args()
    run_preprocess(
        audio_dir=args.audio_dir,
        out_prefix=args.out_prefix,
        pretrained=args.pretrained,
        vae_id=args.vae_id,
        vae_weight_path=args.vae_weight_path,
        latent_nav_k=args.latent_nav_k,
        encode_chunk_sec=args.encode_chunk_sec,
        encode_chunk_overlap_sec=args.encode_chunk_overlap_sec,
        trim_silence=args.trim_silence,
        silence_threshold_db=args.silence_threshold_db,
        silence_min_duration_sec=args.silence_min_duration_sec,
        silence_keep_sec=args.silence_keep_sec,
        compute_decoder_targets=args.compute_decoder_targets,
        manual_reducer=args.manual_reducer,
        manual_embed_dim=args.manual_embed_dim,
        manual_umap_n_neighbors=args.manual_umap_n_neighbors,
        manual_umap_min_dist=args.manual_umap_min_dist,
        manual_umap_metric=args.manual_umap_metric,
        manual_umap_random_state=args.manual_umap_random_state,
        reorg_min_sec=args.reorg_min_sec,
        reorg_max_sec=args.reorg_max_sec,
        reorg_target_sec=args.reorg_target_sec,
        reorg_candidate_k=args.reorg_candidate_k,
        reorg_graph_k=args.reorg_graph_k,
        reorg_weight_entry=args.reorg_weight_entry,
        reorg_weight_delta=args.reorg_weight_delta,
        reorg_crossfile_penalty=args.reorg_crossfile_penalty,
        reorg_boundary_smoothness_weight=args.reorg_boundary_smoothness_weight,
    )


if __name__ == "__main__":
    main()
