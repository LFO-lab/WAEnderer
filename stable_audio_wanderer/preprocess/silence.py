"""
Silence trimming helpers for corpus preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_RMS_FLOOR_DB = -120.0


@dataclass(frozen=True)
class SilenceTrimConfig:
    """Configuration for latent-frame silence trimming."""

    enabled: bool = True
    threshold_db: float = -45.0
    min_silence_sec: float = 0.25
    keep_silence_sec: float = 0.10
    rms_floor_db: float = DEFAULT_RMS_FLOOR_DB


@dataclass(frozen=True)
class SilenceTrimResult:
    """Summary of a silence-trimming pass."""

    keep_mask: np.ndarray
    frame_rms_db: np.ndarray
    original_frames: int
    kept_frames: int
    removed_frames: int
    raw_active_frames: int


def _compute_frame_rms_db(
    wav_stereo: np.ndarray,
    target_frames: int,
    floor_db: float = DEFAULT_RMS_FLOOR_DB,
) -> np.ndarray:
    """
    Compute a per-frame RMS envelope in dBFS with exactly target_frames samples.

    Frames are aligned by splitting the waveform into equal sample spans so the
    output length always matches the latent trajectory length.
    """
    wav = np.asarray(wav_stereo, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    if wav.ndim != 2:
        raise ValueError(f"Expected waveform [T, C], got {wav.shape}")

    n_frames = int(max(0, target_frames))
    if n_frames == 0:
        return np.zeros((0,), dtype=np.float32)

    total_samples = int(wav.shape[0])
    if total_samples <= 0:
        return np.full((n_frames,), float(floor_db), dtype=np.float32)

    edges = np.linspace(0, total_samples, n_frames + 1, dtype=np.int64)
    frame_db = np.empty((n_frames,), dtype=np.float32)
    floor_amp = float(10.0 ** (float(floor_db) / 20.0))

    for i in range(n_frames):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            start = int(min(start, total_samples - 1))
            end = int(min(total_samples, start + 1))
        frame = wav[start:end]
        rms_channels = np.sqrt(np.mean(np.square(frame, dtype=np.float32), axis=0) + 1e-12)
        # Use max-channel RMS so hard-panned material is not treated as silence.
        rms = float(np.max(rms_channels))
        rms = max(rms, floor_amp)
        frame_db[i] = np.float32(20.0 * np.log10(rms))

    return frame_db


def _fill_short_silent_gaps(active_mask: np.ndarray, min_silence_frames: int) -> np.ndarray:
    """
    Keep short silent gaps between active regions so small pauses survive intact.
    """
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    if active.size == 0 or not active.any():
        return active.copy()

    min_gap = int(max(1, min_silence_frames))
    filled = active.copy()
    idx = 0
    n = int(active.shape[0])

    while idx < n:
        if active[idx]:
            idx += 1
            continue

        start = idx
        while idx < n and not active[idx]:
            idx += 1
        end = idx
        gap_len = int(end - start)

        prev_active = start > 0 and bool(active[start - 1])
        next_active = end < n and bool(active[end])
        if prev_active and next_active and gap_len < min_gap:
            filled[start:end] = True

    return filled


def _pad_active_regions(active_mask: np.ndarray, keep_frames: int) -> np.ndarray:
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    if active.size == 0 or keep_frames <= 0:
        return active.copy()
    radius = int(keep_frames)
    kernel = np.ones((2 * radius + 1,), dtype=np.int32)
    # NumPy's "same" returns max(signal length, kernel length), so short
    # clips need an explicit centered crop to retain one mask entry per frame.
    full = np.convolve(active.astype(np.int32), kernel, mode="full")
    padded = full[radius:radius + active.size] > 0
    return padded.astype(bool)


def trim_silent_frames(
    wav_stereo: np.ndarray,
    latents: np.ndarray,
    descriptors: np.ndarray,
    cfg: SilenceTrimConfig,
    latent_hz: float,
) -> tuple[np.ndarray, np.ndarray, SilenceTrimResult]:
    """
    Trim long silent runs from latent and descriptor sequences using a shared mask.
    """
    z = np.asarray(latents, dtype=np.float32)
    desc = np.asarray(descriptors, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected latents [T, D], got {z.shape}")
    if desc.ndim != 2:
        raise ValueError(f"Expected descriptors [T, D], got {desc.shape}")
    if z.shape[0] != desc.shape[0]:
        raise ValueError(
            f"Latents/descriptors length mismatch for silence trimming: {z.shape[0]} vs {desc.shape[0]}"
        )

    frame_rms_db = _compute_frame_rms_db(
        wav_stereo=wav_stereo,
        target_frames=int(z.shape[0]),
        floor_db=float(cfg.rms_floor_db),
    )

    if not bool(cfg.enabled) or z.shape[0] == 0:
        keep_mask = np.ones((z.shape[0],), dtype=bool)
        result = SilenceTrimResult(
            keep_mask=keep_mask,
            frame_rms_db=frame_rms_db,
            original_frames=int(z.shape[0]),
            kept_frames=int(z.shape[0]),
            removed_frames=0,
            raw_active_frames=int(np.count_nonzero(frame_rms_db >= float(cfg.threshold_db))),
        )
        return z, desc, result

    raw_active = frame_rms_db >= float(cfg.threshold_db)
    if not np.any(raw_active):
        keep_mask = np.ones((z.shape[0],), dtype=bool)
        result = SilenceTrimResult(
            keep_mask=keep_mask,
            frame_rms_db=frame_rms_db,
            original_frames=int(z.shape[0]),
            kept_frames=int(z.shape[0]),
            removed_frames=0,
            raw_active_frames=0,
        )
        return z, desc, result

    min_silence_frames = int(max(1, round(float(cfg.min_silence_sec) * float(latent_hz))))
    keep_frames = int(max(0, round(float(cfg.keep_silence_sec) * float(latent_hz))))

    merged_active = _fill_short_silent_gaps(raw_active, min_silence_frames)
    keep_mask = _pad_active_regions(merged_active, keep_frames)
    if not np.any(keep_mask):
        keep_mask = raw_active.copy()

    z_trimmed = np.ascontiguousarray(z[keep_mask])
    desc_trimmed = np.ascontiguousarray(desc[keep_mask])
    result = SilenceTrimResult(
        keep_mask=keep_mask,
        frame_rms_db=frame_rms_db,
        original_frames=int(z.shape[0]),
        kept_frames=int(np.count_nonzero(keep_mask)),
        removed_frames=int(z.shape[0] - np.count_nonzero(keep_mask)),
        raw_active_frames=int(np.count_nonzero(raw_active)),
    )
    return z_trimmed, desc_trimmed, result
