"""
VAE encoding utilities for offline preprocessing.
Runtime playback uses live decoding.
"""
from typing import Optional, Union
import numpy as np
import torch
from ..config import DEVICE, DTYPE, SR, LATENT_HZ
from .base import VAEAdapter

# Long MPS convolutions fail for multi-minute inputs; encode in smaller windows instead.
MPS_AUTO_CHUNK_THRESHOLD_SEC = 240.0
MPS_AUTO_CHUNK_SEC = 60.0
MPS_CHUNK_OVERLAP_SEC = 1.0


def load_vae(repo_or_path="stabilityai/stable-audio-open-1.0") -> VAEAdapter:
    """Load the default Stable Audio Open VAE and return a VAEAdapter.

    This is a backward-compatible shim. New code should use
    ``load_vae_adapter(vae_id, ...)`` from ``stable_audio_wanderer.vae``.
    """
    from .adapters.stable_audio_open import StableAudioOpenAdapter
    return StableAudioOpenAdapter.load(repo_or_path=repo_or_path)


def _min_vae_samples(adapter: VAEAdapter) -> int:
    """Minimum audio samples to avoid encoder conv kernel crash."""
    info = adapter.info()
    return max(16, int(np.ceil(info.sample_rate / float(info.latent_hz))))


def _pad_wav_end(wav_np: np.ndarray, min_samples: int) -> np.ndarray:
    """Zero-pad waveform at the end to ensure wav_np.shape[0] >= min_samples."""
    wav_np = np.asarray(wav_np, dtype=np.float32)
    if wav_np.ndim == 1:
        wav_np = wav_np[:, None]
    if wav_np.shape[0] >= min_samples:
        return wav_np
    pad_len = int(min_samples - wav_np.shape[0])
    pad = np.zeros((pad_len, wav_np.shape[1]), dtype=wav_np.dtype)
    return np.concatenate([wav_np, pad], axis=0)


def _encode_chunk(adapter: VAEAdapter, wav_np: np.ndarray) -> np.ndarray:
    min_samp = _min_vae_samples(adapter)
    wav_np = _pad_wav_end(wav_np, min_samp)
    x = torch.from_numpy(wav_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1, C, T]
    z = adapter.encode(x)  # [1, latent_dim, T_lat]
    return z.squeeze(0).permute(1, 0).cpu().numpy()  # [T_lat, latent_dim]


def encode_full(
    adapter: VAEAdapter,
    wav_np: np.ndarray,
    chunk_sec: Optional[float] = None,
    overlap_sec: float = MPS_CHUNK_OVERLAP_SEC,
    sr: Optional[int] = None,
    latent_hz: Optional[float] = None,
) -> np.ndarray:
    """Encode audio to latent space, with optional chunking for long inputs.

    Args:
        adapter: A loaded VAEAdapter.
        wav_np: Audio waveform [T, channels] float32.
        chunk_sec: Force chunking at this duration (seconds). None = auto.
        overlap_sec: Overlap between chunks (seconds).
        sr: Sample rate override. Defaults to adapter.info().sample_rate.
        latent_hz: Latent rate override. Defaults to adapter.info().latent_hz.
    """
    info = adapter.info()
    sr = sr or info.sample_rate
    latent_hz = latent_hz or info.latent_hz
    latent_dim = info.latent_dim

    wav_np = np.asarray(wav_np, dtype=np.float32)
    total_samples = wav_np.shape[0]
    total_sec = total_samples / float(sr)
    samples_per_latent = sr / latent_hz

    use_chunking = False
    chunk_seconds = chunk_sec
    overlap_seconds = max(0.0, float(overlap_sec))

    if chunk_seconds is not None and chunk_seconds > 0.0:
        use_chunking = total_sec > chunk_seconds
    elif DEVICE == "mps" and total_sec > MPS_AUTO_CHUNK_THRESHOLD_SEC:
        chunk_seconds = MPS_AUTO_CHUNK_SEC
        overlap_seconds = MPS_CHUNK_OVERLAP_SEC
        use_chunking = True

    if not use_chunking:
        expected_total = max(1, int(round(max(total_samples, 0) / samples_per_latent)))
        z = _encode_chunk(adapter, wav_np)
        z_dim = z.shape[1] if (z.ndim == 2 and z.shape[0] > 0) else latent_dim
        if z.shape[0] == 0:
            return np.zeros((expected_total, z_dim), dtype=np.float32)
        if z.shape[0] > expected_total:
            return z[:expected_total]
        if z.shape[0] < expected_total:
            pad = np.repeat(z[-1:], expected_total - z.shape[0], axis=0)
            return np.concatenate([z, pad], axis=0)
        return z

    chunk_seconds = max(float(chunk_seconds), 1.0)
    chunk_samples = int(round(chunk_seconds * sr))
    overlap_samples = int(round(min(overlap_seconds, max(chunk_seconds - 1e-3, 0.0)) * sr))
    step = max(1, chunk_samples - overlap_samples)

    latents = []
    z_dim = latent_dim
    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_np[start:end]
        z_chunk = _encode_chunk(adapter, chunk)
        if z_chunk.ndim == 2 and z_chunk.shape[1] > 0:
            z_dim = z_chunk.shape[1]

        if start > 0 and overlap_samples > 0:
            overlap_lat = int(round(overlap_samples / samples_per_latent))
            overlap_lat = min(overlap_lat, max(z_chunk.shape[0] - 1, 0))
            if overlap_lat > 0:
                z_chunk = z_chunk[overlap_lat:]

        keep_samples = end - start - (overlap_samples if start > 0 else 0)
        target_len = max(1, int(round(keep_samples / samples_per_latent)))
        if z_chunk.shape[0] == 0:
            z_chunk = np.zeros((target_len, z_dim), dtype=np.float32)
        elif z_chunk.shape[0] > target_len:
            z_chunk = z_chunk[:target_len]
        elif z_chunk.shape[0] < target_len:
            pad = np.repeat(z_chunk[-1:], target_len - z_chunk.shape[0], axis=0)
            z_chunk = np.concatenate([z_chunk, pad], axis=0)

        latents.append(z_chunk)

    z_full = np.concatenate(latents, axis=0) if latents else np.zeros((0, z_dim), dtype=np.float32)
    expected_total = max(1, int(round(total_samples / samples_per_latent)))
    if z_full.shape[0] > expected_total:
        z_full = z_full[:expected_total]
    elif z_full.shape[0] < expected_total and z_full.shape[0] > 0:
        pad = np.repeat(z_full[-1:], expected_total - z_full.shape[0], axis=0)
        z_full = np.concatenate([z_full, pad], axis=0)
    return z_full
