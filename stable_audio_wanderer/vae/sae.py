"""
VAE encoding utilities for offline preprocessing.
Runtime playback uses pre-rendered grains instead of live decoding.
"""
from typing import Optional
import numpy as np
import torch
from diffusers import AutoencoderOobleck
from ..config import DEVICE, DTYPE, SR, LATENT_HZ
from ..io.audio_io import load_wav  # Re-export for backward compatibility

# Long MPS convolutions fail for multi-minute inputs; encode in smaller windows instead.
MPS_AUTO_CHUNK_THRESHOLD_SEC = 240.0
MPS_AUTO_CHUNK_SEC = 60.0
MPS_CHUNK_OVERLAP_SEC = 1.0

# The Stable Audio Open VAE downsamples audio to LATENT_HZ. Very short inputs (or tail chunks)
# can be shorter than the encoder's first conv kernel and crash. Zero-pad to at least ~1 latent step.
MIN_VAE_SAMPLES = max(16, int(np.ceil(SR / float(LATENT_HZ))))

def load_vae(repo_or_path="stabilityai/stable-audio-open-1.0"):
    return AutoencoderOobleck.from_pretrained(repo_or_path, subfolder="vae").to(DEVICE).eval()

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

def _encode_chunk(ae, wav_np: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        wav_np = _pad_wav_end(wav_np, MIN_VAE_SAMPLES)
        x = torch.from_numpy(wav_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,2,T]
        out = ae.encode(x)
        z = out.latent_dist.sample().squeeze(0).permute(1,0).cpu().numpy()   # [T_lat, 64]
    return z

def encode_full(ae, wav_np: np.ndarray, chunk_sec: Optional[float] = None, overlap_sec: float = MPS_CHUNK_OVERLAP_SEC) -> np.ndarray:
    wav_np = np.asarray(wav_np, dtype=np.float32)
    total_samples = wav_np.shape[0]
    total_sec = total_samples / float(SR)
    samples_per_latent = SR / LATENT_HZ

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
        # Pad short inputs to avoid encoder conv kernel crashes, but keep output length consistent
        # with the original (unpadded) audio duration.
        expected_total = max(1, int(round(max(total_samples, 0) / samples_per_latent)))
        z = _encode_chunk(ae, wav_np)
        z_dim = z.shape[1] if (z.ndim == 2 and z.shape[0] > 0) else 64
        if z.shape[0] == 0:
            return np.zeros((expected_total, z_dim), dtype=np.float32)
        if z.shape[0] > expected_total:
            return z[:expected_total]
        if z.shape[0] < expected_total:
            pad = np.repeat(z[-1:], expected_total - z.shape[0], axis=0)
            return np.concatenate([z, pad], axis=0)
        return z

    chunk_seconds = max(float(chunk_seconds), 1.0)
    chunk_samples = int(round(chunk_seconds * SR))
    overlap_samples = int(round(min(overlap_seconds, max(chunk_seconds - 1e-3, 0.0)) * SR))
    step = max(1, chunk_samples - overlap_samples)

    latents = []
    z_dim = 64
    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_np[start:end]
        z_chunk = _encode_chunk(ae, chunk)
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

# decode_window removed - runtime uses pre-rendered grains instead of live VAE decoding
# load_wav moved to stable_audio_wanderer.io.audio_io (re-exported above for compatibility)
