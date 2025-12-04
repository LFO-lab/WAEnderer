from typing import Optional
import numpy as np
import soundfile as sf
import torch, torchaudio
from diffusers import AutoencoderOobleck
from ..config import DEVICE, DTYPE, SR, LATENT_HZ

# Long MPS convolutions fail for multi-minute inputs; encode in smaller windows instead.
MPS_AUTO_CHUNK_THRESHOLD_SEC = 240.0
MPS_AUTO_CHUNK_SEC = 60.0
MPS_CHUNK_OVERLAP_SEC = 1.0

def load_vae(repo_or_path="stabilityai/stable-audio-open-1.0"):
    return AutoencoderOobleck.from_pretrained(repo_or_path, subfolder="vae").to(DEVICE).eval()

def _encode_chunk(ae, wav_np: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
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
        return _encode_chunk(ae, wav_np)

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

def decode_window(ae, z_win_np: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        zt = torch.from_numpy(z_win_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,64,T_lat]
        x = ae.decode(zt).sample                                                # [1,2,T_audio]
        x = x.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)         # [T_audio,2]
    return x

def load_wav(path: str, target_sr=SR) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True)
    if sr != target_sr:
        wt = torch.from_numpy(x.T).unsqueeze(0).float()
        res = torchaudio.functional.resample(wt, sr, target_sr)
        x = res.squeeze(0).numpy().T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32)
