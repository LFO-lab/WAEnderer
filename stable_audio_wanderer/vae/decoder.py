"""
VAE decoder helpers for runtime audio synthesis.
"""
from typing import Union
import numpy as np
import torch

from ..config import DEVICE, DTYPE
from .base import VAEAdapter


def decode_latents(adapter: VAEAdapter, z_raw: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Decode latent vectors into audio.

    Args:
        adapter: A loaded VAEAdapter instance.
        z_raw: [D] or [T, D] latent vectors (numpy or torch), where D = adapter latent_dim.

    Returns:
        Audio array with shape [T_audio, channels].
    """
    latent_dim = adapter.info().latent_dim

    if isinstance(z_raw, torch.Tensor):
        z_np = z_raw.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_raw, dtype=np.float32)

    if z_np.ndim == 1:
        z_np = z_np[None, :]
    if z_np.ndim != 2 or z_np.shape[1] != latent_dim:
        raise ValueError(
            f"Expected z_raw shape [{latent_dim}] or [T,{latent_dim}], got {z_np.shape}"
        )

    z_t = torch.from_numpy(z_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1, D, T]
    audio = adapter.decode(z_t)  # [1, C, T_audio]
    audio_np = audio.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)
    return audio_np
