"""
VAE decoder helpers for runtime audio synthesis.
"""
from typing import Union
import numpy as np
import torch

from ..config import DEVICE, DTYPE


@torch.inference_mode()
def decode_latents(ae, z_raw: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Decode latent vectors into audio.

    Args:
        ae: Loaded AutoencoderOobleck instance
        z_raw: [64] or [T, 64] latent vectors (numpy or torch)

    Returns:
        Audio array with shape [T_audio, 2]
    """
    if isinstance(z_raw, torch.Tensor):
        z_np = z_raw.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_raw, dtype=np.float32)

    if z_np.ndim == 1:
        z_np = z_np[None, :]
    if z_np.ndim != 2 or z_np.shape[1] != 64:
        raise ValueError(f"Expected z_raw shape [64] or [T,64], got {z_np.shape}")

    z_t = torch.from_numpy(z_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1, 64, T]
    audio = ae.decode(z_t).sample  # [1, 2, T_audio]
    audio_np = audio.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)
    return audio_np
