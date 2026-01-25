import os
import torch

SR = 44100
# LATENT_HZ: VAE encoder downsampling rate (used during preprocess only)
LATENT_HZ = 21.5
DTYPE = torch.float32

def pick_device():
    # Allow forcing CPU via environment variable (workaround for MPS shader bugs)
    if os.environ.get("STABLE_AUDIO_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = pick_device()
torch.set_float32_matmul_precision("high")
