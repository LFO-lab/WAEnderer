import torch

SR = 44100
LATENT_HZ = 21.5
NORM_CLAMP = 3.0
DTYPE = torch.float32

def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = pick_device()
torch.set_float32_matmul_precision("high")
