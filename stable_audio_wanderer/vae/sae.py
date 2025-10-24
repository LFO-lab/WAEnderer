import numpy as np
import soundfile as sf
import torch, torchaudio
from diffusers import AutoencoderOobleck
from ..config import DEVICE, DTYPE, SR

def load_vae(repo_or_path="stabilityai/stable-audio-open-1.0"):
    return AutoencoderOobleck.from_pretrained(repo_or_path, subfolder="vae").to(DEVICE).eval()

def encode_full(ae, wav_np: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        x = torch.from_numpy(wav_np.T).unsqueeze(0).to(DEVICE, dtype=DTYPE)  # [1,2,T]
        out = ae.encode(x)
        z = out.latent_dist.sample().squeeze(0).permute(1,0).cpu().numpy()   # [T_lat, 64]
    return z

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
