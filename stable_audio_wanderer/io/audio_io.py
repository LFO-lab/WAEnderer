"""Audio I/O utilities."""
import numpy as np
import soundfile as sf
import torch
import torchaudio

from ..config import SR


def load_wav(path: str, target_sr: int = SR) -> np.ndarray:
    """
    Load a WAV file and resample to target sample rate.
    
    Returns stereo float32 array with shape [T, 2].
    """
    x, sr = sf.read(path, always_2d=True)
    if sr != target_sr:
        wt = torch.from_numpy(x.T).unsqueeze(0).float()
        res = torchaudio.functional.resample(wt, sr, target_sr)
        x = res.squeeze(0).numpy().T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    """
    Save audio array to WAV file.
    
    Args:
        path: Output file path
        audio: Audio array with shape [T, channels]
        sr: Sample rate
    """
    sf.write(path, audio, sr)
