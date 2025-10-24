import numpy as np
import torch, torchaudio

SR = 44100
N_MFCC = 15
N_MELS = 64
N_FFT  = 2048
HOP_STFT = 512

_mfcc = torchaudio.transforms.MFCC(
    sample_rate=SR, n_mfcc=N_MFCC,
    melkwargs={"n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_STFT, "center": False}
)

def segment_mfcc_no_c0(x_np: np.ndarray, seg_start_samp: int, seg_len_samp: int, add_rms=False) -> np.ndarray:
    T = x_np.shape[0]
    s = int(np.clip(seg_start_samp, 0, max(0, T - 1)))
    e = int(np.clip(s + seg_len_samp, 0, T))
    seg = x_np[s:e]
    if seg.shape[0] < seg_len_samp:
        pad = np.zeros((seg_len_samp - seg.shape[0], seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg.astype(np.float32), pad], axis=0)
    if seg.shape[0] < N_FFT:
        pad2 = np.zeros((N_FFT - seg.shape[0], seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg, pad2], axis=0)

    x = torch.from_numpy(seg.T).unsqueeze(0)     # [1,2,L]
    x_mono = x.mean(dim=1, keepdim=True)         # [1,1,L]
    mfcc = _mfcc(x_mono).squeeze(0)              # [N_MFCC, frames]
    mfcc_no0 = mfcc[1:, :] if mfcc.shape[0] > 1 else mfcc

    m_mean = mfcc_no0.mean(dim=-1).numpy().astype(np.float32)
    m_var  = mfcc_no0.var(dim=-1).numpy().astype(np.float32)
    feats = np.concatenate([m_mean, m_var], axis=0)    # [2*(N_MFCC-1)]

    if add_rms:
        rms = np.sqrt((seg.astype(np.float32) ** 2).mean()).astype(np.float32)
        feats = np.concatenate([feats, [rms]], axis=0)

    return feats.reshape(-1).astype(np.float32)

def ensure_2d_stack(feat_list):
    F = np.vstack([np.asarray(f, dtype=np.float32).reshape(1, -1) for f in feat_list]).astype(np.float32)
    if not np.isfinite(F).all():
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return F
