#!/usr/bin/env python3
import os, argparse, datetime, glob, numpy as np
from typing import List
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stable_audio_wanderer.config import SR, LATENT_HZ, DEVICE
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.features.mfcc import segment_mfcc_no_c0, ensure_2d_stack
from stable_audio_wanderer.dr.pca import fit_transform
from stable_audio_wanderer.io.corpus_io import save_latents_bundle, save_corpus


class LatentDataset(Dataset):
    def __init__(self, latents: np.ndarray):
        self.latents = torch.from_numpy(latents)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        return self.latents[idx]


class LatentProjector(nn.Module):
    def __init__(self, input_dim: int = 64, proj_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, proj_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1, eps=1e-8)


def augment_latents(x: torch.Tensor, noise_std: float, drop_prob: float, mix_alpha: float) -> torch.Tensor:
    y = x.clone()
    if noise_std > 0.0:
        y = y + noise_std * torch.randn_like(y)
    if drop_prob > 0.0:
        keep = (torch.rand_like(y) > drop_prob).to(y.dtype)
        y = y * keep
    if mix_alpha > 0.0 and y.size(0) > 1:
        perm = torch.randperm(y.size(0), device=y.device)
        y = (1.0 - mix_alpha) * y + mix_alpha * y[perm]
    return y


def info_nce_loss(u: torch.Tensor, v: torch.Tensor, tau: float) -> torch.Tensor:
    logits = torch.matmul(u, v.transpose(0, 1)) / tau
    labels = torch.arange(u.size(0), device=u.device)
    loss_uv = F.cross_entropy(logits, labels)
    loss_vu = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_uv + loss_vu)


def train_contrastive_projector(
    latents_np: np.ndarray,
    proj_dim: int,
    tau: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    noise_std: float,
    drop_prob: float,
    mix_alpha: float,
):
    dataset = LatentDataset(latents_np)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = LatentProjector(input_dim=latents_np.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            anchor = batch
            positive = augment_latents(batch, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=mix_alpha)
            u = model(anchor)
            v = model(positive)
            loss = info_nce_loss(u, v, tau=tau)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)
        epoch_loss = running_loss / max(total, 1)
        losses.append(float(epoch_loss))
        print(f"[g] epoch {epoch + 1}/{epochs} - InfoNCE loss: {epoch_loss:.4f}")
    return model, losses


def project_latents(model: LatentProjector, latents_np: np.ndarray, device: torch.device, batch_size: int = 4096):
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latents_np.shape[0], batch_size):
            chunk = torch.from_numpy(latents_np[start:start + batch_size]).to(device)
            proj = model(chunk).cpu().numpy()
            outputs.append(proj)
    return np.concatenate(outputs, axis=0)


def compute_segment_latents(latents_dict: dict, meta: np.ndarray, fallback_len: int) -> np.ndarray:
    seg_latents = []
    for fid, start, seg_len in meta:
        key = f"z_{int(fid)}"
        z_full = latents_dict[key]
        start = int(start)
        seg_len = int(seg_len) if int(seg_len) > 0 else int(fallback_len)
        end = min(z_full.shape[0], start + seg_len)
        z_slice = z_full[start:end]
        if z_slice.shape[0] == 0:
            z_slice = np.repeat(z_full[:1], seg_len, axis=0)
        if z_slice.shape[0] < seg_len:
            pad = np.repeat(z_slice[-1:], seg_len - z_slice.shape[0], axis=0)
            z_slice = np.concatenate([z_slice, pad], axis=0)
        seg_latents.append(z_slice.mean(axis=0))
    return np.stack(seg_latents, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Préprocess: MFCC→PCA + bundle latents + contrastive projector g.")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--pca_dim", type=int, default=2)
    ap.add_argument("--add_rms", type=int, default=0)
    ap.add_argument("--proj_dim", type=int, default=64, help="Output dimension d ≤ 64 for g(z).")
    ap.add_argument("--proj_epochs", type=int, default=10, help="Number of epochs for contrastive training.")
    ap.add_argument("--proj_batch", type=int, default=512, help="Batch size for contrastive projector training.")
    ap.add_argument("--proj_tau", type=float, default=0.1, help="InfoNCE temperature (0.05–0.2).")
    ap.add_argument("--proj_lr", type=float, default=1e-3, help="Learning rate for projector training.")
    ap.add_argument("--proj_noise", type=float, default=0.05, help="Gaussian noise std for latent augmentation.")
    ap.add_argument("--proj_drop", type=float, default=0.1, help="Drop probability for latent augmentation.")
    ap.add_argument("--proj_mix", type=float, default=0.15, help="Mixup coefficient for latent augmentation.")
    args = ap.parse_args()

    if not (1 <= args.proj_dim <= 64):
        ap.error("--proj_dim must be in [1, 64].")
    if args.proj_epochs < 1:
        ap.error("--proj_epochs must be ≥ 1.")
    if args.proj_batch < 1:
        ap.error("--proj_batch must be ≥ 1.")
    tau = float(np.clip(args.proj_tau, 0.05, 0.2))
    if abs(tau - args.proj_tau) > 1e-6:
        print(f"[warn] Clamped --proj_tau to {tau:.3f}")
    args.proj_tau = tau

    prefix = os.path.basename(args.out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= 2048, "seg_sec must be >= 2048/44100"

    ae = load_vae(args.pretrained)

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    feat_list, meta_list = [], []
    latent_sequences: List[np.ndarray] = []
    latents_dict = {}

    print("Encoding + feature extraction…")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        z_full = encode_full(ae, wav).astype(np.float32)  # [T_lat, 64]
        z_full = np.ascontiguousarray(z_full)
        latents_dict[f"z_{fid}"] = z_full
        latent_sequences.append(z_full)

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * SR))
            feats = segment_mfcc_no_c0(wav, samp_start, seg_len_samp, add_rms=bool(args.add_rms))
            feat_list.append(feats)
            meta_list.append((fid, t_lat, win_lat))

    if not latent_sequences:
        raise RuntimeError("No latent sequences were extracted.")

    F = ensure_2d_stack(feat_list)
    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    latent_stack = np.concatenate(latent_sequences, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    latent_stack_norm = (latent_stack - Z_mean[None, :]) / Z_std[None, :]
    latent_stack_norm = np.ascontiguousarray(latent_stack_norm.astype(np.float32))
    if latent_stack_norm.shape[0] < 2:
        raise RuntimeError("Need at least two latent frames to train the contrastive projector.")

    proj_batch = min(args.proj_batch, latent_stack_norm.shape[0])
    proj_batch = max(2, proj_batch)
    device = torch.device(DEVICE)

    print("Training contrastive projector g…")
    projector, g_losses = train_contrastive_projector(
        latent_stack_norm,
        args.proj_dim,
        args.proj_tau,
        args.proj_epochs,
        proj_batch,
        args.proj_lr,
        device,
        args.proj_noise,
        args.proj_drop,
        args.proj_mix,
    )

    seg_latents = compute_segment_latents(latents_dict, meta, win_lat)
    seg_latents_norm = (seg_latents - Z_mean[None, :]) / Z_std[None, :]
    seg_latents_norm = np.ascontiguousarray(seg_latents_norm.astype(np.float32))

    g_batch = max(proj_batch, 1024)
    g_embeddings = project_latents(projector, seg_latents_norm, device, batch_size=g_batch)

    g_min = g_embeddings.min(axis=0).astype(np.float32)
    g_max = g_embeddings.max(axis=0).astype(np.float32)
    g_range = np.maximum(g_max - g_min, 1e-6)
    GG = ((g_embeddings - g_min[None, :]) / g_range[None, :]).astype(np.float32)
    GG = np.clip(GG, 0.0, 1.0)

    projector_state = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
    projector_path = os.path.join(out_dir, f"{prefix}_projector_{ts}.pt")
    torch.save(
        {
            "state_dict": projector_state,
            "input_dim": 64,
            "proj_dim": int(args.proj_dim),
            "tau": float(args.proj_tau),
            "augment": {
                "noise_std": float(args.proj_noise),
                "drop_prob": float(args.proj_drop),
                "mix_alpha": float(args.proj_mix),
            },
            "loss_history": g_losses,
        },
        projector_path,
    )

    ZZ, dr_meta = fit_transform(F, args.pca_dim)

    latents_path = os.path.join(out_dir, f"{prefix}_latents_{ts}.npz")
    save_latents_bundle(latents_path, latents_dict, paths_arr)

    corpus_path = os.path.join(out_dir, f"{prefix}_corpus_{ts}.npz")
    dr_arrays = {k: (v if v is None else np.asarray(v, dtype=np.float32)) for k, v in dr_meta.items()}
    save_corpus(
        corpus_path,
        ZZ=ZZ,
        GG=GG,
        pca_dim=np.array(int(args.pca_dim), dtype=np.int32),
        g_dim=np.array(int(args.proj_dim), dtype=np.int32),
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        g_min=g_min,
        g_max=g_max,
        g_loss_history=np.asarray(g_losses, dtype=np.float32),
        g_model_path=np.array(projector_path),
        g_tau=np.array(float(args.proj_tau), dtype=np.float32),
        g_train_epochs=np.array(int(args.proj_epochs), dtype=np.int32),
        g_train_batch=np.array(int(proj_batch), dtype=np.int32),
        g_train_lr=np.array(float(args.proj_lr), dtype=np.float32),
        g_aug_noise=np.array(float(args.proj_noise), dtype=np.float32),
        g_aug_drop=np.array(float(args.proj_drop), dtype=np.float32),
        g_aug_mix=np.array(float(args.proj_mix), dtype=np.float32),
        latent_bundle_path=np.array(latents_path),
        **dr_arrays,
    )

    print("\nSaved:")
    print("  Latents   :", latents_path)
    print("  Projector :", projector_path)
    print("  Corpus    :", corpus_path)
    print("  Folder    :", out_dir)


if __name__ == "__main__":
    main()
