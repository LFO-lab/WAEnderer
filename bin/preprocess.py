#!/usr/bin/env python3
import os, argparse, datetime, glob, numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors

from stable_audio_wanderer.config import SR, LATENT_HZ, DEVICE
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.features.mfcc import segment_mfcc_no_c0, ensure_2d_stack
from stable_audio_wanderer.dr.pca import fit_transform
from stable_audio_wanderer.io.corpus_io import save_latents_bundle, save_corpus
from stable_audio_wanderer.models.latent_ar import LatentAutoregressiveCNN


class LatentDataset(Dataset):
    def __init__(self, latents: np.ndarray, return_index: bool = False):
        self.latents = torch.from_numpy(latents)
        self.return_index = return_index

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        if self.return_index:
            return self.latents[idx], idx
        return self.latents[idx]


class LatentSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        context: int,
        max_future: int,
        jitter: int = 0,
        mix_prob: float = 0.0,
    ):
        self.sequences = [torch.from_numpy(seq) for seq in sequences]
        self.context = int(context)
        self.max_future = max(1, int(max_future))
        self.jitter = max(0, int(jitter))
        self.mix_prob = max(0.0, float(mix_prob))
        self.index: List[Tuple[int, int]] = []
        for si, seq in enumerate(self.sequences):
            if seq.shape[0] <= self.context + self.max_future:
                continue
            for start in range(0, seq.shape[0] - (self.context + self.max_future) + 1):
                self.index.append((si, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_id, start = self.index[idx]
        seq = self.sequences[seq_id]
        start_jitter = 0
        if self.jitter > 0:
            start_jitter = int(np.random.randint(-self.jitter, self.jitter + 1))
        start = int(np.clip(start + start_jitter, 0, max(seq.shape[0] - (self.context + self.max_future), 0)))

        x = seq[start : start + self.context]
        future = seq[start + self.context : start + self.context + self.max_future]

        if future.shape[0] < self.max_future:
            pad = future[-1:] if future.shape[0] > 0 else seq[start + self.context - 1 : start + self.context]
            pad = pad.repeat(self.max_future - future.shape[0], 1)
            future = torch.cat([future, pad], dim=0)

        if self.mix_prob > 0.0 and len(self.sequences) > 1 and np.random.rand() < self.mix_prob:
            alt_id = (seq_id + np.random.choice([-1, 1])) % len(self.sequences)
            alt_seq = self.sequences[alt_id]
            min_len = self.context + self.max_future + 1
            if alt_seq.shape[0] >= min_len:
                cut = int(np.random.randint(1, self.context))
                alt_start_max = int(alt_seq.shape[0] - (self.context - cut + self.max_future))
                alt_start = int(np.random.randint(0, max(1, alt_start_max)))
                alt_ctx = alt_seq[alt_start : alt_start + (self.context - cut)]
                alt_future = alt_seq[
                    alt_start + (self.context - cut) : alt_start + (self.context - cut) + self.max_future
                ]
                x = torch.cat([x[:cut], alt_ctx], dim=0)
                if alt_future.shape[0] == self.max_future:
                    future = alt_future
        return x, future


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


def build_latent_knn(latents_np: np.ndarray, k: int) -> List[List[int]]:
    n = latents_np.shape[0]
    if n <= 1:
        return [[]]
    k_eff = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=min(k_eff + 1, n), metric="euclidean")
    nn.fit(latents_np)
    _, indices = nn.kneighbors(latents_np, return_distance=True)
    neighbors: List[List[int]] = []
    for i, row in enumerate(indices):
        neigh = [int(j) for j in row if int(j) != i][:k_eff]
        neighbors.append(neigh)
    return neighbors


def sample_positive_indices(batch_indices: torch.Tensor, neighbor_graph: List[List[int]]) -> torch.Tensor:
    pos = []
    for idx in batch_indices.tolist():
        neigh = neighbor_graph[idx]
        if neigh:
            choice = np.random.choice(neigh)
        else:
            choice = idx
        pos.append(int(choice))
    return torch.tensor(pos, device=batch_indices.device, dtype=torch.long)


def uniformity_loss(u: torch.Tensor) -> torch.Tensor:
    if u.size(0) <= 1:
        return u.new_tensor(0.0)
    # For unit vectors: ||u_i - u_j||^2 = 2 - 2 cos(theta)
    sim = torch.matmul(u, u.transpose(0, 1))
    mask = ~torch.eye(u.size(0), dtype=torch.bool, device=u.device)
    dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    pairs = dist2[mask]
    if pairs.numel() == 0:
        return u.new_tensor(0.0)
    return torch.log(torch.exp(-2.0 * pairs).mean() + 1e-12)


def covariance_regularizer(
    u: torch.Tensor, gamma: float, var_weight: float, decorr_weight: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if u.size(0) <= 1:
        zeros = u.new_tensor(0.0)
        return zeros, zeros, zeros
    mu = u.mean(dim=0, keepdim=True)
    zc = u - mu
    cov = torch.matmul(zc.transpose(0, 1), zc) / float(u.size(0))
    var = torch.diag(cov)
    var_term = F.relu(gamma - var).pow(2)
    var_loss = var_term.sum() / max(var.numel(), 1)

    # Decorrelate off-diagonal terms.
    off_diag = cov - torch.diag(var)
    off_denom = max(off_diag.numel() - var.numel(), 1)
    decorr_loss = off_diag.pow(2).sum() / off_denom

    cov_loss = var_weight * var_loss + decorr_weight * decorr_loss
    return cov_loss, var_loss, decorr_loss


def residual_ref(x: torch.Tensor, span: int) -> torch.Tensor:
    span = max(1, min(int(span), x.size(1)))
    return x[:, -span:, :].mean(dim=1)


def scheduled_replace_with_preds(
    model: LatentAutoregressiveCNN,
    x: torch.Tensor,
    max_steps: int,
    prob: float,
) -> torch.Tensor:
    if prob <= 0.0 or max_steps <= 0 or x.size(1) < 2:
        return x
    if torch.rand(1, device=x.device).item() > prob:
        return x
    steps = int(np.random.randint(1, max_steps + 1))
    steps = max(1, min(steps, x.size(1)))
    with torch.no_grad():
        ctx = x.clone()
        preds = []
        for _ in range(steps):
            out = model(ctx, return_aux=False, return_delta=False)
            pred = out["pred"] if isinstance(out, dict) else out
            preds.append(pred.detach())
            ctx = torch.cat([ctx[:, 1:, :], pred.unsqueeze(1)], dim=1)
    preds_stack = torch.stack(preds, dim=1)  # [B, steps, D]
    x_aug = x.clone()
    x_aug[:, -steps:, :] = preds_stack
    return x_aug


def rollout_predictions(
    model: LatentAutoregressiveCNN,
    ctx: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    if steps <= 0:
        return ctx.new_zeros((ctx.size(0), 0, ctx.size(2)))
    preds = []
    roll_ctx = ctx
    for _ in range(steps):
        out = model(roll_ctx, return_aux=False, return_delta=True)
        pred = out["pred"] if isinstance(out, dict) else out
        preds.append(pred)
        roll_ctx = torch.cat([roll_ctx[:, 1:, :], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)  # [B, steps, D]


def train_autoregressive_cnn(
    sequences: List[np.ndarray],
    context: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    hidden_dim: int,
    layers: int,
    kernel_size: int,
    dropout: float,
    rollout_min: int,
    rollout_max: int,
    scheduled_prob: float,
    context_noise_std: float,
    residual_span: int,
    aux_weight: float,
    rollout_weight: float,
    delta_weight: float,
    norm_reg_weight: float,
    var_reg_weight: float,
    target_norm: float,
    target_var: np.ndarray,
    jitter: int,
    mix_prob: float,
):
    dataset = LatentSequenceDataset(
        sequences,
        context=context,
        max_future=rollout_max,
        jitter=jitter,
        mix_prob=mix_prob,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples for autoregressive model (sequences shorter than context).")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    latent_dim = int(sequences[0].shape[1])
    model = LatentAutoregressiveCNN(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        kernel_size=kernel_size,
        dropout=dropout,
        predict_residual=True,
        residual_center_span=residual_span,
        aux_head=aux_weight > 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history: List[Dict[str, float]] = []
    target_norm_t = torch.as_tensor(float(target_norm), device=device, dtype=torch.float32)
    target_var_t = torch.from_numpy(target_var).to(device=device, dtype=torch.float32)
    target_var_t = target_var_t.view(1, -1)

    for epoch in range(epochs):
        model.train()
        totals = {
            "total": 0.0,
            "main": 0.0,
            "delta": 0.0,
            "rollout": 0.0,
            "aux": 0.0,
            "reg_norm": 0.0,
            "reg_var": 0.0,
        }
        total_samples = 0
        for x, future in loader:
            x = x.to(device)
            future = future.to(device)
            if context_noise_std > 0.0:
                x = x + context_noise_std * torch.randn_like(x)
            x = scheduled_replace_with_preds(model, x, max_steps=rollout_max, prob=scheduled_prob)

            optimizer.zero_grad(set_to_none=True)

            out = model(x, return_aux=True, return_delta=True)
            pred_next = out["pred"]
            delta_next = out.get("delta", pred_next)
            ref = residual_ref(x, residual_span)

            target_next = future[:, 0, :]
            target_delta = target_next - ref

            main_loss = F.mse_loss(pred_next, target_next)
            delta_loss = F.mse_loss(delta_next, target_delta)

            aux_loss = pred_next.new_tensor(0.0)
            if out.get("aux_pred", None) is not None and future.size(1) > 1:
                target_aux = future[:, 1, :]
                aux_pred = out["aux_pred"]
                aux_delta = out.get("aux_delta", aux_pred)
                target_aux_delta = target_aux - ref
                aux_loss = 0.5 * (F.mse_loss(aux_pred, target_aux) + F.mse_loss(aux_delta, target_aux_delta))

            roll_steps = int(np.random.randint(max(1, rollout_min), rollout_max + 1))
            roll_steps = min(roll_steps, future.size(1))
            rollout_pred = rollout_predictions(model, x, steps=roll_steps)
            rollout_target = future[:, :roll_steps, :]
            rollout_loss = F.mse_loss(rollout_pred, rollout_target)

            reg_frames = torch.cat([pred_next.unsqueeze(1), rollout_pred], dim=1)
            norms = reg_frames.norm(dim=-1)
            norm_reg = ((norms - target_norm_t) ** 2).mean()
            batch_var = reg_frames.reshape(-1, reg_frames.size(-1)).var(dim=0, unbiased=False).view(1, -1)
            var_reg = ((batch_var - target_var_t) ** 2).mean()

            total_loss = (
                main_loss
                + delta_weight * delta_loss
                + rollout_weight * rollout_loss
                + aux_weight * aux_loss
                + norm_reg_weight * norm_reg
                + var_reg_weight * var_reg
            )

            total_loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_samples += bs
            totals["total"] += float(total_loss.item()) * bs
            totals["main"] += float(main_loss.item()) * bs
            totals["delta"] += float(delta_loss.item()) * bs
            totals["rollout"] += float(rollout_loss.item()) * bs
            totals["aux"] += float(aux_loss.item()) * bs
            totals["reg_norm"] += float(norm_reg.item()) * bs
            totals["reg_var"] += float(var_reg.item()) * bs

        denom = max(total_samples, 1)
        epoch_stats = {k: v / denom for k, v in totals.items()}
        loss_history.append(epoch_stats)
        print(
            f"[ar] epoch {epoch + 1}/{epochs} - total: {epoch_stats['total']:.6f} | "
            f"main: {epoch_stats['main']:.6f} | roll: {epoch_stats['rollout']:.6f} | aux: {epoch_stats['aux']:.6f}"
        )
    return model, loss_history


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
    uniformize: bool,
    knn_k: int,
    lambda_unif: float,
    lambda_cov: float,
    cov_var_weight: float,
    cov_decorr_weight: float,
    cov_gamma: float,
    mean_weight: float,
):
    dataset = LatentDataset(latents_np, return_index=uniformize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = LatentProjector(input_dim=latents_np.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_history: List[Dict[str, float]] = []

    neighbor_graph = build_latent_knn(latents_np, k=knn_k) if uniformize else None
    latents_all = torch.from_numpy(latents_np).to(device) if uniformize else None

    for epoch in range(epochs):
        model.train()
        totals = {
            "total": 0.0,
            "sim": 0.0,
            "unif": 0.0,
            "cov": 0.0,
            "cov_var": 0.0,
            "cov_decorr": 0.0,
            "mean": 0.0,
        }
        total_samples = 0

        for batch in loader:
            if uniformize:
                batch, batch_idx = batch
                batch = batch.to(device)
                batch_idx = batch_idx.to(device)
                pos_idx = sample_positive_indices(batch_idx, neighbor_graph)
                anchor = augment_latents(batch, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=0.0)
                pos_latents = latents_all[pos_idx]
                positive = augment_latents(pos_latents, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=0.0)
            else:
                batch = batch.to(device)
                anchor = batch
                positive = augment_latents(batch, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=mix_alpha)

            optimizer.zero_grad(set_to_none=True)
            u = model(anchor)
            v = model(positive)

            sim_loss = info_nce_loss(u, v, tau=tau)
            total_loss = sim_loss

            unif_loss = u.new_tensor(0.0)
            cov_loss = u.new_tensor(0.0)
            cov_var_loss = u.new_tensor(0.0)
            cov_decorr_loss = u.new_tensor(0.0)
            mean_loss = u.new_tensor(0.0)

            if uniformize:
                combined = torch.cat([u, v], dim=0)
                unif_loss = uniformity_loss(combined)
                cov_loss, cov_var_loss, cov_decorr_loss = covariance_regularizer(
                    combined, gamma=cov_gamma, var_weight=cov_var_weight, decorr_weight=cov_decorr_weight
                )
                total_loss = total_loss + lambda_unif * unif_loss + lambda_cov * cov_loss
                if mean_weight > 0.0:
                    mean_loss = combined.mean(dim=0).pow(2).sum()
                    total_loss = total_loss + mean_weight * mean_loss

            total_loss.backward()
            optimizer.step()

            bs = anchor.size(0)
            total_samples += bs
            totals["total"] += float(total_loss.item()) * bs
            totals["sim"] += float(sim_loss.item()) * bs
            totals["unif"] += float(unif_loss.item()) * bs
            totals["cov"] += float(cov_loss.item()) * bs
            totals["cov_var"] += float(cov_var_loss.item()) * bs
            totals["cov_decorr"] += float(cov_decorr_loss.item()) * bs
            totals["mean"] += float(mean_loss.item()) * bs

        denom = max(total_samples, 1)
        epoch_stats = {k: v / denom for k, v in totals.items()}
        loss_history.append(epoch_stats)
        print(
            f"[g] epoch {epoch + 1}/{epochs} - "
            f"total: {epoch_stats['total']:.4f} | sim: {epoch_stats['sim']:.4f} | "
            f"unif: {epoch_stats['unif']:.4f} | cov: {epoch_stats['cov']:.4f}"
        )
    return model, loss_history


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
    ap.add_argument(
        "--proj_mix",
        type=float,
        default=0.15,
        help="Mixup coefficient for latent augmentation (ignored when --uniform_proj is enabled).",
    )
    ap.add_argument(
        "--uniform_proj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spherical uniformization + VAE-KNN positives for the projector (default: on).",
    )
    ap.add_argument(
        "--proj_knn",
        type=int,
        default=10,
        help="Number of nearest neighbors (in VAE latent space) used as positives when uniform_proj is on.",
    )
    ap.add_argument("--proj_lambda_unif", type=float, default=1.0, help="Weight for spherical uniformity loss.")
    ap.add_argument("--proj_lambda_cov", type=float, default=0.5, help="Weight for covariance regularizer.")
    ap.add_argument("--proj_cov_var", type=float, default=0.5, help="Weight for the variance floor term inside cov reg.")
    ap.add_argument(
        "--proj_cov_decorr",
        type=float,
        default=1.0,
        help="Weight for the decorrelation term inside covariance regularizer.",
    )
    ap.add_argument(
        "--proj_cov_gamma",
        type=float,
        default=-1.0,
        help="Minimum per-dim variance target for covariance regularizer (<= 1/proj_dim). Use -1 for auto.",
    )
    ap.add_argument(
        "--proj_mean_weight",
        type=float,
        default=0.0,
        help="Optional weight for ||mean||^2 penalty on projector outputs (helps recentring if needed).",
    )
    ap.add_argument(
        "--train_ar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train an autoregressive CNN on normalized latent sequences (default: on).",
    )
    ap.add_argument("--ar_context", type=int, default=32, help="Context window (latent frames) for AR conditioning.")
    ap.add_argument("--ar_hidden", type=int, default=256, help="Hidden channels inside the AR CNN.")
    ap.add_argument("--ar_layers", type=int, default=4, help="Number of dilated conv layers in the AR CNN.")
    ap.add_argument("--ar_kernel", type=int, default=3, help="Kernel size for AR convolutions.")
    ap.add_argument("--ar_dropout", type=float, default=0.05, help="Dropout applied inside the AR CNN.")
    ap.add_argument("--ar_epochs", type=int, default=10, help="Training epochs for the autoregressive model.")
    ap.add_argument("--ar_batch", type=int, default=256, help="Batch size for AR model training.")
    ap.add_argument("--ar_lr", type=float, default=1e-3, help="Learning rate for AR model training.")
    ap.add_argument("--ar_weight_decay", type=float, default=1e-4, help="Weight decay for AR optimizer.")
    ap.add_argument("--ar_rollout_min", type=int, default=2, help="Minimum rollout steps used inside AR loss.")
    ap.add_argument("--ar_rollout_max", type=int, default=4, help="Maximum rollout steps used inside AR loss.")
    ap.add_argument(
        "--ar_rollout_weight",
        type=float,
        default=0.5,
        help="Weight applied to rollout loss terms (keep small to avoid over-penalizing long horizons).",
    )
    ap.add_argument(
        "--ar_sched_prob",
        type=float,
        default=0.2,
        help="Probability of replacing the tail of the context with model predictions (scheduled sampling).",
    )
    ap.add_argument(
        "--ar_context_noise",
        type=float,
        default=0.01,
        help="Gaussian noise std added to AR context (in normalized latent units).",
    )
    ap.add_argument(
        "--ar_residual_span",
        type=int,
        default=1,
        help="Number of tail frames averaged as the residual anchor for Δz predictions.",
    )
    ap.add_argument(
        "--ar_aux_weight",
        type=float,
        default=0.25,
        help="Weight for the auxiliary 2-step prediction head inside the AR loss.",
    )
    ap.add_argument(
        "--ar_delta_weight",
        type=float,
        default=0.5,
        help="Weight for supervising Δz directly in addition to absolute latent MSE.",
    )
    ap.add_argument(
        "--ar_norm_reg",
        type=float,
        default=0.01,
        help="Regularizer weight for keeping predicted norms close to training stats (in normalized space).",
    )
    ap.add_argument(
        "--ar_var_reg",
        type=float,
        default=0.01,
        help="Regularizer weight for keeping per-dim variance close to training stats (in normalized space).",
    )
    ap.add_argument(
        "--ar_jitter",
        type=int,
        default=2,
        help="Temporal jitter (frames) applied when sampling AR training windows for seed diversity.",
    )
    ap.add_argument(
        "--ar_mix_prob",
        type=float,
        default=0.1,
        help="Probability of splicing the tail of a random neighboring sequence into the AR context.",
    )
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
    if args.proj_knn < 1:
        ap.error("--proj_knn must be ≥ 1.")
    if args.ar_context < 1:
        ap.error("--ar_context must be ≥ 1.")
    if args.ar_hidden < 1:
        ap.error("--ar_hidden must be ≥ 1.")
    if args.ar_layers < 1:
        ap.error("--ar_layers must be ≥ 1.")
    if args.ar_kernel < 2:
        ap.error("--ar_kernel must be ≥ 2.")
    if args.ar_epochs < 1:
        ap.error("--ar_epochs must be ≥ 1.")
    if args.ar_batch < 1:
        ap.error("--ar_batch must be ≥ 1.")
    if args.ar_lr <= 0.0:
        ap.error("--ar_lr must be > 0.")
    if args.ar_weight_decay < 0.0:
        ap.error("--ar_weight_decay must be ≥ 0.")
    if args.ar_rollout_min < 1:
        ap.error("--ar_rollout_min must be ≥ 1.")
    if args.ar_rollout_max < args.ar_rollout_min:
        ap.error("--ar_rollout_max must be ≥ --ar_rollout_min.")
    if args.ar_residual_span < 1:
        ap.error("--ar_residual_span must be ≥ 1.")

    # Clamp weights to sane ranges.
    args.proj_lambda_unif = max(0.0, float(args.proj_lambda_unif))
    args.proj_lambda_cov = max(0.0, float(args.proj_lambda_cov))
    args.proj_cov_var = max(0.0, float(args.proj_cov_var))
    args.proj_cov_decorr = max(0.0, float(args.proj_cov_decorr))
    args.proj_mean_weight = max(0.0, float(args.proj_mean_weight))
    args.ar_dropout = min(max(0.0, float(args.ar_dropout)), 0.95)
    max_gamma = 1.0 / float(args.proj_dim)
    if args.proj_cov_gamma <= 0.0 or args.proj_cov_gamma > max_gamma:
        if args.proj_cov_gamma > max_gamma:
            print(f"[warn] Clamped --proj_cov_gamma to {max_gamma:.4f} to keep targets feasible on the unit sphere.")
        args.proj_cov_gamma = max_gamma
    if args.uniform_proj and args.proj_mix > 0.0:
        print("[info] --proj_mix is ignored when --uniform_proj is enabled (using VAE-space neighbors as positives).")
    args.ar_sched_prob = float(np.clip(args.ar_sched_prob, 0.0, 1.0))
    args.ar_context_noise = max(0.0, float(args.ar_context_noise))
    args.ar_rollout_weight = max(0.0, float(args.ar_rollout_weight))
    args.ar_aux_weight = max(0.0, float(args.ar_aux_weight))
    args.ar_delta_weight = max(0.0, float(args.ar_delta_weight))
    args.ar_norm_reg = max(0.0, float(args.ar_norm_reg))
    args.ar_var_reg = max(0.0, float(args.ar_var_reg))
    args.ar_jitter = max(0, int(args.ar_jitter))
    args.ar_mix_prob = float(np.clip(args.ar_mix_prob, 0.0, 1.0))

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

    latent_sequences_norm = []
    for seq in latent_sequences:
        seq_norm = (seq - Z_mean[None, :]) / Z_std[None, :]
        latent_sequences_norm.append(np.ascontiguousarray(seq_norm.astype(np.float32)))

    latent_stack_norm = (latent_stack - Z_mean[None, :]) / Z_std[None, :]
    latent_stack_norm = np.ascontiguousarray(latent_stack_norm.astype(np.float32))
    if latent_stack_norm.shape[0] < 2:
        raise RuntimeError("Need at least two latent frames to train the contrastive projector.")
    train_norm = float(np.linalg.norm(latent_stack_norm, axis=1).mean())
    train_var = np.maximum(latent_stack_norm.var(axis=0).astype(np.float32), 1e-6)

    device = torch.device(DEVICE)

    ar_checkpoint_path = None
    ar_loss_history: List[Dict[str, float]] = []
    ar_loss_total: List[float] = []
    ar_config = None
    if args.train_ar:
        try:
            print("[info] Training autoregressive latent CNN…")
            ar_model, ar_loss_history = train_autoregressive_cnn(
                latent_sequences_norm,
                context=int(args.ar_context),
                epochs=int(args.ar_epochs),
                batch_size=int(args.ar_batch),
                lr=float(args.ar_lr),
                weight_decay=float(args.ar_weight_decay),
                device=device,
                hidden_dim=int(args.ar_hidden),
                layers=int(args.ar_layers),
                kernel_size=int(args.ar_kernel),
                dropout=float(args.ar_dropout),
                rollout_min=int(args.ar_rollout_min),
                rollout_max=int(args.ar_rollout_max),
                scheduled_prob=float(args.ar_sched_prob),
                context_noise_std=float(args.ar_context_noise),
                residual_span=int(args.ar_residual_span),
                aux_weight=float(args.ar_aux_weight),
                rollout_weight=float(args.ar_rollout_weight),
                delta_weight=float(args.ar_delta_weight),
                norm_reg_weight=float(args.ar_norm_reg),
                var_reg_weight=float(args.ar_var_reg),
                target_norm=float(train_norm),
                target_var=train_var,
                jitter=int(args.ar_jitter),
                mix_prob=float(args.ar_mix_prob),
            )
            ar_loss_total = [float(entry["total"]) for entry in ar_loss_history]
            ar_state = {k: v.detach().cpu() for k, v in ar_model.state_dict().items()}
            ar_checkpoint_path = os.path.join(out_dir, f"{prefix}_ar_{ts}.pt")
            ar_config = ar_model.to_config()
            torch.save(
                {
                    "state_dict": ar_state,
                    "config": ar_config,
                    "context": int(args.ar_context),
                    "loss_history": ar_loss_history,
                    "loss_total": ar_loss_total,
                    "epochs": int(args.ar_epochs),
                    "batch_size": int(args.ar_batch),
                    "lr": float(args.ar_lr),
                    "weight_decay": float(args.ar_weight_decay),
                    "rollout": {
                        "min": int(args.ar_rollout_min),
                        "max": int(args.ar_rollout_max),
                        "weight": float(args.ar_rollout_weight),
                    },
                    "scheduled_prob": float(args.ar_sched_prob),
                    "context_noise": float(args.ar_context_noise),
                    "residual_span": int(args.ar_residual_span),
                    "aux_weight": float(args.ar_aux_weight),
                    "delta_weight": float(args.ar_delta_weight),
                    "norm_reg": float(args.ar_norm_reg),
                    "var_reg": float(args.ar_var_reg),
                    "target_norm": float(train_norm),
                    "target_var": train_var,
                    "jitter": int(args.ar_jitter),
                    "mix_prob": float(args.ar_mix_prob),
                },
                ar_checkpoint_path,
            )
        except RuntimeError as exc:
            print(f"[warn] Skipping autoregressive training: {exc}")

    proj_batch = min(args.proj_batch, latent_stack_norm.shape[0])
    proj_batch = max(2, proj_batch)

    print(f"Training contrastive projector g… (uniform_proj={bool(args.uniform_proj)})")
    projector, g_loss_history = train_contrastive_projector(
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
        uniformize=bool(args.uniform_proj),
        knn_k=int(args.proj_knn),
        lambda_unif=float(args.proj_lambda_unif),
        lambda_cov=float(args.proj_lambda_cov),
        cov_var_weight=float(args.proj_cov_var),
        cov_decorr_weight=float(args.proj_cov_decorr),
        cov_gamma=float(args.proj_cov_gamma),
        mean_weight=float(args.proj_mean_weight),
    )
    g_losses = [float(entry["total"]) for entry in g_loss_history]

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
            "uniformize": bool(args.uniform_proj),
            "knn_k": int(args.proj_knn),
            "lambda_unif": float(args.proj_lambda_unif),
            "lambda_cov": float(args.proj_lambda_cov),
            "cov_var_weight": float(args.proj_cov_var),
            "cov_decorr_weight": float(args.proj_cov_decorr),
            "cov_gamma": float(args.proj_cov_gamma),
            "mean_weight": float(args.proj_mean_weight),
            "loss_history": g_losses,
            "loss_breakdown": g_loss_history,
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
        g_uniformize=np.array(bool(args.uniform_proj)),
        g_knn_k=np.array(int(args.proj_knn), dtype=np.int32),
        g_lambda_unif=np.array(float(args.proj_lambda_unif), dtype=np.float32),
        g_lambda_cov=np.array(float(args.proj_lambda_cov), dtype=np.float32),
        g_cov_var_weight=np.array(float(args.proj_cov_var), dtype=np.float32),
        g_cov_decorr_weight=np.array(float(args.proj_cov_decorr), dtype=np.float32),
        g_cov_gamma=np.array(float(args.proj_cov_gamma), dtype=np.float32),
        g_mean_weight=np.array(float(args.proj_mean_weight), dtype=np.float32),
        g_loss_breakdown=np.array(g_loss_history, dtype=object),
        ar_model_path=np.array(ar_checkpoint_path if ar_checkpoint_path is not None else ""),
        ar_context=np.array(int(args.ar_context), dtype=np.int32),
        ar_loss_history=np.asarray(ar_loss_total, dtype=np.float32),
        ar_loss_breakdown=np.array(ar_loss_history, dtype=object),
        ar_config=np.array(ar_config if ar_config is not None else {}, dtype=object),
        ar_train_epochs=np.array(int(args.ar_epochs), dtype=np.int32),
        ar_train_batch=np.array(int(args.ar_batch), dtype=np.int32),
        ar_train_lr=np.array(float(args.ar_lr), dtype=np.float32),
        ar_weight_decay=np.array(float(args.ar_weight_decay), dtype=np.float32),
        ar_rollout_min=np.array(int(args.ar_rollout_min), dtype=np.int32),
        ar_rollout_max=np.array(int(args.ar_rollout_max), dtype=np.int32),
        ar_rollout_weight=np.array(float(args.ar_rollout_weight), dtype=np.float32),
        ar_sched_prob=np.array(float(args.ar_sched_prob), dtype=np.float32),
        ar_context_noise=np.array(float(args.ar_context_noise), dtype=np.float32),
        ar_residual_span=np.array(int(args.ar_residual_span), dtype=np.int32),
        ar_aux_weight=np.array(float(args.ar_aux_weight), dtype=np.float32),
        ar_delta_weight=np.array(float(args.ar_delta_weight), dtype=np.float32),
        ar_norm_reg=np.array(float(args.ar_norm_reg), dtype=np.float32),
        ar_var_reg=np.array(float(args.ar_var_reg), dtype=np.float32),
        ar_jitter=np.array(int(args.ar_jitter), dtype=np.int32),
        ar_mix_prob=np.array(float(args.ar_mix_prob), dtype=np.float32),
        ar_target_norm=np.array(float(train_norm), dtype=np.float32),
        ar_target_var=train_var,
        ar_trained=np.array(bool(ar_checkpoint_path is not None)),
        latent_bundle_path=np.array(latents_path),
        **dr_arrays,
    )

    print("\nSaved:")
    print("  Latents   :", latents_path)
    print("  Projector :", projector_path)
    if ar_checkpoint_path is not None:
        print("  AR model  :", ar_checkpoint_path)
    print("  Corpus    :", corpus_path)
    print("  Folder    :", out_dir)


if __name__ == "__main__":
    main()
