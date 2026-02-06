#!/usr/bin/env python3
"""
Train GRU latent policy over 64D trajectories.
Latent-only training (no index policy, no contrastive loss).
"""
import os
import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stable_audio_wanderer.config import DEVICE
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import (
    LatentPolicy,
    LatentPolicyConfig,
    load_geometry_from_dict,
    group_meta_by_file,
)


def emit_json_progress(data: dict):
    """Emit JSON progress update to stdout for GUI consumption."""
    print(json.dumps(data), flush=True)


class LatentTrajectoryDataset:
    """
    Dataset for training LatentPolicy over 64D latent trajectories.

    Each item returns tensors for:
        - z: [seq_len, 64] latent positions
        - v: [seq_len, 64] velocities
        - z_next: [seq_len, 64] next latent positions (target)
        - local_features: [seq_len, 16] local geometry features
        - controls: [seq_len, control_dim] control parameters
        - knn_centroid: [seq_len, 64] kNN centroids for manifold loss
        - file_ids: [seq_len] source file IDs
        - t_lat: [seq_len] time positions
        - window_targets: [seq_len] continuous log2 targets (float32)
    """

    def __init__(
        self,
        sequences: list,
        GG: np.ndarray,
        geometry,
        seq_len: int = 32,
        control_dim: int = 6,
        window_targets: np.ndarray = None,
    ):
        self.sequences = [np.asarray(seq, dtype=np.int32) for seq in sequences if len(seq) > 0]
        if not self.sequences:
            raise ValueError("No sequences available for latent policy training.")
        self.GG = np.asarray(GG, dtype=np.float32)
        self.geometry = geometry
        self.seq_len = int(seq_len)
        self.control_dim = int(control_dim)
        self.N = self.GG.shape[0]
        self.D = self.GG.shape[1]
        # Window targets for adaptive decoding (continuous log2)
        if window_targets is not None:
            self.window_targets = window_targets.astype(np.float32)
        else:
            self.window_targets = np.full(self.N, 3.0, dtype=np.float32)  # Default: log2(8)

    def __len__(self):
        return len(self.sequences)

    def _sample_window(self, seq: np.ndarray):
        if seq.size < self.seq_len + 1:
            pad_needed = self.seq_len + 1 - seq.size
            seq = np.concatenate([seq, np.repeat(seq[-1:], pad_needed)]).astype(np.int32)
        if seq.size == self.seq_len + 1:
            start = 0
        else:
            start = np.random.randint(0, seq.size - self.seq_len - 1)
        curr = seq[start : start + self.seq_len]
        nxt = seq[start + 1 : start + self.seq_len + 1]
        return curr, nxt

    def __getitem__(self, idx: int):
        seq = self.sequences[idx % len(self.sequences)]
        curr_idx, next_idx = self._sample_window(seq)

        z = self.GG[curr_idx].astype(np.float32)
        z_next = self.GG[next_idx].astype(np.float32)

        v = np.zeros_like(z)
        v[1:] = z[1:] - z[:-1]

        local_features = np.zeros((self.seq_len, 16), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            local_features[i, 0] = self.geometry.local_sigma[idx_i]
            local_features[i, 1] = self.geometry.local_density[idx_i]
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid = self.GG[knn_idx].mean(axis=0)
            local_features[i, 2] = np.linalg.norm(z[i] - knn_centroid)
            time_grad = self.geometry.time_gradients[idx_i]
            z_norm_i = z[i] / (np.linalg.norm(z[i]) + 1e-6)
            local_features[i, 3] = np.dot(z_norm_i, time_grad)
            local_features[i, 4] = float(self.geometry.t_lat[idx_i]) / 1000.0
            local_features[i, 5] = float(self.geometry.file_ids[idx_i]) / 100.0

        controls = np.zeros((self.seq_len, self.control_dim), dtype=np.float32)
        controls[:, :4] = 0.5  # width, energy, gravity, memory defaults

        knn_centroid = np.zeros((self.seq_len, self.D), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid[i] = self.GG[knn_idx].mean(axis=0)

        file_ids = self.geometry.file_ids[curr_idx].astype(np.int64)
        t_lat = self.geometry.t_lat[curr_idx].astype(np.float32)

        # Window targets for adaptive decoding (continuous log2)
        window_targets = self.window_targets[curr_idx].astype(np.float32)

        return {
            "z": z,
            "v": v,
            "z_next": z_next,
            "local_features": local_features,
            "controls": controls,
            "knn_centroid": knn_centroid,
            "file_ids": file_ids,
            "t_lat": t_lat,
            "window_targets": window_targets,
        }


def compute_latent_losses(
    delta_mean: torch.Tensor,
    delta_log_std: torch.Tensor,
    delta_weights: torch.Tensor,
    vel_delta: torch.Tensor,
    window_log2: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    z_next: torch.Tensor,
    knn_centroid: torch.Tensor,
    window_targets: torch.Tensor,
    weights: dict,
) -> dict:
    """Compute losses for latent policy training (no contrastive term)."""
    B, T, M, D = delta_mean.shape

    mix_weights = F.softmax(delta_weights, dim=-1)
    delta_z_pred = (mix_weights.unsqueeze(-1) * delta_mean).sum(dim=2)

    alpha = 0.95
    v_new = alpha * v + vel_delta
    z_pred = z + v_new + delta_z_pred

    recon_loss = F.mse_loss(z_pred, z_next)

    if T > 1:
        accel = v_new[:, 1:] - v_new[:, :-1]
        smooth_loss = torch.mean(accel ** 2)
    else:
        smooth_loss = torch.tensor(0.0, device=delta_mean.device)

    manifold_loss = F.mse_loss(z_pred, knn_centroid)

    z_pred_var = z_pred.view(-1, D).var(dim=0).mean()
    diversity_loss = -z_pred_var

    # Window loss: Huber loss in log2 space
    window_loss = F.huber_loss(
        window_log2.view(-1), window_targets.view(-1), delta=1.0
    )

    # Window metrics: MAE in log2 and frame space
    with torch.no_grad():
        window_mae_log2 = (window_log2.view(-1) - window_targets.view(-1)).abs().mean()
        pred_frames = 2.0 ** window_log2.detach().view(-1)
        target_frames = 2.0 ** window_targets.view(-1)
        window_mae_frames = (pred_frames - target_frames).abs().mean()

    total_loss = (
        weights.get("recon", 1.0) * recon_loss +
        weights.get("smooth", 0.1) * smooth_loss +
        weights.get("manifold", 0.1) * manifold_loss +
        weights.get("diversity", 0.01) * diversity_loss +
        weights.get("window", 0.5) * window_loss
    )

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "smooth_loss": smooth_loss,
        "manifold_loss": manifold_loss,
        "diversity_loss": diversity_loss,
        "window_loss": window_loss,
        "window_mae_log2": window_mae_log2,
        "window_mae_frames": window_mae_frames,
    }


def run_latent_epoch(model, loader, device, weights, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    scalar_keys = ["loss", "recon_loss", "smooth_loss", "manifold_loss", "diversity_loss",
                    "window_loss", "window_mae_log2", "window_mae_frames"]
    totals = {k: 0.0 for k in scalar_keys}
    total_samples = 0

    for batch in loader:
        z = torch.tensor(batch["z"], device=device)
        v = torch.tensor(batch["v"], device=device)
        z_next = torch.tensor(batch["z_next"], device=device)
        local_features = torch.tensor(batch["local_features"], device=device)
        controls = torch.tensor(batch["controls"], device=device)
        knn_centroid = torch.tensor(batch["knn_centroid"], device=device)
        window_targets = torch.tensor(batch["window_targets"], device=device)

        if train:
            optimizer = weights["optimizer"]
            optimizer.zero_grad(set_to_none=True)

        delta_mean, delta_log_std, delta_weights, vel_delta, window_log2, _ = model(
            z=z,
            v=v,
            controls=controls,
            local_features=local_features,
        )

        losses = compute_latent_losses(
            delta_mean=delta_mean,
            delta_log_std=delta_log_std,
            delta_weights=delta_weights,
            vel_delta=vel_delta,
            window_log2=window_log2,
            z=z,
            v=v,
            z_next=z_next,
            knn_centroid=knn_centroid,
            window_targets=window_targets,
            weights=weights,
        )

        if train:
            losses["loss"].backward()
            optimizer.step()

        bs = z.size(0)
        total_samples += bs
        for k in scalar_keys:
            totals[k] += float(losses[k].item()) * bs

    for k in scalar_keys:
        totals[k] /= max(1, total_samples)

    return totals


def build_latent_loaders(sequences, GG, geometry, args, window_targets=None):
    sequences = list(sequences)
    np.random.shuffle(sequences)
    val_count = int(round(len(sequences) * float(args.val_split)))
    val_count = max(0, min(len(sequences) - 1, val_count)) if len(sequences) > 1 else 0

    val_seqs = sequences[:val_count]
    train_seqs = sequences[val_count:] if val_count > 0 else sequences

    train_ds = LatentTrajectoryDataset(
        train_seqs, GG, geometry,
        seq_len=int(args.seq_len),
        control_dim=int(args.control_dim),
        window_targets=window_targets,
    )
    val_ds = LatentTrajectoryDataset(
        val_seqs if val_seqs else train_seqs, GG, geometry,
        seq_len=int(args.seq_len),
        control_dim=int(args.control_dim),
        window_targets=window_targets,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser(
        description="Train GRU latent policy over 64D trajectories."
    )
    ap.add_argument("--corpus_dir", required=True, help="Folder containing corpus.npz")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--val_split", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--control_dim", type=int, default=6,
                    help="width, energy, gravity, memory, coherence, exploration")

    ap.add_argument("--lambda_recon", type=float, default=1.0)
    ap.add_argument("--lambda_smooth", type=float, default=0.1)
    ap.add_argument("--lambda_manifold", type=float, default=0.1)
    ap.add_argument("--lambda_diversity", type=float, default=0.01)
    ap.add_argument("--lambda_window", type=float, default=0.5,
                    help="Weight for window size Huber loss.")

    ap.add_argument("--out_path", default=None, help="Override policy checkpoint path.")
    ap.add_argument("--verbose", action="store_true", help="Print detailed metrics each epoch.")
    ap.add_argument("--json_progress", action="store_true", help="Emit JSON progress updates.")

    args = ap.parse_args()

    try:
        corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        corpus_npz = os.path.join(args.corpus_dir, "corpus.npz")
        if not os.path.exists(corpus_npz):
            raise FileNotFoundError(f"No corpus found in {args.corpus_dir}")

    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError("Corpus missing latent geometry. Re-run preprocess.py.")

    GG = data["GG"].astype(np.float32)
    meta = data["meta"]

    # Load continuous window targets (log2 space)
    window_targets_log2 = data.get("window_targets_log2", None)
    if window_targets_log2 is not None:
        window_targets = window_targets_log2.astype(np.float32)
        print(f"[info] Loaded window targets (log2): "
              f"min={window_targets.min():.2f}, max={window_targets.max():.2f}, mean={window_targets.mean():.2f}")
        # Blend with decoder quality targets if available
        decoder_targets = data.get("decoder_quality_targets_log2", None)
        if decoder_targets is not None:
            alpha = 0.7  # Velocity weight
            decoder_targets = decoder_targets.astype(np.float32)
            window_targets = alpha * window_targets + (1 - alpha) * decoder_targets
            print(f"[info] Blended with decoder quality targets (alpha={alpha})")
    else:
        window_targets = None
        print("[warn] No window_targets_log2 in corpus. Using defaults.")

    sequences = group_meta_by_file(meta)
    print(f"[info] Sequences: {len(sequences)} (total points={GG.shape[0]})")

    train_loader, val_loader = build_latent_loaders(
        sequences, GG, geometry, args, window_targets=window_targets,
    )

    cfg = LatentPolicyConfig(
        latent_dim=int(GG.shape[1]),
        hidden_size=int(args.hidden),
        num_layers=int(args.layers),
        num_mixture_components=4,
        control_dim=int(args.control_dim),
        local_feature_dim=16,
    )

    model = LatentPolicy(cfg).to(torch.device(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights = {
        "recon": float(args.lambda_recon),
        "smooth": float(args.lambda_smooth),
        "manifold": float(args.lambda_manifold),
        "diversity": float(args.lambda_diversity),
        "window": float(args.lambda_window),
        "optimizer": optimizer,
    }

    print(
        f"[info] Training LatentPolicy with: recon={args.lambda_recon}, "
        f"smooth={args.lambda_smooth}, manifold={args.lambda_manifold}, "
        f"diversity={args.lambda_diversity}, window={args.lambda_window}"
    )

    history = {"train": [], "val": []}
    device = torch.device(DEVICE)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_stats = run_latent_epoch(model, train_loader, device, weights, train=True)
        val_stats = run_latent_epoch(model, val_loader, device, weights, train=False)
        train_hist = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in train_stats.items()}
        val_hist = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in val_stats.items()}
        history["train"].append(train_hist)
        history["val"].append(val_hist)

        if args.json_progress:
            train_json = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in train_stats.items()}
            val_json = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in val_stats.items()}
            emit_json_progress({
                "epoch": epoch + 1,
                "train": train_json,
                "val": val_json,
            })

        if args.verbose:
            print(
                f"[epoch {epoch+1:03d}] "
                f"loss={train_stats['loss']:.4f} "
                f"win_loss={train_stats['window_loss']:.3f} "
                f"win_mae_log2={train_stats['window_mae_log2']:.3f} "
                f"win_mae_frames={train_stats['window_mae_frames']:.1f}"
            )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]

    out_path = args.out_path
    if out_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.corpus_dir, f"latent_policy_{ts}.pt")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "history": history,
    }
    torch.save(ckpt, out_path)
    print(f"[done] Saved policy checkpoint: {out_path}")


if __name__ == "__main__":
    main()
