#!/usr/bin/env python3
"""
Train GRU latent policy and/or build manual navigation KD-tree artifact.
"""
import os
import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from stable_audio_wanderer.config import DEVICE
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import (
    LatentPolicy,
    LatentPolicyConfig,
    load_geometry_from_dict,
    compute_causal_ema_summaries,
)


def emit_json_progress(data: dict):
    """Emit JSON progress update to stdout for GUI consumption."""
    print(json.dumps(data), flush=True)


def _to_device_float_tensor(x, device):
    """
    Convert batch field to float tensor on target device without redundant copies.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def build_sequences_from_offsets(file_offsets: np.ndarray) -> list:
    """Build per-file frame index sequences from cumulative offsets [M+1]."""
    file_offsets = np.asarray(file_offsets, dtype=np.int64)
    if file_offsets.ndim != 1 or file_offsets.size < 2:
        raise ValueError("file_offsets must be shape [num_files + 1].")

    sequences = []
    for fid in range(file_offsets.size - 1):
        start = int(file_offsets[fid])
        end = int(file_offsets[fid + 1])
        if end <= start:
            continue
        seq = np.arange(start, end, dtype=np.int32)
        sequences.append(seq)
    return sequences


class LatentTrajectoryDataset:
    """
    Dataset for training LatentPolicy over 64D latent trajectories.

    Each item returns tensors for:
        - z: [seq_len, 64] latent positions
        - v: [seq_len, 64] velocities
        - z_next: [seq_len, 64] next latent positions (target)
        - local_features: [seq_len, 16] local geometry features
        - context_summaries: [seq_len, C] causal EMA summaries
        - controls: [seq_len, control_dim] control parameters
        - knn_centroid: [seq_len, 64] kNN centroids for manifold loss
        - file_ids: [seq_len] source file IDs
        - t_lat: [seq_len] time positions
        - window_targets: [seq_len] continuous log2 targets (float32)
    """

    def __init__(
        self,
        sequences: list,
        Z: np.ndarray,
        geometry,
        seq_len: int = 32,
        control_dim: int = 6,
        window_targets: np.ndarray = None,
    ):
        self.sequences = [np.asarray(seq, dtype=np.int32) for seq in sequences if len(seq) > 0]
        if not self.sequences:
            raise ValueError("No sequences available for latent policy training.")
        self.Z = np.asarray(Z, dtype=np.float32)
        self.geometry = geometry
        self.seq_len = int(seq_len)
        self.control_dim = int(control_dim)
        self.N = self.Z.shape[0]
        self.D = self.Z.shape[1]
        self.use_ema_mid = bool(getattr(self.geometry, "use_ema_mid", False))
        self.ema_alpha_fast = float(getattr(self.geometry, "ema_alpha_fast", 0.60))
        self.ema_alpha_mid = float(getattr(self.geometry, "ema_alpha_mid", 0.80))
        self.ema_alpha_slow = float(getattr(self.geometry, "ema_alpha_slow", 0.95))
        self.context_summary_dim = self.D * (3 if self.use_ema_mid else 2)
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

    def _derive_controls(
        self,
        z: np.ndarray,
        v: np.ndarray,
        curr_idx: np.ndarray,
        knn_centroids: np.ndarray,
    ) -> np.ndarray:
        """
        Derive control values from trajectory properties so the policy learns
        a meaningful control-response mapping.

        Returns: [seq_len, control_dim] controls in [0, 1].
        """
        T = z.shape[0]
        controls = np.full((T, self.control_dim), 0.5, dtype=np.float32)

        # --- energy: normalized velocity magnitude ---
        v_mag = np.linalg.norm(v, axis=1)  # [T]
        v_max = v_mag.max() + 1e-8
        controls[:, 1] = np.clip(v_mag / v_max, 0.0, 1.0)

        # --- width: local trajectory spread (rolling std of positions) ---
        if T >= 3:
            half_w = max(1, T // 8)
            spread = np.zeros(T, dtype=np.float32)
            for i in range(T):
                lo, hi = max(0, i - half_w), min(T, i + half_w + 1)
                spread[i] = z[lo:hi].std()
            s_max = spread.max() + 1e-8
            controls[:, 0] = np.clip(spread / s_max, 0.0, 1.0)

        # --- gravity: alignment of velocity with local time gradient ---
        for i in range(T):
            tg = self.geometry.time_gradients[curr_idx[i]]
            tg_norm = np.linalg.norm(tg) + 1e-8
            v_norm = np.linalg.norm(v[i]) + 1e-8
            alignment = np.dot(v[i], tg) / (v_norm * tg_norm)
            controls[i, 2] = np.clip(0.5 + 0.5 * alignment, 0.0, 1.0)

        # --- memory: trajectory autocorrelation (smoothness) ---
        if T >= 3:
            diffs = np.linalg.norm(np.diff(v, axis=0), axis=1)
            d_max = diffs.max() + 1e-8
            smoothness = 1.0 - np.clip(diffs / d_max, 0.0, 1.0)
            controls[0, 3] = smoothness[0]
            controls[1:, 3] = smoothness

        # --- coherence: fraction of kNN in the same file ---
        for i in range(T):
            knn_idx = self.geometry.knn_indices[curr_idx[i]]
            fid = self.geometry.file_ids[curr_idx[i]]
            same = (self.geometry.file_ids[knn_idx] == fid).mean()
            controls[i, 4] = float(same)

        # --- exploration: distance from kNN centroid (normalized) ---
        dist = np.linalg.norm(z - knn_centroids, axis=1)
        sigmas = self.geometry.local_sigma[curr_idx]
        controls[:, 5] = np.clip(dist / (sigmas + 1e-8), 0.0, 1.0)

        return controls

    def __getitem__(self, idx: int):
        seq = self.sequences[idx % len(self.sequences)]
        curr_idx, next_idx = self._sample_window(seq)

        z = self.Z[curr_idx].astype(np.float32)
        z_next = self.Z[next_idx].astype(np.float32)

        v = np.zeros_like(z)
        v[1:] = z[1:] - z[:-1]

        m_fast, m_mid, m_slow = compute_causal_ema_summaries(
            z,
            alpha_fast=self.ema_alpha_fast,
            alpha_mid=self.ema_alpha_mid,
            alpha_slow=self.ema_alpha_slow,
            use_ema_mid=self.use_ema_mid,
        )
        if self.use_ema_mid:
            context_summaries = np.concatenate([m_fast, m_mid, m_slow], axis=1).astype(np.float32)
        else:
            context_summaries = np.concatenate([m_fast, m_slow], axis=1).astype(np.float32)

        local_features = np.zeros((self.seq_len, 16), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            local_features[i, 0] = self.geometry.local_sigma[idx_i]
            local_features[i, 1] = self.geometry.local_density[idx_i]
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid = self.Z[knn_idx].mean(axis=0)
            local_features[i, 2] = np.linalg.norm(z[i] - knn_centroid)
            time_grad = self.geometry.time_gradients[idx_i]
            z_norm_i = z[i] / (np.linalg.norm(z[i]) + 1e-6)
            local_features[i, 3] = np.dot(z_norm_i, time_grad)
            local_features[i, 4] = float(self.geometry.t_lat[idx_i]) / 1000.0
            local_features[i, 5] = float(self.geometry.file_ids[idx_i]) / 100.0

        knn_centroid = np.zeros((self.seq_len, self.D), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid[i] = self.Z[knn_idx].mean(axis=0)

        controls = self._derive_controls(z, v, curr_idx, knn_centroid)

        file_ids = self.geometry.file_ids[curr_idx].astype(np.int64)
        t_lat = self.geometry.t_lat[curr_idx].astype(np.float32)

        # Window targets for adaptive decoding (continuous log2)
        window_targets = self.window_targets[curr_idx].astype(np.float32)

        return {
            "z": z,
            "v": v,
            "z_next": z_next,
            "local_features": local_features,
            "context_summaries": context_summaries,
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
        z = _to_device_float_tensor(batch["z"], device)
        v = _to_device_float_tensor(batch["v"], device)
        z_next = _to_device_float_tensor(batch["z_next"], device)
        local_features = _to_device_float_tensor(batch["local_features"], device)
        context_summaries = _to_device_float_tensor(batch["context_summaries"], device)
        controls = _to_device_float_tensor(batch["controls"], device)
        knn_centroid = _to_device_float_tensor(batch["knn_centroid"], device)
        window_targets = _to_device_float_tensor(batch["window_targets"], device)

        if train:
            optimizer = weights["optimizer"]
            optimizer.zero_grad(set_to_none=True)

        delta_mean, delta_log_std, delta_weights, vel_delta, window_log2, _ = model(
            z=z,
            v=v,
            controls=controls,
            local_features=local_features,
            context_summaries=context_summaries,
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


def build_latent_loaders(sequences, Z, geometry, args, window_targets=None):
    sequences = list(sequences)
    np.random.shuffle(sequences)
    val_count = int(round(len(sequences) * float(args.val_split)))
    val_count = max(0, min(len(sequences) - 1, val_count)) if len(sequences) > 1 else 0

    val_seqs = sequences[:val_count]
    train_seqs = sequences[val_count:] if val_count > 0 else sequences

    train_ds = LatentTrajectoryDataset(
        train_seqs, Z, geometry,
        seq_len=int(args.seq_len),
        control_dim=int(args.control_dim),
        window_targets=window_targets,
    )
    val_ds = LatentTrajectoryDataset(
        val_seqs if val_seqs else train_seqs, Z, geometry,
        seq_len=int(args.seq_len),
        control_dim=int(args.control_dim),
        window_targets=window_targets,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _resolve_corpus_path(corpus_dir: str) -> str:
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        corpus_npz = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(corpus_npz):
            raise FileNotFoundError(f"No corpus found in {corpus_dir}")
        return corpus_npz


def _save_manual_navigation_artifact(
    data: dict,
    corpus_npz: str,
    out_path: str,
    leafsize: int,
):
    required = [
        "manual_pca_points",
        "manual_fader_p01",
        "manual_fader_p99",
        "frame_file_ids",
        "frame_t",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(
            "Corpus missing manual navigation fields. Re-run preprocess.py. "
            f"Missing keys: {missing}"
        )

    points = np.asarray(data["manual_pca_points"], dtype=np.float32)
    p01 = np.asarray(data["manual_fader_p01"], dtype=np.float32).reshape(-1)
    p99 = np.asarray(data["manual_fader_p99"], dtype=np.float32).reshape(-1)
    frame_file_ids = np.asarray(data["frame_file_ids"], dtype=np.int32).reshape(-1)
    frame_t = np.asarray(data["frame_t"], dtype=np.int32).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 8:
        raise RuntimeError(f"manual_pca_points must be [N, 8], got {points.shape}")
    if points.shape[0] != frame_file_ids.shape[0] or points.shape[0] != frame_t.shape[0]:
        raise RuntimeError("manual points and frame metadata length mismatch.")
    if p01.shape[0] != 8 or p99.shape[0] != 8:
        raise RuntimeError("manual_fader_p01/p99 must both be shape [8].")

    leaf = max(1, int(leafsize))
    tree = cKDTree(points, leafsize=leaf)

    # Validation query: nearest-neighbor distance on identity samples.
    sample_count = min(256, points.shape[0])
    sample_idx = np.linspace(0, points.shape[0] - 1, sample_count, dtype=np.int64)
    dists, nn_idx = tree.query(points[sample_idx], k=1)
    identical_ratio = float((nn_idx == sample_idx).mean())
    dists = np.asarray(dists, dtype=np.float32)
    print(
        f"[info] Manual KD-tree validation: samples={sample_count}, "
        f"identity={identical_ratio:.3f}, dist_mean={dists.mean():.6f}, dist_max={dists.max():.6f}"
    )

    source_corpus = np.array([os.path.abspath(corpus_npz)], dtype=np.str_)
    artifact = {
        "version": np.array(1, dtype=np.int32),
        "manual_pca_points": points.astype(np.float32),
        "manual_fader_p01": p01.astype(np.float32),
        "manual_fader_p99": p99.astype(np.float32),
        "frame_file_ids": frame_file_ids.astype(np.int32),
        "frame_t": frame_t.astype(np.int32),
        "kdtree_leafsize": np.array(int(leaf), dtype=np.int32),
        "source_corpus_path": source_corpus,
    }
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, **artifact)
    print(f"[done] Saved manual navigation artifact: {out_path}")


def _train_policy(
    args,
    data: dict,
):
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError("Corpus missing latent geometry. Re-run preprocess.py.")

    Z = data["Z_concat"].astype(np.float32)
    file_offsets = data["file_offsets"].astype(np.int64)

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

    sequences = build_sequences_from_offsets(file_offsets)
    print(f"[info] Sequences: {len(sequences)} (total points={Z.shape[0]})")

    train_loader, val_loader = build_latent_loaders(
        sequences, Z, geometry, args, window_targets=window_targets,
    )

    context_summary_dim = int(Z.shape[1]) * (3 if bool(geometry.use_ema_mid) else 2)

    cfg = LatentPolicyConfig(
        latent_dim=int(Z.shape[1]),
        hidden_size=int(args.hidden),
        num_layers=int(args.layers),
        num_mixture_components=4,
        control_dim=int(args.control_dim),
        local_feature_dim=16,
        context_summary_dim=context_summary_dim,
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
        else:
            log_every = max(1, int(args.log_every))
            epoch_num = epoch + 1
            if epoch_num == 1 or epoch_num % log_every == 0 or epoch_num == int(args.epochs):
                print(
                    f"[epoch {epoch_num:04d}/{args.epochs}] "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"val_loss={val_stats['loss']:.4f} "
                    f"win_mae={val_stats['window_mae_frames']:.2f}f"
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


def main():
    ap = argparse.ArgumentParser(
        description="Train latent policy and/or build manual navigation KD-tree artifact."
    )
    ap.add_argument("--corpus_dir", required=True, help="Folder containing corpus.npz")
    ap.add_argument(
        "--navigation_mode",
        choices=["policy", "manual", "both"],
        default="policy",
        help="Training mode: policy network, manual KD artifact, or both.",
    )
    ap.add_argument(
        "--manual_out_path",
        default=None,
        help="Output path for manual navigation artifact (.npz). Defaults to <corpus_dir>/manual_navigation.npz",
    )
    ap.add_argument(
        "--manual_kdtree_leafsize",
        type=int,
        default=32,
        help="Leaf size used when fitting manual cKDTree.",
    )
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
    ap.add_argument("--log_every", type=int, default=50,
                    help="Print progress every N epochs when --verbose is not set.")
    ap.add_argument("--verbose", action="store_true", help="Print detailed metrics each epoch.")
    ap.add_argument("--json_progress", action="store_true", help="Emit JSON progress updates.")

    args = ap.parse_args()
    corpus_npz = _resolve_corpus_path(args.corpus_dir)
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)

    mode = str(args.navigation_mode)
    if mode in ("manual", "both"):
        manual_out = args.manual_out_path
        if manual_out is None:
            manual_out = os.path.join(args.corpus_dir, "manual_navigation.npz")
        _save_manual_navigation_artifact(
            data=data,
            corpus_npz=corpus_npz,
            out_path=manual_out,
            leafsize=int(args.manual_kdtree_leafsize),
        )

    if mode in ("policy", "both"):
        _train_policy(args=args, data=data)


if __name__ == "__main__":
    main()
