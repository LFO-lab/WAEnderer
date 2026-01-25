#!/usr/bin/env python3
"""
Train GRU policy over index trajectories with comprehensive metrics.
Supports entropy regularization, smoothness loss, and novelty bonuses.
"""
import os, argparse, datetime, json, sys, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from stable_audio_wanderer.config import DEVICE


def emit_json_progress(data: dict):
    """Emit JSON progress update to stdout for GUI consumption."""
    print(json.dumps(data), flush=True)
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import (
    IndexPolicy,
    PolicyConfig,
    IndexTrajectoryDataset,
    compute_annotations,
    group_meta_by_file,
    LatentPolicy,
    LatentPolicyConfig,
    load_geometry_from_dict,
)


def compute_metrics(
    delta_logits: torch.Tensor,
    delta_class: torch.Tensor,
    dv_pred: torch.Tensor,
    dv_target: torch.Tensor,
    delta_max: int,
) -> dict:
    """
    Compute comprehensive training metrics.

    Returns:
        dict with keys:
            - delta_acc_1: Top-1 accuracy for delta prediction
            - delta_acc_3: Top-3 accuracy for delta prediction
            - delta_entropy: Entropy of delta prediction distribution
            - vel_mae: Mean absolute error for velocity prediction
            - mean_displacement: Average |Δi| from predictions
    """
    batch_size = delta_logits.size(0)
    seq_len = delta_logits.size(1)
    num_classes = delta_logits.size(-1)

    # Flatten for metrics computation
    delta_logits_flat = delta_logits.view(-1, num_classes)
    delta_class_flat = delta_class.view(-1)

    # Delta accuracy (top-1)
    delta_pred = delta_logits_flat.argmax(dim=-1)
    delta_correct = (delta_pred == delta_class_flat).float()
    delta_acc_1 = delta_correct.mean().item()

    # Delta accuracy (top-3)
    _, top3_indices = delta_logits_flat.topk(min(3, num_classes), dim=-1)
    top3_correct = (top3_indices == delta_class_flat.unsqueeze(-1)).any(dim=-1).float()
    delta_acc_3 = top3_correct.mean().item()

    # Delta entropy (higher = more uncertain predictions)
    delta_probs = F.softmax(delta_logits_flat, dim=-1)
    delta_entropy = -(delta_probs * torch.log(delta_probs + 1e-8)).sum(dim=-1).mean().item()

    # Velocity MAE
    vel_mae = F.l1_loss(dv_pred.view(-1), dv_target.view(-1)).item()

    # Mean displacement from predictions
    # Convert delta class predictions back to displacement values
    delta_vals = delta_pred.float() - float(delta_max)
    mean_displacement = torch.abs(delta_vals).mean().item()

    return {
        "delta_acc_1": delta_acc_1,
        "delta_acc_3": delta_acc_3,
        "delta_entropy": delta_entropy,
        "vel_mae": vel_mae,
        "mean_displacement": mean_displacement,
    }


def compute_trajectory_metrics(
    delta_class: torch.Tensor,
    delta_max: int,
    total_corpus_size: int,
) -> dict:
    """
    Compute trajectory-level metrics from ground truth data.
    
    Returns:
        dict with:
            - coverage: Fraction of unique indices visited
            - recurrence_rate: How often trajectories revisit indices
    """
    # Reconstruct trajectory indices from delta classes
    batch_size = delta_class.size(0)
    seq_len = delta_class.size(1)
    
    total_unique = 0
    total_steps = 0
    total_revisits = 0
    
    for b in range(batch_size):
        deltas = delta_class[b].cpu().numpy() - delta_max
        # Simulate trajectory
        indices = [0]
        for d in deltas:
            next_idx = int(np.clip(indices[-1] + d, 0, total_corpus_size - 1))
            if next_idx in indices:
                total_revisits += 1
            indices.append(next_idx)
        
        total_unique += len(set(indices))
        total_steps += len(indices)
    
    coverage = total_unique / max(1, batch_size * (seq_len + 1))
    recurrence_rate = total_revisits / max(1, total_steps)
    
    return {
        "coverage": coverage,
        "recurrence_rate": recurrence_rate,
    }


class LatentTrajectoryDataset:
    """
    Dataset for training LatentPolicy over 64D latent trajectories.

    Samples windows over latent sequences for policy learning.
    Each item returns tensors for:
        - z: [seq_len, 64] latent positions
        - v: [seq_len, 64] velocities
        - z_next: [seq_len, 64] next latent positions (target)
        - local_features: [seq_len, 16] local geometry features
        - controls: [seq_len, 7] control parameters
        - knn_centroid: [seq_len, 64] kNN centroids for manifold loss
        - file_ids: [seq_len] source file IDs
        - t_lat: [seq_len] time positions
    """

    def __init__(
        self,
        sequences: list,
        GG: np.ndarray,
        geometry,
        seq_len: int = 32,
        control_dim: int = 7,
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

        # L2 normalize for consistent distances
        norms = np.linalg.norm(self.GG, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        self.GG_l2 = self.GG / norms

    def __len__(self):
        return len(self.sequences)

    def _sample_window(self, seq: np.ndarray):
        """Sample a window from a sequence."""
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

        # Latent positions
        z = self.GG[curr_idx].astype(np.float32)
        z_next = self.GG[next_idx].astype(np.float32)

        # Velocities (in latent space)
        v = np.zeros_like(z)
        v[1:] = z[1:] - z[:-1]

        # Local features
        local_features = np.zeros((self.seq_len, 16), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            local_features[i, 0] = self.geometry.local_sigma[idx_i]
            local_features[i, 1] = self.geometry.local_density[idx_i]
            # Distance to centroid
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid = self.GG[knn_idx].mean(axis=0)
            local_features[i, 2] = np.linalg.norm(z[i] - knn_centroid)
            # Time alignment
            time_grad = self.geometry.time_gradients[idx_i]
            z_norm_i = z[i] / (np.linalg.norm(z[i]) + 1e-6)
            local_features[i, 3] = np.dot(z_norm_i, time_grad)
            # Time and file info
            local_features[i, 4] = float(self.geometry.t_lat[idx_i]) / 1000.0
            local_features[i, 5] = float(self.geometry.file_ids[idx_i]) / 100.0

        # Controls (default values)
        controls = np.zeros((self.seq_len, self.control_dim), dtype=np.float32)
        controls[:, :4] = 0.5  # width, energy, gravity, memory at 0.5

        # kNN centroids for manifold loss
        knn_centroid = np.zeros((self.seq_len, self.D), dtype=np.float32)
        for i, idx_i in enumerate(curr_idx):
            knn_idx = self.geometry.knn_indices[idx_i]
            knn_centroid[i] = self.GG[knn_idx].mean(axis=0)

        # File IDs and time positions
        file_ids = self.geometry.file_ids[curr_idx].astype(np.int64)
        t_lat = self.geometry.t_lat[curr_idx].astype(np.float32)

        return {
            "z": z,
            "v": v,
            "z_next": z_next,
            "local_features": local_features,
            "controls": controls,
            "knn_centroid": knn_centroid,
            "file_ids": file_ids,
            "t_lat": t_lat,
        }


def compute_latent_losses(
    delta_mean: torch.Tensor,
    delta_log_std: torch.Tensor,
    delta_weights: torch.Tensor,
    vel_delta: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    z_next: torch.Tensor,
    knn_centroid: torch.Tensor,
    file_ids: torch.Tensor,
    t_lat: torch.Tensor,
    weights: dict,
) -> dict:
    """
    Compute all losses for latent policy training.

    Args:
        delta_mean: [B, T, M, D] mixture means
        delta_log_std: [B, T, M, D] mixture log-stds
        delta_weights: [B, T, M] mixture log-weights
        vel_delta: [B, T, D] velocity updates
        z: [B, T, D] current positions
        v: [B, T, D] current velocities
        z_next: [B, T, D] target next positions
        knn_centroid: [B, T, D] kNN centroids
        file_ids: [B, T] file IDs
        t_lat: [B, T] time positions
        weights: dict with loss weights

    Returns:
        dict with loss components
    """
    B, T, M, D = delta_mean.shape
    device = delta_mean.device

    # Compute predicted next position using weighted mixture mean
    # z_pred = z + v_new + delta_z
    mix_weights = F.softmax(delta_weights, dim=-1)  # [B, T, M]
    delta_z_pred = (mix_weights.unsqueeze(-1) * delta_mean).sum(dim=2)  # [B, T, D]

    # Velocity update
    alpha = 0.95
    v_new = alpha * v + vel_delta
    z_pred = z + v_new + delta_z_pred

    # 1. Reconstruction loss (MSE to true next position)
    recon_loss = F.mse_loss(z_pred, z_next)

    # 2. Contrastive loss (InfoNCE) - push z_pred toward z_next, away from batch negatives
    # Flatten batch and time dimensions
    z_pred_flat = z_pred.view(-1, D)  # [B*T, D]
    z_next_flat = z_next.view(-1, D)  # [B*T, D]

    # Normalize for cosine similarity
    z_pred_norm = F.normalize(z_pred_flat, dim=-1)
    z_next_norm = F.normalize(z_next_flat, dim=-1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z_pred_norm, z_next_norm.t())  # [B*T, B*T]
    sim_matrix = sim_matrix / weights.get("contrastive_temp", 0.1)

    # Positive pairs are on the diagonal
    labels = torch.arange(sim_matrix.size(0), device=device)
    contrastive_loss = F.cross_entropy(sim_matrix, labels)

    # 3. Smoothness loss (penalize acceleration)
    if T > 1:
        accel = v_new[:, 1:] - v_new[:, :-1]  # [B, T-1, D]
        # Scale by energy control (higher energy allows more acceleration)
        smooth_loss = torch.mean(accel ** 2)
    else:
        smooth_loss = torch.tensor(0.0, device=device)

    # 4. Manifold proximity loss (stay close to kNN centroid)
    manifold_loss = F.mse_loss(z_pred, knn_centroid)

    # 5. Coherence-time loss (penalize file/time jumps when coherence is high)
    # This would require coherence control from the input, skip for now
    coherence_loss = torch.tensor(0.0, device=device)

    # 6. Diversity loss (negative variance regularizer - prevent collapse)
    # Encourage spread in predictions across batch
    z_pred_var = z_pred_flat.var(dim=0).mean()
    diversity_loss = -z_pred_var  # Negative because we maximize variance

    # Total loss
    total_loss = (
        weights.get("recon", 1.0) * recon_loss +
        weights.get("contrastive", 0.1) * contrastive_loss +
        weights.get("smooth", 0.1) * smooth_loss +
        weights.get("manifold", 0.1) * manifold_loss +
        weights.get("coherence", 0.0) * coherence_loss +
        weights.get("diversity", 0.01) * diversity_loss
    )

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "contrastive_loss": contrastive_loss,
        "smooth_loss": smooth_loss,
        "manifold_loss": manifold_loss,
        "coherence_loss": coherence_loss,
        "diversity_loss": diversity_loss,
    }


def run_latent_epoch(model, loader, device, weights, train: bool):
    """
    Run one epoch of latent policy training or validation.

    Args:
        model: LatentPolicy model
        loader: DataLoader
        device: torch device
        weights: dict with loss weights and optimizer
        train: whether to train (True) or evaluate (False)

    Returns:
        dict with loss components
    """
    if train:
        model.train()
    else:
        model.eval()

    totals = defaultdict(float)
    totals["samples"] = 0

    for batch in loader:
        z = batch["z"].to(device)
        v = batch["v"].to(device)
        z_next = batch["z_next"].to(device)
        local_features = batch["local_features"].to(device)
        controls = batch["controls"].to(device)
        knn_centroid = batch["knn_centroid"].to(device)
        file_ids = batch["file_ids"].to(device)
        t_lat = batch["t_lat"].to(device)

        if train:
            optimizer = weights["optimizer"]
            optimizer.zero_grad(set_to_none=True)

        # Forward pass
        delta_mean, delta_log_std, delta_weights, vel_delta, _ = model(
            z=z, v=v, controls=controls, local_features=local_features
        )

        # Compute losses
        losses = compute_latent_losses(
            delta_mean, delta_log_std, delta_weights, vel_delta,
            z, v, z_next, knn_centroid, file_ids, t_lat, weights
        )

        if train:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = z.size(0)
        for k, v_loss in losses.items():
            totals[k] += float(v_loss.item()) * bs
        totals["samples"] += bs

    # Compute averages
    denom = max(1, totals["samples"])
    results = {k: totals[k] / denom for k in totals if k != "samples"}

    return results


def build_loaders(sequences, annotations, ZZ, args):
    split = int(round(len(sequences) * (1.0 - args.val_split)))
    split = max(1, min(split, len(sequences) - 1)) if len(sequences) > 1 else len(sequences)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:] if len(sequences) > 1 else sequences[:]

    train_ds = IndexTrajectoryDataset(train_seqs, annotations, ZZ, seq_len=args.seq_len, delta_max=args.delta_max, control_dim=args.control_dim)
    val_ds = IndexTrajectoryDataset(val_seqs, annotations, ZZ, seq_len=args.seq_len, delta_max=args.delta_max, control_dim=args.control_dim)

    def collate(batch):
        out = {}
        for key in batch[0]:
            arr = np.stack([b[key] for b in batch], axis=0)
            if arr.dtype.kind in ("i", "u"):
                out[key] = torch.from_numpy(arr).long()
            else:
                out[key] = torch.from_numpy(arr).float()
        return out

    # Keep small datasets (e.g., a single sequence) by not dropping the last batch.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    return train_loader, val_loader


def build_latent_loaders(sequences, GG, geometry, args):
    """Build data loaders for latent policy training."""
    split = int(round(len(sequences) * (1.0 - args.val_split)))
    split = max(1, min(split, len(sequences) - 1)) if len(sequences) > 1 else len(sequences)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:] if len(sequences) > 1 else sequences[:]

    train_ds = LatentTrajectoryDataset(train_seqs, GG, geometry, seq_len=args.seq_len, control_dim=args.control_dim)
    val_ds = LatentTrajectoryDataset(val_seqs, GG, geometry, seq_len=args.seq_len, control_dim=args.control_dim)

    def collate(batch):
        out = {}
        for key in batch[0]:
            arr = np.stack([b[key] for b in batch], axis=0)
            if arr.dtype.kind in ("i", "u"):
                out[key] = torch.from_numpy(arr).long()
            else:
                out[key] = torch.from_numpy(arr).float()
        return out

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    return train_loader, val_loader


def run_epoch(model, loader, device, weights, train: bool, corpus_size: int):
    """
    Run one epoch of training or validation with comprehensive metrics.

    Args:
        model: IndexPolicy model
        loader: DataLoader
        device: torch device
        weights: dict with loss weights and optimizer
        train: whether to train (True) or evaluate (False)
        corpus_size: total number of segments in corpus (for coverage metrics)

    Returns:
        dict with loss components and metrics
    """
    if train:
        model.train()
    else:
        model.eval()

    # Accumulators
    totals = defaultdict(float)
    totals["samples"] = 0

    delta_max = model.cfg.delta_max

    for batch in loader:
        index_norm = batch["index_norm"].to(device)
        velocity = batch["velocity"].to(device)
        descriptors = batch["descriptors"].to(device)
        delta_class = batch["delta_class"].to(device)
        dv = batch["dv"].to(device)
        embed = batch["embedding"].to(device)
        controls = batch["controls"].to(device)

        if train:
            optimizer = weights["optimizer"]
            optimizer.zero_grad(set_to_none=True)

        delta_logits, dv_pred, _ = model(
            index_norm=index_norm,
            velocity=velocity,
            descriptors=descriptors,
            embedding=embed,
            controls=controls,
        )

        # Core losses
        label_smoothing = weights.get("label_smoothing", 0.0)
        delta_loss = F.cross_entropy(
            delta_logits.view(-1, delta_logits.size(-1)),
            delta_class.view(-1),
            label_smoothing=label_smoothing,
        )
        vel_loss = F.smooth_l1_loss(dv_pred.view(-1), dv.view(-1))

        # Entropy regularization (encourage exploration)
        if weights.get("entropy", 0.0) > 0:
            delta_probs = F.softmax(delta_logits.view(-1, delta_logits.size(-1)), dim=-1)
            entropy = -(delta_probs * torch.log(delta_probs + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -weights["entropy"] * entropy  # Negative because we want to maximize entropy
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        # Smoothness loss (penalize acceleration)
        if weights.get("smooth", 0.0) > 0 and dv_pred.size(1) > 1:
            accel = dv_pred[:, 1:] - dv_pred[:, :-1]
            smooth_loss = weights["smooth"] * torch.mean(accel ** 2)
        else:
            smooth_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = (
            delta_loss
            + weights["vel"] * vel_loss
            + entropy_loss
            + smooth_loss
        )

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients")
            optimizer.step()

        bs = index_norm.size(0)

        # Accumulate losses
        totals["loss"] += float(loss.item()) * bs
        totals["delta_loss"] += float(delta_loss.item()) * bs
        totals["vel_loss"] += float(vel_loss.item()) * bs
        totals["entropy_loss"] += float(entropy_loss.item()) * bs
        totals["smooth_loss"] += float(smooth_loss.item()) * bs

        # Compute and accumulate metrics
        with torch.no_grad():
            metrics = compute_metrics(
                delta_logits, delta_class, dv_pred, dv, delta_max
            )
            for k, v in metrics.items():
                totals[k] += v * bs

            # Trajectory metrics
            traj_metrics = compute_trajectory_metrics(delta_class, delta_max, corpus_size)
            for k, v in traj_metrics.items():
                totals[k] += v * bs

        totals["samples"] += bs

    # Compute averages
    denom = max(1, totals["samples"])
    results = {
        "loss": totals["loss"] / denom,
        "delta_loss": totals["delta_loss"] / denom,
        "vel_loss": totals["vel_loss"] / denom,
        "entropy_loss": totals["entropy_loss"] / denom,
        "smooth_loss": totals["smooth_loss"] / denom,
        "delta_acc_1": totals["delta_acc_1"] / denom,
        "delta_acc_3": totals["delta_acc_3"] / denom,
        "delta_entropy": totals["delta_entropy"] / denom,
        "vel_mae": totals["vel_mae"] / denom,
        "mean_displacement": totals["mean_displacement"] / denom,
        "coverage": totals["coverage"] / denom,
        "recurrence_rate": totals["recurrence_rate"] / denom,
    }

    return results


def format_epoch_summary(epoch: int, total_epochs: int, train_stats: dict, val_stats: dict) -> str:
    """Format a comprehensive epoch summary for printing."""
    lines = [
        f"[epoch {epoch+1}/{total_epochs}]",
        f"  Loss: train={train_stats['loss']:.4f} val={val_stats['loss']:.4f}",
        f"    delta={train_stats['delta_loss']:.4f} vel={train_stats['vel_loss']:.4f}",
        f"  Accuracy: delta_top1={train_stats['delta_acc_1']:.3f} delta_top3={train_stats['delta_acc_3']:.3f}",
        f"  Metrics: entropy={train_stats['delta_entropy']:.2f} vel_mae={train_stats['vel_mae']:.3f} disp={train_stats['mean_displacement']:.2f}",
        f"  Trajectory: coverage={train_stats['coverage']:.3f} recur_rate={train_stats['recurrence_rate']:.3f}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Train GRU policy over index trajectories (Δi logits + velocity updates)."
    )
    # Data
    ap.add_argument("--corpus_dir", required=True, help="Folder containing *_corpus_*.npz")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--delta_max", type=int, default=8, help="Max |Δi| class span => bins = 2*delta_max + 1.")
    ap.add_argument("--val_split", type=float, default=0.1)
    
    # Training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--input_proj", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--control_dim", type=int, default=6,
                    help="Number of performer control channels (width, energy, gravity, memory, coherence, exploration).")
    
    # Loss weights (index policy)
    ap.add_argument("--lambda_vel", type=float, default=0.5, help="Weight for velocity loss.")
    ap.add_argument("--lambda_entropy", type=float, default=0.01,
                    help="Weight for entropy regularization (higher = more exploration).")
    ap.add_argument("--lambda_smooth", type=float, default=0.1,
                    help="Weight for smoothness loss (penalize acceleration).")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing factor for delta class predictions (0 = none, 0.1 = typical).")

    # Loss weights (latent policy)
    ap.add_argument("--lambda_recon", type=float, default=1.0, help="Weight for reconstruction loss (latent).")
    ap.add_argument("--lambda_contrastive", type=float, default=0.1, help="Weight for contrastive loss (latent).")
    ap.add_argument("--lambda_manifold", type=float, default=0.1, help="Weight for manifold proximity loss (latent).")
    ap.add_argument("--lambda_diversity", type=float, default=0.01, help="Weight for diversity loss (latent).")
    ap.add_argument("--contrastive_temp", type=float, default=0.1, help="Temperature for contrastive loss.")
    
    # Annotations
    ap.add_argument("--recur_k", type=int, default=6)
    ap.add_argument("--recur_exclude", type=int, default=2)
    
    # Output
    ap.add_argument("--out_path", default=None, help="Override policy checkpoint path.")
    ap.add_argument("--verbose", action="store_true", help="Print detailed metrics each epoch.")
    ap.add_argument("--json_progress", action="store_true",
                    help="Emit JSON progress updates for GUI consumption.")

    args = ap.parse_args()

    # Try pattern-based naming first, then fall back to simple corpus.npz
    try:
        corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        corpus_npz = os.path.join(args.corpus_dir, "corpus.npz")
        if not os.path.exists(corpus_npz):
            raise FileNotFoundError(f"No corpus found in {args.corpus_dir}")
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)
    meta = data["meta"]

    # Detect corpus format: latent (has geometry) or legacy (index-based)
    geometry = load_geometry_from_dict(data)
    is_latent_mode = geometry is not None

    if is_latent_mode:
        print("[info] Detected latent navigation corpus - training LatentPolicy")
        GG = data["GG"].astype(np.float32)
        corpus_size = GG.shape[0]
        latent_dim = GG.shape[1]

        sequences = group_meta_by_file(meta)
        print(f"[info] Sequences: {len(sequences)} (total points={corpus_size}, latent_dim={latent_dim})")

        train_loader, val_loader = build_latent_loaders(sequences, GG, geometry, args)

        cfg = LatentPolicyConfig(
            latent_dim=latent_dim,
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
            "contrastive": float(args.lambda_contrastive),
            "smooth": float(args.lambda_smooth),
            "manifold": float(args.lambda_manifold),
            "diversity": float(args.lambda_diversity),
            "contrastive_temp": float(args.contrastive_temp),
            "optimizer": optimizer,
        }

        print(f"[info] Training LatentPolicy with: recon={args.lambda_recon}, "
              f"contrastive={args.lambda_contrastive}, smooth={args.lambda_smooth}, "
              f"manifold={args.lambda_manifold}, diversity={args.lambda_diversity}")
    else:
        print("[info] Detected legacy index corpus - training IndexPolicy")
        ZZ = data["ZZ"].astype(np.float32)
        corpus_size = ZZ.shape[0]

        print("[info] Annotating index trajectories...")
        annotations = compute_annotations(meta, ZZ, recur_k=args.recur_k, recur_exclude=args.recur_exclude)
        sequences = group_meta_by_file(meta)
        print(f"[info] Sequences: {len(sequences)} (total points={corpus_size})")

        train_loader, val_loader = build_loaders(sequences, annotations, ZZ, args)

        cfg = PolicyConfig(
            delta_max=int(args.delta_max),
            desc_dim=int(annotations.desc.shape[1]),
            embed_dim=int(ZZ.shape[1]),
            control_dim=int(args.control_dim),
            hidden_size=int(args.hidden),
            input_proj=int(args.input_proj),
            num_layers=int(args.layers),
        )
        model = IndexPolicy(cfg).to(torch.device(DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        weights = {
            "vel": float(args.lambda_vel),
            "entropy": float(args.lambda_entropy),
            "smooth": float(args.lambda_smooth),
            "label_smoothing": float(args.label_smoothing),
            "optimizer": optimizer,
        }

        print(f"[info] Training IndexPolicy with: vel={args.lambda_vel}, "
              f"entropy={args.lambda_entropy}, smooth={args.lambda_smooth}, label_smooth={args.label_smoothing}")

    history = {"train": [], "val": []}
    device = torch.device(DEVICE)
    print(f"[info] Control dimensions: {args.control_dim}")
    print()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if is_latent_mode:
            train_stats = run_latent_epoch(model, train_loader, device, weights, train=True)
            val_stats = run_latent_epoch(model, val_loader, device, weights, train=False)
        else:
            train_stats = run_epoch(model, train_loader, device, weights, train=True, corpus_size=corpus_size)
            val_stats = run_epoch(model, val_loader, device, weights, train=False, corpus_size=corpus_size)
        history["train"].append(train_stats)
        history["val"].append(val_stats)

        if args.json_progress:
            # Emit JSON progress for GUI consumption
            progress_data = {
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "message": f"Epoch {epoch+1}/{args.epochs} - Loss: {train_stats['loss']:.4f}"
            }
            if not is_latent_mode:
                progress_data.update({
                    "delta_acc": train_stats.get("delta_acc_1", 0),
                    "entropy": train_stats.get("delta_entropy", 0),
                    "coverage": train_stats.get("coverage", 0),
                    "recurrence": train_stats.get("recurrence_rate", 0),
                    "vel_mae": train_stats.get("vel_mae", 0),
                })
            else:
                progress_data.update({
                    "recon_loss": train_stats.get("recon_loss", 0),
                    "contrastive_loss": train_stats.get("contrastive_loss", 0),
                    "manifold_loss": train_stats.get("manifold_loss", 0),
                })
            emit_json_progress(progress_data)
        elif args.verbose and not is_latent_mode:
            print(format_epoch_summary(epoch, args.epochs, train_stats, val_stats))
            print()
        else:
            # Compact output
            if is_latent_mode:
                print(
                    f"[epoch {epoch+1}/{args.epochs}] "
                    f"loss {train_stats['loss']:.4f} | "
                    f"recon {train_stats.get('recon_loss', 0):.4f} | "
                    f"contr {train_stats.get('contrastive_loss', 0):.4f} | "
                    f"manif {train_stats.get('manifold_loss', 0):.4f} | "
                    f"val {val_stats['loss']:.4f}"
                )
            else:
                print(
                    f"[epoch {epoch+1}/{args.epochs}] "
                    f"loss {train_stats['loss']:.4f} | "
                    f"acc {train_stats['delta_acc_1']:.3f} | "
                    f"ent {train_stats['delta_entropy']:.2f} | "
                    f"cov {train_stats['coverage']:.3f} | "
                    f"recur {train_stats['recurrence_rate']:.3f} | "
                    f"val {val_stats['loss']:.4f}"
                )

        # Track best model
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_path:
        out_path = args.out_path
    else:
        policy_type = "latent_policy" if is_latent_mode else "policy"
        out_path = os.path.join(os.getcwd(), f"{policy_type}_{ts}.pt")

    # Build checkpoint dict based on mode
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "seq_len": int(args.seq_len),
        "history": history,
        "train_args": vars(args),
        "best_val_loss": best_val_loss,
        "policy_type": "latent" if is_latent_mode else "index",
    }

    if is_latent_mode:
        checkpoint["latent_dim"] = latent_dim
    else:
        checkpoint.update({
            "desc_mean": annotations.desc_mean,
            "desc_std": annotations.desc_std,
            "embed_min": annotations.embed_min,
            "embed_range": annotations.embed_range,
            "delta_max": int(args.delta_max),
        })

    torch.save(checkpoint, out_path)
    print()
    print(f"[done] Saved policy checkpoint: {out_path}")
    print(f"[done] Best validation loss: {best_val_loss:.4f}")

    # Convert to CoreML for macOS app
    if args.json_progress:
        emit_json_progress({"message": "Converting to CoreML..."})
    else:
        print("[info] Converting to CoreML...")

    try:
        import subprocess
        import sys

        # Find convert_to_coreml.py - could be in same dir, bin dir, or PythonScripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "convert_to_coreml.py"),
            os.path.join(os.path.dirname(script_dir), "bin", "convert_to_coreml.py"),
            os.path.join(script_dir, "..", "bin", "convert_to_coreml.py"),
        ]

        convert_script = None
        for p in possible_paths:
            if os.path.exists(p):
                convert_script = p
                break

        if convert_script is None:
            raise FileNotFoundError("convert_to_coreml.py not found")

        # Convert .pt to .mlpackage using subprocess
        mlpackage_path = out_path.replace(".pt", ".mlpackage")
        convert_result = subprocess.run(
            [sys.executable, convert_script, "--checkpoint", out_path, "--output", mlpackage_path],
            capture_output=True,
            text=True
        )

        if convert_result.returncode != 0:
            print(f"[warn] CoreML conversion failed: {convert_result.stderr}")
            raise RuntimeError(f"Conversion failed: {convert_result.stderr}")

        print(f"[done] Converted to CoreML package: {mlpackage_path}")

        # Compile .mlpackage to .mlmodelc using xcrun
        mlmodelc_path = out_path.replace(".pt", ".mlmodelc")
        output_dir = os.path.dirname(mlmodelc_path) or "."

        compile_result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir],
            capture_output=True,
            text=True
        )

        if compile_result.returncode == 0:
            print(f"[done] Compiled CoreML model: {mlmodelc_path}")
            if args.json_progress:
                emit_json_progress({"message": f"CoreML model ready: {mlmodelc_path}"})
        else:
            print(f"[warn] CoreML compilation failed: {compile_result.stderr}")
            if args.json_progress:
                emit_json_progress({"message": "CoreML compilation failed - model will need manual conversion"})

    except FileNotFoundError as e:
        print(f"[warn] CoreML conversion skipped (script not found): {e}")
    except Exception as e:
        print(f"[warn] CoreML conversion failed: {e}")


if __name__ == "__main__":
    main()
