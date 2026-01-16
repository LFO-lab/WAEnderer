#!/usr/bin/env python3
"""
Train GRU policy over index trajectories with comprehensive metrics.
Supports entropy regularization, smoothness loss, and novelty bonuses.
"""
import os, argparse, datetime, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from stable_audio_wanderer.config import DEVICE
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import (
    IndexPolicy,
    PolicyConfig,
    IndexTrajectoryDataset,
    compute_annotations,
    group_meta_by_file,
)


def compute_metrics(
    delta_logits: torch.Tensor,
    delta_class: torch.Tensor,
    dv_pred: torch.Tensor,
    dv_target: torch.Tensor,
    regime_logits: torch.Tensor,
    regime_target: torch.Tensor,
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
            - regime_acc: Accuracy for regime prediction
            - mean_displacement: Average |Δi| from predictions
            - regime_dist: Distribution across regime classes
    """
    batch_size = delta_logits.size(0)
    seq_len = delta_logits.size(1)
    num_classes = delta_logits.size(-1)
    
    # Flatten for metrics computation
    delta_logits_flat = delta_logits.view(-1, num_classes)
    delta_class_flat = delta_class.view(-1)
    regime_logits_flat = regime_logits.view(-1, regime_logits.size(-1))
    regime_target_flat = regime_target.view(-1)
    
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
    
    # Regime accuracy
    regime_pred = regime_logits_flat.argmax(dim=-1)
    regime_correct = (regime_pred == regime_target_flat).float()
    regime_acc = regime_correct.mean().item()
    
    # Mean displacement from predictions
    # Convert delta class predictions back to displacement values
    delta_vals = delta_pred.float() - float(delta_max)
    mean_displacement = torch.abs(delta_vals).mean().item()
    
    # Regime distribution (what fraction in each regime)
    num_regime_classes = regime_logits.size(-1)
    regime_counts = torch.zeros(num_regime_classes, device=regime_pred.device)
    for r in range(num_regime_classes):
        regime_counts[r] = (regime_pred == r).sum()
    regime_dist = (regime_counts / regime_counts.sum()).cpu().numpy().tolist()
    
    return {
        "delta_acc_1": delta_acc_1,
        "delta_acc_3": delta_acc_3,
        "delta_entropy": delta_entropy,
        "vel_mae": vel_mae,
        "regime_acc": regime_acc,
        "mean_displacement": mean_displacement,
        "regime_dist": regime_dist,
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
        regime = batch["regime"].to(device)
        regime_next = batch["regime_next"].to(device)
        delta_class = batch["delta_class"].to(device)
        dv = batch["dv"].to(device)
        embed = batch["embedding"].to(device)
        controls = batch["controls"].to(device)

        if train:
            optimizer = weights["optimizer"]
            optimizer.zero_grad(set_to_none=True)

        delta_logits, dv_pred, regime_logits, _ = model(
            index_norm=index_norm,
            velocity=velocity,
            descriptors=descriptors,
            regime=regime,
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
        regime_loss = F.cross_entropy(regime_logits.view(-1, regime_logits.size(-1)), regime_next.view(-1))
        
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
            + weights["regime"] * regime_loss 
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
        totals["regime_loss"] += float(regime_loss.item()) * bs
        totals["entropy_loss"] += float(entropy_loss.item()) * bs
        totals["smooth_loss"] += float(smooth_loss.item()) * bs
        
        # Compute and accumulate metrics
        with torch.no_grad():
            metrics = compute_metrics(
                delta_logits, delta_class, dv_pred, dv, regime_logits, regime_next, delta_max
            )
            for k, v in metrics.items():
                if k != "regime_dist":  # Handle regime_dist separately
                    totals[k] += v * bs
            
            # Accumulate regime distribution
            if "regime_dist_sum" not in totals:
                totals["regime_dist_sum"] = np.zeros(len(metrics["regime_dist"]))
            totals["regime_dist_sum"] += np.array(metrics["regime_dist"]) * bs
            
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
        "regime_loss": totals["regime_loss"] / denom,
        "entropy_loss": totals["entropy_loss"] / denom,
        "smooth_loss": totals["smooth_loss"] / denom,
        "delta_acc_1": totals["delta_acc_1"] / denom,
        "delta_acc_3": totals["delta_acc_3"] / denom,
        "delta_entropy": totals["delta_entropy"] / denom,
        "vel_mae": totals["vel_mae"] / denom,
        "regime_acc": totals["regime_acc"] / denom,
        "mean_displacement": totals["mean_displacement"] / denom,
        "coverage": totals["coverage"] / denom,
        "recurrence_rate": totals["recurrence_rate"] / denom,
        "regime_dist": (totals["regime_dist_sum"] / denom).tolist(),
    }
    
    return results


def format_epoch_summary(epoch: int, total_epochs: int, train_stats: dict, val_stats: dict) -> str:
    """Format a comprehensive epoch summary for printing."""
    lines = [
        f"[epoch {epoch+1}/{total_epochs}]",
        f"  Loss: train={train_stats['loss']:.4f} val={val_stats['loss']:.4f}",
        f"    delta={train_stats['delta_loss']:.4f} vel={train_stats['vel_loss']:.4f} regime={train_stats['regime_loss']:.4f}",
        f"  Accuracy: delta_top1={train_stats['delta_acc_1']:.3f} delta_top3={train_stats['delta_acc_3']:.3f} regime={train_stats['regime_acc']:.3f}",
        f"  Metrics: entropy={train_stats['delta_entropy']:.2f} vel_mae={train_stats['vel_mae']:.3f} disp={train_stats['mean_displacement']:.2f}",
        f"  Trajectory: coverage={train_stats['coverage']:.3f} recur_rate={train_stats['recurrence_rate']:.3f}",
        f"  Regime dist: drift={train_stats['regime_dist'][0]:.2f} turn={train_stats['regime_dist'][1]:.2f} linger={train_stats['regime_dist'][2]:.2f}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Train GRU policy over index trajectories (Δi logits + velocity/regime updates)."
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
    ap.add_argument("--control_dim", type=int, default=7, 
                    help="Number of performer control channels (width, energy, gravity, memory, coherence, exploration, regime_bias).")
    
    # Loss weights
    ap.add_argument("--lambda_vel", type=float, default=0.5, help="Weight for velocity loss.")
    ap.add_argument("--lambda_regime", type=float, default=0.5, help="Weight for regime loss.")
    ap.add_argument("--lambda_entropy", type=float, default=0.01, 
                    help="Weight for entropy regularization (higher = more exploration).")
    ap.add_argument("--lambda_smooth", type=float, default=0.1, 
                    help="Weight for smoothness loss (penalize acceleration).")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing factor for delta class predictions (0 = none, 0.1 = typical).")
    
    # Annotations
    ap.add_argument("--recur_k", type=int, default=6)
    ap.add_argument("--recur_exclude", type=int, default=2)
    
    # Output
    ap.add_argument("--out_path", default=None, help="Override policy checkpoint path.")
    ap.add_argument("--verbose", action="store_true", help="Print detailed metrics each epoch.")
    
    args = ap.parse_args()

    corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)
    ZZ = data["ZZ"].astype(np.float32)
    meta = data["meta"]
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
        regime_classes=int(annotations.regime.max()) + 1,
        control_dim=int(args.control_dim),
        hidden_size=int(args.hidden),
        input_proj=int(args.input_proj),
        num_layers=int(args.layers),
    )
    model = IndexPolicy(cfg).to(torch.device(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights = {
        "vel": float(args.lambda_vel),
        "regime": float(args.lambda_regime),
        "entropy": float(args.lambda_entropy),
        "smooth": float(args.lambda_smooth),
        "label_smoothing": float(args.label_smoothing),
        "optimizer": optimizer,
    }
    history = {"train": [], "val": []}
    device = torch.device(DEVICE)

    print(f"[info] Training with: vel={args.lambda_vel}, regime={args.lambda_regime}, "
          f"entropy={args.lambda_entropy}, smooth={args.lambda_smooth}, label_smooth={args.label_smoothing}")
    print(f"[info] Control dimensions: {args.control_dim}")
    print()

    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_stats = run_epoch(model, train_loader, device, weights, train=True, corpus_size=corpus_size)
        val_stats = run_epoch(model, val_loader, device, weights, train=False, corpus_size=corpus_size)
        history["train"].append(train_stats)
        history["val"].append(val_stats)
        
        if args.verbose:
            print(format_epoch_summary(epoch, args.epochs, train_stats, val_stats))
            print()
        else:
            # Compact output
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
        out_path = os.path.join(os.getcwd(), f"policy_{ts}.pt")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "desc_mean": annotations.desc_mean,
            "desc_std": annotations.desc_std,
            "embed_min": annotations.embed_min,
            "embed_range": annotations.embed_range,
            "delta_max": int(args.delta_max),
            "seq_len": int(args.seq_len),
            "history": history,
            "train_args": vars(args),
            "best_val_loss": best_val_loss,
        },
        out_path,
    )
    print()
    print(f"[done] Saved policy checkpoint: {out_path}")
    print(f"[done] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
