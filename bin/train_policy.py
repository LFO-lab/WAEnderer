#!/usr/bin/env python3
import os, argparse, datetime, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stable_audio_wanderer.config import DEVICE
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import (
    IndexPolicy,
    PolicyConfig,
    IndexTrajectoryDataset,
    compute_annotations,
    group_meta_by_file,
)


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


def run_epoch(model, loader, device, weights, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    totals = {"loss": 0.0, "delta": 0.0, "vel": 0.0, "regime": 0.0, "samples": 0}

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
        delta_loss = F.cross_entropy(delta_logits.view(-1, delta_logits.size(-1)), delta_class.view(-1))
        vel_loss = F.smooth_l1_loss(dv_pred.view(-1), dv.view(-1))
        regime_loss = F.cross_entropy(regime_logits.view(-1, regime_logits.size(-1)), regime_next.view(-1))

        loss = delta_loss + weights["vel"] * vel_loss + weights["regime"] * regime_loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients")
            optimizer.step()

        bs = index_norm.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["delta"] += float(delta_loss.item()) * bs
        totals["vel"] += float(vel_loss.item()) * bs
        totals["regime"] += float(regime_loss.item()) * bs
        totals["samples"] += bs

    denom = max(1, totals["samples"])
    return {
        "loss": totals["loss"] / denom,
        "delta": totals["delta"] / denom,
        "vel": totals["vel"] / denom,
        "regime": totals["regime"] / denom,
    }


def main():
    ap = argparse.ArgumentParser(description="Train GRU policy over index trajectories (Δi logits + velocity/regime updates).")
    ap.add_argument("--corpus_dir", required=True, help="Folder containing *_corpus_*.npz")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--delta_max", type=int, default=8, help="Max |Δi| class span ⇒ bins = 2*delta_max + 1.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--input_proj", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--lambda_vel", type=float, default=0.5)
    ap.add_argument("--lambda_regime", type=float, default=0.5)
    ap.add_argument("--recur_k", type=int, default=6)
    ap.add_argument("--recur_exclude", type=int, default=2)
    ap.add_argument("--control_dim", type=int, default=4, help="Number of performer control channels (width, energy, gravity, memory).")
    ap.add_argument("--out_path", default=None, help="Override policy checkpoint path.")
    args = ap.parse_args()

    corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)
    ZZ = data["ZZ"].astype(np.float32)
    meta = data["meta"]

    print("[info] Annotating index trajectories…")
    annotations = compute_annotations(meta, ZZ, recur_k=args.recur_k, recur_exclude=args.recur_exclude)
    sequences = group_meta_by_file(meta)
    print(f"[info] Sequences: {len(sequences)} (total points={ZZ.shape[0]})")

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

    weights = {"vel": float(args.lambda_vel), "regime": float(args.lambda_regime), "optimizer": optimizer}
    history = {"train": [], "val": []}
    device = torch.device(DEVICE)

    for epoch in range(args.epochs):
        train_stats = run_epoch(model, train_loader, device, weights, train=True)
        val_stats = run_epoch(model, val_loader, device, weights, train=False)
        history["train"].append(train_stats)
        history["val"].append(val_stats)
        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train loss {train_stats['loss']:.4f} (Δ {train_stats['delta']:.4f} | v {train_stats['vel']:.4f} | m {train_stats['regime']:.4f}) "
            f"val {val_stats['loss']:.4f}"
        )

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
        },
        out_path,
    )
    print(f"[done] Saved policy checkpoint: {out_path}")


if __name__ == "__main__":
    main()
