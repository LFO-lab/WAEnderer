#!/usr/bin/env python3
"""
Render extracted V2 units to WAV for offline listening.
"""
import argparse
import os
from typing import List

import numpy as np

from stable_audio_wanderer.config import SR
from stable_audio_wanderer.io.audio_io import save_wav
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.vae.decoder import decode_latents
from stable_audio_wanderer.vae.sae import load_vae


def _resolve_corpus_path(corpus_dir: str) -> str:
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        fallback = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"No corpus file found in {corpus_dir}")
        return fallback


def _parse_indices(value: str) -> List[int]:
    out: List[int] = []
    for token in value.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _select_units(
    n_units: int,
    mode: str,
    count: int,
    seed: int,
) -> np.ndarray:
    n_units = int(n_units)
    count = int(max(1, min(count, n_units)))
    mode = str(mode).lower().strip()

    if mode == "spread":
        idx = np.linspace(0, n_units - 1, count, dtype=np.int32)
        return np.unique(idx).astype(np.int32)
    if mode == "first":
        return np.arange(count, dtype=np.int32)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n_units, size=count, replace=False).astype(np.int32))
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    ap = argparse.ArgumentParser(
        description="Decode V2 units into WAV files for listening."
    )
    ap.add_argument("--corpus_dir", required=True, help="Corpus directory containing corpus.npz")
    ap.add_argument(
        "--v2_artifact",
        default=None,
        help="Path to policy_v2_units.npz (default: <corpus_dir>/policy_v2_units.npz)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output folder for rendered WAVs (default: <corpus_dir>/v2_unit_renders)",
    )
    ap.add_argument(
        "--mode",
        choices=["spread", "random", "first"],
        default="spread",
        help="Automatic unit selection mode when --unit_indices is not provided.",
    )
    ap.add_argument(
        "--num_units",
        type=int,
        default=12,
        help="Number of units to render in automatic mode.",
    )
    ap.add_argument(
        "--unit_indices",
        type=str,
        default=None,
        help="Comma-separated explicit unit ids to render (overrides --mode/--num_units).",
    )
    ap.add_argument("--seed", type=int, default=7, help="Random seed for --mode random.")
    ap.add_argument("--gap_sec", type=float, default=0.25, help="Gap in preview montage (seconds).")
    ap.add_argument(
        "--pretrained",
        default="stabilityai/stable-audio-open-1.0",
        help="Stable Audio VAE checkpoint path or hub repo.",
    )
    ap.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print selected units and output names without decoding.",
    )
    args = ap.parse_args()

    corpus_npz = _resolve_corpus_path(args.corpus_dir)
    artifact_path = args.v2_artifact or os.path.join(args.corpus_dir, "policy_v2_units.npz")
    artifact_path = os.path.abspath(artifact_path)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"V2 artifact not found: {artifact_path}")

    corpus = load_corpus(corpus_npz)
    art = np.load(artifact_path, allow_pickle=True)

    required_corpus = ["Z_concat", "Z_mean", "Z_std", "frame_file_ids", "frame_t"]
    missing_corpus = [k for k in required_corpus if k not in corpus]
    if missing_corpus:
        raise RuntimeError(f"Corpus missing required fields: {missing_corpus}")

    required_art = ["unit_start_idx", "unit_end_idx", "unit_file_id", "unit_start_t", "unit_end_t"]
    missing_art = [k for k in required_art if k not in art]
    if missing_art:
        raise RuntimeError(f"V2 artifact missing required fields: {missing_art}")

    z_norm = np.asarray(corpus["Z_concat"], dtype=np.float32)
    z_mean = np.asarray(corpus["Z_mean"], dtype=np.float32).reshape(-1)
    z_std = np.asarray(corpus["Z_std"], dtype=np.float32).reshape(-1)

    unit_start = np.asarray(art["unit_start_idx"], dtype=np.int32)
    unit_end = np.asarray(art["unit_end_idx"], dtype=np.int32)
    unit_file = np.asarray(art["unit_file_id"], dtype=np.int32)
    unit_t0 = np.asarray(art["unit_start_t"], dtype=np.int32)
    unit_t1 = np.asarray(art["unit_end_t"], dtype=np.int32)

    n_units = int(unit_start.shape[0])
    if n_units <= 0:
        raise RuntimeError("No units found in V2 artifact.")

    if args.unit_indices:
        selected = np.asarray(_parse_indices(args.unit_indices), dtype=np.int32)
    else:
        selected = _select_units(
            n_units=n_units,
            mode=str(args.mode),
            count=int(args.num_units),
            seed=int(args.seed),
        )
    selected = selected[(selected >= 0) & (selected < n_units)]
    selected = np.unique(selected).astype(np.int32)
    if selected.size == 0:
        raise RuntimeError("No valid unit indices selected.")

    out_dir = args.out_dir or os.path.join(args.corpus_dir, "v2_unit_renders")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] Corpus: {corpus_npz}")
    print(f"[info] V2 artifact: {artifact_path}")
    print(f"[info] Selected units ({selected.size}): {selected.tolist()}")
    print(f"[info] Output dir: {out_dir}")

    if args.dry_run:
        for uid in selected.tolist():
            name = (
                f"unit_{uid:05d}_f{int(unit_file[uid]):03d}_"
                f"t{int(unit_t0[uid]):05d}-{int(unit_t1[uid]):05d}.wav"
            )
            print(f"[dry-run] {name}")
        return

    print("[info] Loading VAE decoder...")
    try:
        ae = load_vae(args.pretrained)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load VAE decoder for unit rendering. "
            "If network access is unavailable, pass a local pretrained path via --pretrained."
        ) from exc

    rendered = []
    for uid in selected.tolist():
        s = int(unit_start[uid])
        e = int(unit_end[uid])
        if not (0 <= s < e <= z_norm.shape[0]):
            print(f"[warn] Skipping invalid unit bounds uid={uid}, [{s}, {e})")
            continue
        z_raw = z_norm[s:e] * z_std[None, :] + z_mean[None, :]
        audio = decode_latents(ae, z_raw)
        name = (
            f"unit_{uid:05d}_f{int(unit_file[uid]):03d}_"
            f"t{int(unit_t0[uid]):05d}-{int(unit_t1[uid]):05d}.wav"
        )
        out_path = os.path.join(out_dir, name)
        save_wav(out_path, audio, sr=SR)
        rendered.append(audio)
        print(f"[done] {out_path}")

    if not rendered:
        raise RuntimeError("No units were rendered.")

    gap = np.zeros((int(max(0.0, float(args.gap_sec)) * SR), 2), dtype=np.float32)
    montage_parts = []
    for i, audio in enumerate(rendered):
        montage_parts.append(audio.astype(np.float32))
        if i + 1 < len(rendered) and gap.shape[0] > 0:
            montage_parts.append(gap)
    montage = np.concatenate(montage_parts, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(montage)))
    if peak > 0.98:
        montage = montage * (0.98 / peak)
    montage_path = os.path.join(out_dir, "units_preview_montage.wav")
    save_wav(montage_path, montage, sr=SR)
    print(f"[done] Preview montage: {montage_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
