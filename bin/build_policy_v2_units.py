#!/usr/bin/env python3
"""
Build V2 unit-graph artifact for trained morphology sequencing.
"""
import argparse
import os

import numpy as np

from stable_audio_wanderer.io.corpus_io import load_corpus, resolve_corpus_path
from stable_audio_wanderer.policy import UnitGraphConfig, build_v2_unit_artifact


def main():
    ap = argparse.ArgumentParser(
        description="Build unit-level graph artifact for V2 trained policy."
    )
    ap.add_argument("--corpus_dir", required=True, help="Folder containing corpus.npz")
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for V2 unit artifact (.npz). Defaults to <corpus_dir>/policy_v2_units.npz",
    )
    ap.add_argument("--min_sec", type=float, default=2.0, help="Minimum unit duration (seconds).")
    ap.add_argument("--max_sec", type=float, default=10.0, help="Maximum unit duration (seconds).")
    ap.add_argument("--target_sec", type=float, default=5.0, help="Target unit duration (seconds).")
    ap.add_argument("--candidate_k", type=int, default=64, help="Candidate neighbor pool size.")
    ap.add_argument("--graph_k", type=int, default=24, help="Saved outgoing transitions per unit.")
    ap.add_argument("--weight_entry", type=float, default=0.70, help="Exit->entry timbre continuity weight.")
    ap.add_argument("--weight_delta", type=float, default=0.30, help="Descriptor delta compatibility weight.")
    ap.add_argument(
        "--crossfile_penalty",
        type=float,
        default=0.10,
        help="Penalty added to cross-file transitions.",
    )
    ap.add_argument(
        "--boundary_smoothness_weight",
        type=float,
        default=0.35,
        help="Boundary preference for low descriptor-velocity cuts.",
    )
    args = ap.parse_args()

    corpus_npz = resolve_corpus_path(args.corpus_dir)
    data = load_corpus(corpus_npz)

    required = ["file_offsets", "frame_file_ids", "frame_t", "manual_desc_weighted"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(
            "Corpus missing fields required for V2 unit artifact. "
            f"Missing keys: {missing}"
        )

    file_offsets = np.asarray(data["file_offsets"], dtype=np.int64)
    frame_file_ids = np.asarray(data["frame_file_ids"], dtype=np.int32)
    frame_t = np.asarray(data["frame_t"], dtype=np.int32)
    desc_weighted = np.asarray(data["manual_desc_weighted"], dtype=np.float32)

    latent_hz = float(np.asarray(data.get("latent_hz", np.array(21.5, dtype=np.float32))).reshape(-1)[0])
    cfg = UnitGraphConfig(
        min_sec=float(args.min_sec),
        max_sec=float(args.max_sec),
        target_sec=float(args.target_sec),
        latent_hz=latent_hz,
        candidate_k=int(args.candidate_k),
        graph_k=int(args.graph_k),
        weight_entry=float(args.weight_entry),
        weight_delta=float(args.weight_delta),
        crossfile_penalty=float(args.crossfile_penalty),
        boundary_smoothness_weight=float(args.boundary_smoothness_weight),
    )

    artifact = build_v2_unit_artifact(
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        desc_weighted=desc_weighted,
        cfg=cfg,
        source_corpus_path=os.path.abspath(corpus_npz),
    )

    out_path = args.out or os.path.join(args.corpus_dir, "policy_v2_units.npz")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **artifact)

    n_units = int(artifact["unit_start_idx"].shape[0])
    k = int(artifact["unit_graph_neighbors"].shape[1])
    mean_len = float(np.asarray(artifact["unit_len"], dtype=np.float32).mean())
    min_len = int(np.asarray(artifact["unit_len"], dtype=np.int32).min())
    max_len = int(np.asarray(artifact["unit_len"], dtype=np.int32).max())
    print(f"[done] Saved V2 unit artifact: {out_path}")
    print(
        f"[info] Units={n_units}, graph_k={k}, "
        f"len_frames(min/mean/max)={min_len}/{mean_len:.1f}/{max_len}"
    )


if __name__ == "__main__":
    main()
