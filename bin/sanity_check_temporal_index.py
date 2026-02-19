#!/usr/bin/env python3
"""Quick sanity checks for temporal context corpus/index artifacts."""
import os
import argparse
import numpy as np
from scipy.spatial import cKDTree

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy import load_geometry_from_dict, compute_causal_ema_summaries


def resolve_corpus(corpus_dir: str) -> str:
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        fallback = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"No corpus file found in {corpus_dir}")
        return fallback


def main():
    ap = argparse.ArgumentParser(description="Sanity-check temporal context index and mappings")
    ap.add_argument("--corpus_dir", required=True, help="Directory containing corpus.npz")
    ap.add_argument("--knn_k", type=int, default=8, help="k for a sample retrieval query")
    args = ap.parse_args()

    corpus_npz = resolve_corpus(args.corpus_dir)
    data = load_corpus(corpus_npz)
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError("Missing geometry in corpus.")

    z = data["Z_concat"].astype(np.float32)
    offsets = data["file_offsets"].astype(np.int64)

    assert z.ndim == 2 and z.shape[1] == 64, f"Expected Z_concat [N,64], got {z.shape}"
    assert geometry.embeddings.shape[0] == z.shape[0], "E rows must match number of indexed frames"
    assert geometry.idx_to_file_id.shape[0] == z.shape[0], "idx_to_file_id length mismatch"
    assert geometry.idx_to_t.shape[0] == z.shape[0], "idx_to_t length mismatch"
    assert offsets.ndim == 1 and offsets.size >= 2, "file_offsets must be [num_files+1]"

    # Context summary shape sanity (same logic used by training input construction).
    file0_start = int(offsets[0])
    file0_end = int(offsets[1])
    z0 = z[file0_start:file0_end]
    m_fast, m_mid, m_slow = compute_causal_ema_summaries(
        z0,
        alpha_fast=float(geometry.ema_alpha_fast),
        alpha_mid=float(geometry.ema_alpha_mid),
        alpha_slow=float(geometry.ema_alpha_slow),
        use_ema_mid=bool(geometry.use_ema_mid),
    )
    if geometry.use_ema_mid:
        ctx_summary_dim = m_fast.shape[1] + m_mid.shape[1] + m_slow.shape[1]
    else:
        ctx_summary_dim = m_fast.shape[1] + m_slow.shape[1]
    assert ctx_summary_dim in (128, 192), f"Unexpected context summary dim: {ctx_summary_dim}"

    # Query validity sanity through causal context projection + nearest-neighbor lookup.
    q = geometry.embed_query(
        z_t=z0[0],
        m_fast_t=m_fast[0],
        m_slow_t=m_slow[0],
        m_mid_t=(m_mid[0] if (geometry.use_ema_mid and m_mid is not None) else None),
    )
    emb = geometry.embeddings.astype(np.float32)
    emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = np.maximum(emb_norm, 1e-6)
    emb_l2 = emb / emb_norm
    tree = cKDTree(emb_l2)
    dists, indices = tree.query(q, k=max(1, int(args.knn_k)))
    dists = np.atleast_1d(dists).astype(np.float32)
    indices = np.atleast_1d(indices).astype(np.int32)

    assert indices.size > 0, "Query returned no neighbors"
    assert np.all(indices >= 0) and np.all(indices < z.shape[0]), "Query returned invalid frame index"

    top_idx = int(indices[0])
    top_file = int(geometry.idx_to_file_id[top_idx])
    top_t = int(geometry.idx_to_t[top_idx])
    top_cosine_distance = float(0.5 * (dists[0] ** 2))

    print(f"[ok] corpus={corpus_npz}")
    print(f"[ok] indexed_frames={z.shape[0]} embed_dim={geometry.embeddings.shape[1]}")
    print(f"[ok] context_summary_dim={ctx_summary_dim} use_ema_mid={bool(geometry.use_ema_mid)}")
    print(f"[ok] top_query=(idx={top_idx}, file_id={top_file}, t={top_t}, cosine_dist={top_cosine_distance:.5f})")


if __name__ == "__main__":
    main()
