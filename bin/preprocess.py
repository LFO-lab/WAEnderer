#!/usr/bin/env python3
"""
Preprocess audio files into a 64D latent corpus with geometry and grains.
Pipeline:
  - Encode audio with Stable Audio Open VAE
  - Mean-pool latents per segment -> GG (64D)
  - Normalize GG with global Z_mean/Z_std
  - Compute latent geometry (kNN + PCA for visualization)
  - Render grains and save manifest
"""
import os
import argparse
import datetime
import glob
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

from stable_audio_wanderer.config import SR, LATENT_HZ
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.io.corpus_io import save_corpus, save_grain_manifest
from stable_audio_wanderer.io.audio_io import save_wav
from stable_audio_wanderer.policy import compute_latent_geometry, save_geometry_to_dict


def render_grains(
    wav_dict: Dict[int, np.ndarray],
    meta: np.ndarray,
    grain_sec: float,
    out_dir: str,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Render raw grains to disk (no envelope - enveloping happens at playback)."""
    grain_len_samp = int(round(grain_sec * sr))

    # Group segments by file
    file_segments: Dict[int, List[Tuple[int, int]]] = {}
    for seg_idx, (fid, t_lat, _) in enumerate(meta):
        fid = int(fid)
        if fid not in file_segments:
            file_segments[fid] = []
        file_segments[fid].append((seg_idx, int(t_lat)))

    offsets_list: List[int] = []
    lengths_list: List[int] = []
    file_ids_list: List[int] = []
    segment_ids_list: List[int] = []
    grain_paths: List[str] = []

    grains_dir = os.path.join(out_dir, "grains")
    os.makedirs(grains_dir, exist_ok=True)

    file_ids = sorted(file_segments.keys())
    for fid in file_ids:
        wav = wav_dict[fid]
        segments = file_segments[fid]

        grain_buffer_list: List[np.ndarray] = []
        current_offset = 0

        for seg_idx, t_lat in segments:
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * sr))
            samp_end = min(samp_start + grain_len_samp, wav.shape[0])
            grain = wav[samp_start:samp_end].copy()

            if grain.shape[0] < grain_len_samp:
                pad_len = grain_len_samp - grain.shape[0]
                pad = np.zeros((pad_len, grain.shape[1]), dtype=np.float32)
                grain = np.concatenate([grain, pad], axis=0)

            offsets_list.append(current_offset)
            lengths_list.append(grain.shape[0])
            file_ids_list.append(fid)
            segment_ids_list.append(seg_idx)

            grain_buffer_list.append(grain)
            current_offset += grain.shape[0]

        if grain_buffer_list:
            file_grain_buffer = np.concatenate(grain_buffer_list, axis=0)
            grain_path = os.path.join(grains_dir, f"file_{fid:04d}.wav")
            save_wav(grain_path, file_grain_buffer, sr)
            grain_paths.append(grain_path)
        else:
            grain_paths.append("")

    return (
        np.array(offsets_list, dtype=np.int64),
        np.array(lengths_list, dtype=np.int64),
        np.array(file_ids_list, dtype=np.int32),
        np.array(segment_ids_list, dtype=np.int32),
        grain_paths,
    )


def compute_segment_latents(latents_dict: Dict[str, np.ndarray], meta: np.ndarray, win_lat: int) -> np.ndarray:
    """Mean-pool latent windows into per-segment embeddings (GG)."""
    seg_latents = []
    for fid, start, seg_len in meta:
        key = f"z_{int(fid)}"
        z_full = latents_dict[key]
        start = int(start)
        seg_len = int(seg_len) if int(seg_len) > 0 else int(win_lat)
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
    ap = argparse.ArgumentParser(
        description="Preprocess: VAE latents -> 64D corpus + geometry + grains."
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--latent_nav_k", type=int, default=32, help="k for latent kNN geometry.")
    ap.add_argument(
        "--render_grains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render pre-baked grain audio buffers for runtime playback (default: on).",
    )
    ap.add_argument(
        "--grain_sec",
        type=float,
        default=1.0 / LATENT_HZ,
        help="Grain duration in seconds (default: 1/LATENT_HZ).",
    )

    args = ap.parse_args()

    prefix = os.path.basename(args.out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= 2048, "seg_sec must be >= 2048/44100"

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    ae = load_vae(args.pretrained)

    meta_list = []
    latent_sequences: List[np.ndarray] = []
    latents_dict = {}
    wav_dict: Dict[int, np.ndarray] = {}

    print("Encoding audio with VAE...")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        wav_dict[fid] = wav
        z_full = encode_full(ae, wav).astype(np.float32)
        z_full = np.ascontiguousarray(z_full)
        latents_dict[f"z_{fid}"] = z_full
        latent_sequences.append(z_full)

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            meta_list.append((fid, t_lat, win_lat))

    if not latent_sequences:
        raise RuntimeError("No latent sequences were extracted.")

    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    # Compute normalization stats from all VAE latents
    latent_stack = np.concatenate(latent_sequences, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    # Compute GG (segment latents) from VAE embeddings - mean pooling per segment
    GG = compute_segment_latents(latents_dict, meta, win_lat)

    # Normalize GG using VAE latent statistics
    GG_norm = (GG - Z_mean[None, :]) / Z_std[None, :]
    GG = np.clip(GG_norm, -5.0, 5.0).astype(np.float32)

    # Compute latent geometry (kNN + PCA projection)
    GG_norms = np.linalg.norm(GG, axis=1, keepdims=True)
    GG_norms = np.maximum(GG_norms, 1e-6)
    GG_l2 = (GG / GG_norms).astype(np.float32)

    print(f"Computing latent geometry (k={args.latent_nav_k})...")
    geometry = compute_latent_geometry(GG_l2, meta, k=int(args.latent_nav_k))
    geometry_arrays = save_geometry_to_dict(geometry)

    # Save corpus
    corpus_path = os.path.join(out_dir, "corpus.npz")
    save_corpus(
        corpus_path,
        GG=GG,
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        sr=np.array(int(SR), dtype=np.int32),
        latent_hz=np.array(float(LATENT_HZ), dtype=np.float32),
        segment_dur=np.array(float(args.seg_sec), dtype=np.float32),
        hop_dur=np.array(float(args.hop_sec), dtype=np.float32),
        latent_nav_k=np.array(int(args.latent_nav_k), dtype=np.int32),
        **geometry_arrays,
    )

    # Grain rendering
    grain_manifest_path = None
    if args.render_grains:
        print(f"Rendering grains (grain_sec={args.grain_sec})...")
        offsets, lengths, file_ids, segment_ids, grain_paths = render_grains(
            wav_dict=wav_dict,
            meta=meta,
            grain_sec=args.grain_sec,
            out_dir=out_dir,
            sr=SR,
        )
        grain_manifest_path = os.path.join(out_dir, "grains", "manifest.npz")
        save_grain_manifest(
            path=grain_manifest_path,
            offsets=offsets,
            lengths=lengths,
            file_ids=file_ids,
            segment_ids=segment_ids,
            grain_paths=grain_paths,
            grain_sec=args.grain_sec,
            grain_hop_sec=args.hop_sec,
            sr=SR,
        )

    # Clear wav_dict to free memory
    wav_dict.clear()

    print("\nSaved:")
    print("  Corpus         :", corpus_path)
    if grain_manifest_path is not None:
        print("  Grain manifest :", grain_manifest_path)


if __name__ == "__main__":
    main()
