#!/usr/bin/env python3
"""
Preprocessing library for macOS app integration.
Wraps the preprocessing pipeline with JSON progress output for Swift UI.
"""
import os
import sys
import json
import argparse
import datetime
import glob
import traceback
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

from stable_audio_wanderer.config import SR, LATENT_HZ, DEVICE
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
# MFCC import removed - ZZ is now derived from GG (VAE latents) via PCA
from stable_audio_wanderer.dr.pca import fit_transform
from stable_audio_wanderer.io.corpus_io import save_corpus, save_grain_manifest
from stable_audio_wanderer.io.audio_io import save_wav
from stable_audio_wanderer.policy.latent_geometry import (
    compute_latent_geometry,
    save_geometry_to_dict,
)


class JSONProgress:
    """Emit JSON progress updates to stdout for Swift app consumption."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def emit(self, stage: str, progress: float, message: str = "", **extra):
        if not self.enabled:
            print(f"[{stage}] {progress*100:.0f}% - {message}")
            return

        data = {
            "stage": stage,
            "progress": round(progress, 4),
            "message": message,
            **extra
        }
        print(json.dumps(data), flush=True)

    def error(self, message: str, stage: str = "error"):
        self.emit(stage, 0.0, message, error=True)


def find_audio_files(input_path: str) -> List[str]:
    """Find all audio files in input path (file or directory)."""
    extensions = [".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"]

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(files))

    # Comma-separated paths
    if "," in input_path:
        return [p.strip() for p in input_path.split(",") if os.path.isfile(p.strip())]

    return []


def render_grains(
    wav_dict: Dict[int, np.ndarray],
    meta: np.ndarray,
    grain_sec: float,
    out_dir: str,
    sr: int,
    progress: JSONProgress,
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
    for idx, fid in enumerate(file_ids):
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

            # No envelope applied here - GrainPlayer applies TrigEnv at playback

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

        progress.emit(
            "grain_rendering",
            (idx + 1) / len(file_ids),
            f"Rendered grains for file {idx + 1}/{len(file_ids)}"
        )

    return (
        np.array(offsets_list, dtype=np.int64),
        np.array(lengths_list, dtype=np.int64),
        np.array(file_ids_list, dtype=np.int32),
        np.array(segment_ids_list, dtype=np.int32),
        grain_paths,
    )


def preprocess(
    input_path: str,
    output_path: str,
    segment_dur: float = 0.2,
    hop_dur: float = 0.05,
    pca_dim: int = 2,
    train_projector: bool = True,
    projector_epochs: int = 100,
    grain_sec: Optional[float] = None,
    json_progress: bool = True,
    latent_nav: bool = False,
    latent_nav_k: int = 32,
) -> dict:
    """
    Main preprocessing function for macOS app.

    Args:
        input_path: Path to audio file, folder, or comma-separated paths
        output_path: Output .sawmodel bundle path
        segment_dur: Segment duration in seconds
        hop_dur: Hop duration in seconds
        pca_dim: PCA output dimension (usually 2 for visualization)
        train_projector: Whether to train contrastive projector
        projector_epochs: Number of training epochs
        grain_sec: Grain duration (default: 1/LATENT_HZ)
        json_progress: Whether to emit JSON progress
        latent_nav: Whether to enable 64D latent navigation mode
        latent_nav_k: Number of nearest neighbors for latent navigation (default 32)

    Returns:
        dict with result info (segments, paths, etc.)
    """
    progress = JSONProgress(enabled=json_progress)

    if grain_sec is None:
        grain_sec = 1.0 / LATENT_HZ

    # Find audio files
    progress.emit("initializing", 0.0, "Finding audio files...")
    audio_files = find_audio_files(input_path)
    if not audio_files:
        progress.error(f"No audio files found in {input_path}")
        raise FileNotFoundError(f"No audio files found in {input_path}")

    progress.emit("initializing", 0.1, f"Found {len(audio_files)} audio files")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load VAE
    progress.emit("initializing", 0.2, "Loading VAE model...")
    ae = load_vae("stabilityai/stable-audio-open-1.0")
    progress.emit("initializing", 0.5, "VAE loaded")

    seg_len_samp = int(round(segment_dur * SR))
    win_lat = max(1, int(round(segment_dur * LATENT_HZ)))
    hop_lat = max(1, int(round(hop_dur * LATENT_HZ)))

    meta_list = []
    latent_sequences: List[np.ndarray] = []
    latents_dict = {}
    wav_dict: Dict[int, np.ndarray] = {}

    # VAE encoding
    progress.emit("vae_encoding", 0.0, "Starting VAE encoding...")
    for fid, path in enumerate(audio_files):
        progress.emit(
            "vae_encoding",
            fid / len(audio_files),
            f"Encoding {os.path.basename(path)} ({fid + 1}/{len(audio_files)})"
        )

        wav = load_wav(path)
        wav_dict[fid] = wav
        z_full = encode_full(ae, wav).astype(np.float32)
        z_full = np.ascontiguousarray(z_full)
        latents_dict[f"z_{fid}"] = z_full
        latent_sequences.append(z_full)

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            meta_list.append((fid, t_lat, win_lat))

    progress.emit("vae_encoding", 1.0, f"Encoded {len(audio_files)} files")

    if not latent_sequences:
        progress.error("No latent sequences extracted")
        raise RuntimeError("No latent sequences were extracted")

    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(audio_files)

    # Compute normalization stats from all VAE latents
    progress.emit("pca_fitting", 0.0, "Computing segment latents (GG)...")
    latent_stack = np.concatenate(latent_sequences, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    # Compute GG (segment latents) from VAE embeddings - mean pooling per segment
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
    GG = np.stack(seg_latents, axis=0).astype(np.float32)

    # Normalize GG using VAE latent statistics
    GG_norm = (GG - Z_mean[None, :]) / Z_std[None, :]
    GG = np.clip(GG_norm, -5.0, 5.0).astype(np.float32)

    # Derive ZZ from GG via PCA (64D → 2D)
    progress.emit("pca_fitting", 0.5, "Fitting PCA on GG...")
    ZZ, dr_meta = fit_transform(GG, pca_dim)
    progress.emit("pca_fitting", 1.0, f"PCA complete: GG(64D) -> ZZ({pca_dim}D)")

    # Compute latent geometry for 64D navigation if enabled
    geometry_arrays = {}
    if latent_nav:
        progress.emit("latent_geometry", 0.0, "Computing latent geometry for 64D navigation...")

        # L2 normalize GG for cosine similarity kNN
        GG_norms = np.linalg.norm(GG, axis=1, keepdims=True)
        GG_norms = np.maximum(GG_norms, 1e-6)
        GG_l2 = (GG / GG_norms).astype(np.float32)

        progress.emit("latent_geometry", 0.3, "Building FAISS index...")
        geometry = compute_latent_geometry(GG_l2, meta, k=latent_nav_k)
        geometry_arrays = save_geometry_to_dict(geometry)

        progress.emit("latent_geometry", 1.0, f"Latent geometry computed: K={latent_nav_k} neighbors")

    # Projector training (optional)
    if train_projector:
        progress.emit("projector_training", 0.0, "Starting projector training...")
        # Try to import projector training - may not be available in bundled app
        try:
            from bin.preprocess import train_contrastive_projector, project_latents, LatentProjector
        except ImportError:
            progress.emit("projector_training", 1.0, "Projector training skipped (not available in app bundle)")
            train_projector = False

    if train_projector:
        latent_stack_norm = (latent_stack - Z_mean[None, :]) / Z_std[None, :]
        latent_stack_norm = np.ascontiguousarray(latent_stack_norm.astype(np.float32))

        device = torch.device(DEVICE)
        projector, g_loss_history = train_contrastive_projector(
            latent_stack_norm,
            proj_dim=64,
            tau=0.1,
            epochs=projector_epochs,
            batch_size=min(512, latent_stack_norm.shape[0]),
            lr=1e-3,
            device=device,
            noise_std=0.05,
            drop_prob=0.1,
            mix_alpha=0.0,
            uniformize=True,
            knn_k=10,
            lambda_unif=1.0,
            lambda_cov=0.5,
            cov_var_weight=0.5,
            cov_decorr_weight=1.0,
            cov_gamma=1.0 / 64,
            mean_weight=0.0,
        )
        progress.emit("projector_training", 1.0, "Projector training complete")

        # Project segment latents
        g_embeddings = project_latents(projector, GG, device, batch_size=1024)
        g_min = g_embeddings.min(axis=0).astype(np.float32)
        g_max = g_embeddings.max(axis=0).astype(np.float32)
        g_range = np.maximum(g_max - g_min, 1e-6)
        GG = ((g_embeddings - g_min[None, :]) / g_range[None, :]).astype(np.float32)
        GG = np.clip(GG, 0.0, 1.0)

        # Save projector
        projector_path = os.path.join(output_path, "projector.pt")
        torch.save(projector.state_dict(), projector_path)

    # Grain rendering
    progress.emit("grain_rendering", 0.0, "Rendering grains...")
    offsets, lengths, file_ids, segment_ids, grain_paths = render_grains(
        wav_dict=wav_dict,
        meta=meta,
        grain_sec=grain_sec,
        out_dir=output_path,
        sr=SR,
        progress=progress,
    )
    progress.emit("grain_rendering", 1.0, f"Rendered {len(offsets)} grains")

    # Clear wav_dict
    wav_dict.clear()

    # Save manifest
    progress.emit("saving", 0.0, "Saving corpus...")
    corpus_path = os.path.join(output_path, "corpus.npz")
    dr_arrays = {k: (v if v is None else np.asarray(v, dtype=np.float32)) for k, v in dr_meta.items()}
    save_corpus(
        corpus_path,
        ZZ=ZZ,
        GG=GG,
        pca_dim=np.array(int(pca_dim), dtype=np.int32),
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        file_indices=meta[:, 0],
        start_secs=(meta[:, 1] / LATENT_HZ).astype(np.float32),
        duration_secs=np.full(len(meta), segment_dur, dtype=np.float32),
        **dr_arrays,
        **geometry_arrays,
    )

    # Save grain manifest
    progress.emit("saving", 0.5, "Saving grain manifest...")
    grain_manifest_path = os.path.join(output_path, "grains", "manifest.npz")
    save_grain_manifest(
        path=grain_manifest_path,
        offsets=offsets,
        lengths=lengths,
        file_ids=file_ids,
        segment_ids=segment_ids,
        grain_paths=grain_paths,
        grain_sec=grain_sec,
        grain_hop_sec=hop_dur,
        sr=SR,
    )

    # Save manifest.json
    progress.emit("saving", 0.8, "Saving manifest...")
    manifest = {
        "version": "1.0",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_files": [os.path.basename(p) for p in audio_files],
        "segments": len(meta),
        "grain_sec": grain_sec,
        "sample_rate": SR,
        "nav_dim": pca_dim,
        "policy_trained": False,
        "training_epochs": None,
        "navigation_mode": "latent" if latent_nav else "index",
        "latent_nav_k": latent_nav_k if latent_nav else None,
    }
    manifest_path = os.path.join(output_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    progress.emit("complete", 1.0, "Preprocessing complete", segments=len(meta))

    return {
        "segments": len(meta),
        "files": len(audio_files),
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for Stable Audio Wanderer")
    parser.add_argument("--input", required=True, help="Input audio file, folder, or comma-separated paths")
    parser.add_argument("--output", required=True, help="Output .sawmodel bundle path")
    parser.add_argument("--segment_dur", type=float, default=0.2, help="Segment duration in seconds")
    parser.add_argument("--hop_dur", type=float, default=0.05, help="Hop duration in seconds")
    parser.add_argument("--pca_dim", type=int, default=2, help="PCA dimension")
    parser.add_argument("--train_projector", action="store_true", help="Train contrastive projector")
    parser.add_argument("--projector_epochs", type=int, default=100, help="Projector training epochs")
    parser.add_argument("--grain_sec", type=float, default=None, help="Grain duration in seconds")
    parser.add_argument("--json_progress", action="store_true", help="Emit JSON progress to stdout")
    parser.add_argument("--latent_nav", action="store_true", help="Enable 64D latent navigation mode")
    parser.add_argument("--latent_nav_k", type=int, default=32, help="Number of nearest neighbors for latent nav")

    args = parser.parse_args()

    try:
        result = preprocess(
            input_path=args.input,
            output_path=args.output,
            segment_dur=args.segment_dur,
            hop_dur=args.hop_dur,
            pca_dim=args.pca_dim,
            train_projector=args.train_projector,
            projector_epochs=args.projector_epochs,
            grain_sec=args.grain_sec,
            json_progress=args.json_progress,
            latent_nav=args.latent_nav,
            latent_nav_k=args.latent_nav_k,
        )
        print(json.dumps({"status": "success", **result}), flush=True)
    except Exception as e:
        progress = JSONProgress(enabled=args.json_progress)
        progress.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
