"""Corpus and grain manifest I/O utilities."""
import os
import glob
from typing import Dict, List, Optional
import numpy as np


def find_latest(pattern_dir: str, pattern: str) -> str:
    """Find the most recently modified file matching pattern in directory."""
    matches = glob.glob(os.path.join(pattern_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No match for pattern '{pattern}' in {pattern_dir}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def save_latents_bundle(path: str, latents_dict: dict, paths_arr):
    """Save latent embeddings bundle (preprocess only, deprecated for runtime)."""
    latents_dict["paths"] = paths_arr
    np.savez_compressed(path, **latents_dict)


def save_corpus(path: str, **arrays):
    """Save corpus data to NPZ file."""
    np.savez_compressed(path, **arrays)


def load_corpus(corpus_npz_path: str):
    """Load corpus data from NPZ file."""
    return np.load(corpus_npz_path, allow_pickle=True)


def load_latents_bundle(bundle_path: str):
    """Load latent embeddings bundle (preprocess only, deprecated for runtime)."""
    return np.load(bundle_path, allow_pickle=True)


# --- Grain Manifest I/O ---

def save_grain_manifest(
    path: str,
    offsets: np.ndarray,
    lengths: np.ndarray,
    file_ids: np.ndarray,
    segment_ids: np.ndarray,
    grain_paths: List[str],
    grain_sec: float,
    grain_hop_sec: float,
    sr: int,
) -> None:
    """
    Save grain manifest to NPZ file.
    
    Args:
        path: Output manifest path
        offsets: Sample offset into each file's concatenated grain buffer [N_grains]
        lengths: Length in samples for each grain [N_grains]
        file_ids: Source file ID for each grain [N_grains]
        segment_ids: Maps each grain to corpus segment index [N_grains]
        grain_paths: Path to each file's concatenated grain WAV [N_files]
        grain_sec: Grain duration in seconds
        grain_hop_sec: Hop between grain starts in seconds
        sr: Sample rate
    """
    np.savez_compressed(
        path,
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        file_ids=np.asarray(file_ids, dtype=np.int32),
        segment_ids=np.asarray(segment_ids, dtype=np.int32),
        grain_paths=np.asarray(grain_paths, dtype=object),
        grain_sec=np.array(grain_sec, dtype=np.float32),
        grain_hop_sec=np.array(grain_hop_sec, dtype=np.float32),
        sr=np.array(sr, dtype=np.int32),
    )


def load_grain_manifest(manifest_path: str) -> Dict:
    """
    Load grain manifest from NPZ file.
    
    Returns:
        Dictionary with keys: offsets, lengths, file_ids, segment_ids, 
        grain_paths, grain_sec, grain_hop_sec, sr
    """
    data = np.load(manifest_path, allow_pickle=True)
    return {
        "offsets": data["offsets"],
        "lengths": data["lengths"],
        "file_ids": data["file_ids"],
        "segment_ids": data["segment_ids"],
        "grain_paths": list(data["grain_paths"]),
        "grain_sec": float(data["grain_sec"]),
        "grain_hop_sec": float(data["grain_hop_sec"]),
        "sr": int(data["sr"]),
    }


def find_grain_manifest(corpus_dir: str) -> Optional[str]:
    """Find grain manifest in corpus directory, returns None if not found."""
    try:
        return find_latest(os.path.join(corpus_dir, "grains"), "manifest.npz")
    except FileNotFoundError:
        # Try looking in the corpus directory directly
        try:
            return find_latest(corpus_dir, "*_grains_manifest.npz")
        except FileNotFoundError:
            return None
