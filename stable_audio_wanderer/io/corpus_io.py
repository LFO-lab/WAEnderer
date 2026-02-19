"""Corpus I/O utilities."""
import os
import glob
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


def get_file_latents(corpus_data: dict, file_id: int) -> np.ndarray:
    """
    Return latent trajectory [T, 64] for a file from canonical frame corpus fields.
    """
    z_concat = corpus_data["Z_concat"]
    offsets = corpus_data["file_offsets"]
    file_id = int(file_id)
    if file_id < 0 or file_id >= offsets.shape[0] - 1:
        raise IndexError(f"file_id out of range: {file_id}")
    start = int(offsets[file_id])
    end = int(offsets[file_id + 1])
    return z_concat[start:end]


def iter_file_latents(corpus_data: dict):
    """
    Yield (file_id, z_file) for each file in corpus.
    """
    offsets = corpus_data["file_offsets"]
    for file_id in range(int(offsets.shape[0] - 1)):
        yield file_id, get_file_latents(corpus_data, file_id)


def load_latents_bundle(bundle_path: str):
    """Load latent embeddings bundle (preprocess only, deprecated for runtime)."""
    return np.load(bundle_path, allow_pickle=True)
