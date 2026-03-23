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


def save_corpus(path: str, **arrays):
    """Save corpus data to NPZ file."""
    np.savez_compressed(path, **arrays)


def load_corpus(corpus_npz_path: str):
    """Load corpus data from NPZ file."""
    return np.load(corpus_npz_path, allow_pickle=True)


def resolve_corpus_path(corpus_dir: str) -> str:
    """Resolve the latest corpus NPZ path in *corpus_dir*."""
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        fallback = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"No corpus file found in {corpus_dir}")
        return fallback


def read_scalar(data: dict, key: str, default):
    """Read a scalar value from *data*, handling numpy arrays."""
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    return value


