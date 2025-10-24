import os, glob, numpy as np

def find_latest(pattern_dir: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(pattern_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No match for pattern '{pattern}' in {pattern_dir}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def save_latents_bundle(path: str, latents_dict: dict, paths_arr):
    latents_dict["paths"] = paths_arr
    np.savez_compressed(path, **latents_dict)

def save_corpus(path: str, **arrays):
    np.savez_compressed(path, **arrays)

def load_corpus(corpus_npz_path: str):
    return np.load(corpus_npz_path, allow_pickle=True)

def load_latents_bundle(bundle_path: str):
    return np.load(bundle_path, allow_pickle=True)
