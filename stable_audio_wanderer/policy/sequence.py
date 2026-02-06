import numpy as np
from typing import Dict, List, Tuple


def group_meta_by_file(meta: np.ndarray) -> List[np.ndarray]:
    """
    Group meta rows into index sequences ordered by start time within each file.
    Returns a list of arrays containing global indices for each file.
    """
    per_file: Dict[int, List[Tuple[int, int]]] = {}
    for idx, row in enumerate(meta):
        fid, t_lat = int(row[0]), int(row[1])
        per_file.setdefault(fid, []).append((t_lat, idx))
    sequences: List[np.ndarray] = []
    for _, items in sorted(per_file.items(), key=lambda kv: kv[0]):
        items.sort(key=lambda x: x[0])
        seq = np.array([idx for _, idx in items], dtype=np.int32)
        if seq.size > 0:
            sequences.append(seq)
    return sequences


def compute_velocity_magnitudes(GG: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """
    Compute per-segment latent velocity magnitude using adjacent segments per file.

    Velocity is estimated from GG differences between neighboring segments
    (ordered by t_lat) and averaged for interior points to reduce jitter.
    """
    GG = np.asarray(GG, dtype=np.float32)
    N = GG.shape[0]
    vel_mag = np.zeros(N, dtype=np.float32)
    sequences = group_meta_by_file(meta)
    for seq in sequences:
        if seq.size == 0:
            continue
        if seq.size == 1:
            vel_mag[seq[0]] = 0.0
            continue
        for j, idx in enumerate(seq):
            if j == 0:
                diff = GG[seq[1]] - GG[seq[0]]
                vel_mag[idx] = float(np.linalg.norm(diff))
            elif j == seq.size - 1:
                diff = GG[seq[-1]] - GG[seq[-2]]
                vel_mag[idx] = float(np.linalg.norm(diff))
            else:
                diff_prev = GG[seq[j]] - GG[seq[j - 1]]
                diff_next = GG[seq[j + 1]] - GG[seq[j]]
                vel_mag[idx] = float(0.5 * (np.linalg.norm(diff_prev) + np.linalg.norm(diff_next)))
    return vel_mag
