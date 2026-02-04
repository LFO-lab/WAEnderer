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
