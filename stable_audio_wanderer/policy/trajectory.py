import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from scipy.spatial import cKDTree


@dataclass
class TrajectoryAnnotations:
    velocities: np.ndarray
    speed: np.ndarray
    curvature: np.ndarray
    recurrence: np.ndarray
    regime: np.ndarray
    desc: np.ndarray
    desc_mean: np.ndarray
    desc_std: np.ndarray
    embed_min: np.ndarray
    embed_range: np.ndarray


def group_meta_by_file(meta: np.ndarray) -> List[np.ndarray]:
    """
    Groups meta rows into index sequences ordered by start time within each file.
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


def _recurrence_scores(seq_idx: np.ndarray, ZZ: np.ndarray, k: int = 6, exclude: int = 2) -> np.ndarray:
    """Recurrence proxy: nearest non-adjacent neighbor distance in embedding space (smaller ⇒ more recurrent)."""
    if seq_idx.size == 0:
        return np.zeros((0,), dtype=np.float32)
    pts = ZZ[seq_idx]
    if pts.shape[0] <= 1:
        return np.zeros((pts.shape[0],), dtype=np.float32)
    tree = cKDTree(pts)
    k_eff = min(max(k, 1) + 1, pts.shape[0])
    dists, inds = tree.query(pts, k=k_eff)
    dists = np.atleast_2d(dists)
    inds = np.atleast_2d(inds)
    scores = np.zeros((pts.shape[0],), dtype=np.float32)
    for t in range(pts.shape[0]):
        best = None
        for dist, j in zip(dists[t], inds[t]):
            if j == t:
                continue
            if abs(int(j) - t) <= exclude:
                continue
            best = float(dist)
            break
        if best is None:
            best = float(np.max(dists[t]))
        # Convert distance to bounded [0,1] recurrence weight (smaller dist ⇒ closer to 1).
        scores[t] = float(np.exp(-best))
    return scores


def _normalize(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + eps
    return (arr - mean[None, :]) / std[None, :], mean, std


def compute_annotations(meta: np.ndarray, ZZ: np.ndarray, recur_k: int = 6, recur_exclude: int = 2) -> TrajectoryAnnotations:
    """
    Compute per-index descriptors derived from index trajectories:
      - velocity (Δi), speed |Δi|, curvature Δv, recurrence
      - heuristic regime labels based on descriptors
    """
    N = meta.shape[0]
    velocities = np.zeros((N,), dtype=np.float32)
    speed = np.zeros((N,), dtype=np.float32)
    curvature = np.zeros((N,), dtype=np.float32)
    recurrence = np.zeros((N,), dtype=np.float32)

    sequences = group_meta_by_file(meta)
    for seq in sequences:
        if seq.size < 2:
            continue
        diffs = np.diff(seq.astype(np.float32))
        velocities[seq[1:]] = diffs
        speed[seq[1:]] = np.abs(diffs)
        if diffs.size > 1:
            curv = np.diff(diffs, prepend=diffs[:1])
            curvature[seq[1:]] = curv
        recur_vals = _recurrence_scores(seq, ZZ, k=recur_k, exclude=recur_exclude)
        recurrence[seq] = recur_vals

    desc = np.stack([speed, curvature, recurrence], axis=1).astype(np.float32)
    desc_norm, desc_mean, desc_std = _normalize(desc)

    # Regime heuristic thresholds.
    speed_abs = np.abs(speed)
    curv_abs = np.abs(curvature)
    # Avoid degenerate quantiles by falling back to fixed small numbers.
    sp_q = np.quantile(speed_abs, [0.3, 0.7]) if speed_abs.size > 0 else np.array([0.0, 0.1], dtype=np.float32)
    curv_q = np.quantile(curv_abs, [0.6]) if curv_abs.size > 0 else np.array([0.05], dtype=np.float32)
    recur_q = np.quantile(recurrence, [0.6]) if recurrence.size > 0 else np.array([0.2], dtype=np.float32)

    regime = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        r_val = recurrence[i]
        s_val = speed_abs[i]
        c_val = curv_abs[i]
        reg = 0  # forward drift
        if r_val >= recur_q[0] or s_val <= sp_q[0]:
            reg = 2  # memory / linger
        elif c_val >= curv_q[0] or velocities[i] < 0.0:
            reg = 1  # turning / reversing
        regime[i] = reg

    embed_min = ZZ.min(axis=0)
    embed_range = np.maximum(ZZ.max(axis=0) - embed_min, 1e-6)

    return TrajectoryAnnotations(
        velocities=velocities,
        speed=speed,
        curvature=curvature,
        recurrence=recurrence,
        regime=regime,
        desc=desc_norm,
        desc_mean=desc_mean.astype(np.float32),
        desc_std=desc_std.astype(np.float32),
        embed_min=embed_min.astype(np.float32),
        embed_range=embed_range.astype(np.float32),
    )


class IndexTrajectoryDataset:
    """
    Samples windows over index trajectories for policy learning.
    Each item returns tensors for inputs (i_t, v_t, descriptors, regime) and targets (Δi class, Δv, next regime).
    """

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        annotations: TrajectoryAnnotations,
        ZZ: np.ndarray,
        seq_len: int,
        delta_max: int,
        control_dim: int = 4,
    ):
        self.sequences = [np.asarray(seq, dtype=np.int32) for seq in sequences if len(seq) > 0]
        if not self.sequences:
            raise ValueError("No index sequences available for policy training.")
        self.ann = annotations
        self.ZZ = np.asarray(ZZ, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.delta_max = int(delta_max)
        self.control_dim = int(control_dim)
        self.N = self.ZZ.shape[0]

        # Normalize embedding for stable training.
        self.ZZ_norm = (self.ZZ - self.ann.embed_min[None, :]) / self.ann.embed_range[None, :]
        self.ZZ_norm = np.clip(self.ZZ_norm, 0.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def _sample_window(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure we have at least seq_len+1 points; pad with the last index if needed.
        if seq.size < self.seq_len + 1:
            pad_needed = self.seq_len + 1 - seq.size
            seq = np.concatenate([seq, np.repeat(seq[-1:], pad_needed)]).astype(np.int32)
        if seq.size == self.seq_len + 1:
            start = 0
        else:
            start = np.random.randint(0, seq.size - self.seq_len - 1)
        curr = seq[start : start + self.seq_len]
        nxt = seq[start + 1 : start + self.seq_len + 1]
        return curr, nxt

    def __getitem__(self, idx: int):
        seq = self.sequences[idx % len(self.sequences)]
        curr_idx, next_idx = self._sample_window(seq)

        index_norm = curr_idx.astype(np.float32) / max(float(self.N - 1), 1.0)
        velocity = self.ann.velocities[curr_idx].astype(np.float32)
        descriptors = self.ann.desc[curr_idx].astype(np.float32)
        regime = self.ann.regime[curr_idx].astype(np.int64)
        regime_next = self.ann.regime[next_idx].astype(np.int64)

        delta = next_idx.astype(np.float32) - curr_idx.astype(np.float32)
        delta_round = np.round(delta).astype(np.int32)
        delta_clamped = np.clip(delta_round, -self.delta_max, self.delta_max)
        delta_class = (delta_clamped + self.delta_max).astype(np.int64)

        dv = self.ann.velocities[next_idx].astype(np.float32) - velocity

        embed = self.ZZ_norm[curr_idx]
        controls = np.zeros((self.seq_len, self.control_dim), dtype=np.float32)

        return {
            "index_norm": index_norm,
            "velocity": velocity,
            "descriptors": descriptors,
            "regime": regime,
            "regime_next": regime_next,
            "delta_class": delta_class,
            "delta_raw": delta.astype(np.float32),
            "dv": dv,
            "embedding": embed,
            "controls": controls,
        }
