import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional
from scipy.spatial import cKDTree


@dataclass
class TrajectoryAnnotations:
    """Per-index trajectory descriptors and statistics."""
    velocities: np.ndarray
    speed: np.ndarray
    curvature: np.ndarray
    recurrence: np.ndarray
    novelty: np.ndarray  # NEW: novelty score (higher = less recently visited)
    coverage: np.ndarray  # NEW: local coverage score (fraction of nearby space visited)
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
    """Recurrence proxy: nearest non-adjacent neighbor distance in embedding space (smaller => more recurrent)."""
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
        # Convert distance to bounded [0,1] recurrence weight (smaller dist => closer to 1).
        scores[t] = float(np.exp(-best))
    return scores


def _novelty_scores(seq_idx: np.ndarray, window: int = 64) -> np.ndarray:
    """
    Novelty score based on visit history within a sliding window.
    Higher score means the index hasn't been visited recently (more novel).
    
    Args:
        seq_idx: Array of indices in the sequence
        window: Size of the sliding window to check for revisits
    
    Returns:
        Array of novelty scores in [0, 1] where 1 = completely novel, 0 = just visited
    """
    if seq_idx.size == 0:
        return np.zeros((0,), dtype=np.float32)
    
    scores = np.ones(len(seq_idx), dtype=np.float32)
    visited: Dict[int, int] = {}  # index -> last visit time
    
    for t, idx in enumerate(seq_idx):
        idx_int = int(idx)
        if idx_int in visited:
            # Compute novelty based on time since last visit
            time_since = t - visited[idx_int]
            # Novelty increases with time since last visit, saturates at window
            scores[t] = min(1.0, time_since / window)
        else:
            scores[t] = 1.0  # First visit is completely novel
        
        # Update visit time
        visited[idx_int] = t
        
        # Clean up old entries outside window
        if t >= window:
            old_idx = int(seq_idx[t - window])
            if old_idx in visited and visited[old_idx] == t - window:
                del visited[old_idx]
    
    return scores


def _coverage_scores(seq_idx: np.ndarray, ZZ: np.ndarray, k: int = 16, window: int = 64) -> np.ndarray:
    """
    Local coverage score: how much of the nearby embedding space has been visited recently.
    Lower coverage = more unexplored territory nearby, opportunity for exploration.
    
    Args:
        seq_idx: Array of indices in the sequence
        ZZ: Embedding matrix [N, D]
        k: Number of nearest neighbors to check for coverage
        window: How far back to look for visits
    
    Returns:
        Array of coverage scores in [0, 1] where 1 = fully explored nearby, 0 = unexplored
    """
    if seq_idx.size == 0:
        return np.zeros((0,), dtype=np.float32)
    
    pts = ZZ[seq_idx]
    if pts.shape[0] <= 1:
        return np.zeros((pts.shape[0],), dtype=np.float32)
    
    # Build tree over all points in sequence
    tree = cKDTree(pts)
    k_eff = min(k + 1, pts.shape[0])
    _, neighbor_indices = tree.query(pts, k=k_eff)
    neighbor_indices = np.atleast_2d(neighbor_indices)
    
    scores = np.zeros((pts.shape[0],), dtype=np.float32)
    recent_visits = set()
    
    for t in range(pts.shape[0]):
        # Add current position to recent visits
        recent_visits.add(t)
        
        # Remove positions outside window
        if t >= window:
            recent_visits.discard(t - window)
        
        # Count how many of the k nearest neighbors have been visited recently
        neighbors = neighbor_indices[t]
        visited_count = 0
        total_neighbors = 0
        
        for n in neighbors:
            if n == t:
                continue
            total_neighbors += 1
            if n in recent_visits:
                visited_count += 1
        
        if total_neighbors > 0:
            scores[t] = visited_count / total_neighbors
        else:
            scores[t] = 0.0
    
    return scores


def _normalize(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + eps
    return (arr - mean[None, :]) / std[None, :], mean, std


def compute_annotations(
    meta: np.ndarray,
    ZZ: np.ndarray,
    recur_k: int = 6,
    recur_exclude: int = 2,
    novelty_window: int = 64,
    coverage_k: int = 16,
    coverage_window: int = 64,
) -> TrajectoryAnnotations:
    """
    Compute per-index descriptors derived from index trajectories:
      - velocity (Δi), speed |Δi|, curvature Δv, recurrence
      - novelty: how new/unexpected the position is
      - coverage: how explored the nearby space is
      - heuristic regime labels based on descriptors
    
    Args:
        meta: Segment metadata [N, 3] with (file_id, t_lat, win_lat)
        ZZ: Embedding matrix [N, D]
        recur_k: Number of neighbors for recurrence computation
        recur_exclude: Exclude neighbors within this temporal distance
        novelty_window: Window size for novelty computation
        coverage_k: Number of neighbors for coverage computation
        coverage_window: Window size for coverage computation
    
    Returns:
        TrajectoryAnnotations with all computed descriptors
    """
    N = meta.shape[0]
    velocities = np.zeros((N,), dtype=np.float32)
    speed = np.zeros((N,), dtype=np.float32)
    curvature = np.zeros((N,), dtype=np.float32)
    recurrence = np.zeros((N,), dtype=np.float32)
    novelty = np.zeros((N,), dtype=np.float32)
    coverage = np.zeros((N,), dtype=np.float32)

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
        
        # Recurrence scores
        recur_vals = _recurrence_scores(seq, ZZ, k=recur_k, exclude=recur_exclude)
        recurrence[seq] = recur_vals
        
        # Novelty scores
        novelty_vals = _novelty_scores(seq, window=novelty_window)
        novelty[seq] = novelty_vals
        
        # Coverage scores
        coverage_vals = _coverage_scores(seq, ZZ, k=coverage_k, window=coverage_window)
        coverage[seq] = coverage_vals

    # Include novelty and coverage in descriptors (total 5 now)
    desc = np.stack([speed, curvature, recurrence, novelty, coverage], axis=1).astype(np.float32)
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
        novelty=novelty,
        coverage=coverage,
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
    
    Now includes novelty and coverage descriptors for diversity-aware training.
    """

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        annotations: TrajectoryAnnotations,
        ZZ: np.ndarray,
        seq_len: int,
        delta_max: int,
        control_dim: int = 7,
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
        
        # Controls initialized to default values
        # [width, energy, gravity, memory, coherence, exploration, regime_bias]
        controls = np.zeros((self.seq_len, self.control_dim), dtype=np.float32)
        # Set default neutral values
        controls[:, :4] = 0.5  # width, energy, gravity, memory at 0.5
        # coherence, exploration, regime_bias stay at 0
        
        # Include novelty and coverage as additional outputs for loss weighting
        novelty = self.ann.novelty[curr_idx].astype(np.float32)
        coverage = self.ann.coverage[curr_idx].astype(np.float32)

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
            "novelty": novelty,
            "coverage": coverage,
        }
