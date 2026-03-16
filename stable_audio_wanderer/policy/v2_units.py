"""
V2 unit-graph artifact builder for corpus-locked morphology sequencing.

This module builds:
1) Variable-duration morphology units over frame indices.
2) Unit descriptors (entry/exit/mean/delta timbre).
3) A nearest-neighbor transition graph in unit space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class UnitGraphConfig:
    """Configuration for V2 unit segmentation and graph build."""

    min_sec: float = 2.0
    max_sec: float = 10.0
    target_sec: float = 5.0
    latent_hz: float = 21.5
    candidate_k: int = 64
    graph_k: int = 24
    weight_entry: float = 0.70
    weight_delta: float = 0.30
    crossfile_penalty: float = 0.10
    boundary_smoothness_weight: float = 0.35


def _norm01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    vmax = float(np.max(arr))
    if vmax <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / vmax


def _segment_file_units(
    desc_file: np.ndarray,
    min_frames: int,
    max_frames: int,
    target_frames: int,
    boundary_smoothness_weight: float,
) -> List[Tuple[int, int]]:
    """
    Segment one file's descriptor sequence into variable-length units.

    Boundaries are selected in [min_frames, max_frames] windows while preferring
    low descriptor-velocity cuts near target_frames.
    """
    n = int(desc_file.shape[0])
    if n <= 0:
        return []
    if n <= max_frames:
        return [(0, n)]

    vel = np.linalg.norm(np.diff(desc_file, axis=0), axis=1).astype(np.float32)
    units: List[Tuple[int, int]] = []
    cursor = 0

    while cursor < n:
        remaining = n - cursor
        if remaining <= max_frames:
            if remaining < min_frames and units:
                s_prev, _ = units[-1]
                units[-1] = (s_prev, n)
            else:
                units.append((cursor, n))
            break

        lo = cursor + min_frames
        hi = min(cursor + max_frames, n)
        if hi <= lo:
            units.append((cursor, n))
            break

        target = min(cursor + target_frames, hi)
        candidates = np.arange(lo, hi + 1, dtype=np.int32)

        proximity_cost = np.abs(candidates.astype(np.float32) - float(target))
        proximity_cost = _norm01(proximity_cost)

        if vel.size > 0:
            v_idx = np.clip(candidates - 1, 0, vel.size - 1)
            smooth_cost = _norm01(vel[v_idx])
        else:
            smooth_cost = np.zeros_like(proximity_cost, dtype=np.float32)

        total_cost = proximity_cost + float(boundary_smoothness_weight) * smooth_cost
        end = int(candidates[int(np.argmin(total_cost))])

        tail = n - end
        if 0 < tail < min_frames:
            end = n

        end = max(cursor + min_frames, min(end, n))
        units.append((cursor, end))
        cursor = end

    # Defensive clean-up to guarantee valid units.
    cleaned: List[Tuple[int, int]] = []
    for s, e in units:
        s_i = int(max(0, min(s, n - 1)))
        e_i = int(max(s_i + 1, min(e, n)))
        cleaned.append((s_i, e_i))

    if cleaned and (cleaned[-1][1] - cleaned[-1][0]) < min_frames and len(cleaned) > 1:
        s_prev, _ = cleaned[-2]
        _, e_last = cleaned[-1]
        cleaned[-2] = (s_prev, e_last)
        cleaned.pop()

    return cleaned


def build_units_from_frames(
    file_offsets: np.ndarray,
    frame_file_ids: np.ndarray,
    frame_t: np.ndarray,
    desc_weighted: np.ndarray,
    cfg: UnitGraphConfig,
) -> Dict[str, np.ndarray]:
    """Build unit boundaries and descriptor summaries from frame-level arrays."""
    offsets = np.asarray(file_offsets, dtype=np.int64).reshape(-1)
    file_ids = np.asarray(frame_file_ids, dtype=np.int32).reshape(-1)
    frame_t = np.asarray(frame_t, dtype=np.int32).reshape(-1)
    desc = np.asarray(desc_weighted, dtype=np.float32)

    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("file_offsets must be shape [num_files+1].")
    if desc.ndim != 2:
        raise ValueError(f"desc_weighted must be [N, D], got {desc.shape}.")
    n_frames = int(desc.shape[0])
    if file_ids.shape[0] != n_frames or frame_t.shape[0] != n_frames:
        raise ValueError("frame metadata length mismatch with descriptors.")

    min_frames = max(2, int(round(float(cfg.min_sec) * float(cfg.latent_hz))))
    max_frames = max(min_frames + 1, int(round(float(cfg.max_sec) * float(cfg.latent_hz))))
    target_frames = int(
        np.clip(
            round(float(cfg.target_sec) * float(cfg.latent_hz)),
            min_frames,
            max_frames,
        )
    )

    unit_start_idx: List[int] = []
    unit_end_idx: List[int] = []
    unit_file_id: List[int] = []
    unit_start_t: List[int] = []
    unit_end_t: List[int] = []
    unit_len: List[int] = []
    unit_entry_desc: List[np.ndarray] = []
    unit_exit_desc: List[np.ndarray] = []
    unit_mean_desc: List[np.ndarray] = []
    unit_delta_desc: List[np.ndarray] = []
    frame_to_unit = np.full((n_frames,), -1, dtype=np.int32)

    for fid in range(int(offsets.size - 1)):
        g0 = int(offsets[fid])
        g1 = int(offsets[fid + 1])
        if g1 <= g0:
            continue
        desc_file = desc[g0:g1]
        local_units = _segment_file_units(
            desc_file=desc_file,
            min_frames=min_frames,
            max_frames=max_frames,
            target_frames=target_frames,
            boundary_smoothness_weight=float(cfg.boundary_smoothness_weight),
        )

        for s_loc, e_loc in local_units:
            s = int(g0 + s_loc)
            e = int(g0 + e_loc)
            if e <= s:
                continue

            uid = len(unit_start_idx)
            unit_start_idx.append(s)
            unit_end_idx.append(e)
            unit_file_id.append(int(file_ids[s]))
            unit_start_t.append(int(frame_t[s]))
            unit_end_t.append(int(frame_t[e - 1]))
            unit_len.append(int(e - s))

            entry = desc[s]
            exit_ = desc[e - 1]
            mean = desc[s:e].mean(axis=0).astype(np.float32)
            delta = (exit_ - entry).astype(np.float32)

            unit_entry_desc.append(entry.astype(np.float32))
            unit_exit_desc.append(exit_.astype(np.float32))
            unit_mean_desc.append(mean)
            unit_delta_desc.append(delta)
            frame_to_unit[s:e] = int(uid)

    if not unit_start_idx:
        raise RuntimeError("Failed to build units from corpus frames.")
    if np.any(frame_to_unit < 0):
        raise RuntimeError("Some frames were not assigned to units.")

    return {
        "unit_start_idx": np.asarray(unit_start_idx, dtype=np.int32),
        "unit_end_idx": np.asarray(unit_end_idx, dtype=np.int32),
        "unit_file_id": np.asarray(unit_file_id, dtype=np.int32),
        "unit_start_t": np.asarray(unit_start_t, dtype=np.int32),
        "unit_end_t": np.asarray(unit_end_t, dtype=np.int32),
        "unit_len": np.asarray(unit_len, dtype=np.int32),
        "unit_entry_desc": np.asarray(unit_entry_desc, dtype=np.float32),
        "unit_exit_desc": np.asarray(unit_exit_desc, dtype=np.float32),
        "unit_mean_desc": np.asarray(unit_mean_desc, dtype=np.float32),
        "unit_delta_desc": np.asarray(unit_delta_desc, dtype=np.float32),
        "frame_to_unit": frame_to_unit.astype(np.int32),
        "unit_min_frames": np.array(int(min_frames), dtype=np.int32),
        "unit_max_frames": np.array(int(max_frames), dtype=np.int32),
        "unit_target_frames": np.array(int(target_frames), dtype=np.int32),
    }


def build_unit_graph(
    unit_entry_desc: np.ndarray,
    unit_exit_desc: np.ndarray,
    unit_delta_desc: np.ndarray,
    unit_file_id: np.ndarray,
    cfg: UnitGraphConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build fixed-size KNN transition candidates for each unit."""
    entry = np.asarray(unit_entry_desc, dtype=np.float32)
    exit_ = np.asarray(unit_exit_desc, dtype=np.float32)
    delta = np.asarray(unit_delta_desc, dtype=np.float32)
    file_id = np.asarray(unit_file_id, dtype=np.int32).reshape(-1)
    n_units = int(entry.shape[0])
    if n_units <= 0:
        raise ValueError("No units available for graph construction.")

    graph_k = int(max(1, min(int(cfg.graph_k), n_units)))
    cand_k = int(max(graph_k, min(int(cfg.candidate_k), n_units)))
    neighbors = np.full((n_units, graph_k), -1, dtype=np.int32)
    scores = np.full((n_units, graph_k), np.inf, dtype=np.float32)

    if n_units == 1:
        neighbors[0, 0] = 0
        scores[0, 0] = 0.0
        return neighbors, scores

    tree = cKDTree(entry)

    for i in range(n_units):
        dists, inds = tree.query(exit_[i], k=cand_k)
        inds = np.atleast_1d(inds).astype(np.int32).reshape(-1)

        # Keep uniqueness and remove impossible self-loop from first pass.
        uniq = []
        seen = set()
        for j in inds.tolist():
            j_i = int(j)
            if j_i == i:
                continue
            if j_i in seen:
                continue
            seen.add(j_i)
            uniq.append(j_i)
        if not uniq:
            uniq = [int((i + 1) % n_units)]

        cand = np.asarray(uniq, dtype=np.int32)
        c_entry = np.linalg.norm(entry[cand] - exit_[i][None, :], axis=1).astype(np.float32)
        c_delta = np.linalg.norm(delta[cand] - delta[i][None, :], axis=1).astype(np.float32)
        c_file = (file_id[cand] != int(file_id[i])).astype(np.float32)

        c_entry_n = _norm01(c_entry)
        c_delta_n = _norm01(c_delta)
        total = (
            float(cfg.weight_entry) * c_entry_n
            + float(cfg.weight_delta) * c_delta_n
            + float(cfg.crossfile_penalty) * c_file
        )

        order = np.argsort(total)
        take = min(graph_k, order.size)
        chosen = cand[order[:take]]
        chosen_scores = total[order[:take]]

        neighbors[i, :take] = chosen.astype(np.int32)
        scores[i, :take] = chosen_scores.astype(np.float32)

        if take < graph_k:
            fill_idx = int(chosen[0]) if take > 0 else int((i + 1) % n_units)
            fill_score = float(chosen_scores[0]) if take > 0 else float("inf")
            neighbors[i, take:] = fill_idx
            scores[i, take:] = fill_score

    return neighbors, scores


def build_v2_unit_artifact(
    file_offsets: np.ndarray,
    frame_file_ids: np.ndarray,
    frame_t: np.ndarray,
    desc_weighted: np.ndarray,
    cfg: UnitGraphConfig,
    source_corpus_path: str,
) -> Dict[str, np.ndarray]:
    """Build complete V2 unit artifact payload ready for NPZ serialization."""
    unit_data = build_units_from_frames(
        file_offsets=file_offsets,
        frame_file_ids=frame_file_ids,
        frame_t=frame_t,
        desc_weighted=desc_weighted,
        cfg=cfg,
    )
    neighbors, scores = build_unit_graph(
        unit_entry_desc=unit_data["unit_entry_desc"],
        unit_exit_desc=unit_data["unit_exit_desc"],
        unit_delta_desc=unit_data["unit_delta_desc"],
        unit_file_id=unit_data["unit_file_id"],
        cfg=cfg,
    )

    out: Dict[str, np.ndarray] = {
        "version": np.array(1, dtype=np.int32),
        "artifact_type": np.array(["policy_v2_units"], dtype=np.str_),
        "source_corpus_path": np.array([source_corpus_path], dtype=np.str_),
        "latent_hz": np.array(float(cfg.latent_hz), dtype=np.float32),
        "min_sec": np.array(float(cfg.min_sec), dtype=np.float32),
        "max_sec": np.array(float(cfg.max_sec), dtype=np.float32),
        "target_sec": np.array(float(cfg.target_sec), dtype=np.float32),
        "candidate_k": np.array(int(cfg.candidate_k), dtype=np.int32),
        "graph_k": np.array(int(cfg.graph_k), dtype=np.int32),
        "weight_entry": np.array(float(cfg.weight_entry), dtype=np.float32),
        "weight_delta": np.array(float(cfg.weight_delta), dtype=np.float32),
        "crossfile_penalty": np.array(float(cfg.crossfile_penalty), dtype=np.float32),
        "boundary_smoothness_weight": np.array(
            float(cfg.boundary_smoothness_weight),
            dtype=np.float32,
        ),
        "unit_graph_neighbors": neighbors.astype(np.int32),
        "unit_graph_scores": scores.astype(np.float32),
    }
    out.update(unit_data)
    return out
