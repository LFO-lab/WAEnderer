"""
Lightweight unit-transition scorer for V2 policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class V2TransitionModelConfig:
    """Configuration for the V2 unit transition scorer."""

    hidden_dim: int = 192
    layers: int = 3
    dropout: float = 0.10


class V2UnitTransitionScorer(nn.Module):
    """MLP scorer that outputs one logit per candidate transition."""

    def __init__(self, input_dim: int, cfg: V2TransitionModelConfig):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be >0, got {input_dim}")
        h = int(max(32, cfg.hidden_dim))
        n_layers = int(max(1, cfg.layers))
        p_drop = float(np.clip(cfg.dropout, 0.0, 0.8))

        blocks = []
        prev = int(input_dim)
        for _ in range(n_layers):
            blocks.append(nn.Linear(prev, h))
            blocks.append(nn.SiLU())
            if p_drop > 0:
                blocks.append(nn.Dropout(p_drop))
            prev = h
        blocks.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_features: [B, K, F] or [K, F]
        Returns:
            logits: [B, K] or [K]
        """
        if pair_features.dim() == 2:
            x = pair_features.unsqueeze(0)
            squeeze = True
        elif pair_features.dim() == 3:
            x = pair_features
            squeeze = False
        else:
            raise ValueError(f"Expected rank-2/3 features, got {pair_features.shape}")

        b, k, f = x.shape
        y = self.net(x.reshape(b * k, f)).reshape(b, k)
        return y.squeeze(0) if squeeze else y


def build_v2_pair_features(
    unit_entry_desc: np.ndarray,
    unit_exit_desc: np.ndarray,
    unit_delta_desc: np.ndarray,
    unit_len: np.ndarray,
    unit_file_id: np.ndarray,
    current_unit: int,
    candidate_units: np.ndarray,
) -> np.ndarray:
    """
    Build pairwise features for (current_unit -> candidate_units).
    """
    entry = np.asarray(unit_entry_desc, dtype=np.float32)
    exit_ = np.asarray(unit_exit_desc, dtype=np.float32)
    delta = np.asarray(unit_delta_desc, dtype=np.float32)
    unit_len = np.asarray(unit_len, dtype=np.float32).reshape(-1)
    file_id = np.asarray(unit_file_id, dtype=np.int32).reshape(-1)
    cand = np.asarray(candidate_units, dtype=np.int32).reshape(-1)

    cur = int(current_unit)
    if cand.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    n_units = entry.shape[0]
    if not (0 <= cur < n_units):
        raise IndexError(f"current_unit out of range: {cur}")
    if np.any(cand < 0) or np.any(cand >= n_units):
        raise IndexError("candidate_units contains out-of-range indices")

    len_max = float(max(1.0, float(np.max(unit_len))))
    cur_len = float(unit_len[cur]) / len_max
    cand_len = unit_len[cand] / len_max

    cur_exit = exit_[cur][None, :].repeat(cand.shape[0], axis=0)
    cur_delta = delta[cur][None, :].repeat(cand.shape[0], axis=0)
    cand_entry = entry[cand]
    cand_delta = delta[cand]

    dist_entry = np.linalg.norm(cand_entry - cur_exit, axis=1, keepdims=True).astype(np.float32)
    dist_delta = np.linalg.norm(cand_delta - cur_delta, axis=1, keepdims=True).astype(np.float32)
    same_file = (file_id[cand] == int(file_id[cur])).astype(np.float32).reshape(-1, 1)
    cand_len_col = cand_len.reshape(-1, 1).astype(np.float32)
    len_diff = np.abs(cand_len - cur_len).reshape(-1, 1).astype(np.float32)

    features = np.concatenate(
        [
            cur_exit.astype(np.float32),
            cur_delta.astype(np.float32),
            cand_entry.astype(np.float32),
            cand_delta.astype(np.float32),
            dist_entry,
            dist_delta,
            same_file,
            cand_len_col,
            len_diff,
        ],
        axis=1,
    ).astype(np.float32)
    return features


def infer_v2_input_dim(unit_desc_dim: int) -> int:
    """
    Feature dimensionality for build_v2_pair_features.
    """
    d = int(unit_desc_dim)
    if d <= 0:
        raise ValueError(f"unit_desc_dim must be >0, got {d}")
    return int(4 * d + 5)


def load_v2_transition_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[V2UnitTransitionScorer, Dict]:
    """
    Load trained V2 transition scorer from checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = V2TransitionModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else V2TransitionModelConfig()
    input_dim = int(ckpt["input_dim"])
    model = V2UnitTransitionScorer(input_dim=input_dim, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
