from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    delta_max: int = 8
    desc_dim: int = 5  # speed, curvature, recurrence, novelty, coverage
    embed_dim: int = 2
    control_dim: int = 6  # width, energy, gravity, memory, coherence, exploration
    hidden_size: int = 128
    input_proj: int = 32
    num_layers: int = 1


class IndexPolicy(nn.Module):
    """
    GRU policy over 1D index trajectories.
    Inputs per step: normalized index, previous velocity, descriptors,
    optional local embedding + performer controls.
    Outputs per step: logits over Δi classes, velocity delta.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.input_proj
        self.index_proj = nn.Linear(1, p)
        self.vel_proj = nn.Linear(1, p)
        self.desc_proj = nn.Linear(cfg.desc_dim, p)
        self.embed_proj = nn.Linear(cfg.embed_dim, p) if cfg.embed_dim > 0 else None
        self.ctrl_proj = nn.Linear(cfg.control_dim, p) if cfg.control_dim > 0 else None

        feat_blocks = 3  # index, velocity, desc
        if self.embed_proj is not None:
            feat_blocks += 1
        if self.ctrl_proj is not None:
            feat_blocks += 1
        input_dim = feat_blocks * p

        self.gru = nn.GRU(input_dim, cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True)
        self.delta_head = nn.Linear(cfg.hidden_size, 2 * cfg.delta_max + 1)
        self.vel_head = nn.Linear(cfg.hidden_size, 1)

    def forward(
        self,
        index_norm: torch.Tensor,
        velocity: torch.Tensor,
        descriptors: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        controls: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        index_norm: [B, T]
        velocity:   [B, T]
        descriptors:[B, T, desc_dim]
        embedding:  [B, T, embed_dim] or None
        controls:   [B, T, control_dim] or None
        hidden:     [num_layers, B, H] or None
        """
        x_parts = [
            self.index_proj(index_norm.unsqueeze(-1)),
            self.vel_proj(velocity.unsqueeze(-1)),
            self.desc_proj(descriptors),
        ]
        if self.embed_proj is not None:
            if embedding is None:
                embedding = torch.zeros(index_norm.size(0), index_norm.size(1), self.cfg.embed_dim, device=index_norm.device)
            x_parts.append(self.embed_proj(embedding))
        if self.ctrl_proj is not None:
            if controls is None:
                controls = torch.zeros(index_norm.size(0), index_norm.size(1), self.cfg.control_dim, device=index_norm.device)
            x_parts.append(self.ctrl_proj(controls))

        x = torch.cat(x_parts, dim=-1)
        out, h_next = self.gru(x, hidden)
        delta_logits = self.delta_head(out)
        vel_delta = self.vel_head(out).squeeze(-1)
        return delta_logits, vel_delta, h_next

    @torch.inference_mode()
    def step(
        self,
        state,
        controls: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One-step inference helper.
        state: dict with keys index_norm, velocity, descriptors, embedding (optional).
        controls: optional control tensor [1,1,C].
        """
        delta_logits, vel_delta, h_next = self.forward(
            index_norm=state["index_norm"],
            velocity=state["velocity"],
            descriptors=state["descriptors"],
            embedding=state.get("embedding"),
            controls=controls,
            hidden=hidden,
        )
        return delta_logits[:, -1], vel_delta[:, -1], h_next

    def delta_class_to_value(self, cls: torch.Tensor) -> torch.Tensor:
        return cls.to(torch.float32) - float(self.cfg.delta_max)
