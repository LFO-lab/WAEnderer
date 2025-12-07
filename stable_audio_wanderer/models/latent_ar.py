import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional


class LatentAutoregressiveCNN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)

        self.input_proj = nn.Conv1d(self.latent_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            dilation = 2 ** i
            conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, dilation=dilation
            )
            self.blocks.append(conv)
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if x.dim() != 3:
            raise ValueError(f"Expected x to have 3 dims [B,T,D], got {x.shape}")
        h = x.transpose(1, 2)  # [B, D, T]
        h = self.input_proj(h)
        for conv in self.blocks:
            pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            h_conv = conv(F.pad(h, (pad, 0)))  # causal padding on the left
            h = F.gelu(h_conv + h)
            if self.dropout is not None:
                h = self.dropout(h)
        h_last = h[:, :, -1]  # [B, hidden]
        return self.output_proj(h_last)

    def to_config(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_p,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LatentAutoregressiveCNN":
        return cls(
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            layers=int(cfg.get("layers", 4)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.0)),
        )


def load_latent_ar_model(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[LatentAutoregressiveCNN, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device or "cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    cfg = {}
    if isinstance(payload, dict):
        cfg = payload.get("config", {})
    model = LatentAutoregressiveCNN.from_config(cfg)
    model.load_state_dict(state)
    model.to(device or "cpu")
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"} if isinstance(payload, dict) else {"state_dict": state}
    return model, meta
