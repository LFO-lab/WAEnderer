import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Union


class LatentAutoregressiveCNN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        predict_residual: bool = True,
        residual_center_span: int = 1,
        aux_head: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.predict_residual = bool(predict_residual)
        self.residual_center_span = max(1, int(residual_center_span))
        self.aux_head = bool(aux_head)

        self.input_proj = nn.Conv1d(self.latent_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            dilation = 2 ** i
            conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, dilation=dilation
            )
            self.blocks.append(conv)
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.output_proj_aux = nn.Linear(self.hidden_dim, self.latent_dim) if self.aux_head else None
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else None

    def forward(
        self, x: torch.Tensor, return_aux: bool = False, return_delta: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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
        delta_main = self.output_proj(h_last)
        delta_aux = self.output_proj_aux(h_last) if self.output_proj_aux is not None else None

        ref = None
        if self.predict_residual:
            span = min(self.residual_center_span, x.size(1))
            ref = x[:, -span:, :].mean(dim=1)
        pred_main = delta_main + ref if ref is not None else delta_main
        pred_aux = delta_aux + ref if (ref is not None and delta_aux is not None) else delta_aux

        if not (return_aux or return_delta):
            return pred_main

        out = {"pred": pred_main}
        if return_delta:
            out["delta"] = delta_main
        if self.aux_head and (return_aux or return_delta) and pred_aux is not None:
            out["aux_pred"] = pred_aux
            if return_delta:
                out["aux_delta"] = delta_aux
        return out

    def to_config(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_p,
            "predict_residual": self.predict_residual,
            "residual_center_span": self.residual_center_span,
            "aux_head": self.aux_head,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LatentAutoregressiveCNN":
        return cls(
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            layers=int(cfg.get("layers", 4)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.0)),
            predict_residual=bool(cfg.get("predict_residual", False)),
            residual_center_span=int(cfg.get("residual_center_span", 1)),
            aux_head=bool(cfg.get("aux_head", False)),
        )


def load_latent_ar_model(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[LatentAutoregressiveCNN, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
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
