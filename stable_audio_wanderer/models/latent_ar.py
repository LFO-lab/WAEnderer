import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union


# =============================================================================
# Distribution Regularization Losses
# =============================================================================

def kl_divergence_to_standard_normal(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute KL divergence from empirical batch distribution to N(0, I).
    
    For a batch of samples z ~ q(z), this estimates KL(q || p) where p = N(0, I).
    Uses the analytical form: KL = 0.5 * (tr(Σ) + μᵀμ - d - log|Σ|)
    
    Args:
        z: Tensor of shape [B, D] (normalized latent samples)
        eps: Small constant for numerical stability
        
    Returns:
        Scalar KL divergence estimate
    """
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    
    # Empirical mean and covariance
    mu = z.mean(dim=0)  # [D]
    z_centered = z - mu.unsqueeze(0)
    cov = torch.matmul(z_centered.t(), z_centered) / (z.size(0) - 1)  # [D, D]
    
    d = z.size(1)
    
    # tr(Σ)
    trace_cov = torch.trace(cov)
    
    # μᵀμ
    mu_sq = torch.sum(mu ** 2)
    
    # log|Σ| - use eigenvalues for numerical stability
    # For diagonal approximation (faster): log_det = log(diag(cov)).sum()
    diag_cov = torch.diag(cov)
    log_det = torch.log(diag_cov + eps).sum()
    
    # KL = 0.5 * (tr(Σ) + μᵀμ - d - log|Σ|)
    kl = 0.5 * (trace_cov + mu_sq - d - log_det)
    
    return kl


def moment_matching_loss(
    generated: torch.Tensor,
    target_mean: torch.Tensor,
    target_var: torch.Tensor,
    order: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute moment matching loss between generated samples and target statistics.
    
    Args:
        generated: Generated samples [B, T, D] or [B*T, D]
        target_mean: Target mean [D] (typically 0 for normalized latents)
        target_var: Target variance [D] (typically 1 for normalized latents)
        order: Maximum moment order to match (1=mean, 2=mean+var, etc.)
        
    Returns:
        Tuple of (mean_loss, var_loss)
    """
    if generated.dim() == 3:
        generated = generated.reshape(-1, generated.size(-1))
    
    # First moment (mean)
    gen_mean = generated.mean(dim=0)
    mean_loss = F.mse_loss(gen_mean, target_mean)
    
    # Second moment (variance)
    var_loss = generated.new_tensor(0.0)
    if order >= 2:
        gen_var = generated.var(dim=0, unbiased=False)
        var_loss = F.mse_loss(gen_var, target_var)
    
    return mean_loss, var_loss


def spherical_projection(z: torch.Tensor, target_norm: float, soft: bool = True) -> torch.Tensor:
    """
    Project latents toward a target norm (spherical shell).
    
    Args:
        z: Latent tensor [B, D] or [B, T, D]
        target_norm: Target L2 norm
        soft: If True, use soft projection (interpolation); if False, hard projection
        
    Returns:
        Projected tensor with same shape as input
    """
    original_shape = z.shape
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
    
    norms = torch.norm(z, dim=-1, keepdim=True)
    
    if soft:
        # Soft projection: interpolate toward target norm
        scale = target_norm / (norms + 1e-8)
        # Blend: keep direction but adjust magnitude
        projected = z * (0.5 + 0.5 * scale)
    else:
        # Hard projection: exactly target norm
        projected = z * (target_norm / (norms + 1e-8))
    
    return projected.view(original_shape)


def adaptive_clamping(
    z: torch.Tensor,
    running_mean: torch.Tensor,
    running_std: torch.Tensor,
    n_std: float = 3.0,
) -> torch.Tensor:
    """
    Adaptively clamp latents based on running statistics.
    
    Args:
        z: Latent tensor [B, D] or [B, T, D]
        running_mean: Running mean [D]
        running_std: Running std [D]
        n_std: Number of standard deviations for clamping
        
    Returns:
        Clamped tensor
    """
    lower = running_mean - n_std * running_std
    upper = running_mean + n_std * running_std
    return torch.clamp(z, min=lower, max=upper)


# =============================================================================
# Manifold Projection Module
# =============================================================================

class ManifoldProjector(nn.Module):
    """
    Learned projection back onto the encoder manifold.
    
    This module learns to correct drift in generated sequences by projecting
    them back toward the training distribution. Can be trained jointly with
    the AR model or separately.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        
        layers = []
        in_dim = latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        
        self.net = nn.Sequential(*layers)
        
        # Gating mechanism for residual connection
        if residual:
            self.gate = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Sigmoid(),
            )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latents back toward manifold.
        
        Args:
            z: Input latents [B, D] or [B, T, D]
            
        Returns:
            Projected latents with same shape
        """
        original_shape = z.shape
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))
        
        correction = self.net(z)
        
        if self.residual:
            gate = self.gate(z)
            out = z + gate * correction
        else:
            out = correction
        
        return out.view(original_shape)
    
    def to_config(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "residual": self.residual,
        }
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ManifoldProjector":
        return cls(
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            residual=bool(cfg.get("residual", True)),
        )


# =============================================================================
# Free-Running Rollout Utilities
# =============================================================================

class RolloutScheduler:
    """
    Schedules rollout length during training for curriculum learning.
    
    Starts with short rollouts (close to teacher forcing) and gradually
    increases to longer free-running sequences.
    """
    
    def __init__(
        self,
        initial_steps: int = 1,
        final_steps: int = 32,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        schedule: str = "linear",  # "linear", "cosine", "exponential"
    ):
        self.initial_steps = max(1, initial_steps)
        self.final_steps = max(initial_steps, final_steps)
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = max(1, total_epochs)
        self.schedule = schedule
    
    def get_rollout_steps(self, epoch: int) -> int:
        """Get rollout steps for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_steps
        
        # Progress from warmup to end
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule == "linear":
            factor = progress
        elif self.schedule == "cosine":
            factor = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.schedule == "exponential":
            factor = (math.exp(progress) - 1) / (math.e - 1)
        else:
            factor = progress
        
        steps = int(self.initial_steps + factor * (self.final_steps - self.initial_steps))
        return max(1, steps)
    
    def get_teacher_forcing_prob(self, epoch: int) -> float:
        """Get teacher forcing probability (decreases over training)."""
        if epoch < self.warmup_epochs:
            return 1.0
        
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule == "linear":
            return 1.0 - progress
        elif self.schedule == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return 1.0 - progress


def free_running_rollout(
    model: nn.Module,
    seed_context: torch.Tensor,
    num_steps: int,
    teacher_forcing_prob: float = 0.0,
    ground_truth: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    projector: Optional[ManifoldProjector] = None,
    clamp_std: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform free-running rollout with optional teacher forcing and projection.
    
    Args:
        model: AR model
        seed_context: Initial context [B, T, D]
        num_steps: Number of steps to generate
        teacher_forcing_prob: Probability of using ground truth instead of prediction
        ground_truth: Optional ground truth for teacher forcing [B, num_steps, D]
        noise_std: Noise to add to predictions
        projector: Optional manifold projector for drift correction
        clamp_std: Optional std threshold for clamping
        
    Returns:
        Tuple of (predictions [B, num_steps, D], aux_dict)
    """
    if num_steps <= 0:
        return seed_context.new_zeros((seed_context.size(0), 0, seed_context.size(2))), {}
    
    preds = []
    deltas = []
    roll_ctx = seed_context
    
    for t in range(num_steps):
        out = model(roll_ctx, return_aux=False, return_delta=True)
        pred = out["pred"] if isinstance(out, dict) else out
        delta = out.get("delta", pred) if isinstance(out, dict) else pred
        
        # Add noise if specified
        if noise_std > 0.0:
            pred = pred + noise_std * torch.randn_like(pred)
        
        # Apply manifold projection if available
        if projector is not None:
            pred = projector(pred)
        
        # Apply clamping if specified
        if clamp_std is not None:
            pred = torch.clamp(pred, -clamp_std, clamp_std)
        
        preds.append(pred)
        deltas.append(delta)
        
        # Decide next input: teacher forcing or free running
        if teacher_forcing_prob > 0.0 and ground_truth is not None and t < ground_truth.size(1):
            if torch.rand(1).item() < teacher_forcing_prob:
                next_input = ground_truth[:, t:t+1, :]
            else:
                next_input = pred.unsqueeze(1)
        else:
            next_input = pred.unsqueeze(1)
        
        roll_ctx = torch.cat([roll_ctx[:, 1:, :], next_input], dim=1)
    
    predictions = torch.stack(preds, dim=1)  # [B, num_steps, D]
    
    aux = {
        "deltas": torch.stack(deltas, dim=1),
    }
    
    return predictions, aux


# =============================================================================
# AR Models
# =============================================================================

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

    def generate_batch(
        self,
        ctx: torch.Tensor,
        num_frames: int,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Generate multiple frames at once with efficient context management.
        
        Args:
            ctx: Context tensor of shape [B, T, D] (normalized latents)
            num_frames: Number of frames to generate
            noise_std: Optional noise to add to predictions (in normalized space)
            
        Returns:
            Tensor of shape [B, num_frames, D] containing generated frames
        """
        if ctx.dim() != 3:
            raise ValueError(f"Expected ctx to have 3 dims [B,T,D], got {ctx.shape}")
        
        preds = []
        roll_ctx = ctx
        
        for _ in range(num_frames):
            # Forward pass to get prediction
            out = self(roll_ctx, return_aux=False, return_delta=False)
            pred = out["pred"] if isinstance(out, dict) else out
            
            # Add optional noise
            if noise_std > 0.0:
                pred = pred + noise_std * torch.randn_like(pred)
            
            preds.append(pred)
            
            # Efficient context update: shift and append
            roll_ctx = torch.cat([roll_ctx[:, 1:, :], pred.unsqueeze(1)], dim=1)
        
        return torch.stack(preds, dim=1)  # [B, num_frames, D]


class LatentAutoregressiveHybrid(nn.Module):
    """Hybrid CNN + Attention model for latent sequence generation.
    
    Combines dilated causal convolutions for local pattern extraction
    with multi-head self-attention for global context modeling.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        cnn_layers: int = 4,
        kernel_size: int = 3,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        dropout: float = 0.0,
        predict_residual: bool = True,
        residual_center_span: int = 1,
        aux_head: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.cnn_layers = int(cnn_layers)
        self.kernel_size = int(kernel_size)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.dropout_p = float(dropout)
        self.predict_residual = bool(predict_residual)
        self.residual_center_span = max(1, int(residual_center_span))
        self.aux_head = bool(aux_head)

        # Input projection
        self.input_proj = nn.Conv1d(self.latent_dim, self.hidden_dim, kernel_size=1)
        
        # Dilated causal CNN blocks for local patterns
        self.cnn_blocks = nn.ModuleList()
        for i in range(self.cnn_layers):
            dilation = 2 ** i
            conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim, 
                kernel_size=self.kernel_size, 
                dilation=dilation
            )
            self.cnn_blocks.append(conv)
        
        # Layer norm after CNN
        self.cnn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Multi-head self-attention for global context
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(self.hidden_dim)
        
        # Output projections
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.output_proj_aux = nn.Linear(self.hidden_dim, self.latent_dim) if self.aux_head else None
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else None
        
        # Causal attention mask will be registered as buffer
        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        if self._causal_mask is None or self._causal_mask.size(0) < seq_len:
            # Create causal mask: True means position is masked (cannot attend)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self._causal_mask = mask
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self, x: torch.Tensor, return_aux: bool = False, return_delta: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B, T, D]
        if x.dim() != 3:
            raise ValueError(f"Expected x to have 3 dims [B,T,D], got {x.shape}")
        
        B, T, D = x.shape
        
        # CNN path: local feature extraction
        h = x.transpose(1, 2)  # [B, D, T]
        h = self.input_proj(h)  # [B, hidden, T]
        
        for conv in self.cnn_blocks:
            pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            h_conv = conv(F.pad(h, (pad, 0)))  # causal padding
            h = F.gelu(h_conv + h)  # residual connection
            if self.dropout is not None:
                h = self.dropout(h)
        
        h = h.transpose(1, 2)  # [B, T, hidden]
        h = self.cnn_norm(h)
        
        # Attention path: global context
        causal_mask = self._get_causal_mask(T, x.device)
        h_attn, _ = self.attention(h, h, h, attn_mask=causal_mask, need_weights=False)
        h = self.attn_norm(h + h_attn)  # residual + norm
        
        # Feed-forward
        h_ff = self.ff(h)
        h = self.ff_norm(h + h_ff)  # residual + norm
        
        # Output from last time step
        h_last = h[:, -1, :]  # [B, hidden]
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
            "model_type": "hybrid",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "cnn_layers": self.cnn_layers,
            "kernel_size": self.kernel_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout": self.attention_dropout,
            "dropout": self.dropout_p,
            "predict_residual": self.predict_residual,
            "residual_center_span": self.residual_center_span,
            "aux_head": self.aux_head,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LatentAutoregressiveHybrid":
        return cls(
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            cnn_layers=int(cfg.get("cnn_layers", cfg.get("layers", 4))),
            kernel_size=int(cfg.get("kernel_size", 3)),
            num_attention_heads=int(cfg.get("num_attention_heads", 4)),
            attention_dropout=float(cfg.get("attention_dropout", 0.1)),
            dropout=float(cfg.get("dropout", 0.0)),
            predict_residual=bool(cfg.get("predict_residual", False)),
            residual_center_span=int(cfg.get("residual_center_span", 1)),
            aux_head=bool(cfg.get("aux_head", False)),
        )

    def generate_batch(
        self,
        ctx: torch.Tensor,
        num_frames: int,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Generate multiple frames at once with efficient context management."""
        if ctx.dim() != 3:
            raise ValueError(f"Expected ctx to have 3 dims [B,T,D], got {ctx.shape}")
        
        preds = []
        roll_ctx = ctx
        
        for _ in range(num_frames):
            out = self(roll_ctx, return_aux=False, return_delta=False)
            pred = out["pred"] if isinstance(out, dict) else out
            
            if noise_std > 0.0:
                pred = pred + noise_std * torch.randn_like(pred)
            
            preds.append(pred)
            roll_ctx = torch.cat([roll_ctx[:, 1:, :], pred.unsqueeze(1)], dim=1)
        
        return torch.stack(preds, dim=1)


def load_latent_ar_model(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[Union[LatentAutoregressiveCNN, LatentAutoregressiveHybrid], Dict[str, Any]]:
    """Load a latent AR model from checkpoint.
    
    Automatically detects model type (CNN or Hybrid) from config.
    """
    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    cfg = {}
    if isinstance(payload, dict):
        cfg = payload.get("config", {})
    
    # Determine model type from config
    model_type = cfg.get("model_type", "cnn")
    if model_type == "hybrid":
        model = LatentAutoregressiveHybrid.from_config(cfg)
    else:
        model = LatentAutoregressiveCNN.from_config(cfg)
    
    model.load_state_dict(state)
    model.to(device or "cpu")
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"} if isinstance(payload, dict) else {"state_dict": state}
    return model, meta


def load_manifold_projector(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[ManifoldProjector, Dict[str, Any]]:
    """Load a manifold projector from checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    
    projector = ManifoldProjector.from_config(cfg)
    projector.load_state_dict(state)
    projector.to(device or "cpu")
    projector.eval()
    
    meta = {k: v for k, v in payload.items() if k != "state_dict"} if isinstance(payload, dict) else {}
    return projector, meta


def load_ar_model_with_projector(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Tuple[
    Union[LatentAutoregressiveCNN, LatentAutoregressiveHybrid],
    Optional[ManifoldProjector],
    Dict[str, Any]
]:
    """Load AR model and optionally its associated manifold projector.
    
    Returns:
        Tuple of (model, projector or None, metadata)
    """
    model, meta = load_latent_ar_model(checkpoint_path, device)
    
    projector = None
    projector_info = meta.get("projector", {})
    if projector_info.get("trained", False):
        # Projector state is embedded in AR checkpoint
        projector_state = projector_info.get("state_dict")
        projector_config = projector_info.get("config", {})
        if projector_state is not None:
            projector = ManifoldProjector.from_config(projector_config)
            projector.load_state_dict(projector_state)
            projector.to(device or "cpu")
            projector.eval()
    
    return model, projector, meta


class InferenceBuffer:
    """Pre-allocated tensor buffers for efficient real-time inference.
    
    Manages context buffer and prediction buffer to avoid repeated allocations
    during autoregressive generation.
    """
    
    def __init__(
        self,
        context_len: int,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate buffers
        self._ctx_buffer = torch.zeros(
            1, context_len, latent_dim, device=device, dtype=dtype
        )
        self._pred_buffer = torch.zeros(
            1, latent_dim, device=device, dtype=dtype
        )
        self._position = 0
        
    def reset(self, seed_context: Optional[torch.Tensor] = None):
        """Reset buffer, optionally seeding with initial context."""
        self._ctx_buffer.zero_()
        self._position = 0
        if seed_context is not None:
            # seed_context: [T, D] or [1, T, D]
            if seed_context.dim() == 2:
                seed_context = seed_context.unsqueeze(0)
            T = min(seed_context.size(1), self.context_len)
            self._ctx_buffer[:, -T:, :] = seed_context[:, -T:, :].to(
                device=self.device, dtype=self.dtype
            )
            self._position = T
    
    def get_context(self) -> torch.Tensor:
        """Get current context buffer."""
        return self._ctx_buffer
    
    def update(self, new_frame: torch.Tensor):
        """Update context buffer with new frame (efficient shift-and-append)."""
        # new_frame: [1, D] or [D]
        if new_frame.dim() == 1:
            new_frame = new_frame.unsqueeze(0)
        
        # Shift left by 1 and append new frame
        self._ctx_buffer[:, :-1, :] = self._ctx_buffer[:, 1:, :].clone()
        self._ctx_buffer[:, -1, :] = new_frame.to(device=self.device, dtype=self.dtype)
        self._position = min(self._position + 1, self.context_len)
    
    def generate_with_model(
        self,
        model: Union[LatentAutoregressiveCNN, LatentAutoregressiveHybrid],
        num_frames: int,
        noise_std: float = 0.0,
    ) -> List[torch.Tensor]:
        """Generate frames using model with efficient buffer management."""
        preds = []
        with torch.inference_mode():
            for _ in range(num_frames):
                out = model(self._ctx_buffer, return_aux=False, return_delta=False)
                pred = out["pred"] if isinstance(out, dict) else out
                
                if noise_std > 0.0:
                    pred = pred + noise_std * torch.randn_like(pred)
                
                # Store prediction in pre-allocated buffer
                self._pred_buffer.copy_(pred)
                preds.append(self._pred_buffer.clone())
                
                # Update context buffer
                self.update(pred)
        
        return preds
