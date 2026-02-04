"""
Latent space navigation policy model.
GRU policy over 64D latent trajectories with Gaussian mixture outputs.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class LatentPolicyConfig:
    """Configuration for LatentPolicy model."""
    latent_dim: int = 64           # Dimensionality of latent space (GG)
    hidden_size: int = 256         # GRU hidden size
    num_layers: int = 2            # Number of GRU layers
    num_mixture_components: int = 4  # Number of Gaussian mixture components
    control_dim: int = 6           # Controls: width, energy, gravity, memory, coherence, exploration
    local_feature_dim: int = 16    # Local geometry features (sigma, density, etc.)
    dropout: float = 0.1           # Dropout rate
    # Window size prediction (adaptive decoding)
    num_window_classes: int = 5    # Number of discrete window size classes
    window_class_sizes: tuple = (1, 2, 4, 8, 16)  # Latent frames per class


class LatentPolicy(nn.Module):
    """
    GRU policy over 64D latent trajectories.

    Inputs:
        - z: [B, T, 64] current position in latent space
        - v: [B, T, 64] current velocity
        - controls: [B, T, 6] performer control parameters
        - local_features: [B, T, 16] local geometry features

    Outputs:
        - delta_mean: [B, T, M, 64] Gaussian mixture means (M components)
        - delta_log_std: [B, T, M, 64] Gaussian mixture log-stds
        - delta_weights: [B, T, M] mixture weights (log-softmax)
        - vel_delta: [B, T, 64] velocity update
        - hidden: GRU hidden state
    """

    def __init__(self, cfg: LatentPolicyConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        H4 = H // 4

        # Input projections (each to H/4)
        self.z_proj = nn.Linear(cfg.latent_dim, H4)
        self.v_proj = nn.Linear(cfg.latent_dim, H4)
        self.ctrl_proj = nn.Linear(cfg.control_dim, H4)
        self.local_proj = nn.Linear(cfg.local_feature_dim, H4)

        # Input layer norm for stability
        self.input_norm = nn.LayerNorm(H)

        # Core GRU
        self.gru = nn.GRU(
            input_size=H,
            hidden_size=H,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Output layer norm
        self.output_norm = nn.LayerNorm(H)

        # Output heads for Gaussian mixture
        M = cfg.num_mixture_components
        D = cfg.latent_dim

        # Mixture means: H -> M * D
        self.delta_mean_head = nn.Linear(H, M * D)

        # Mixture log-stds: H -> M * D (will be clamped)
        self.delta_log_std_head = nn.Linear(H, M * D)

        # Mixture weights: H -> M (will be log-softmax)
        self.delta_weights_head = nn.Linear(H, M)

        # Velocity update: H -> D
        self.vel_delta_head = nn.Linear(H, D)

        # Window size prediction: H -> num_window_classes
        self.window_head = nn.Linear(H, cfg.num_window_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    # GRU uses orthogonal initialization
                    nn.init.orthogonal_(param, gain=1.0)
                elif param.dim() >= 2:
                    # Xavier uniform for 2D+ tensors (Linear layers)
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    # For 1D tensors (e.g., LayerNorm weights), use uniform initialization
                    nn.init.uniform_(param, -0.1, 0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize log_std to small values (small initial variance)
        nn.init.constant_(self.delta_log_std_head.bias, -2.0)

    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        controls: torch.Tensor,
        local_features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.

        Args:
            z: [B, T, 64] current latent position
            v: [B, T, 64] current velocity
            controls: [B, T, 6] control parameters
            local_features: [B, T, 16] local geometry features
            hidden: [num_layers, B, H] optional initial hidden state

        Returns:
            delta_mean: [B, T, M, 64] mixture component means
            delta_log_std: [B, T, M, 64] mixture component log-stds
            delta_weights: [B, T, M] mixture log-weights
            vel_delta: [B, T, 64] velocity update
            window_logits: [B, T, num_window_classes] window size logits
            hidden: [num_layers, B, H] final hidden state
        """
        B, T, D = z.shape
        M = self.cfg.num_mixture_components

        # Project inputs
        z_feat = self.z_proj(z)           # [B, T, H/4]
        v_feat = self.v_proj(v)           # [B, T, H/4]
        ctrl_feat = self.ctrl_proj(controls)  # [B, T, H/4]
        local_feat = self.local_proj(local_features)  # [B, T, H/4]

        # Concatenate features
        x = torch.cat([z_feat, v_feat, ctrl_feat, local_feat], dim=-1)  # [B, T, H]
        x = self.input_norm(x)

        # Run through GRU
        out, h_next = self.gru(x, hidden)  # out: [B, T, H], h_next: [L, B, H]
        out = self.output_norm(out)

        # Compute outputs
        # Mixture means
        delta_mean = self.delta_mean_head(out)  # [B, T, M*D]
        delta_mean = delta_mean.view(B, T, M, D)

        # Mixture log-stds (clamped for stability)
        delta_log_std = self.delta_log_std_head(out)  # [B, T, M*D]
        delta_log_std = delta_log_std.view(B, T, M, D)
        delta_log_std = torch.clamp(delta_log_std, min=-5.0, max=2.0)

        # Mixture weights (log-softmax for numerical stability)
        delta_weights = F.log_softmax(self.delta_weights_head(out), dim=-1)  # [B, T, M]

        # Velocity update
        vel_delta = self.vel_delta_head(out)  # [B, T, D]

        # Window size prediction
        window_logits = self.window_head(out)  # [B, T, num_window_classes]

        return delta_mean, delta_log_std, delta_weights, vel_delta, window_logits, h_next

    @torch.inference_mode()
    def step(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        controls: torch.Tensor,
        local_features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step inference helper.

        Args:
            z: [1, 1, 64] current position
            v: [1, 1, 64] current velocity
            controls: [1, 1, 6] control parameters
            local_features: [1, 1, 16] local geometry features
            hidden: optional hidden state

        Returns:
            delta_mean: [1, M, 64] mixture means
            delta_log_std: [1, M, 64] mixture log-stds
            delta_weights: [1, M] mixture log-weights
            vel_delta: [1, 64] velocity update
            window_logits: [1, num_window_classes] window size logits
            hidden: updated hidden state
        """
        delta_mean, delta_log_std, delta_weights, vel_delta, window_logits, h_next = self.forward(
            z, v, controls, local_features, hidden
        )
        # Remove time dimension
        return (
            delta_mean[:, -1],      # [B, M, D]
            delta_log_std[:, -1],   # [B, M, D]
            delta_weights[:, -1],   # [B, M]
            vel_delta[:, -1],       # [B, D]
            window_logits[:, -1],   # [B, num_window_classes]
            h_next,
        )

    def sample_delta(
        self,
        delta_mean: torch.Tensor,
        delta_log_std: torch.Tensor,
        delta_weights: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample displacement from the Gaussian mixture.

        Args:
            delta_mean: [B, M, D] mixture means
            delta_log_std: [B, M, D] mixture log-stds
            delta_weights: [B, M] mixture log-weights
            temperature: sampling temperature (higher = more random)

        Returns:
            delta_z: [B, D] sampled displacement
        """
        B, M, D = delta_mean.shape

        # Apply temperature to weights
        weights = F.softmax(delta_weights / temperature, dim=-1)  # [B, M]

        # Sample component index
        component_idx = torch.multinomial(weights, num_samples=1)  # [B, 1]

        # Gather mean and std for selected component
        batch_idx = torch.arange(B, device=delta_mean.device)[:, None]
        mean = delta_mean[batch_idx, component_idx]  # [B, 1, D]
        log_std = delta_log_std[batch_idx, component_idx]  # [B, 1, D]
        std = torch.exp(log_std) * temperature  # [B, 1, D]

        # Sample from Gaussian
        eps = torch.randn_like(std)
        delta_z = mean + std * eps  # [B, 1, D]

        return delta_z.squeeze(1)  # [B, D]

    def get_mixture_mode(
        self,
        delta_mean: torch.Tensor,
        delta_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the mean of the highest-weight mixture component (deterministic).

        Args:
            delta_mean: [B, M, D] mixture means
            delta_weights: [B, M] mixture log-weights

        Returns:
            delta_z: [B, D] mode displacement
        """
        # Find highest weight component
        best_idx = delta_weights.argmax(dim=-1)  # [B]

        # Gather corresponding mean
        B, M, D = delta_mean.shape
        batch_idx = torch.arange(B, device=delta_mean.device)
        return delta_mean[batch_idx, best_idx]  # [B, D]

    def get_window_size(
        self,
        window_logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> int:
        """
        Get predicted window size from logits.

        Args:
            window_logits: [B, num_window_classes] or [num_window_classes] logits
            temperature: Sampling temperature (higher = more random)
            sample: If True, sample from softmax; if False, use argmax

        Returns:
            Window size in latent frames (from window_class_sizes)
        """
        if window_logits.dim() == 1:
            window_logits = window_logits.unsqueeze(0)

        if sample and temperature > 0:
            # Sample from softmax distribution
            probs = F.softmax(window_logits / temperature, dim=-1)
            class_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            # Deterministic argmax
            class_idx = window_logits.argmax(dim=-1).item()

        return self.cfg.window_class_sizes[class_idx]


def build_local_features(
    z: np.ndarray,
    local_sigma: np.ndarray,
    local_density: np.ndarray,
    time_gradient: np.ndarray,
    t_lat: np.ndarray,
    file_id: np.ndarray,
    knn_centroid: np.ndarray,
) -> np.ndarray:
    """
    Build local feature vector for policy input.

    Args:
        z: [64] current latent position
        local_sigma: float, local scale
        local_density: float, local density
        time_gradient: [64] local time direction
        t_lat: float, normalized time position
        file_id: int, source file ID
        knn_centroid: [64] centroid of kNN neighborhood

    Returns:
        [16] local feature vector
    """
    # Distance to kNN centroid
    dist_to_centroid = np.linalg.norm(z - knn_centroid)

    # Alignment with time gradient
    z_normalized = z / (np.linalg.norm(z) + 1e-6)
    time_alignment = np.dot(z_normalized, time_gradient)

    # Compose feature vector
    features = np.array([
        local_sigma,
        local_density,
        dist_to_centroid,
        time_alignment,
        float(t_lat) / 1000.0,  # Normalize time position
        float(file_id) / 100.0,   # Normalize file ID
    ], dtype=np.float32)

    # Pad to 16 dimensions with zeros (reserved for future features)
    padded = np.zeros(16, dtype=np.float32)
    padded[:len(features)] = features

    return padded
