"""VAE adapter abstraction for pluggable encode/decode backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch


@dataclass
class VAEInfo:
    """Static metadata about a VAE, available before model load."""
    vae_id: str
    display_name: str
    sample_rate: int
    latent_hz: float
    latent_dim: int
    channels: int = 2
    requires_path: bool = False
    path_label: str = "Path to model weights"


class VAEAdapter(ABC):
    """Canonical interface for all VAE backends.

    Encode and decode use a standardized tensor layout so downstream code
    (chunked encoding, real-time decoding) never touches model-specific APIs.
    """

    @abstractmethod
    def info(self) -> VAEInfo:
        """Return static metadata about this VAE."""
        ...

    @abstractmethod
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent space.

        Args:
            audio: [1, channels, T_audio] float tensor on model device.
        Returns:
            Latent tensor [1, latent_dim, T_lat].
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to audio.

        Args:
            z: [1, latent_dim, T_lat] float tensor on model device.
        Returns:
            Audio tensor [1, channels, T_audio].
        """
        ...

    @property
    def raw_model(self):
        """Access to the underlying model object (for backward compat)."""
        return getattr(self, "_model", None)
