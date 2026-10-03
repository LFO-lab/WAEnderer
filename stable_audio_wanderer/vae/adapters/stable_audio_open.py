"""Stable Audio Open VAE adapter (AutoencoderOobleck via diffusers)."""
import torch
from ..base import VAEAdapter, VAEInfo
from ..registry import register_vae
from ...config import DEVICE, DTYPE

_INFO = VAEInfo(
    vae_id="stable_audio_open",
    display_name="Stable Audio Open (44.1k)",
    sample_rate=44100,
    latent_hz=21.5,
    latent_dim=64,
    channels=2,
    requires_path=False,
)


class StableAudioOpenAdapter(VAEAdapter):
    """Wraps diffusers AutoencoderOobleck for the canonical VAEAdapter API."""

    def __init__(self, model):
        self._model = model
        self._info = _INFO

    @classmethod
    def load(cls, repo_or_path: str = "stabilityai/stable-audio-open-1.0", device=None, local_files_only=True, **kwargs):
        from diffusers import AutoencoderOobleck
        from ..stable_audio_open_weights import resolve_source, SOURCE_REVISION
        root, _ = resolve_source(repo_or_path, revision=kwargs.get("revision", SOURCE_REVISION),
                                 local_files_only=local_files_only)
        model = AutoencoderOobleck.from_pretrained(
            str(root), subfolder="vae", local_files_only=True
        ).to(device=device if device is not None else DEVICE, dtype=torch.float32).eval()
        return cls(model)

    def info(self) -> VAEInfo:
        return self._info

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        out = self._model.encode(audio)
        return out.latent_dist.sample()  # [1, latent_dim, T_lat]

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._model.decode(z).sample  # [1, channels, T_audio]


# Register with the global VAE registry.
register_vae(
    vae_id=_INFO.vae_id,
    info=_INFO,
    loader=StableAudioOpenAdapter.load,
)
