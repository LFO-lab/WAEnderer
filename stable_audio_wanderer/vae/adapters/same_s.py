"""SAME-S adapter (stabilityai/SAME-S via stable-audio-3)."""
import torch

from ..base import VAEAdapter, VAEInfo
from ..registry import register_vae
from ...config import DEVICE

_SAMPLE_RATE = 44100
_DOWNSAMPLING_RATIO = 4096

_INFO = VAEInfo(
    vae_id="same_s",
    display_name="SAME-S (44.1k)",
    sample_rate=_SAMPLE_RATE,
    latent_hz=_SAMPLE_RATE / _DOWNSAMPLING_RATIO,
    latent_dim=256,
    channels=2,
    requires_path=False,
)


class SameSAdapter(VAEAdapter):
    """Wraps stable-audio-3 SAME-S for the canonical VAEAdapter API."""

    def __init__(self, model):
        self._model = model
        self._info = _INFO

    @classmethod
    def load(cls, repo_or_path: str = "same-s", **_kwargs):
        try:
            from stable_audio_3 import AutoencoderModel
        except ImportError as exc:
            raise ImportError(
                "SAME-S requires compatible native dependencies. Install the default "
                "requirements.txt or the native-same-s package extra in this "
                "interpreter; see docs/INSTALLATION_PROFILES.md. Do not bypass "
                "upstream dependency pins with --no-deps."
            ) from exc

        model = AutoencoderModel.from_pretrained(repo_or_path)
        if hasattr(model, "to"):
            model = model.to(DEVICE)
        if hasattr(model, "eval"):
            model = model.eval()
        return cls(model)

    def info(self) -> VAEInfo:
        return self._info

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[0] != 1:
            raise ValueError(
                f"SAME-S expects audio [1, channels, samples], got {tuple(audio.shape)}"
            )
        # stable-audio-3 AutoencoderModel handles channel conversion and returns [1, D, T].
        return self._model.encode(audio.squeeze(0), self._info.sample_rate)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._model.decode(z)


# Register with the global VAE registry.
register_vae(
    vae_id=_INFO.vae_id,
    info=_INFO,
    loader=SameSAdapter.load,
)
