"""EAR VAE adapter (earlab/EAR_VAE via local repo + .pyt weights)."""
import math

import torch
from ..base import VAEAdapter, VAEInfo
from ..registry import register_vae
from ...config import DEVICE

_INFO_44K = VAEInfo(
    vae_id="ear_vae_44k",
    display_name="EAR VAE (44.1k)",
    sample_rate=44100,
    latent_hz=44100.0 / 1024.0,  # default; overridden at load time
    latent_dim=64,
    channels=2,
    requires_path=True,
    path_label="Path to .pyt weight file",
)

_INFO_48K = VAEInfo(
    vae_id="ear_vae_48k",
    display_name="EAR VAE (48k)",
    sample_rate=48000,
    latent_hz=48000.0 / 960.0,  # default; overridden at load time
    latent_dim=64,
    channels=2,
    requires_path=True,
    path_label="Path to .pyt weight file",
)


class EarVAEAdapter(VAEAdapter):
    """Wraps earlab EAR_VAE for the canonical VAEAdapter API."""

    def __init__(self, model, info: VAEInfo):
        self._model = model
        self._info = info

    @classmethod
    def load(cls, weight_path: str = "", repo_path: str = "", config_path: str = "",
             sample_rate: int = 44100, device=None, expected_source=None, **_kwargs):
        from ..ear_weights import resolve_source, load_model_class
        if sample_rate not in (44100, 48000):
            raise ValueError('EAR sample rate must be 44100 or 48000')
        vae_id = 'ear_vae_44k' if sample_rate == 44100 else 'ear_vae_48k'
        source = resolve_source(vae_id, weight_path, repo_path, config_path, expected_source)
        model_config = source.config
        model = load_model_class(source)(model_config=model_config)
        model.load_state_dict(source.state, strict=True)
        model = model.to(device=device if device is not None else DEVICE, dtype=torch.float32).eval()

        # Compute actual downsampling ratio from the config strides.
        strides = model_config["encoder"]["config"]["strides"]
        ds_ratio = math.prod(strides)
        # The encoder's latent_dim is pre-split (mean + scale); the decoder's
        # latent_dim is the actual latent space dimensionality after VAE sampling.
        latent_dim = model_config["decoder"]["config"].get("latent_dim", 64)
        latent_hz = sample_rate / ds_ratio

        # Build VAEInfo with actual values from the loaded config.
        base = _INFO_44K if sample_rate <= 44100 else _INFO_48K
        info = VAEInfo(
            vae_id=base.vae_id,
            display_name=base.display_name,
            sample_rate=sample_rate,
            latent_hz=latent_hz,
            latent_dim=latent_dim,
            channels=base.channels,
            requires_path=base.requires_path,
            path_label=base.path_label,
        )
        adapter = cls(model, info)
        adapter.source = source.identity
        adapter.source_files = source.code_files
        adapter.effective_config = source.config
        return adapter

    def info(self) -> VAEInfo:
        return self._info

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self._model.encode(audio, use_sample=True)  # [1, latent_dim, T_lat]

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._model.decode(z)  # [1, channels, T_audio]


def _load_ear_44k(**kwargs):
    if kwargs.pop("sample_rate", 44100) != 44100:
        raise ValueError("EAR 44.1 kHz registry sample-rate mismatch")
    kwargs["sample_rate"] = 44100
    return EarVAEAdapter.load(**kwargs)


def _load_ear_48k(**kwargs):
    if kwargs.pop("sample_rate", 48000) != 48000:
        raise ValueError("EAR 48 kHz registry sample-rate mismatch")
    kwargs["sample_rate"] = 48000
    return EarVAEAdapter.load(**kwargs)


# Register both variants.
register_vae(vae_id=_INFO_44K.vae_id, info=_INFO_44K, loader=_load_ear_44k)
register_vae(vae_id=_INFO_48K.vae_id, info=_INFO_48K, loader=_load_ear_48k)