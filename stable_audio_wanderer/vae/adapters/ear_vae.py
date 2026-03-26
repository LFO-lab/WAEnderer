"""EAR VAE adapter (earlab/EAR_VAE via local repo + .pyt weights)."""
import json
import math
import os
import sys

import torch
from ..base import VAEAdapter, VAEInfo
from ..registry import register_vae
from ...config import DEVICE, DTYPE

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
    latent_hz=48000.0 / 1024.0,  # default; overridden at load time
    latent_dim=64,
    channels=2,
    requires_path=True,
    path_label="Path to .pyt weight file",
)


def _pick_config(repo_path: str, weight_path: str) -> dict:
    """Select the right JSON config for the given weight file.

    Heuristic: if the weight filename contains 'v2', use ear_vae_v2.json,
    otherwise use model_config.json.  Falls back to model_config.json if
    the v2 config doesn't exist.
    """
    config_dir = os.path.join(repo_path, "config")
    weight_name = os.path.basename(weight_path).lower()

    if "v2" in weight_name:
        v2_path = os.path.join(config_dir, "ear_vae_v2.json")
        if os.path.isfile(v2_path):
            with open(v2_path, "r") as f:
                return json.load(f)

    default_path = os.path.join(config_dir, "model_config.json")
    if not os.path.isfile(default_path):
        raise FileNotFoundError(
            f"EAR VAE config not found in {config_dir}. "
            "Ensure the EAR_VAE repo has config/model_config.json."
        )
    with open(default_path, "r") as f:
        return json.load(f)


def _reconcile_transformer(model_config: dict, state_dict: dict) -> dict:
    """Remove transformer from config if the checkpoint has no transformer keys.

    The EAR_VAE v2 config file ships with a transformer block defined,
    but the v2 pretrained weights were trained without one.  If there
    are no ``transformers.*`` keys in the state dict, we set the
    transformer config entry to None so the model is built without it.
    """
    has_transformer_keys = any(k.startswith("transformers.") for k in state_dict)
    if not has_transformer_keys:
        model_config["transformer"] = None
    return model_config


class EarVAEAdapter(VAEAdapter):
    """Wraps earlab EAR_VAE for the canonical VAEAdapter API."""

    def __init__(self, model, info: VAEInfo):
        self._model = model
        self._info = info

    @classmethod
    def load(cls, weight_path: str = "", repo_path: str = "",
             sample_rate: int = 44100, **_kwargs):
        """Load EAR VAE from a local repo clone + .pyt weight file.

        Args:
            weight_path: Path to the .pyt checkpoint file.
            repo_path: Path to the EAR_VAE repo root. If empty, inferred
                       from weight_path (assumes weights are inside the repo).
            sample_rate: 44100 or 48000 — selects which VAEInfo to use.
        """
        if not weight_path:
            raise ValueError(
                "EAR VAE requires a weight_path. Please provide the path "
                "to your .pyt checkpoint file."
            )
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"EAR VAE weight file not found: {weight_path}")

        # Infer repo root from weight path if not given.
        if not repo_path:
            candidate = os.path.dirname(os.path.abspath(weight_path))
            for _ in range(4):
                if os.path.isdir(os.path.join(candidate, "model")):
                    repo_path = candidate
                    break
                candidate = os.path.dirname(candidate)
            if not repo_path:
                raise FileNotFoundError(
                    "Could not find EAR_VAE repo root (directory containing 'model/'). "
                    "Please provide repo_path explicitly or ensure your weight file "
                    "is inside the EAR_VAE repository."
                )

        # Add repo to sys.path so we can import the model module.
        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Select config matching the weight file variant.
        model_config = _pick_config(repo_path, weight_path)

        # Load state dict and reconcile transformer presence.
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        model_config = _reconcile_transformer(model_config, state_dict)

        # Import and instantiate.
        try:
            from model.ear_vae import EAR_VAE  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            if "dac" in str(e):
                raise ImportError(
                    "EAR VAE requires the descript-audio-codec package. "
                    "Install it with: pip install descript-audio-codec"
                ) from e
            raise
        model = EAR_VAE(model_config=model_config)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()

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
        return cls(model, info)

    def info(self) -> VAEInfo:
        return self._info

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self._model.encode(audio, use_sample=True)  # [1, latent_dim, T_lat]

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._model.decode(z)  # [1, channels, T_audio]


def _load_ear_44k(**kwargs):
    kwargs.setdefault("sample_rate", 44100)
    return EarVAEAdapter.load(**kwargs)


def _load_ear_48k(**kwargs):
    kwargs.setdefault("sample_rate", 48000)
    return EarVAEAdapter.load(**kwargs)


# Register both variants.
register_vae(vae_id=_INFO_44K.vae_id, info=_INFO_44K, loader=_load_ear_44k)
register_vae(vae_id=_INFO_48K.vae_id, info=_INFO_48K, loader=_load_ear_48k)