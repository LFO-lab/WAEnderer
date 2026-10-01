"""Pinned native SAME-S checkpoint, independent of ONNX and export tooling."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


SOURCE_MODEL = "stabilityai/SAME-S"
SOURCE_REVISION = "fbeb3dcf53a326e5682f38e22e7f740202d44232"
LIBRARY_REVISION = "779434a908193105335fd8d833418603625b2859"
# Identity recorded against the packaged decoder during the phase-0 audit.
CONFIG_SHA256 = "c329dd0a6f61d0b3ea4f23930059a6c00437005692fed4310924eb286253303a"
WEIGHTS_SHA256 = "c19698ce3a0b462acb967ee495e9eb7945221f236968c50206cce8cf22b3d305"


class NativeDecoderLoadError(RuntimeError):
    """Native decoder dependencies, device, checkpoint or warm-up unavailable."""


@dataclass(frozen=True)
class SameSWeights:
    config_path: Path
    model_path: Path
    config_sha256: str
    model_sha256: str
    source_model: str = SOURCE_MODEL
    source_revision: str = SOURCE_REVISION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_same_s_weights(*, local_files_only: bool = True) -> SameSWeights:
    """Resolve the exact standalone checkpoint; do not use registered-model fallbacks.

    Offline by default. Passing local_files_only=False explicitly allows the
    Hugging Face client to obtain the two pinned files using its normal auth.
    No ONNX graph/session is required, including for identity verification.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise NativeDecoderLoadError("Native SAME-S requires huggingface_hub") from exc
    paths = []
    for filename, expected in (("model_config.json", CONFIG_SHA256),
                               ("model.safetensors", WEIGHTS_SHA256)):
        try:
            path = Path(hf_hub_download(
                repo_id=SOURCE_MODEL, filename=filename, revision=SOURCE_REVISION,
                local_files_only=local_files_only,
            ))
            if not path.is_file() or SOURCE_REVISION not in path.parts:
                raise NativeDecoderLoadError(f"SAME-S {filename} did not resolve to the pinned snapshot")
            if _sha256(path) != expected:
                raise NativeDecoderLoadError(f"SAME-S {filename} SHA-256 mismatch")
        except NativeDecoderLoadError:
            raise
        except Exception as exc:
            mode = "offline cache" if local_files_only else "Hugging Face"
            raise NativeDecoderLoadError(
                f"Cannot obtain pinned SAME-S {filename} from {mode}: {exc}"
            ) from exc
        paths.append(path)
    return SameSWeights(paths[0], paths[1], CONFIG_SHA256, WEIGHTS_SHA256)


def load_same_s_model(weights: SameSWeights):
    """Load all native tensors strictly on CPU, then discard the unused encoder.

    Deliberately bypasses upstream copy_state_dict, which silently skips missing
    or incompatible tensors. This checkpoint uses exact standalone AE keys, so
    neither key remapping nor loading from a combined diffusion model is needed.
    """
    try:
        import torch
        from safetensors.torch import load_file
        from stable_audio_3.factory import create_autoencoder_from_config
    except Exception as exc:
        raise NativeDecoderLoadError(
            "Native SAME-S requires compatible torch, torchaudio, safetensors and "
            "stable-audio-3; see docs/DUAL_INFERENCE_PHASE2.md"
        ) from exc
    try:
        config = json.loads(weights.config_path.read_text(encoding="utf-8"))
        with torch.device("cpu"):
            model = create_autoencoder_from_config(config["model"], config["sample_rate"])
        model = model.to(device="cpu", dtype=torch.float32)
        state = load_file(str(weights.model_path), device="cpu")
        # strict=True rejects missing/unexpected keys and mismatched shapes.
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False)
        model.encoder = None
        return model
    except Exception as exc:
        raise NativeDecoderLoadError(f"Cannot strictly load native SAME-S checkpoint: {exc}") from exc
