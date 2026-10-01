"""Explicit decoder selection and artifact-aware cache identity for the pipeline."""

from dataclasses import dataclass, field
import hashlib
from importlib import resources, metadata
import json
from pathlib import Path


@dataclass(frozen=True)
class DecoderSelection:
    backend: str
    device: str
    identity: tuple[str, ...]
    resource_dir: Path | None = field(default=None, compare=False)
    local_files_only: bool = field(default=True, compare=False)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_decoder(config: dict, *, resource_dir=None) -> DecoderSelection:
    """Validate selection before releasing a cached decoder; never infer a GPU."""
    if not isinstance(config, dict):
        raise ValueError("Perform config must be an object")
    backend = config.get("decoder_backend", "onnxruntime")
    if backend == "onnxruntime":
        device = config.get("decoder_device", "cpu")
        if device != "cpu":
            raise ValueError("ONNX decoding requires decoder_device='cpu'")
        root = (Path(resource_dir) if resource_dir is not None else
                Path(str(resources.files("stable_audio_wanderer.resources.same_s")))).resolve()
        metadata_path = root / "decoder.json"
        description = json.loads(metadata_path.read_text(encoding="utf-8"))
        name = description.get("model")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".onnx"):
            raise ValueError("ONNX decoder model must be a local .onnx filename")
        model_path = (root / name).resolve()
        if not model_path.is_relative_to(root):
            raise ValueError("ONNX model path escapes its resource directory")
        model_hash = _hash(model_path)
        if description.get("model_sha256", model_hash) != model_hash:
            raise ValueError("ONNX decoder model SHA-256 mismatch")
        return DecoderSelection(backend, "cpu", (str(root), _hash(metadata_path), model_hash), root)
    if backend == "pytorch":
        import torch
        from .torch_decoder import _resolve_device
        from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION, CONFIG_SHA256, WEIGHTS_SHA256
        device = _resolve_device(torch, config.get("decoder_device"))
        offline = config.get("decoder_local_files_only", True)
        if type(offline) is not bool:
            raise ValueError("decoder_local_files_only must be a boolean")
        distribution = metadata.distribution("stable-audio-3")
        origin = distribution.read_text("direct_url.json") or ""
        identity = (SOURCE_MODEL, SOURCE_REVISION, CONFIG_SHA256, WEIGHTS_SHA256,
                    distribution.version, origin, str(torch.__version__), "float32")
        return DecoderSelection(backend, str(device), identity, local_files_only=offline)
    raise ValueError(f"Unsupported decoder_backend {backend!r}; use onnxruntime or pytorch")


def create_decoder(selection: DecoderSelection):
    """Prepare one decoder. The pipeline validates the corpus on every start."""
    if selection.backend == "onnxruntime":
        from .onnx_decoder import load_same_s_app_decoder
        return load_same_s_app_decoder(resource_dir=selection.resource_dir)
    if selection.backend == "pytorch":
        from .torch_decoder import SameSTorchDecoder
        return SameSTorchDecoder(device=selection.device, local_files_only=selection.local_files_only)
    raise ValueError(f"Unsupported decoder backend {selection.backend!r}")
