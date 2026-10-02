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
    vae_id: str = "same_s"
    adapter_path: str = ""


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_decoder(config: dict, *, resource_dir=None, corpus_spec=None) -> DecoderSelection:
    """Validate selection before releasing a cached decoder; never infer a GPU."""
    if not isinstance(config, dict):
        raise ValueError("Perform config must be an object")
    vae_id = (corpus_spec or {}).get("vae_id", "same_s")
    backend = config.get("decoder_backend", "onnxruntime" if vae_id == "same_s" else "pytorch")
    if vae_id != "same_s":
        return _select_adapter(config, vae_id, backend)
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
    if selection.backend == "pytorch" and selection.vae_id != "same_s":
        from .adapter_decoder import AdapterTorchDecoder
        return AdapterTorchDecoder(vae_id=selection.vae_id, device=selection.device,
            weight_path=selection.adapter_path if selection.vae_id.startswith("ear_") else "",
            repo_or_path=selection.adapter_path if selection.vae_id == "stable_audio_open" else None,
            local_files_only=selection.local_files_only)
    if selection.backend == "pytorch":
        from .torch_decoder import SameSTorchDecoder
        return SameSTorchDecoder(device=selection.device, local_files_only=selection.local_files_only)
    raise ValueError(f"Unsupported decoder backend {selection.backend!r}")


def _select_adapter(config, vae_id, backend):
    if backend != 'pytorch':
        raise ValueError(f'{vae_id} requires PyTorch; the packaged ONNX graph supports SAME-S only')
    import torch
    from .torch_decoder import _resolve_device
    device = str(_resolve_device(torch, config.get('decoder_device', 'cpu')))
    offline = config.get('decoder_local_files_only', True)
    if type(offline) is not bool:
        raise ValueError('decoder_local_files_only must be a boolean')
    if vae_id == 'stable_audio_open':
        from huggingface_hub import hf_hub_download
        try:
            files = [Path(hf_hub_download('stabilityai/stable-audio-open-1.0', name,
                local_files_only=offline)) for name in
                ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors')]
        except Exception as exc:
            raise ValueError('Stable Audio Open weights are unavailable locally. Prepare the VAE cache before Start Perform.') from exc
        if files[0].parent.parent != files[1].parent.parent:
            raise ValueError("Stable Audio Open config and weights must come from one cached revision")
        adapter_path = str(files[0].parent.parent)
        identity = (vae_id, adapter_path, *(_hash(p) for p in files), metadata.version('diffusers'))
    elif vae_id in ('ear_vae_44k', 'ear_vae_48k'):
        path = Path(config.get('vae_weight_path', '')).expanduser()
        if not path.is_file():
            raise ValueError('EAR VAE requires an existing .pyt weight file in its EAR_VAE repository; set VAE weights in Perform')
        path = path.resolve()
        configs = []
        for parent in list(path.parents)[:4]:
            if (parent / 'model').is_dir():
                configs = sorted((parent / 'config').glob('*.json'))
                break
        adapter_path = str(path)
        identity = (vae_id, adapter_path, _hash(path), *(_hash(p) for p in configs))
    else:
        raise ValueError(f'Unsupported corpus VAE {vae_id!r}')
    return DecoderSelection(backend, device, (*identity, str(torch.__version__), 'float32'),
        local_files_only=offline, vae_id=vae_id, adapter_path=adapter_path)
