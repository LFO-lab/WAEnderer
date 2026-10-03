"""Explicit decoder selection and artifact-aware cache identity for the pipeline."""

from dataclasses import dataclass, field
import hashlib
from importlib import metadata
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
    adapter_repo: str = ""
    adapter_config: str = ""
    adapter_source: tuple = ()
    artifact: object = field(default=None, compare=False, repr=False)


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
    if backend == "onnxruntime":
        if config.get("decoder_device", "cpu") != "cpu":
            raise ValueError("ONNX decoding requires decoder_device='cpu'")
        from .onnx_artifacts import resolve_artifact
        expected_source = config.get("decoder_source_identity")
        if expected_source is not None and not isinstance(expected_source, dict):
            raise ValueError("decoder_source_identity must be an object")
        if vae_id.startswith("ear_") and config.get("vae_weight_path"):
            from .ear_weights import selected_file_identity
            selected = selected_file_identity(vae_id, config['vae_weight_path'],
                config.get('vae_repo_path',''), config.get('vae_config_path',''))
            if expected_source and any(k in expected_source and expected_source[k]!=v for k,v in selected.items()):
                raise ValueError('Selected EAR files conflict with expected source identity')
            expected_source = {**(expected_source or {}), **selected}
        artifact = resolve_artifact(vae_id,
            artifact_dir=config.get("decoder_artifact_dir") or resource_dir,
            store_dir=config.get("decoder_store_dir"), artifact_id=config.get("decoder_artifact_id"),
            expected_source=expected_source)
        if corpus_spec:
            artifact.validate_corpus(corpus_spec)
        return DecoderSelection(backend, "cpu", artifact.cache_key, artifact.root,
                                vae_id=vae_id, artifact=artifact)
    if vae_id != "same_s":
        return _select_adapter(config, vae_id, backend, corpus_spec)
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
        from .artifact_decoder import load_artifact_decoder
        from .onnx_artifacts import read_artifact
        artifact = selection.artifact or read_artifact(selection.resource_dir, expected_vae=selection.vae_id)
        return load_artifact_decoder(artifact)
    if selection.backend == "pytorch" and selection.vae_id != "same_s":
        from .adapter_decoder import AdapterTorchDecoder
        return AdapterTorchDecoder(vae_id=selection.vae_id, device=selection.device,
            weight_path=selection.adapter_path if selection.vae_id.startswith("ear_") else "",
            repo_or_path=selection.adapter_path if selection.vae_id == "stable_audio_open" else None,
            local_files_only=selection.local_files_only,
            repo_path=selection.adapter_repo, config_path=selection.adapter_config,
            expected_source=dict(selection.adapter_source) if selection.adapter_source else None)
    if selection.backend == "pytorch":
        from .torch_decoder import SameSTorchDecoder
        return SameSTorchDecoder(device=selection.device, local_files_only=selection.local_files_only)
    raise ValueError(f"Unsupported decoder backend {selection.backend!r}")


def _select_adapter(config, vae_id, backend, corpus_spec=None):
    if backend != 'pytorch':
        raise ValueError(f'Unsupported decoder backend {backend!r} for {vae_id}')
    import torch
    from .torch_decoder import _resolve_device
    device = str(_resolve_device(torch, config.get('decoder_device', 'cpu')))
    offline = config.get('decoder_local_files_only', True)
    if type(offline) is not bool:
        raise ValueError('decoder_local_files_only must be a boolean')
    if vae_id == 'stable_audio_open':
        from .stable_audio_open_weights import resolve_source
        root, source = resolve_source(local_files_only=offline)
        expected = config.get('decoder_source_identity')
        if expected is not None and (not isinstance(expected, dict) or
                any(source.get(k) != v for k, v in expected.items())):
            raise ValueError('Stale source checkpoint/config identity')
        for key in ('config_sha256', 'weights_sha256', 'revision'):
            if key in (corpus_spec or {}) and corpus_spec[key] != source[key]:
                raise ValueError(f'Corpus source {key} does not match native decoder')
        adapter_path = str(root)
        identity = (vae_id, adapter_path, *source.values(), metadata.version('diffusers'))
    elif vae_id in ('ear_vae_44k', 'ear_vae_48k'):
        from .ear_weights import resolve_source
        resolved = resolve_source(vae_id, config.get('vae_weight_path',''),
            config.get('vae_repo_path',''), config.get('vae_config_path',''),
            config.get('decoder_source_identity'))
        for key in resolved.identity:
            if key in (corpus_spec or {}) and corpus_spec[key] != resolved.identity[key]:
                raise ValueError(f'Corpus source {key} does not match native EAR decoder')
        return DecoderSelection(backend, device,
            (vae_id, str(resolved.weights), str(resolved.repo), str(resolved.config_path),
             *resolved.identity.values(), metadata.version('descript-audio-codec'),
             metadata.version('descript-audiotools'), metadata.version('einops'), str(torch.__version__), 'float32'),
            local_files_only=offline, vae_id=vae_id, adapter_path=str(resolved.weights),
            adapter_repo=str(resolved.repo), adapter_config=str(resolved.config_path),
            adapter_source=tuple(sorted(resolved.identity.items())))
    else:
        raise ValueError(f'Unsupported corpus VAE {vae_id!r}')
    return DecoderSelection(backend, device, (*identity, str(torch.__version__), 'float32'),
        local_files_only=offline, vae_id=vae_id, adapter_path=adapter_path)
