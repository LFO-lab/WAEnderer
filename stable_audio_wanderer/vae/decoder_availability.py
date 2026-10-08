"""Read-only, model-specific capability discovery; never load native models."""
import hashlib
import json
from importlib.util import find_spec
from pathlib import Path

SOURCE_FIELDS = ('config_sha256', 'weights_sha256', 'revision', 'effective_config_sha256', 'code_sha256')


def installed(name):
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def decoder_availability(resource_dir=None, *, corpus_spec=None, weight_path='', repo_path='', config_path='', store_dir=None):
    from .onnx_artifacts import resolve_artifact
    spec = corpus_spec or {'vae_id': 'same_s'}
    vae = spec['vae_id']
    if vae not in ('same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k'):
        raise ValueError('Unknown corpus VAE')
    source = {k: spec[k] for k in SOURCE_FIELDS if k in spec}
    native_present, native_error = False, ''
    native_missing_detail = 'Native checkpoint is missing'
    required = ('torch', 'torchaudio', 'stable_audio_3', 'safetensors', 'huggingface_hub') if vae == 'same_s' else (
        ('torch', 'diffusers', 'safetensors', 'huggingface_hub') if vae == 'stable_audio_open' else ('torch', 'dac', 'audiotools', 'einops'))
    selected_error = ''
    if vae.startswith('ear_'):
        native_present = bool(weight_path and Path(weight_path).expanduser().is_file())
        if weight_path:
            source_selected = False
            try:
                from .ear_weights import selected_file_identity, file_identity, VARIANTS
                selected = selected_file_identity(vae, weight_path, repo_path, config_path)
                if any(k in source and source[k] != v for k,v in selected.items()):
                    raise ValueError('Selected EAR source conflicts with corpus provenance')
                source.update(selected)
                source_selected = True
                native = file_identity(vae, weight_path, repo_path, config_path)
                known = next((v for v in VARIANTS.values() if v['weights_sha256'] == native['weights_sha256']), None)
                if known and known != VARIANTS[vae]:
                    raise ValueError('EAR checkpoint belongs to another sample-rate variant')
                if known and known['config_sha256'] != native['config_sha256']:
                    raise ValueError('EAR configuration does not match checkpoint')
                if not known and not config_path:
                    raise ValueError('Custom EAR checkpoint requires an explicit configuration')
            except (ValueError, OSError) as exc:
                native_error = str(exc)
                # A standalone checkpoint still binds ONNX without a native repository.
                if not source_selected:
                    selected_error = native_error
        else:
            native_error = 'Set the local EAR checkpoint, repository and configuration.'
    else:
        try:
            import huggingface_hub
            from huggingface_hub import try_to_load_from_cache
            if vae == 'same_s':
                from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION
                names = ('model_config.json', 'model.safetensors')
            else:
                from .stable_audio_open_weights import SOURCE_MODEL, SOURCE_REVISION
                names = ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors')
            cached = {name: try_to_load_from_cache(SOURCE_MODEL, name, revision=SOURCE_REVISION) for name in names}
            missing_files = [name for name, path in cached.items() if not isinstance(path, str)]
            native_present = not missing_files
            cache = getattr(getattr(huggingface_hub, 'constants', None), 'HF_HUB_CACHE', 'Hugging Face default cache')
            if missing_files:
                command = f'hf download {SOURCE_MODEL} {" ".join(names)} --revision {SOURCE_REVISION}'
                native_missing_detail = (
                    f'Native checkpoint is missing at revision {SOURCE_REVISION}. '
                    f'Missing files: {", ".join(missing_files)}. Cache: {cache}. '
                    f'An older encoder may have cached another revision. In the decoder environment run: {command}; '
                    'then refresh decoder availability. '
                    'ONNX preparation is not required for PyTorch playback.'
                )
        except Exception as exc:
            native_missing_detail = f'Native checkpoint cache lookup failed ({type(exc).__name__}): {exc}. Check huggingface_hub in the decoder interpreter.'
    artifact, artifact_error = None, ''
    artifact_present = False
    artifact_reason = 'missing_artifact'
    if resource_dir:
        artifact_present = (Path(resource_dir)/'decoder.json').is_file()
    else:
        store=Path(store_dir).expanduser() if store_dir else Path.home()/'.cache/waenderer/decoders'
        try:
            pointer=json.loads((store/vae/'current.json').read_text())
            artifact_present=(store/vae/pointer['artifact_id']/'decoder.json').is_file()
        except (OSError,ValueError,KeyError,TypeError):
            if vae == 'same_s':
                from importlib import resources
                artifact_present=(Path(str(resources.files('stable_audio_wanderer.resources.same_s')))/'decoder.json').is_file()
    try:
        if selected_error:
            raise ValueError(selected_error)
        artifact_reason = 'failed_validation'
        artifact = resolve_artifact(vae, artifact_dir=resource_dir, store_dir=store_dir, expected_source=source)
        artifact_present = True
        if 'latent_dim' in spec:
            artifact.validate_corpus(spec)
        # Full graph/external-data validation when the validator is available.
        if installed('onnx'):
            from .onnx_artifacts import inspect_graph
            for graph in artifact.graphs:
                inspect_graph(artifact, graph)
    except Exception as exc:
        artifact_error, artifact = str(exc), None
        if isinstance(exc,FileNotFoundError) or isinstance(exc.__cause__,FileNotFoundError): artifact_reason='missing_artifact'
    identity_source = dict(artifact.source) if artifact else {}
    if not identity_source and vae in ('same_s','stable_audio_open'):
        if vae == 'same_s':
            from .same_s_weights import SOURCE_REVISION,CONFIG_SHA256,WEIGHTS_SHA256
        else:
            from .stable_audio_open_weights import SOURCE_REVISION,CONFIG_SHA256,WEIGHTS_SHA256
        identity_source.update(revision=SOURCE_REVISION,config_sha256=CONFIG_SHA256,weights_sha256=WEIGHTS_SHA256)
    identity_source.update(source)
    # Stable across ONNX presence and exporter settings. EAR's raw source/code
    # hashes determine effective configuration without importing its model.
    identity_keys = ('weights_sha256','config_sha256','code_sha256') if vae.startswith('ear_') else ('weights_sha256','config_sha256','revision')
    identity_source = {k:identity_source[k] for k in identity_keys if k in identity_source}
    token = hashlib.sha256(json.dumps({'vae_id': vae, 'source': identity_source}, sort_keys=True).encode()).hexdigest() if identity_source else None
    devices = [('cpu', True), ('mps', False), ('cuda:0', False)]
    try:
        import torch
        devices = [('cpu', True), ('mps', bool(torch.backends.mps.is_available()))]
        devices += [(f'cuda:{i}', True) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [('cuda:0', False)]
    except Exception:
        pass
    entries = []
    for backend, device, hardware in [('onnxruntime', 'cpu', True), *(('pytorch',d,h) for d,h in devices)]:
        onnx = backend == 'onnxruntime'
        missing = [n for n in (('onnxruntime','onnx') if onnx else required) if not installed(n)]
        reasons = []
        if missing: reasons.append(('missing_dependencies', 'Install: '+', '.join(missing)))
        if not hardware: reasons.append(('unavailable_hardware', f'{device.upper()} hardware unavailable'))
        if onnx and not artifact:
            explanation = artifact_error or 'Prepare ONNX decoder first'
            if not artifact_present:
                explanation = 'No prepared ONNX decoder installed. Use a ready PyTorch decoder or Prepare ONNX decoder. ' + explanation
            reasons.append((artifact_reason, explanation))
        if not onnx and not native_present: reasons.append(('missing_weights', native_missing_detail))
        if not onnx and native_error: reasons.append(('invalid_native_source', native_error))
        entries.append(dict(backend=backend, device=device, vae_id=vae,
            choice_key=f'{vae}|{backend}|{device}', label=f'{"ONNX" if onnx else "PyTorch"} · {device.upper()} · {vae}',
            hardware=hardware, dependencies=not missing, weights=bool(artifact) if onnx else native_present,
            native_source_present=native_present, artifact_present=artifact_present, artifact_verified=bool(artifact),
            artifact_identity=artifact.identity if artifact else None, model_identity=token,
            validated=False, selectable=not reasons, reason_codes=[r[0] for r in reasons],
            detail='; '.join(r[1] for r in reasons) or 'Available; runtime checked on Start Perform. Real-time performance is not implied.'))
    return entries
