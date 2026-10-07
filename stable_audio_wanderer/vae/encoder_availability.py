"""Offline encoder preflight: dependencies and required cached files, no downloads."""
from pathlib import Path
from .decoder_availability import installed


def encoder_availability(vae_id):
    if vae_id not in ('same_s', 'stable_audio_open'):
        return dict(ready=True, detail='EAR requires a local checkpoint; validated when encoding starts.')
    if vae_id == 'same_s':
        from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION
        names = ('model_config.json', 'model.safetensors')
        required = ('huggingface_hub', 'torch', 'torchaudio', 'stable_audio_3', 'safetensors')
    else:
        from .stable_audio_open_weights import SOURCE_MODEL, SOURCE_REVISION
        names = ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors')
        required = ('huggingface_hub', 'torch', 'diffusers', 'safetensors')
    missing = [name for name in required if not installed(name)]
    if missing:
        return dict(ready=False, detail='Missing libraries: ' + ', '.join(missing) + '. Install requirements.txt in the server environment.')
    try:
        from huggingface_hub import try_to_load_from_cache
        paths = [try_to_load_from_cache(SOURCE_MODEL, name, revision=SOURCE_REVISION) for name in names]
        cached = all(isinstance(path, str) and Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)
    except Exception:
        cached = False
    if not cached:
        return dict(ready=True, download_required=True, detail=f'Weights not cached for {SOURCE_MODEL}. Encoding will download them automatically. Internet access and authorized Hugging Face access may be required; see README for hf auth login.')
    return dict(ready=True, detail='Required model files are cached. Model loading and integrity are checked when encoding starts.')


def prepare_encoder_weights(vae_id, progress, cancel_event):
    """Download missing pinned files before the offline encoder loader runs."""
    if vae_id not in ('same_s', 'stable_audio_open'):
        return
    if vae_id == 'same_s':
        from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION
        names = ('model_config.json', 'model.safetensors')
    else:
        from .stable_audio_open_weights import SOURCE_MODEL, SOURCE_REVISION
        names = ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors')
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    for name in names:
        if cancel_event.is_set():
            return
        path = try_to_load_from_cache(SOURCE_MODEL, name, revision=SOURCE_REVISION)
        cached = isinstance(path, str) and Path(path).is_file() and Path(path).stat().st_size > 0
        if not cached:
            progress(dict(event='model_download', model=SOURCE_MODEL, file=name,
                          detail=f'Downloading {SOURCE_MODEL}: {name}. First use may take several minutes.'))
            try:
                hf_hub_download(SOURCE_MODEL, name, revision=SOURCE_REVISION)
            except Exception as exc:
                raise RuntimeError(download_error(SOURCE_MODEL, exc)) from exc
    if not cancel_event.is_set():
        progress(dict(event='model_download_done', detail='Model files available. Loading and validating the encoder…'))


def download_error(model, error):
    """Classify nested Hub/HTTP failures without exposing tokens or URLs."""
    causes = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(current)
        current = current.__cause__ or current.__context__
    names = {type(exc).__name__ for exc in causes}
    statuses = {getattr(getattr(exc, 'response', None), 'status_code', None) for exc in causes}
    prefix = f'Could not download {model}. '
    if 'GatedRepoError' in names or 403 in statuses:
        return prefix + 'Access denied. Accept the model terms/request access on Hugging Face, then run hf auth login with an authorized account.'
    if 401 in statuses:
        return prefix + 'Authentication failed. Run hf auth login in the server environment and verify the token with hf auth whoami.'
    if names & {'ConnectionError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'TimeoutError', 'LocalEntryNotFoundError', 'OfflineModeIsEnabled'}:
        return prefix + 'Network/offline error. Check internet access, proxy settings and HF_HUB_OFFLINE, then retry.'
    if 404 in statuses:
        return prefix + 'Model revision/file unavailable or account access missing. Check model access and the pinned revision in the README.'
    return prefix + 'Download failed. Check Hugging Face access, connectivity and local disk space; see the server log for details.'
