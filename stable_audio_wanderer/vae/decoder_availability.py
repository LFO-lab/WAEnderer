"""Read-only discovery. Presence is not successful model validation."""
from importlib.util import find_spec
from importlib import resources
from pathlib import Path


def decoder_availability(resource_dir=None):
    root = Path(resource_dir) if resource_dir else Path(str(resources.files('stable_audio_wanderer.resources.same_s')))
    def installed(name):
        try:
            return find_spec(name) is not None
        except (ImportError, ValueError):
            return False
    onnx_deps = installed('onnxruntime')
    onnx_files = (root / 'decoder.json').is_file() and any(root.glob('*.onnx'))
    entries = [dict(backend='onnxruntime', device='cpu', label='ONNX · CPU',
                    hardware=True, dependencies=onnx_deps, weights=onnx_files,
                    validated=False, selectable=onnx_deps and onnx_files,
                    detail='Validated when Perform loads.' if onnx_deps and onnx_files else
                    'Install ONNX Runtime and the packaged SAME-S decoder.')]
    native_deps = all(installed(name) for name in ('torch', 'torchaudio', 'stable_audio_3', 'safetensors', 'huggingface_hub'))
    weights = False
    try:
        from huggingface_hub import try_to_load_from_cache
        from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION
        weights = all(isinstance(try_to_load_from_cache(SOURCE_MODEL, name, revision=SOURCE_REVISION), str)
                      for name in ('model_config.json', 'model.safetensors'))
    except Exception:
        pass
    devices = [('mps', False), ('cuda:0', False)]
    hardware_error = ''
    try:
        import torch
        devices = [('mps', bool(torch.backends.mps.is_available()))]
        devices += [(f'cuda:{i}', True) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [('cuda:0', False)]
    except Exception as exc:
        hardware_error = str(exc)
    for device, hardware in devices:
        missing = []
        if not hardware: missing.append('GPU unavailable' + (f': {hardware_error}' if hardware_error else ''))
        if not native_deps: missing.append('Install native SAME-S dependencies (see phase 2 setup)')
        if not weights: missing.append('Cache the pinned SAME-S weights (see phase 2 setup)')
        entries.append(dict(backend='pytorch', device=device,
                            label=f'PyTorch · GPU · {device.upper()}', hardware=hardware,
                            dependencies=native_deps, weights=weights, validated=False,
                            selectable=hardware and native_deps and weights,
                            detail=('; '.join(missing) or 'Present; weights, compatibility and warm-up checked on Start Perform.')
                            + (' CUDA hardware qualification pending.' if device.startswith('cuda') else '')))
    return entries
