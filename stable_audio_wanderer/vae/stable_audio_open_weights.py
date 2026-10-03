"""Immutable Stable Audio Open source shared by native decoding and export."""
from pathlib import Path
from .onnx_artifacts import sha256

SOURCE_MODEL = 'stabilityai/stable-audio-open-1.0'
SOURCE_REVISION = 'f21265c1e2710b3bd2386596943f0007f55f802e'
CONFIG_SHA256 = '858b6cce27aa1b2d1e8d331734d2d9690d9ea1dad2fb1e6bd23ee67a61058ad9'
WEIGHTS_SHA256 = '2131cdb52020b2473707465449d8bdb4f6cca61c93150a947baca02bc58ffd7b'


def resolve_source(repo_or_path=SOURCE_MODEL, *, revision=SOURCE_REVISION, local_files_only=True):
    if revision != SOURCE_REVISION:
        raise ValueError('Stable Audio Open requires the verified immutable revision ' + SOURCE_REVISION)
    root = Path(repo_or_path).expanduser()
    names = ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors')
    if root.is_dir():
        files = [root / name for name in names]
    else:
        if str(repo_or_path) != SOURCE_MODEL:
            raise ValueError('Unsupported Stable Audio Open source')
        from huggingface_hub import hf_hub_download
        files = [Path(hf_hub_download(SOURCE_MODEL, name, revision=revision,
                  local_files_only=local_files_only)) for name in names]
    if files[0].parent.parent != files[1].parent.parent:
        raise ValueError('Stable Audio Open config and weights must come from one cached revision')
    actual = [sha256(p) for p in files]
    if actual != [CONFIG_SHA256, WEIGHTS_SHA256]:
        raise ValueError('Stable Audio Open source checkpoint/config SHA-256 mismatch')
    return files[0].parent.parent, dict(model=SOURCE_MODEL, revision=revision,
        config_sha256=actual[0], weights_sha256=actual[1])
