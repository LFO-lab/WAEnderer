"""Local EAR source identity and strict, variant-specific checkpoint resolution."""
from dataclasses import dataclass
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from .onnx_artifacts import canonical, sha256

VARIANTS = {
    'ear_vae_44k': dict(sample_rate=44100, ratio=1024, config='model_config.json',
        weights_sha256='0362dc7e96566869747dbe079b0a6d71c090b0a3a5d5077779e7be17c096d9d5',
        config_sha256='ef19bcb81717fe1c924316419fc5a7040651e8e3ca47b6e17e29241b6bb36303'),
    'ear_vae_48k': dict(sample_rate=48000, ratio=960, config='ear_vae_v2.json',
        weights_sha256='cc53f3f043aaae01307ffe1439b3a1c9d9e019652e6c13e089cd18d6037c2d3f',
        config_sha256='7797e3d5e7560487690987a0a52e3b2c715ba44dad6caca61ff8d920b93b8ba8'),
}
# Hashes of the user-supplied files; custom checkpoints require an explicit config.
RECONCILIATION = 'checkpoint-transformer-keys-v1'
_IMPORTED_CODE = None


@dataclass
class EarSource:
    vae_id: str
    repo: Path
    weights: Path
    config_path: Path
    config: dict
    state: dict
    identity: dict
    code_files: dict


def code_files(repo):
    root = Path(repo)
    files = {p.relative_to(root).as_posix():sha256(p) for p in sorted((root/'model').rglob('*.py'))}
    for name in ('model/ear_vae.py', 'model/autoencoders.py', 'model/transformer.py'):
        if name not in files:
            raise ValueError(f'EAR repository is missing {name}')
    return files


def code_digest(files):
    return hashlib.sha256(canonical(files)).hexdigest()


def source_paths(vae_id, weight_path, repo_path='', config_path=''):
    if vae_id not in VARIANTS:
        raise ValueError(f'Unknown EAR variant {vae_id!r}')
    weights = Path(weight_path).expanduser().resolve()
    if not weight_path or not weights.is_file():
        raise ValueError('EAR VAE requires an existing .pyt weight file')
    repo = Path(repo_path).expanduser().resolve() if repo_path else next(
        (p for p in list(weights.parents)[:4] if (p/'model').is_dir()), None)
    if repo is None or not (repo/'model').is_dir():
        raise ValueError('EAR repository not found; set vae_repo_path or --repo explicitly')
    config = Path(config_path).expanduser().resolve() if config_path else repo/'config'/VARIANTS[vae_id]['config']
    if not config.is_file():
        raise ValueError(f'EAR configuration missing: {config}')
    return weights, repo, config


def file_identity(vae_id, weight_path, repo_path='', config_path=''):
    """Bind explicitly supplied native files without importing model libraries."""
    weights, repo, config = source_paths(vae_id, weight_path, repo_path, config_path)
    return dict(weights_sha256=sha256(weights), config_sha256=sha256(config),
                code_sha256=code_digest(code_files(repo)))


def selected_file_identity(vae_id, weight_path, repo_path='', config_path=''):
    """Bind known local sources; standalone weight files need no native repo."""
    weights = Path(weight_path).expanduser().resolve()
    if repo_path or config_path or any((p/'model').is_dir() for p in list(weights.parents)[:4]):
        return file_identity(vae_id, weights, repo_path, config_path)
    return {'weights_sha256':sha256(weights)}


def resolve_source(vae_id, weight_path, repo_path='', config_path='', expected_source=None):
    import torch
    weights, repo, config_path_resolved = source_paths(vae_id, weight_path, repo_path, config_path)
    weight_hash, config_hash = sha256(weights), sha256(config_path_resolved)
    known = next((name for name, v in VARIANTS.items() if v['weights_sha256']==weight_hash), None)
    if known is not None and known != vae_id:
        raise ValueError('EAR checkpoint belongs to a different sample-rate variant')
    if known is not None and config_hash != VARIANTS[known]['config_sha256']:
        raise ValueError('Supplied EAR checkpoint/config SHA-256 mismatch')
    if known is None and not config_path:
        raise ValueError('Unrecognized EAR checkpoint: select its matching config explicitly')
    config = json.loads(config_path_resolved.read_text())
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('EAR checkpoint must be an unwrapped tensor state dictionary')
    effective = copy.deepcopy(config)
    has_transformer = any(k.startswith('transformers.') for k in state)
    if has_transformer and not effective.get('transformer'):
        raise ValueError('EAR checkpoint transformer is missing from config')
    if not has_transformer:
        effective['transformer'] = None
    variant = VARIANTS[vae_id]
    enc, dec = effective['encoder']['config'], effective['decoder']['config']
    if (math.prod(enc['strides']) != variant['ratio'] or math.prod(dec['strides']) != variant['ratio']
            or dec['latent_dim'] != 64 or dec['out_channels'] != 2):
        raise ValueError('EAR configuration geometry does not match selected variant')
    files = code_files(repo)
    revision = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or 'unversioned-local-source'
    identity = dict(model='earlab/EAR_VAE', revision=revision, weights_sha256=weight_hash,
        config_sha256=config_hash, effective_config_sha256=hashlib.sha256(canonical(effective)).hexdigest(),
        code_sha256=code_digest(files), reconciliation=RECONCILIATION,
        transformer='present' if has_transformer else 'absent')
    if expected_source is not None and (not isinstance(expected_source,dict) or
            any(identity.get(k)!=v for k,v in expected_source.items())):
        raise ValueError('Stale EAR source checkpoint/config/code identity')
    return EarSource(vae_id,repo,weights,config_path_resolved,effective,state,identity,files)


def load_model_class(source):
    """Prevent generic 'model' imports from selecting a different source tree."""
    global _IMPORTED_CODE
    current = code_digest(code_files(source.repo))
    if current != source.identity['code_sha256']:
        raise ValueError('EAR source code changed after selection')
    root = source.repo/'model'
    for name,module in tuple(sys.modules.items()):
        if name=='model' or name.startswith('model.'):
            paths = ([getattr(module,'__file__',None)] if getattr(module,'__file__',None)
                     else list(getattr(module,'__path__',[])))
            if not paths or any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise ValueError('Conflicting EAR model import; restart with the selected repository')
    if _IMPORTED_CODE is not None and _IMPORTED_CODE != (str(root),current):
        raise ValueError('EAR imported source changed; restart before native loading')
    if str(source.repo) not in sys.path:
        sys.path.insert(0,str(source.repo))
    try:
        model_class = importlib.import_module('model.ear_vae').EAR_VAE
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split('.')[0]=='dac':
            raise ImportError('EAR VAE requires descript-audio-codec in the native/export environment') from exc
        raise
    _IMPORTED_CODE = (str(root),current)
    return model_class
