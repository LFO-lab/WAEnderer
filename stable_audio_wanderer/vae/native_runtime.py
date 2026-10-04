"""Local interpreter/source registry. Never mix packages from different venvs."""
import json
import os
from pathlib import Path
import subprocess
import sys

def configuration_directory():
    if os.name == 'nt':
        return Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'waenderer'
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/waenderer'
    return Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'waenderer'


REGISTRY = configuration_directory() / 'native_runtimes.json'



def local_path(value, *, python=False, base=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    # Resolving a venv Python symlink selects the base interpreter instead.
    return path.absolute() if python else path.resolve()


def native_config(vae_id, config):
    path = Path(os.environ.get('WAENDERER_NATIVE_RUNTIMES', str(REGISTRY))).expanduser().absolute()
    registry = json.loads(path.read_text()) if path.is_file() else {'schema_version':1,'models':{}}
    if registry.get('schema_version') != 1:
        raise ValueError('Unsupported native runtime registry schema')
    entries = registry['models']
    entry = entries.get(vae_id, {})
    result = dict(config)
    # Interpreter paths belong to local server configuration, never Web requests.
    result.pop('decoder_python',None)
    result['decoder_python'] = str(local_path(entry.get('python',sys.executable),python=True,base=path.parent))
    for key, field in [('vae_weight_path','weights'),
                       ('vae_repo_path','repo'),('vae_config_path','config')]:
        if not result.get(key) and entry.get(field):
            result[key] = str(local_path(entry[field],base=path.parent))
    return result


def request_once(python, operation, payload):
    path = local_path(python,python=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'Native interpreter is missing or not executable: {path}')
    result = subprocess.run([str(path), '-m', 'stable_audio_wanderer.vae.native_worker', operation],
        input=json.dumps(payload), text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f'Native runtime {path} failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout)
