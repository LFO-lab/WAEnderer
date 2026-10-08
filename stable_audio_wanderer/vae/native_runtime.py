"""Local interpreter/source registry. Never mix packages from different venvs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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


def worker_directory(python):
    if local_path(python, python=True) == local_path(sys.executable, python=True):
        return str(Path(__file__).resolve().parents[2])
    return None


def worker_environment(python):
    env = dict(os.environ)
    if local_path(python, python=True) == local_path(sys.executable, python=True):
        root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (root, env.get('PYTHONPATH', ''))))
    return env


def request_once(python, operation, payload, *, timeout=120, cancel_event=None):
    path = local_path(python,python=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'Native interpreter is missing or not executable: {path}')
    command = [str(path), '-m', 'stable_audio_wanderer.vae.native_worker', operation]
    if cancel_event is None:
        result = subprocess.run(command, input=json.dumps(payload), text=True,
            capture_output=True, timeout=timeout, env=worker_environment(path), cwd=worker_directory(path))
    else:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=worker_environment(path), cwd=worker_directory(path))
        deadline = time.monotonic() + timeout
        input_text = json.dumps(payload)
        try:
            while True:
                if cancel_event.is_set():
                    raise RuntimeError('Weight download cancelled')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Weight download timed out; check connectivity and retry')
                try:
                    stdout, stderr = process.communicate(input=input_text, timeout=1)
                    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                    break
                except subprocess.TimeoutExpired:
                    input_text = None
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    if result.returncode:
        raise RuntimeError(f'Native runtime {path} failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout)
