"""Short installed/offline decoder checks; no audio device or listening claim."""
import argparse
import importlib.abc
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal
import socket
import time
from urllib.request import urlopen


def offline_boundary():
    class RuntimeOnly(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'stable_audio_3', 'dac', 'audiotools',
                    'diffusers', 'transformers', 'accelerate', 'onnxscript'}:
                raise ModuleNotFoundError('Native/export library disabled by delivery check: ' + fullname)
    sys.meta_path.insert(0, RuntimeOnly())
    def audit(event, args):
        if event == 'socket.getaddrinfo' and args[0] in ('', '0.0.0.0', '127.0.0.1', 'localhost', None):
            return  # Listener address resolution is allowed; outgoing connections are not.
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('Network disabled by delivery check')
    sys.addaudithook(audit)


def probe(vae, store):
    offline_boundary()
    import numpy as np
    from .vae.artifact_decoder import load_artifact_decoder
    from .vae.decoder_preparation import reusable
    from .vae.onnx_artifacts import resolve_artifact
    from .vae.decoder_availability import decoder_availability
    artifact = resolve_artifact(vae, store_dir=store)
    decoder = load_artifact_decoder(artifact)
    window = min(artifact.supported_windows)
    try:
        output = decoder.decode(np.zeros((window, artifact.latent_dim), np.float32))
        assert output.audio.shape == (window * artifact.samples_per_latent, artifact.channels)
        assert output.audio.dtype == np.float32 and np.isfinite(output.audio).all()
        assert decoder.info.provider == 'CPUExecutionProvider'
    finally:
        decoder.close()
    identity = artifact.identity
    reused = reusable({'vae_id': vae, 'store_dir': store})
    assert reused is not None and reused.identity == identity
    choices = decoder_availability(corpus_spec={'vae_id': vae}, store_dir=store)
    onnx = next(row for row in choices if row['backend'] == 'onnxruntime')
    native = next(row for row in choices if row['backend'] == 'pytorch' and row['device'] == 'cpu')
    assert onnx['selectable'] and native['vae_id'] == vae
    if vae.startswith('ear_'):
        assert not native['selectable'] and 'missing_weights' in native['reason_codes']
    return dict(vae_id=vae, artifact_identity=identity, window=window,
                output_shape=list(output.audio.shape), dtype='float32', finite=True,
                provider='CPUExecutionProvider', reuse='already_valid', offline=True,
                native_model_imports=False, export_imports=False,
                native_cpu_visible=True, native_cpu_selectable=native['selectable'],
                native_cpu_reasons=native['reason_codes'])


def check_server(directory, env):
    """Boot the installed server without an audio device, then fetch its assets."""
    def port():
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            return listener.getsockname()[1]
    http_port, ws_port = port(), port()
    code = ('from stable_audio_wanderer.delivery_checks import offline_boundary\n'
            'offline_boundary()\nfrom stable_audio_wanderer.cli.serve import main\nmain()\n')
    with (Path(directory) / 'server.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-c', code, '--http_port', str(http_port),
            '--port', str(ws_port)], cwd=directory, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 25
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError('Installed server failed: ' + log.read()[-4000:])
                try:
                    with urlopen(f'http://127.0.0.1:{http_port}/index.html', timeout=.5) as response:
                        assert b'pipeline.js' in response.read()
                    with socket.create_connection(('127.0.0.1', ws_port), timeout=.5):
                        pass
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Installed server did not serve assets within 25 seconds')
                    time.sleep(.1)
            for asset in ('pipeline.js', 'vendor/p5/LICENSE.txt'):
                with urlopen(f'http://127.0.0.1:{http_port}/{asset}', timeout=2) as response:
                    assert response.status == 200 and response.read()
            # The server has reached its main loop before receiving Stop.
            time.sleep(.1)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode != 0:
            log.seek(0)
            raise RuntimeError('Installed server did not stop cleanly: ' + log.read()[-4000:])


def check_delivery(bundle):
    import stable_audio_wanderer
    location = Path(stable_audio_wanderer.__file__).resolve()
    if 'site-packages' not in location.parts:
        raise RuntimeError('Run delivery checks from a non-editable wheel installation')
    from .assets import web_directory
    from .model_distribution import install_bundle, validate_bundle
    inventory = validate_bundle(bundle)
    packages = {dist.metadata['Name']: dist.version for dist in metadata.distributions()}
    for name in ('stable-audio-3', 'descript-audio-codec', 'descript-audiotools', 'onnxscript',
                 'diffusers', 'transformers', 'accelerate', 'safetensors'):
        if name in {key.lower().replace('_', '-'): value for key, value in packages.items()}:
            raise RuntimeError(f'Clean runtime environment unexpectedly contains {name}')
    results = []
    with tempfile.TemporaryDirectory(prefix='waenderer-installed-check-') as temp:
        store = Path(temp) / 'decoders'
        install_bundle(bundle, store)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        for row in inventory['models']:
            for restart in (False, True):
                completed = subprocess.run([sys.executable, '-m', 'stable_audio_wanderer.delivery_checks',
                    '--probe', row['vae_id'], '--store-dir', str(store)], cwd=temp,
                    env=env, capture_output=True, text=True, timeout=120)
                if completed.returncode:
                    raise RuntimeError(completed.stderr[-4000:])
                result = json.loads(completed.stdout)
                assert result['artifact_identity'] == row['artifact_identity']
                result['fresh_restart'] = restart
                results.append(result)
        startup = subprocess.run([sys.executable, '-c',
            'from stable_audio_wanderer.delivery_checks import offline_boundary\n'
            'offline_boundary()\n'
            'from stable_audio_wanderer.runtime.pipeline_server import PipelineManager\n'
            'from stable_audio_wanderer.cli.serve import main\n'], cwd=temp,
            env=env, capture_output=True, text=True, timeout=30)
        if startup.returncode:
            raise RuntimeError(startup.stderr[-4000:])
        # The installed CLI checks all windows without requiring an exporter.
        for row in inventory['models']:
            cli = subprocess.run([sys.executable, '-m', 'stable_audio_wanderer.cli.prepare_decoders',
                '--vae-id', row['vae_id'], '--store-dir', str(store), '--json'], cwd=temp,
                env=env, capture_output=True, text=True, timeout=120)
            if cli.returncode:
                raise RuntimeError(cli.stderr[-2000:] + cli.stdout[-2000:])
            assert all(item['status'] == 'already_valid' for item in json.loads(cli.stdout))
        request = Path(temp) / 'request.json'
        request.write_text(json.dumps({'vae_id': 'ear_vae_44k'}))
        worker = subprocess.run([sys.executable, '-m', 'stable_audio_wanderer.cli.decoder_preparation_worker',
            '--request', str(request), '--output', str(Path(temp) / 'output')], cwd=temp,
            env=env, capture_output=True, text=True, timeout=30)
        assert worker.returncode == 0 and '"status": "missing_dependencies"' in worker.stdout
        check_server(temp, env)
    return dict(format_version='waenderer.delivery_checks.v1', python=sys.version.split()[0],
                platform=sys.platform, installed_from_wheel=True, packages=packages,
                web_assets_present=(web_directory() / 'vendor/p5/p5.min.js').is_file(),
                offline_application_import=True, offline_server_http=True,
                installed_worker_preflight=True, cli_reuse=True, probes=results,
                scope='Functional synthetic PCM and reuse; no parity, listening or real-time qualification')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    parser.add_argument('--output')
    parser.add_argument('--probe', choices=['same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k'], help=argparse.SUPPRESS)
    parser.add_argument('--store-dir', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        result = probe(args.probe, args.store_dir)
    elif args.bundle:
        result = check_delivery(args.bundle)
    else:
        parser.error('--bundle required')
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
