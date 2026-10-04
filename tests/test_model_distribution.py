"""Delivery checks use tiny real graphs, without private models or audio."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from test_onnx_artifacts import fixture
from stable_audio_wanderer.model_distribution import (
    SELECTION_FORMAT, assemble, install_bundle, validate_bundle, inspect_archive,
)
from stable_audio_wanderer.vae.onnx_artifacts import ArtifactError, read_artifact, resolve_artifact, sha256


def selection(tmp_path, *, dynamic=True):
    root = tmp_path / 'source'
    fixture(root, dynamic=dynamic, external=True)
    artifact = read_artifact(root)
    notices = tmp_path / 'materials'
    notices.mkdir()
    # Test source only: this asserts no license for a production model.
    texts = {'MODEL_NOTICE.md': 'Synthetic test model; license: test-only',
             'TEST_LICENSE.txt': 'Synthetic fixture license text',
             'STABILITY_AI_COMMUNITY_LICENSE.md': 'test-only'}
    texts['MODEL_NOTICE.md'] += '\nThis Stability AI Model is licensed under the Stability AI Community License\nPowered by Stability AI\n'
    for name, value in texts.items():
        (notices / name).write_text(value)
    config = dict(format_version=SELECTION_FORMAT, models=[dict(
        vae_id=artifact.vae_id, artifact_identity=artifact.identity, artifact_dir='source',
        compliance=dict(source=dict(artifact.source), license='Stability AI Community License',
            source_url='https://example.org/synthetic-only', conversion_description='Tiny graph fixture',
            materials={name:dict(path=f'materials/{name}', sha256=sha256(notices / name)) for name in texts}))])
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(config))
    return path, artifact


@pytest.mark.parametrize('dynamic', [True, False])
def test_complete_external_delivery_and_installed_reuse(tmp_path, dynamic):
    config, artifact = selection(tmp_path, dynamic=dynamic)
    bundle = tmp_path / 'bundle'
    inventory = assemble(config, bundle)
    assert inventory == validate_bundle(bundle)
    assert any(name.endswith('.data') for name in inventory['files'])
    assert all('source' not in name for name in inventory['files'])
    store = tmp_path / 'store'
    install_bundle(bundle, store)
    install_bundle(bundle, store)  # Immutable reuse, not replacement.
    assert resolve_artifact(artifact.vae_id, store_dir=store).identity == artifact.identity
    external = next((bundle / artifact.vae_id / artifact.identity).glob('*.data'))
    external.write_bytes(b'corrupt')
    with pytest.raises(ArtifactError, match='changed'):
        install_bundle(bundle, store)
    assert resolve_artifact(artifact.vae_id, store_dir=store).identity == artifact.identity


def test_explicit_repair_retains_corrupt_copy_and_notices(tmp_path):
    config, artifact = selection(tmp_path)
    bundle, store = tmp_path / 'bundle', tmp_path / 'store'
    assemble(config, bundle)
    install_bundle(bundle, store)
    graph = store / artifact.vae_id / artifact.identity / artifact.graphs[0].path
    graph.write_bytes(b'corrupt')
    with pytest.raises(ArtifactError, match='--repair'):
        install_bundle(bundle, store)
    install_bundle(bundle, store, repair=True)
    assert resolve_artifact(artifact.vae_id, store_dir=store).identity == artifact.identity
    assert len(list((store / artifact.vae_id).glob('.corrupt-*'))) == 1
    assert list((store / 'delivery-notices').rglob('MODEL_NOTICE.md'))


@pytest.mark.parametrize('change', ['source', 'notice', 'identity', 'extra', 'missing'])
def test_delivery_rejects_incomplete_or_changed_selection(tmp_path, change):
    config, artifact = selection(tmp_path)
    data = json.loads(config.read_text())
    if change == 'source':
        data['models'][0]['compliance']['source']['weights_sha256'] = '3' * 64
    elif change == 'notice':
        data['models'][0]['compliance']['materials'].pop('MODEL_NOTICE.md')
    elif change == 'identity':
        data['models'][0]['artifact_identity'] = '0' * 64
    elif change == 'missing':
        (artifact.root / artifact.graphs[0].path).unlink()
    config.write_text(json.dumps(data))
    bundle = tmp_path / 'bundle'
    if change == 'extra':
        assemble(config, bundle)
        (bundle / 'private.wav').write_bytes(b'private')
        with pytest.raises(ArtifactError, match='undeclared'):
            validate_bundle(bundle)
    else:
        with pytest.raises(ArtifactError):
            assemble(config, bundle)
        assert not bundle.exists()


def test_worker_structured_missing_model_response(tmp_path):
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'vae_id': 'ear_vae_44k'}))
    # The worker has a useful structured dependency failure without weights.
    result = subprocess.run([sys.executable, '-m',
        'stable_audio_wanderer.cli.decoder_preparation_worker', '--request', str(request),
        '--output', str(tmp_path / 'out')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'WAENDERER_RESULT ' in result.stdout


def test_legacy_delivery_keeps_graph_bound_supplemental_window_report(tmp_path, monkeypatch):
    from stable_audio_wanderer.model_distribution import _artifact_files
    monkeypatch.setattr('stable_audio_wanderer.model_distribution.inspect_graph', lambda *a: None)
    (tmp_path / 'model.onnx').write_bytes(b'test graph')
    source = {'model': 'synthetic-test-only', 'revision': 'a' * 40}
    artifact = SimpleNamespace(root=tmp_path, legacy=True, files=(('model.onnx', sha256(tmp_path / 'model.onnx')),),
        source=tuple(source.items()), graphs=(SimpleNamespace(path='model.onnx'),), supported_windows=(2, 4))
    (tmp_path / 'decoder_parity.json').write_text(json.dumps(dict(source_model=source['model'],
        source_revision=source['revision'], windows={'2': {}})))
    with pytest.raises(ArtifactError, match='every advertised window'):
        _artifact_files(artifact)
    supplement = dict(model_sha256=sha256(tmp_path / 'model.onnx'), source_revision=source['revision'],
        results=[dict(window=4, shape_match=True)])
    (tmp_path / 'even_window_validation.json').write_text(json.dumps(supplement))
    assert 'even_window_validation.json' in _artifact_files(artifact)
    supplement['model_sha256'] = '0' * 64
    (tmp_path / 'even_window_validation.json').write_text(json.dumps(supplement))
    with pytest.raises(ArtifactError, match='selected graph/source'):
        _artifact_files(artifact)


def test_actual_wheel_inspection_checks_license_hashes_and_private_files(tmp_path):
    from stable_audio_wanderer.release_compliance import PINNED_FILE_HASHES
    project = Path(__file__).resolve().parents[1]
    files = {name: (project / name).read_bytes() for name in PINNED_FILE_HASHES}
    for name in ('NOTICE', 'THIRD_PARTY_NOTICES.md', 'stable_audio_wanderer/cli/serve.py',
                 'stable_audio_wanderer/cli/decoder_preparation_worker.py',
                 'stable_audio_wanderer/vae/onnx_artifacts.py', 'stable_audio_wanderer/model_distribution.py',
                 'web/index.html', 'web/pipeline.js'):
        files[name] = (project / name).read_bytes()
    def write(extra=None):
        wheel = tmp_path / 'test.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name, data in {**files, **(extra or {})}.items():
                if name.startswith('web/'):
                    name = 'stable_audio_wanderer/resources/' + name
                elif name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md') or name.startswith('licenses/'):
                    name = 'test.dist-info/licenses/' + name
                archive.writestr(name, data)
        return wheel
    assert inspect_archive(write())['models'] == []
    with pytest.raises(ArtifactError, match='private'):
        inspect_archive(write({'private.wav': b'audio'}))
    with pytest.raises(ArtifactError, match='authoritative'):
        inspect_archive(write({'LICENSE': b'changed'}))
