"""Explicit, source-bound decoder delivery; never collect an entire user cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile
import hashlib
import uuid

from .vae.onnx_artifacts import (
    ArtifactError, inspect_graph, local_path, read_artifact, sha256, VAE_IDS,
)

SELECTION_FORMAT = 'waenderer.model_selection.v1'
BUNDLE_FORMAT = 'waenderer.model_distribution.v1'
PRIVATE_SUFFIXES = {'.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.pyt', '.pt', '.npz', '.safetensors'}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _files(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ArtifactError('Distribution cannot contain symlinks')
        if path.is_file() and path != root / 'distribution.json':
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def validate_materials(artifact, evidence, material_root):
    """Check supplied exact-source evidence, not a model-name license guess."""
    if not isinstance(evidence, dict) or evidence.get('source') != dict(artifact.source):
        raise ArtifactError('Distribution notices must attest the exact model source')
    for field in ('license', 'source_url', 'conversion_description'):
        if not isinstance(evidence.get(field), str) or not evidence[field].strip():
            raise ArtifactError(f'Model distribution missing {field}')
    if not evidence['source_url'].startswith('https://'):
        raise ArtifactError('Model source_url must be HTTPS')
    files = evidence.get('files')
    if not isinstance(files, dict) or not files or 'MODEL_NOTICE.md' not in files:
        raise ArtifactError('Model distribution needs hashed license files and MODEL_NOTICE.md')
    if not any('LICENSE' in name.upper() for name in files):
        raise ArtifactError('Model distribution needs a full license text')
    for name, digest in files.items():
        path = local_path(material_root, name)
        if path.suffix not in ('.txt', '.md') or not path.is_file() or sha256(path) != digest:
            raise ArtifactError(f'Missing or corrupt model notice/license: {name}')
    notice = local_path(material_root, 'MODEL_NOTICE.md').read_text(encoding='utf-8')
    if artifact.vae_id in ('same_s', 'stable_audio_open'):
        if evidence['license'] != 'Stability AI Community License':
            raise ArtifactError('Stability model license evidence is incorrect')
        required = ['STABILITY_AI_COMMUNITY_LICENSE.md']
        if artifact.vae_id == 'same_s':
            required += ['GEMMA_TERMS_OF_USE.md', 'UPSTREAM_NOTICE.txt']
        if any(name not in files for name in required):
            raise ArtifactError('Stability model distribution is missing required license/notice files')
        for text in ('This Stability AI Model is licensed under the Stability AI Community License',
                     'Powered by Stability AI'):
            if text not in notice:
                raise ArtifactError('Stability model notice is incomplete')
    if artifact.vae_id == 'same_s' and 'Apache License 2.0' not in notice:
        raise ArtifactError('SAME-S notice must distinguish the model from application licensing')


def _artifact_files(artifact):
    names = {'decoder.json', *(name for name, _ in artifact.files)}
    if artifact.legacy:
        names.add('decoder_parity.json')
        # Legacy parity is outside its runtime manifest, but is a delivery requirement.
        report = _json(artifact.root / 'decoder_parity.json')
        if not report:
            raise ArtifactError('Legacy SAME-S parity report is empty')
        source = dict(artifact.source)
        if report.get('source_model') != source.get('model') or report.get('source_revision') != source.get('revision'):
            raise ArtifactError('Legacy parity report does not match the selected source')
        covered = set(map(int, report.get('windows', {})))
        supplemental = artifact.root / 'even_window_validation.json'
        if supplemental.is_file():
            validation = _json(supplemental)
            graph = artifact.root / artifact.graphs[0].path
            if validation.get('model_sha256') != sha256(graph) or validation.get('source_revision') != source.get('revision'):
                raise ArtifactError('Supplemental SAME-S window report does not match the selected graph/source')
            covered.update(row['window'] for row in validation.get('results', []) if row.get('shape_match') is True)
            names.add('even_window_validation.json')
        if not set(artifact.supported_windows).issubset(covered):
            raise ArtifactError('Legacy delivery needs validation reports for every advertised window')
    for graph in artifact.graphs:
        inspect_graph(artifact, graph)
    if any(Path(name).suffix.lower() in PRIVATE_SUFFIXES for name in names):
        raise ArtifactError('Artifact declares private audio, corpus arrays or original checkpoints')
    return sorted(names)


def validate_bundle(root):
    root = Path(root).resolve()
    inventory = _json(root / 'distribution.json')
    if inventory.get('format_version') != BUNDLE_FORMAT:
        raise ArtifactError('Unsupported model distribution format')
    if _files(root) != inventory.get('files'):
        raise ArtifactError('Distribution has missing, changed or undeclared files')
    models = inventory.get('models')
    if not isinstance(models, list) or not models:
        raise ArtifactError('Distribution must select at least one model')
    seen, closure = set(), set()
    for row in models:
        vae, identity = row['vae_id'], row['artifact_identity']
        if vae not in VAE_IDS or vae in seen:
            raise ArtifactError('Invalid or duplicate distributed VAE')
        seen.add(vae)
        relative = f'{vae}/{identity}'
        artifact = read_artifact(local_path(root, relative), expected_vae=vae)
        if artifact.identity != identity or artifact.manifest_hash != row['manifest_sha256']:
            raise ArtifactError('Distributed artifact identity changed')
        closure.update(f'{relative}/{name}' for name in _artifact_files(artifact))
        if _json(root / vae / 'current.json') != {'artifact_id': identity}:
            raise ArtifactError('Distribution pointer does not select its model')
        closure.add(f'{vae}/current.json')
        materials = local_path(root, f'notices/{vae}')
        validate_materials(artifact, row['compliance'], materials)
        closure.update(f'notices/{vae}/{name}' for name in row['compliance']['files'])
    if closure != set(inventory['files']):
        raise ArtifactError('Distribution includes files outside selected artifacts and notices')
    return inventory


def assemble(selection_file, destination):
    """Stage and verify before publishing a new directory; refuse replacement."""
    selection_file = Path(selection_file).resolve()
    selection = _json(selection_file)
    if selection.get('format_version') != SELECTION_FORMAT:
        raise ArtifactError('Unsupported model selection format')
    models = selection.get('models')
    if not isinstance(models, list) or not models:
        raise ArtifactError('Select at least one artifact explicitly')
    destination = Path(destination).expanduser().absolute()
    if destination.exists():
        raise ArtifactError('Distribution destination already exists; choose a new path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-delivery-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        rows, seen = [], set()
        for selected in models:
            vae = selected['vae_id']
            if vae not in VAE_IDS or vae in seen:
                raise ArtifactError('Invalid or duplicate selected VAE')
            seen.add(vae)
            source = Path(selected['artifact_dir']).expanduser()
            if not source.is_absolute():
                source = selection_file.parent / source
            artifact = read_artifact(source, expected_vae=vae)
            if selected.get('artifact_identity') != artifact.identity:
                raise ArtifactError('Selection must pin the artifact identity')
            target = staging / vae / artifact.identity
            target.mkdir(parents=True)
            for name in _artifact_files(artifact):
                output = local_path(target, name)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_path(artifact.root, name), output)
            evidence = dict(selected['compliance'])
            material_sources = evidence.pop('materials')
            if not isinstance(material_sources, dict):
                raise ArtifactError('Compliance materials must map notice names to paths and hashes')
            materials = staging / 'notices' / vae
            evidence['files'] = {}
            for name, entry in material_sources.items():
                output = local_path(materials, name)
                origin = Path(entry['path']).expanduser()
                if not origin.is_absolute():
                    origin = selection_file.parent / origin
                if not origin.is_file() or sha256(origin) != entry['sha256']:
                    raise ArtifactError(f'Selected notice/license changed: {name}')
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, output)
                evidence['files'][name] = entry['sha256']
            validate_materials(artifact, evidence, materials)
            _write(staging / vae / 'current.json', {'artifact_id': artifact.identity})
            rows.append(dict(vae_id=vae, artifact_identity=artifact.identity,
                             manifest_sha256=artifact.manifest_hash, compliance=evidence))
        _write(staging / 'distribution.json', dict(format_version=BUNDLE_FORMAT, models=rows, files=_files(staging)))
        inventory = validate_bundle(staging)
        staging.rename(destination)
    return inventory


def install_bundle(bundle, store, *, repair=False):
    """Validate all delivered files before selecting any immutable artifact."""
    from .runtime.decoder_preparation_jobs import store_lock
    bundle, store = Path(bundle).resolve(), Path(store).expanduser().resolve()
    inventory = validate_bundle(bundle)
    installed = []
    for row in inventory['models']:
        vae, identity = row['vae_id'], row['artifact_identity']
        with store_lock(store, vae):
            parent = store / vae
            destination = parent / identity
            with tempfile.TemporaryDirectory(prefix='.delivery-', dir=parent) as temp:
                copied = Path(temp) / identity
                shutil.copytree(bundle / vae / identity, copied)
                artifact = read_artifact(copied, expected_vae=vae)
                if artifact.identity != identity:
                    raise ArtifactError('Delivery changed during installation')
                _artifact_files(artifact)
                if destination.exists():
                    try:
                        if read_artifact(destination, expected_vae=vae).identity != identity:
                            raise ArtifactError('Existing artifact conflicts with delivered identity')
                        _artifact_files(read_artifact(destination, expected_vae=vae))
                    except (ArtifactError, OSError):
                        if not repair:
                            raise ArtifactError('Existing artifact is corrupt; stop the application and install with --repair')
                        quarantine = parent / f'.corrupt-{identity}-{uuid.uuid4().hex}'
                        destination.rename(quarantine)
                        try:
                            copied.rename(destination)
                        except Exception:
                            quarantine.rename(destination)
                            raise
                else:
                    copied.rename(destination)
                # Preserve delivery notices outside immutable runtime identities.
                receipt_id = hashlib.sha256(json.dumps(row['compliance'], sort_keys=True).encode()).hexdigest()
                notices = store / 'delivery-notices' / vae / identity / receipt_id
                if not notices.exists():
                    staged_notices = Path(temp) / 'notices'
                    shutil.copytree(bundle / 'notices' / vae, staged_notices)
                    validate_materials(artifact, row['compliance'], staged_notices)
                    _write(staged_notices / 'compliance.json', row['compliance'])
                    notices.parent.mkdir(parents=True, exist_ok=True)
                    staged_notices.rename(notices)
                else:
                    validate_materials(artifact, row['compliance'], notices)
                pointer = Path(temp) / 'current.json'
                _write(pointer, {'artifact_id': identity})
                pointer.replace(parent / 'current.json')
            installed.append(dict(vae_id=vae, artifact_identity=identity))
    return installed


def inspect_archive(path, *, model_bearing=False):
    """Inspect actual source/wheel output without extracting untrusted paths."""
    path = Path(path)
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            if len(set(archive.namelist())) != len(archive.namelist()):
                raise ArtifactError('Wheel contains duplicate entries')
            entries = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
        kind = 'wheel'
    else:
        with tarfile.open(path) as archive:
            entries = {}
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise ArtifactError('Source archive contains links')
                if member.isfile():
                    name = PurePosixPath(member.name)
                    if '..' in name.parts or name.is_absolute():
                        raise ArtifactError('Source archive contains unsafe paths')
                    relative = str(PurePosixPath(*name.parts[1:]))
                    if relative in entries:
                        raise ArtifactError('Source archive contains duplicate entries')
                    entries[relative] = archive.extractfile(member).read()
        kind = 'source'
    for name in entries:
        parts = PurePosixPath(name).parts
        if '..' in parts or name.startswith('/'):
            raise ArtifactError('Archive contains unsafe paths')
        if any(part in ('corpus', 'user_audio', 'eval_out', '.waenderer', 'jobs') or part.startswith('.venv') for part in parts):
            raise ArtifactError(f'Archive includes private inputs/state: {name}')
        if Path(name).suffix.lower() in PRIVATE_SUFFIXES:
            raise ArtifactError(f'Archive includes private audio/checkpoints/corpus: {name}')
    required = ['stable_audio_wanderer/cli/serve.py', 'stable_audio_wanderer/cli/decoder_preparation_worker.py',
                'stable_audio_wanderer/vae/onnx_artifacts.py', 'stable_audio_wanderer/model_distribution.py']
    web = 'stable_audio_wanderer/resources/web' if kind == 'wheel' else 'web'
    required += [f'{web}/index.html', f'{web}/pipeline.js', f'{web}/vendor/p5/LICENSE.txt']
    if kind == 'source':
        required += ['ROADMAP_MULTI_VAE_ONNX.md', 'docs/MULTI_VAE_ONNX_PHASE6_PLAN.md']
    if any(name not in entries for name in required):
        raise ArtifactError('Archive missing installed commands, assets or documentation')
    for license_name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md',
                         'STABILITY_AI_COMMUNITY_LICENSE.md', 'GEMMA_TERMS_OF_USE.md'):
        if not any(PurePosixPath(name).name == license_name for name in entries):
            raise ArtifactError(f'Archive missing license payload: {license_name}')
    from .release_compliance import PINNED_FILE_HASHES
    canonical_entries = {}
    for name, content in entries.items():
        if kind == 'wheel' and '.dist-info/licenses/' in name:
            name = name.split('.dist-info/licenses/', 1)[1]
        elif kind == 'wheel' and name.startswith('stable_audio_wanderer/resources/web/'):
            name = 'web/' + name.removeprefix('stable_audio_wanderer/resources/web/')
        if name in canonical_entries:
            raise ArtifactError('Archive has ambiguous license/asset entries')
        canonical_entries[name] = content
    for name, expected in PINNED_FILE_HASHES.items():
        if name not in canonical_entries or hashlib.sha256(canonical_entries[name]).hexdigest() != expected:
            raise ArtifactError(f'Archive missing or changed authoritative license/asset: {name}')
    graphs = sorted(name for name in entries if name.endswith('.onnx'))
    if model_bearing:
        # Legacy compatibility profile is deliberately SAME-S only.
        from .release_compliance import validate_model_release
        with tempfile.TemporaryDirectory(prefix='waenderer-release-check-') as temp:
            root = Path(temp)
            for name, content in entries.items():
                target_name = name
                if kind == 'wheel' and '.dist-info/licenses/' in name:
                    target_name = name.split('.dist-info/licenses/', 1)[1]
                target = local_path(root, target_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            validate_model_release(root)
        if graphs != ['stable_audio_wanderer/resources/same_s/same_s_decoder_dynamic.onnx']:
            raise ArtifactError('Legacy model-bearing archive includes unexpected models')
    elif graphs or any(PurePosixPath(name).name in ('decoder.json', 'decoder_parity.json') or name.endswith('.data') for name in entries):
        raise ArtifactError('Ordinary archive contains an undeclared model payload')
    return dict(kind=kind, sha256=sha256(path), file_count=len(entries), models=graphs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('assemble')
    build.add_argument('--selection', required=True)
    build.add_argument('--output', required=True)
    check = commands.add_parser('check')
    check.add_argument('bundle')
    install = commands.add_parser('install')
    install.add_argument('bundle')
    install.add_argument('--store-dir', required=True)
    install.add_argument('--repair', action='store_true', help='Restore corrupt copies from this verified bundle; stop the application first')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('archive')
    inspect.add_argument('--model-bearing', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'assemble':
            result = assemble(args.selection, args.output)
        elif args.command == 'check':
            result = validate_bundle(args.bundle)
        elif args.command == 'install':
            result = install_bundle(args.bundle, args.store_dir, repair=args.repair)
        else:
            result = inspect_archive(args.archive, model_bearing=args.model_bearing)
    except (ArtifactError, ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f'Delivery failed: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
