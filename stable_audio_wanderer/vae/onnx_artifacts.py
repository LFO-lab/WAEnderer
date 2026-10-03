"""Versioned ONNX artifacts, offline resolution and atomic validated publication."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from importlib import resources

FORMAT = 'waenderer.onnx_decoder.v1'
PROVIDER = 'CPUExecutionProvider'
EXECUTION = ('cpu', 'float32', 'intra_threads=1', 'inter_threads=1', 'optimization=basic', 'session_cache=2')
VAE_IDS = ('same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k')


class ArtifactError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ArtifactError(f'{name} must be a positive integer')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f'{name} must be a nonempty string')
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ArtifactError('A lowercase SHA-256 digest is required')
    return value


def local_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ArtifactError('Artifact path must be a relative POSIX path')
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or str(rel) != name or name == '.':
        raise ArtifactError(f'Unsafe artifact path: {name!r}')
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError(f'Artifact path escapes root: {name}')
    return path


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as exc:
        raise ArtifactError(f'Invalid or missing metadata {path}: {exc}') from exc


@dataclass(frozen=True)
class Graph:
    path: str
    windows: tuple[int, ...]
    dynamic: bool


@dataclass(frozen=True)
class Artifact:
    root: Path
    identity: str
    manifest_hash: str
    vae_id: str
    legacy: bool
    opset: int | None
    sample_rate: int
    channels: int
    latent_dim: int
    samples_per_latent: int
    corpus_latent_hz: tuple[float, ...]
    supported_windows: tuple[int, ...]
    default_window: int
    input_name: str
    output_name: str
    input_layout: str
    output_layout: str
    graphs: tuple[Graph, ...]
    files: tuple[tuple[str, str], ...]
    source: tuple[tuple[str, str], ...]

    @property
    def cache_key(self):
        return (str(self.root), self.identity, self.manifest_hash, *EXECUTION)

    def validate_corpus(self, spec):
        if spec['vae_id'] != self.vae_id:
            raise ArtifactError('Corpus VAE does not match ONNX artifact')
        # SAME-S legacy corpus validator has already established its dimension.
        if spec.get('latent_dim', 256 if self.vae_id == 'same_s' else None) != self.latent_dim or spec['sample_rate'] != self.sample_rate:
            raise ArtifactError('Corpus geometry does not match ONNX artifact')
        if not any(math.isclose(spec['latent_hz'], rate, rel_tol=0, abs_tol=1e-7) for rate in self.corpus_latent_hz):
            raise ArtifactError('Corpus latent rate does not match ONNX artifact')
        for name in ('config_sha256', 'weights_sha256', 'revision', 'effective_config_sha256', 'code_sha256'):
            if name in spec and spec[name] != dict(self.source).get(name):
                raise ArtifactError(f'Corpus source {name} does not match ONNX artifact')

    def verify_files(self):
        if sha256(self.root/'decoder.json') != self.manifest_hash:
            raise ArtifactError('Artifact manifest changed after selection')
        for name, expected in self.files:
            path = local_path(self.root, name)
            if not path.is_file() or sha256(path) != expected:
                raise ArtifactError(f'Artifact file missing or SHA-256 mismatch: {name}')


def read_artifact(root, *, expected_vae=None, expected_source=None):
    root = Path(root).expanduser().resolve()
    description = read_json(root/'decoder.json')
    vae_id = description.get('vae_id')
    if vae_id not in VAE_IDS or (expected_vae is not None and vae_id != expected_vae):
        raise ArtifactError(f'Wrong artifact VAE: {vae_id!r}; expected {expected_vae!r}')
    legacy = description.get('format_version') == 'same_s.web_decoder.v1'
    if legacy:
        expected = dict(vae_id='same_s', backend='onnxruntime', provider=PROVIDER,
            sample_rate=44100, channels=2, latent_dim=256, samples_per_latent=4096, ola_mode='full_overlap_add')
        if any(description.get(k) != v for k,v in expected.items()):
            raise ArtifactError('Invalid legacy SAME-S geometry or backend')
        name = description.get('model')
        if not isinstance(name,str) or Path(name).name != name or not name.endswith('.onnx'):
            raise ArtifactError('Legacy model must be a local .onnx filename')
        model = local_path(root,name)
        if not model.is_file(): raise ArtifactError('Legacy model is missing')
        actual = sha256(model)
        if description.get('model_sha256', actual) != actual:
            raise ArtifactError('Legacy model SHA-256 mismatch')
        files = {name:actual}
        source = {target:description[key] for key,target in (('source_model','model'),('source_revision','revision')) if key in description}
        from .same_s_weights import SOURCE_MODEL, SOURCE_REVISION, CONFIG_SHA256, WEIGHTS_SHA256
        if source.get('model')==SOURCE_MODEL and source.get('revision')==SOURCE_REVISION:
            source.update(config_sha256=CONFIG_SHA256,weights_sha256=WEIGHTS_SHA256)
        opset = None
        input_layout, output_layout = 'BDT', 'BCT'
        rates = (44100/4096,)
    else:
        if description.get('format_version') != FORMAT or description.get('backend') != 'onnxruntime' or description.get('provider') != PROVIDER:
            raise ArtifactError('Unsupported ONNX artifact format/backend/provider')
        if description.get('ola_mode') != 'full_overlap_add':
            raise ArtifactError('Unsupported overlap-add mode')
        source = description.get('source')
        if not isinstance(source,dict): raise ArtifactError('Source identity is required')
        _text(source.get('model'), 'source model')
        for name in ('config_sha256','weights_sha256'): _digest(source.get(name))
        if 'revision' in source: _text(source['revision'], 'source revision')
        if any(not isinstance(v,str) for v in source.values()): raise ArtifactError('Source values must be strings')
        export = description.get('export',{})
        opset = _positive(export.get('opset'), 'opset')
        versions = export.get('tool_versions')
        if not isinstance(versions,dict) or not versions or any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in versions.items()):
            raise ArtifactError('Export tool versions are required')
        files = description.get('files')
        if not isinstance(files,dict) or not files: raise ArtifactError('Hashed artifact files are required')
        for name,digest in files.items():
            local_path(root,name)
            _digest(digest)
            if name=='decoder.json': raise ArtifactError('Manifest cannot hash itself')
        input_layout, output_layout = description.get('input_layout'), description.get('output_layout')
        if input_layout not in ('BDT','BTD') or output_layout not in ('BCT','BTC'):
            raise ArtifactError('Unsupported tensor layout')
        rates = description.get('corpus_latent_hz')
        if not isinstance(rates,list) or not rates or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in rates):
            raise ArtifactError('Valid corpus latent rates are required')
    if expected_source is not None and not isinstance(expected_source,dict):
        raise ArtifactError('Expected source identity must be an object')
    if expected_source and any(source.get(k)!=v for k,v in expected_source.items()):
        raise ArtifactError('Stale source checkpoint/config identity')
    sr, channels, dim, ratio = (_positive(description.get(k),k) for k in ('sample_rate','channels','latent_dim','samples_per_latent'))
    if channels != 2: raise ArtifactError('The Web PCM transport requires stereo')
    if any(not math.isclose(rate, sr/ratio, rel_tol=.002, abs_tol=0) for rate in rates):
        raise ArtifactError('Corpus rate is inconsistent with measured PCM geometry')
    windows = description.get('supported_windows')
    if not isinstance(windows,list) or not windows or any(type(w) is not int or w<2 or w%2 for w in windows) or windows != sorted(set(windows)):
        raise ArtifactError('Windows must be sorted distinct positive even integers')
    if legacy and any(w>32 for w in windows): raise ArtifactError('Unsupported legacy SAME-S window')
    default = description.get('default_window')
    if type(default) is not int or default not in windows: raise ArtifactError('Invalid default window')
    input_name, output_name = (_text(description.get(k),k) for k in ('input_name','output_name'))
    graphs = []
    if legacy:
        graphs.append(Graph(name,tuple(windows),True))
    else:
        entries = description.get('graphs')
        if not isinstance(entries,list) or not entries: raise ArtifactError('Graph definitions are required')
        covered=[]
        for entry in entries:
            if not isinstance(entry,dict): raise ArtifactError('Invalid graph definition')
            path = entry.get('path')
            if not isinstance(path,str) or path not in files or not path.endswith('.onnx'): raise ArtifactError('Graph must be declared and hashed')
            gw = entry.get('windows')
            dynamic = entry.get('dynamic')
            if type(dynamic) is not bool or not isinstance(gw,list) or not gw or any(type(w) is not int or w not in windows for w in gw) or (not dynamic and len(gw)!=1):
                raise ArtifactError('Invalid graph window map')
            covered.extend(gw)
            graphs.append(Graph(path,tuple(gw),dynamic))
        if sorted(covered)!=windows or len({g.path for g in graphs})!=len(graphs):
            raise ArtifactError('Graphs must cover each supported window exactly once')
        report_name=description.get('validation_report')
        if report_name not in files: raise ArtifactError('Hashed validation report is required')
    identity=hashlib.sha256(canonical({'manifest':description,'files':files})).hexdigest()
    artifact=Artifact(root,identity,sha256(root/'decoder.json'),vae_id,legacy,opset,sr,channels,dim,ratio,
        tuple(rates),tuple(windows),default,input_name,output_name,input_layout,output_layout,
        tuple(graphs),tuple(sorted(files.items())),tuple(sorted(source.items())))
    artifact.verify_files()
    if not legacy:
        report=read_json(local_path(root,description['validation_report']))
        if report.get('passed') is not True or report.get('source')!=source or report.get('graphs')!={g.path:files[g.path] for g in graphs}:
            raise ArtifactError('Validation report does not attest these source/graph identities')
    return artifact


def inspect_graph(artifact, graph):
    """Parse without loading external tensors, before ORT can follow any paths."""
    import onnx
    model=onnx.load_model_from_string(local_path(artifact.root,graph.path).read_bytes())
    if not artifact.legacy and [entry.version for entry in model.opset_import if entry.domain in ('','ai.onnx')] != [artifact.opset]:
        raise ArtifactError('Graph opset does not match artifact metadata')
    declared=dict(artifact.files)
    def tensors(message):
        if message.DESCRIPTOR.full_name=='onnx.TensorProto': yield message
        for field,value in message.ListFields():
            if field.type==field.TYPE_MESSAGE:
                repeated = getattr(field, "is_repeated", None)
                if repeated is None:  # protobuf 4 (EAR) predates is_repeated.
                    repeated = field.label == field.LABEL_REPEATED
                for child in value if repeated else (value,):
                    yield from tensors(child)
    for tensor in tensors(model):
        if tensor.data_location!=onnx.TensorProto.EXTERNAL and not tensor.external_data: continue
        entries={item.key:item.value for item in tensor.external_data}
        if len(entries)!=len(tensor.external_data) or set(entries)-{'location','offset','length','checksum'}:
            raise ArtifactError('Invalid ONNX external-data metadata')
        location=entries.get('location')
        # Resolve relative to the graph, but disallow traversal even within root.
        local_path(artifact.root,location)
        name=str(PurePosixPath(graph.path).parent / location)
        if name not in declared: raise ArtifactError(f'Undeclared external weights: {name}')
        path=local_path(artifact.root,name)
        try:
            offset=int(entries.get('offset','0'))
            length=int(entries.get('length',str(path.stat().st_size-offset)))
        except (ValueError,OSError) as exc: raise ArtifactError('Invalid external weights range') from exc
        if offset<0 or length<=0 or offset+length>path.stat().st_size:
            raise ArtifactError('External weights range exceeds declared file')
    return model


def resolve_artifact(vae_id, *, artifact_dir=None, store_dir=None, artifact_id=None, expected_source=None):
    if vae_id not in VAE_IDS: raise ArtifactError('Unknown VAE')
    if artifact_dir is not None:
        root=Path(artifact_dir)
    elif vae_id=='same_s' and artifact_id is None:
        root=Path(str(resources.files('stable_audio_wanderer.resources.same_s')))
    else:
        store=Path(store_dir).expanduser() if store_dir is not None else Path.home()/'.cache/waenderer/decoders'
        identity=artifact_id or read_json(store/vae_id/'current.json').get('artifact_id')
        _digest(identity)
        root=local_path(store, f'{vae_id}/{identity}')
    result=read_artifact(root,expected_vae=vae_id,expected_source=expected_source)
    if artifact_dir is None and not (vae_id=='same_s' and artifact_id is None) and result.identity!=identity:
        raise ArtifactError('Stored artifact identity does not match its directory')
    return result


def publish_artifact(prepared_dir, store_dir=None):
    """Copy, verify, check and smoke-test before atomically selecting a new artifact.

    The exporter supplies a hashed parity report. Publication adds structural and
    all-window PCM validation; it does not pretend to perform Torch parity itself.
    Immutable directories and atomic current.json updates preserve prior exports.
    """
    artifact=read_artifact(prepared_dir)
    if artifact.legacy: raise ArtifactError('Publish new exports using the shared schema')
    store=Path(store_dir).expanduser() if store_dir is not None else Path.home()/'.cache/waenderer/decoders'
    parent=local_path(store,artifact.vae_id)
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-',dir=parent) as temporary:
        staging=Path(temporary)/'artifact'
        staging.mkdir()
        for name in ('decoder.json',*(name for name,_ in artifact.files)):
            target=local_path(staging,name)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(local_path(artifact.root,name),target)
        copied=read_artifact(staging)
        if copied.identity!=artifact.identity: raise ArtifactError('Source changed while publishing')
        import onnx
        for graph in copied.graphs:
            inspect_graph(copied,graph)
            # Full shape inference cannot read external shape constants in some ONNX
            # versions. ORT compilation and every-window decode below validate shapes.
            onnx.checker.check_model(str(local_path(staging,graph.path)))
        from .artifact_decoder import ArtifactOnnxDecoder
        import numpy as np
        decoder=ArtifactOnnxDecoder(copied)
        try:
            for window in copied.supported_windows:
                decoder.decode(np.zeros((window,copied.latent_dim),np.float32))
        finally: decoder.close()
        copied.verify_files()
        destination=parent/copied.identity
        if destination.exists():
            if read_artifact(destination).identity!=copied.identity: raise ArtifactError('Existing export identity conflict')
        else:
            try: staging.rename(destination)
            except OSError:
                if not destination.exists() or read_artifact(destination).identity!=copied.identity: raise
        pointer=Path(temporary)/'current.json'
        pointer.write_text(json.dumps({'artifact_id':copied.identity})+'\n')
        pointer.replace(parent/'current.json')
    return destination
