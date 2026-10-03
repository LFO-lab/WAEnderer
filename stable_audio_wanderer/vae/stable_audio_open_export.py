"""Explicit, offline Stable Audio Open decoder preparation and parity validation."""
import json
import math
from importlib.metadata import version
from pathlib import Path
import tempfile
import numpy as np
from .stable_audio_open_weights import SOURCE_MODEL, SOURCE_REVISION, resolve_source
from .onnx_artifacts import publish_artifact, resolve_artifact, sha256

WINDOWS = tuple(range(2, 33, 2))
# Frozen before the first real ONNX evaluation. Absolute plus aggregate gates
# handle waveform zero crossings without unstable pointwise relative errors.
TOLERANCES = {'max_abs_error': 2e-4, 'rmse': 2e-5}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parity(reference, actual):
    if actual.dtype != np.float32 or actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError('Decoder parity shape/dtype/finiteness failure')
    error = actual.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    signal = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    result = dict(max_abs_error=float(np.max(np.abs(error))), rmse=rmse,
                  snr_db=20 * math.log10(signal / rmse) if signal > 0 and rmse > 0 else None)
    if any(result[k] > limit for k, limit in TOLERANCES.items()):
        raise ValueError(f'Decoder numerical parity failed: {result}; limits={TOLERANCES}')
    return result


def load_wrapper(repo_or_path=SOURCE_MODEL, revision=SOURCE_REVISION):
    import torch
    from diffusers import AutoencoderOobleck
    root, source = resolve_source(repo_or_path, revision=revision)
    config = json.loads((root / 'vae/config.json').read_text())
    ratio = math.prod(config['downsampling_ratios'])
    if (ratio, config['sampling_rate'], config['decoder_input_channels'], config['audio_channels']) != (2048, 44100, 64, 2):
        raise ValueError('Unexpected pinned Stable Audio Open geometry')
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, latents):
            return self.model.decode(latents).sample
    model = AutoencoderOobleck.from_pretrained(str(root), subfolder='vae', local_files_only=True)
    return DecoderWrapper(model.to(device='cpu', dtype=torch.float32).eval()).eval(), source, ratio


def corpus_samples(corpus):
    from .corpus_decoder import corpus_decoder_spec
    path = Path(corpus)
    spec = corpus_decoder_spec(path)
    if spec['vae_id'] != 'stable_audio_open':
        raise ValueError('Preparation requires a Stable Audio Open corpus')
    with np.load(path / 'corpus.npz' if path.is_dir() else path, allow_pickle=False) as data:
        z, mean, std = (np.asarray(data[k], dtype=np.float32) for k in ('Z_concat', 'Z_mean', 'Z_std'))
        if len(z) < max(WINDOWS):
            raise ValueError('Parity corpus needs at least 32 latent frames')
        return {w: [np.ascontiguousarray((z[start:start+w]*std+mean).T[None])
                    for start in sorted({0, (len(z)-w)//2, len(z)-w})] for w in WINDOWS}


def export_graph(wrapper, path, sample, *, dynamic, opset):
    import torch
    # Legacy tracing preserves Oobleck's convolutional decoder and dynamic time.
    torch.onnx.export(wrapper, (torch.from_numpy(sample),), str(path),
        input_names=['latents'], output_names=['audio'], opset_version=opset,
        dynamo=False, external_data=True,
        dynamic_axes={'latents': {2: 'latent_time'}, 'audio': {2: 'audio_time'}} if dynamic else None)


def validate_graph(wrapper, path, samples):
    import torch
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    results = {}
    for window, probes in samples.items():
        rows = []
        for sample in probes:
            with torch.inference_mode():
                ref = wrapper(torch.from_numpy(sample)).numpy()
                repeat = wrapper(torch.from_numpy(sample)).numpy()
            if ref.shape != (1, 2, window*2048) or ref.dtype != np.float32 or not np.isfinite(ref).all():
                raise ValueError('Native geometry/finiteness failure')
            variability = parity(ref, repeat)
            actual = session.run(['audio'], {'latents': sample})[0]
            rows.append(dict(shape=list(actual.shape), native_repeat=variability, **parity(ref, actual)))
        results[str(window)] = rows
        print(f'Validated T{window}: {len(rows)} probes', flush=True)
    return results


def prepare(corpus, *, store_dir=None, revision=SOURCE_REVISION, opset=18, force=False):
    import torch
    torch.set_num_threads(1)
    from .corpus_decoder import corpus_decoder_spec
    spec = corpus_decoder_spec(corpus)
    if spec["vae_id"] != "stable_audio_open":
        raise ValueError("Preparation requires a Stable Audio Open corpus")
    _, source = resolve_source(revision=revision)
    if spec.get('latent_dim') != 64 or spec.get('sample_rate') != 44100 or not any(
            math.isclose(spec.get('latent_hz', 0), rate, rel_tol=0, abs_tol=1e-7)
            for rate in (21.5, 44100/2048)):
        raise ValueError('Corpus geometry does not match pinned Stable Audio Open')
    for key in ('config_sha256', 'weights_sha256', 'revision'):
        if key in spec and spec[key] != source[key]:
            raise ValueError(f'Corpus source {key} does not match pinned Stable Audio Open')
    settings = dict(opset=opset, tool_versions={name: version(name) for name in
        ('torch', 'diffusers', 'onnx', 'onnxruntime')}, strategy='dynamic_then_fixed',
        tolerances=TOLERANCES)
    if not force:
        try:
            artifact = resolve_artifact('stable_audio_open', store_dir=store_dir, expected_source=source)
            artifact.validate_corpus(spec)
            manifest = json.loads((artifact.root/'decoder.json').read_text())
            if manifest['export'] == settings and artifact.supported_windows == WINDOWS:
                from .artifact_decoder import load_artifact_decoder
                decoder = load_artifact_decoder(artifact)
                try:
                    for w in WINDOWS: decoder.decode(np.zeros((w,64),np.float32))
                finally: decoder.close()
                print(f"Already valid: {artifact.root}", flush=True)
                return artifact.root
        except (ValueError, FileNotFoundError):
            pass
    real = corpus_samples(corpus)
    rng = np.random.default_rng(20261002)
    samples = {w: [np.zeros((1,64,w),np.float32), rng.standard_normal((1,64,w)).astype(np.float32), *real[w]] for w in WINDOWS}
    wrapper, source, ratio = load_wrapper(revision=revision)
    with tempfile.TemporaryDirectory(prefix='waenderer-sao-') as temp:
        root = Path(temp)
        graphs, results, failure = [], {}, None
        try:
            print('Exporting dynamic Stable Audio Open decoder', flush=True)
            path = root/'decoder_dynamic.onnx'
            export_graph(wrapper, path, samples[8][0], dynamic=True, opset=opset)
            results[path.name] = validate_graph(wrapper, path, samples)
            graphs.append(dict(path=path.name, dynamic=True, windows=list(WINDOWS)))
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
            print(f'Dynamic export failed; testing fixed graphs: {failure}', flush=True)
            # Isolate failed dynamic files from the published set.
            for p in root.iterdir():
                if p.is_file(): p.unlink()
            for w in WINDOWS:
                path = root/f'decoder_T{w}.onnx'
                export_graph(wrapper, path, samples[w][0], dynamic=False, opset=opset)
                results[path.name] = validate_graph(wrapper, path, {w:samples[w]})
                graphs.append(dict(path=path.name, dynamic=False, windows=[w]))
        files = {p.name:sha256(p) for p in root.iterdir() if p.is_file()}
        report = dict(passed=True, source=source, graphs={g['path']:files[g['path']] for g in graphs},
            tolerances=TOLERANCES, results=results, dynamic_failure=failure,
            corpus=str(Path(corpus).resolve()), corpus_sha256=sha256(Path(corpus)/'corpus.npz' if Path(corpus).is_dir() else Path(corpus)),
            scope='CPU float32 native/ORT parity; no real-time qualification')
        write_json(root/'parity.json', report)
        files['parity.json'] = sha256(root/'parity.json')
        write_json(root/'decoder.json', dict(format_version='waenderer.onnx_decoder.v1',
            vae_id='stable_audio_open', backend='onnxruntime', provider='CPUExecutionProvider',
            source=source, export=settings, sample_rate=44100, channels=2, latent_dim=64,
            samples_per_latent=ratio, corpus_latent_hz=[21.5,44100/ratio],
            supported_windows=list(WINDOWS), default_window=8, input_name='latents', output_name='audio',
            input_layout='BDT', output_layout='BCT', ola_mode='full_overlap_add', graphs=graphs,
            files=files, validation_report='parity.json'))
        return publish_artifact(root, store_dir=store_dir)
