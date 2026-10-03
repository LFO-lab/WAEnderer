"""Explicit, offline Stable Audio Open decoder preparation and parity validation."""
import json
import math
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .stable_audio_open_weights import SOURCE_MODEL, SOURCE_REVISION, resolve_source
from .onnx_artifacts import publish_artifact, resolve_artifact, sha256

WINDOWS = tuple(range(2, 33, 2))
# Frozen before the first real ONNX evaluation. Absolute plus aggregate gates
# handle waveform zero crossings without unstable pointwise relative errors.
TOLERANCES = {'max_abs_error': 2e-4, 'rmse': 2e-5}


from .export_common import write_json, export_graph, build_artifact
from . import export_common


def parity(reference, actual):
    return export_common.parity(reference, actual, TOLERANCES)


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


def validate_graph(wrapper, path, samples):
    return export_common.validate_graph(wrapper, path, samples, ratio=2048, channels=2, tolerances=TOLERANCES)


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
    return build_artifact(wrapper=wrapper, source=source, vae_id='stable_audio_open', samples=samples,
        geometry=dict(sample_rate=44100, channels=2, latent_dim=64, samples_per_latent=ratio,
                      corpus_latent_hz=[21.5,44100/ratio]),
        settings=settings, tolerances=TOLERANCES, store_dir=store_dir,
        evidence=dict(corpus=str(Path(corpus).resolve()),
            corpus_sha256=sha256(Path(corpus)/'corpus.npz' if Path(corpus).is_dir() else Path(corpus))),
        export_fn=export_graph, validate_fn=validate_graph, publish_fn=publish_artifact)
