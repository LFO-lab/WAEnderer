"""Explicit local EAR decoder export for independently identified 44k/48k models."""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .ear_weights import VARIANTS, resolve_source
from .onnx_artifacts import resolve_artifact, sha256
from .export_common import build_artifact

WINDOWS = tuple(range(2,33,2))
# Frozen before either EAR graph was evaluated. CPU repeats were exact for both
# checkpoints across T2-T32. GPU comparisons are reported separately by the audit.
TOLERANCES = {
    'ear_vae_44k': dict(max_abs_error=1e-4, rmse=1e-5),
    'ear_vae_48k': dict(max_abs_error=1e-4, rmse=1e-5),
}


def load_wrapper(vae_id, weights, repo='', config='', expected_source=None):
    import torch
    from .registry import load_vae_adapter
    adapter = load_vae_adapter(vae_id,weight_path=str(weights),repo_path=str(repo),
        config_path=str(config),device='cpu',expected_source=expected_source)
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, latents):
            return self.model.decode(latents)
    return DecoderWrapper(adapter._model).eval(), adapter


def export_graph(wrapper, path, sample, *, dynamic, opset):
    import torch
    if not dynamic:
        from .export_common import export_graph as fixed_export
        return fixed_export(wrapper, path, sample, dynamic=False, opset=opset)
    # torch.export handles the EAR transformer's dynamic LayerNorm shape. The
    # legacy tracer rejects that shape even though fixed-window export works.
    torch.onnx.export(wrapper, (torch.from_numpy(sample),), str(path), dynamo=True,
        input_names=['latents'], output_names=['audio'], opset_version=opset,
        external_data=True,
        dynamic_shapes=({2:torch.export.Dim('latent_time', min=2, max=32)},))


def probe_samples(vae_id, source, fixture=None, corpus=None):
    from .corpus_decoder import corpus_decoder_spec
    rng=np.random.default_rng(20261003)
    samples={w:[np.zeros((1,64,w),np.float32),rng.standard_normal((1,64,w)).astype(np.float32)] for w in WINDOWS}
    evidence={ 'probe_seed':20261003, 'probe_classes':['zero','seeded_random'] }
    if fixture:
        with np.load(fixture,allow_pickle=False) as data:
            if str(data['vae_id'].item()) != vae_id or json.loads(str(data['source_identity'].item())) != source:
                raise ValueError('EAR encoded fixture source does not match selected decoder')
            raw=data['raw_latents']
            if raw.dtype!=np.float32 or raw.ndim!=3 or raw.shape[1]!=64 or raw.shape[2]<32 or not np.isfinite(raw).all():
                raise ValueError('Invalid EAR encoded fixture geometry')
            for w in WINDOWS:
                samples[w].extend(np.ascontiguousarray(z[None,:,:w]) for z in raw)
        evidence.update(fixture_sha256=sha256(fixture),fixture=str(Path(fixture).resolve()))
        evidence['probe_classes'].append('source_bound_encoded_audio')
    if corpus:
        spec=corpus_decoder_spec(corpus)
        v=VARIANTS[vae_id]
        if (spec['vae_id']!=vae_id or spec['sample_rate']!=v['sample_rate'] or spec['latent_dim']!=64 or
                not np.isclose(spec['latent_hz'],v['sample_rate']/v['ratio'],rtol=0,atol=1e-7)):
            raise ValueError('EAR parity corpus geometry/variant mismatch')
        for key in source:
            if key in spec and spec[key]!=source[key]:raise ValueError(f'EAR corpus source mismatch: {key}')
        path=Path(corpus)/'corpus.npz' if Path(corpus).is_dir() else Path(corpus)
        with np.load(path,allow_pickle=False) as data:
            z=data['Z_concat']*data['Z_std']+data['Z_mean']
            if len(z)<32:raise ValueError('EAR parity corpus needs at least 32 frames')
            for w in WINDOWS:
                for start in sorted({0,(len(z)-w)//2,len(z)-w}):
                    samples[w].append(np.ascontiguousarray(z[start:start+w].T[None]))
        evidence.update(corpus_sha256=sha256(path),corpus=str(path.resolve()),
                        corpus_checkpoint_provenance=all(k in spec for k in ('weights_sha256','config_sha256')))
        evidence['probe_classes'].append('corpus_latents')
    evidence['synthetic_only']=not bool(fixture or corpus)
    return samples,evidence


def prepare(vae_id, weights, *, repo='', config='', fixture=None, corpus=None,
            store_dir=None, opset=18, force=False, fixed_only=False):
    import torch
    torch.set_num_threads(1)
    resolved=resolve_source(vae_id,weights,repo,config)
    source=resolved.identity
    v=VARIANTS[vae_id]
    settings=dict(opset=opset,tool_versions={n:version(n) for n in
        ('torch','descript-audio-codec','descript-audiotools','einops','onnx','onnxruntime','onnxscript')},
        strategy='fixed' if fixed_only else 'dynamic_then_fixed',tolerances=TOLERANCES[vae_id],
        wrapper='EAR_VAE.decode-v1', exporter='torch.export-dynamic/legacy-fixed-v1')
    samples,evidence=probe_samples(vae_id,source,fixture,corpus)
    # Requested evidence must be covered too; a synthetic-only export must not
    # masquerade as validated on newly supplied encoded audio/corpus probes.
    settings['probe_identity']={k:evidence[k] for k in ('fixture_sha256','corpus_sha256') if k in evidence}
    if not force:
        try:
            artifact=resolve_artifact(vae_id,store_dir=store_dir,expected_source=source)
            manifest=json.loads((artifact.root/'decoder.json').read_text())
            if manifest['export']==settings and artifact.supported_windows==WINDOWS:
                from .artifact_decoder import load_artifact_decoder
                decoder=load_artifact_decoder(artifact)
                try:
                    for w in WINDOWS:decoder.decode(np.zeros((w,64),np.float32))
                finally:decoder.close()
                print(f'Already valid: {artifact.root}',flush=True)
                return artifact.root
        except (ValueError,OSError):pass
    evidence.update(code_files=resolved.code_files,effective_config=resolved.config)
    del resolved
    wrapper,adapter=load_wrapper(vae_id,weights,repo,config,expected_source=source)
    del adapter
    return build_artifact(wrapper=wrapper,source=source,vae_id=vae_id,samples=samples,
        geometry=dict(sample_rate=v['sample_rate'],channels=2,latent_dim=64,samples_per_latent=v['ratio'],
                      corpus_latent_hz=[v['sample_rate']/v['ratio']]),settings=settings,
        tolerances=TOLERANCES[vae_id],evidence=evidence,store_dir=store_dir,fixed_only=fixed_only,export_fn=export_graph)
