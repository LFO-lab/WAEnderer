"""Offline preparation contract shared by command line and server jobs."""
import json
from pathlib import Path
from .onnx_artifacts import VAE_IDS, resolve_artifact, sha256

ALLOWED = {'vae_id','weights','repo','config','fixture','corpus','compatibility_corpus','store_dir','opset','force','fixed_only','revision'}


def validate_request(value):
    if not isinstance(value, dict) or set(value) - ALLOWED:
        raise ValueError('Invalid preparation fields')
    request = dict(value)
    if request.get('vae_id') not in VAE_IDS:
        raise ValueError('Unknown preparation VAE')
    if request['vae_id'] == 'same_s' and (request.get('fixed_only') or request.get('fixture')):
        raise ValueError('SAME-S uses its existing dynamic export and stochastic probe policy')
    if not request['vae_id'].startswith('ear_') and any(request.get(k) for k in ('weights','repo','config','fixture','fixed_only')):
        raise ValueError('EAR source/fixture/fixed options require an EAR VAE')
    for key in ('force','fixed_only'):
        if key in request and type(request[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if 'opset' in request and (type(request['opset']) is not int or request['opset'] < 18):
        raise ValueError('opset must be an integer >= 18')
    for key in ALLOWED - {'vae_id','opset','force','fixed_only'}:
        if key in request and request[key] is not None and not isinstance(request[key], str):
            raise ValueError(f'{key} must be text')
    for key in ('weights','repo','config','fixture','corpus','compatibility_corpus','store_dir'):
        if request.get(key): request[key] = str(Path(request[key]).expanduser().resolve())
    return request


def expected_source(request):
    source = {}
    for corpus_key in ('corpus','compatibility_corpus'):
        if not request.get(corpus_key): continue
        from .corpus_decoder import corpus_decoder_spec
        spec = corpus_decoder_spec(request[corpus_key])
        if spec['vae_id'] != request['vae_id']:
            raise ValueError('Corpus belongs to another VAE')
        from .decoder_availability import SOURCE_FIELDS
        selected = {k:spec[k] for k in SOURCE_FIELDS if k in spec}
        if any(k in source and source[k] != v for k,v in selected.items()):
            raise ValueError('Evidence and selected corpus have different sources')
        source.update(selected)
    if request['vae_id'].startswith('ear_') and request.get('weights'):
        from .ear_weights import selected_file_identity
        selected = selected_file_identity(request['vae_id'], request['weights'], request.get('repo',''), request.get('config',''))
        if any(k in source and source[k] != v for k,v in selected.items()):
            raise ValueError('Checkpoint conflicts with corpus provenance')
        source.update(selected)
    if request.get('revision'):
        source['revision'] = request['revision']
    return source


def input_snapshot(request):
    result = {'source': expected_source(request)}
    for key in ('fixture','corpus','compatibility_corpus'):
        if request.get(key):
            p = Path(request[key]); p = p/'corpus.npz' if p.is_dir() else p
            result[key] = sha256(p)
    return result


def reusable(request):
    """Reuse without requiring native libraries, but never weaken requested evidence."""
    if request.get('force'): return None
    artifact = resolve_artifact(request['vae_id'], store_dir=request.get('store_dir'), expected_source=expected_source(request))
    from .onnx_artifacts import inspect_graph
    for graph in artifact.graphs: inspect_graph(artifact,graph)
    manifest = json.loads((artifact.root/'decoder.json').read_text())
    if request.get('opset') and artifact.opset != request['opset']: return None
    if request.get('fixed_only') and any(g.dynamic for g in artifact.graphs): return None
    evidence = json.loads((artifact.root/manifest.get('validation_report','decoder_parity.json')).read_text())
    snapshot = input_snapshot(request)
    for key in ('fixture','corpus'):
        if key in snapshot and request['vae_id'] != 'same_s' and evidence.get(key+'_sha256') != snapshot[key]: return None
    for key in ('corpus','compatibility_corpus'):
        if request.get(key):
            from .corpus_decoder import corpus_decoder_spec
            artifact.validate_corpus(corpus_decoder_spec(request[key]))
    # Validate runtime availability as well as stored hashes; no native imports.
    import numpy as np
    from .artifact_decoder import load_artifact_decoder
    decoder = load_artifact_decoder(artifact)
    try:
        for window in artifact.supported_windows:
            decoder.decode(np.zeros((window, artifact.latent_dim), np.float32))
    finally:
        decoder.close()
    return artifact


def stage(request, output_dir):
    """Executed only in an isolated exporter process. Never updates current.json."""
    import shutil
    from .decoder_availability import installed
    vae = request['vae_id']
    required = ['torch','onnx','onnxruntime'] + (['dac','audiotools','einops','onnxscript'] if vae.startswith('ear_') else ['stable_audio_3','onnxscript'] if vae=='same_s' else ['diffusers'])
    missing = [name for name in required if not installed(name)]
    if missing: return dict(status='missing_dependencies', detail='Exporter needs: '+', '.join(missing))
    output = Path(output_dir)
    corpus = request.get('corpus') or request.get('compatibility_corpus')
    def save(root, **kwargs):
        shutil.copytree(root, output)
        return output
    if vae.startswith('ear_'):
        if not request.get('weights') or not Path(request['weights']).is_file():
            return dict(status='missing_weights', detail='Set the EAR checkpoint path')
        from .ear_export import prepare
        prepare(vae, request['weights'], repo=request.get('repo',''), config=request.get('config',''),
                fixture=request.get('fixture'), corpus=corpus, opset=request.get('opset',18),
                force=True, fixed_only=request.get('fixed_only',False), publish_fn=save)
    elif vae == 'stable_audio_open':
        if not corpus: return dict(status='missing_inputs',detail='Stable Audio Open export requires a matching evidence corpus')
        from .stable_audio_open_weights import resolve_source, SOURCE_REVISION
        try: resolve_source(revision=request.get('revision',SOURCE_REVISION))
        except (ValueError,OSError,RuntimeError) as exc: return dict(status='missing_weights', detail=str(exc))
        from .stable_audio_open_export import prepare
        prepare(corpus, revision=request.get('revision',SOURCE_REVISION), opset=request.get('opset',18), force=True, publish_fn=save)
    else:
        from .same_s_weights import resolve_same_s_weights, NativeDecoderLoadError
        try: resolve_same_s_weights(local_files_only=True)
        except (OSError,ValueError,NativeDecoderLoadError) as exc:
            return dict(status='missing_weights',detail=str(exc))
        from .same_s_preparation import stage_same_s
        stage_same_s(output, opset=request.get('opset',20), revision=request.get('revision'))
    return dict(status='staged', artifact_dir=str(output))
