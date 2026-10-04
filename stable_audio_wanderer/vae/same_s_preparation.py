"""Adapt the pinned SAME-S release exporter to a local shared-schema artifact."""
import json
import math
from importlib.metadata import version
from .onnx_artifacts import sha256


def stage_same_s(output, *, opset=20, revision=None):
    from .same_s_weights import resolve_same_s_weights, SOURCE_MODEL, SOURCE_REVISION
    from stable_audio_wanderer.cli.export_web_decoder import export_web_decoder
    from .export_common import write_json
    if revision and revision != SOURCE_REVISION:
        raise ValueError('SAME-S preparation requires the pinned revision')
    weights = resolve_same_s_weights(local_files_only=True)
    metadata = export_web_decoder(output, source_revision=SOURCE_REVISION, opset=opset)
    report = json.loads((output/'decoder_parity.json').read_text())
    # SAME-S decoder is stochastic. Retain the native comparison policy used
    # by validate_torch_decoder rather than EAR's deterministic tolerance.
    windows = list(range(2,33,2))
    if sorted(map(int,report['windows'])) != windows: raise ValueError('Incomplete SAME-S parity evidence')
    for w, row in report['windows'].items():
        if row['output_shape'] != [1,2,int(w)*4096]: raise ValueError('Invalid SAME-S geometry')
        if not math.isfinite(row['rmse']) or row['rmse'] >= .005 or not math.isfinite(row['snr_db']) or row['snr_db'] <= 20:
            raise ValueError('SAME-S parity policy failed')
    source = dict(model=SOURCE_MODEL, revision=SOURCE_REVISION, config_sha256=weights.config_sha256, weights_sha256=weights.model_sha256)
    graph = metadata['model']
    files = {p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'decoder.json'}
    write_json(output/'parity.json',dict(passed=True, source=source, graphs={graph:files[graph]},
        tolerances={'rmse_less_than':.005,'snr_db_greater_than':20}, results=report,
        scope='SAME-S stochastic native/ONNX comparison; no new real-time qualification'))
    files['parity.json'] = sha256(output/'parity.json')
    write_json(output/'decoder.json',dict(format_version='waenderer.onnx_decoder.v1',vae_id='same_s',
        backend='onnxruntime',provider='CPUExecutionProvider',source=source,
        export={'opset':opset,'tool_versions':{n:version(n) for n in ('torch','onnx','onnxruntime','stable-audio-3')}},
        sample_rate=44100,channels=2,latent_dim=256,samples_per_latent=4096,corpus_latent_hz=[44100/4096],
        supported_windows=windows,default_window=2,input_name='latents',output_name='audio',input_layout='BDT',output_layout='BCT',
        ola_mode='full_overlap_add',graphs=[dict(path=graph,dynamic=True,windows=windows)],files=files,validation_report='parity.json',
        source_license=metadata['source_license'],conversion_description=metadata['conversion_description']))
