"""Shared decoder export, numerical validation and atomic publication runner."""
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .onnx_artifacts import sha256, publish_artifact

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parity(reference, actual, tolerances):
    if (reference.dtype != np.float32 or not np.isfinite(reference).all() or
            actual.dtype != np.float32 or actual.shape != reference.shape or not np.isfinite(actual).all()):
        raise ValueError('Decoder parity shape/dtype/finiteness failure')
    error = actual.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    signal = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    result = dict(max_abs_error=float(np.max(np.abs(error))), rmse=rmse,
                  snr_db=20 * math.log10(signal / rmse) if signal > 0 and rmse > 0 else None)
    if any(result[k] > limit for k, limit in tolerances.items()):
        raise ValueError(f'Decoder numerical parity failed: {result}; limits={tolerances}')
    return result


def export_graph(wrapper, path, sample, *, dynamic, opset):
    import torch
    # Validate every advertised length: successful tracing alone is insufficient.
    torch.onnx.export(wrapper, (torch.from_numpy(sample),), str(path),
        input_names=['latents'], output_names=['audio'], opset_version=opset,
        dynamo=False, external_data=True,
        dynamic_axes={'latents': {2: 'latent_time'}, 'audio': {2: 'audio_time'}} if dynamic else None)


def validate_graph(wrapper, path, samples, *, ratio, channels, tolerances):
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
            if ref.shape != (1, channels, window*ratio) or ref.dtype != np.float32 or not np.isfinite(ref).all():
                raise ValueError('Native geometry/finiteness failure')
            variability = parity(ref, repeat, tolerances)
            actual = session.run(['audio'], {'latents': sample})[0]
            rows.append(dict(shape=list(actual.shape), native_repeat=variability, **parity(ref, actual, tolerances)))
        results[str(window)] = rows
        print(f'Validated T{window}: {len(rows)} probes', flush=True)
    return results



def build_artifact(*, wrapper, source, vae_id, samples, geometry, settings,
                   tolerances, evidence=None, store_dir=None, default_window=8,
                   export_fn=export_graph, validate_fn=None, publish_fn=publish_artifact,
                   fixed_only=False):
    windows = sorted(samples)
    if not windows or default_window not in windows:
        raise ValueError('Default export window must be covered')
    if validate_fn is None:
        def validate_fn(model, path, probes):
            return validate_graph(model, path, probes, ratio=geometry['samples_per_latent'],
                                  channels=geometry['channels'], tolerances=tolerances)
    with tempfile.TemporaryDirectory(prefix='waenderer-export-') as temp:
        root = Path(temp)
        graphs, results, failure = [], {}, None
        if not fixed_only:
            try:
                print(f'Exporting dynamic {vae_id} decoder', flush=True)
                path = root/'decoder_dynamic.onnx'
                export_fn(wrapper, path, samples[default_window][0], dynamic=True, opset=settings['opset'])
                results[path.name] = validate_fn(wrapper, path, samples)
                graphs.append(dict(path=path.name, dynamic=True, windows=windows))
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
                print(f'Dynamic export failed; testing fixed graphs: {failure}', flush=True)
                for p in root.iterdir():
                    if p.is_dir(): shutil.rmtree(p)
                    else: p.unlink()
        if not graphs:
            for w in windows:
                path = root/f'decoder_T{w}.onnx'
                export_fn(wrapper, path, samples[w][0], dynamic=False, opset=settings['opset'])
                results[path.name] = validate_fn(wrapper, path, {w:samples[w]})
                graphs.append(dict(path=path.name, dynamic=False, windows=[w]))
        files = {p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file()}
        report = dict(passed=True, source=source, graphs={g['path']:files[g['path']] for g in graphs},
            tolerances=tolerances, results=results, dynamic_failure=failure,
            scope='CPU float32 native/ORT parity; no real-time qualification', **(evidence or {}))
        write_json(root/'parity.json', report)
        files['parity.json'] = sha256(root/'parity.json')
        write_json(root/'decoder.json', dict(format_version='waenderer.onnx_decoder.v1',
            vae_id=vae_id, backend='onnxruntime', provider='CPUExecutionProvider',
            source=source, export=settings, **geometry, supported_windows=windows,
            default_window=default_window, input_name='latents', output_name='audio',
            input_layout='BDT', output_layout='BCT', ola_mode='full_overlap_add', graphs=graphs,
            files=files, validation_report='parity.json'))
        return publish_fn(root, store_dir=store_dir)
