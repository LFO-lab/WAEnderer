"""Small shared helpers for Phase 5 scripts; no model is loaded on import."""
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np

PROTOCOL = Path(__file__).resolve().parents[1] / 'docs/multi_vae_phase5_protocol.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def add_selection_arguments(parser):
    parser.add_argument('--weights', default='', help='EAR checkpoint')
    parser.add_argument('--repo', default='', help='EAR repository')
    parser.add_argument('--config', default='', help='EAR matching configuration')
    parser.add_argument('--store-dir', default='')
    parser.add_argument('--artifact-dir', default='')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)


def selection_config(args, backend, device):
    return dict(decoder_backend=backend, decoder_device=device,
                decoder_local_files_only=True, vae_weight_path=args.weights,
                vae_repo_path=args.repo, vae_config_path=args.config,
                decoder_store_dir=args.store_dir or None,
                decoder_artifact_dir=args.artifact_dir or None)


def normalize_device(device):
    return 'mps' if str(device) in ('mps', 'mps:0') else str(device)


def validate_native_source(decoder, artifact):
    # Adapter sources are checked by select_decoder; SAME-S uses pinned weights.
    if artifact.vae_id == 'same_s':
        for key, attribute in (('revision','source_revision'),
                               ('weights_sha256','model_sha256'),('config_sha256','config_sha256')):
            expected = dict(artifact.source).get(key)
            if expected and getattr(decoder.info, attribute, None) != expected:
                raise ValueError(f'Native SAME-S source differs from ONNX: {key}')


def evidence_context(args, spec, selection):
    from dataclasses import asdict
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ('numpy', 'torch', 'onnxruntime', 'diffusers', 'stable-audio-3', 'descript-audio-codec'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    artifact = selection.artifact
    return dict(schema_version=1, run_id=str(uuid.uuid4()), vae_id=selection.vae_id,
                backend=selection.backend, device=normalize_device(selection.device),
                policy_sha256=digest(args.protocol), corpus_spec=spec,
                platform=platform.platform(), hardware=platform.machine(),
                python=platform.python_version(), versions=versions,
                selection_identity=list(selection.identity),
                artifact_identity=artifact.identity if artifact else None,
                artifact_files=dict(artifact.files) if artifact else {},
                model_source=dict(artifact.source) if artifact else {},
                geometry=asdict(artifact_geometry(artifact)) if artifact else None)


def artifact_geometry(artifact):
    from stable_audio_wanderer.vae.decoder_contract import DecoderWindowMetadata
    w = artifact.default_window
    return DecoderWindowMetadata(w, artifact.latent_dim, artifact.sample_rate,
        artifact.channels, artifact.samples_per_latent, w * artifact.samples_per_latent,
        w // 2, w // 2 * artifact.samples_per_latent, 'full_overlap_add')


def compare(reference, actual):
    if reference.shape != actual.shape or reference.dtype != np.float32 or actual.dtype != np.float32:
        raise ValueError('PCM shape/dtype mismatch')
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Empty or nonfinite PCM')
    delta = reference.astype(np.float64) - actual.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return dict(rmse=rmse, max_abs_error=float(np.max(np.abs(delta))),
                snr_db=float(20 * np.log10(rms / rmse)) if rms and rmse else None)


def numerical_gate(policy, cross, within_onnx, within_native, *, synthetic=False):
    if policy is None:
        return dict(passed=False, reason='Native-reference GPU policy pending')
    if synthetic and 'synthetic_rmse_lt' in policy:
        passed = cross['rmse'] < policy['synthetic_rmse_lt'] and cross['snr_db'] is not None and cross['snr_db'] > policy['synthetic_snr_gt']
        return dict(passed=passed)
    if 'repeat_multiplier' in policy:
        limit = policy['repeat_multiplier'] * max(within_onnx['rmse'], within_native['rmse'], policy['repeat_floor'])
        return dict(passed=cross['rmse'] <= limit, rmse_limit=limit)
    return dict(passed=all(cross[key] <= policy[key] for key in ('rmse', 'max_abs_error')),
                limits=policy)


def checked_decode(decoder, raw):
    result = decoder.decode(raw)
    meta = decoder.metadata_for(len(raw))
    if result.audio.shape != (meta.audio_window_samples, meta.channels):
        raise ValueError('Decoded sample count/channels differ from model metadata')
    compare(result.audio, result.audio)
    return result
