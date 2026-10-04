"""Compare identical raw latents and production OLA for any registered VAE.

No playback or export. WAVs remain local. Two calls per engine preserve model
noise. Use --synthetic for the eight-seed contract probes in addition to corpus.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import soundfile as sf
import torch
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder
from stable_audio_wanderer.runtime.overlap_add import StreamingFullOverlapAdd
from eval_scripts.multi_vae_validation_common import (
    add_selection_arguments, selection_config, evidence_context, digest,
    write_report, compare, numerical_gate, checked_decode,
    normalize_device, validate_native_source,
)


def probe(onnx, native, raw, policy, synthetic=False):
    a, b = checked_decode(onnx, raw), checked_decode(native, raw)
    cross = compare(a.audio, b.audio)
    within_onnx = compare(a.audio, checked_decode(onnx, raw).audio)
    within_native = compare(b.audio, checked_decode(native, raw).audio)
    return dict(input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        shape=list(a.audio.shape), cross=cross, within_onnx=within_onnx,
        within_native=within_native, onnx_ms=a.decode_time_ms,
        native_ms=b.decode_time_ms,
        **numerical_gate(policy, cross, within_onnx, within_native, synthetic=synthetic))


def assemble(decoder, raw, window, hop):
    meta = decoder.metadata_for(window)
    ola = StreamingFullOverlapAdd(hop * meta.samples_per_latent, channels=meta.channels)
    return np.concatenate([ola.push(checked_decode(decoder, raw[i:i+window]).audio.T).T
                           for i in range(0, len(raw)-window+1, hop)])


def run(args):
    protocol = json.loads(args.protocol.read_text())
    spec = corpus_decoder_spec(args.corpus)
    path = args.corpus / 'corpus.npz' if args.corpus.is_dir() else args.corpus
    with np.load(path, allow_pickle=False) as data:
        z = np.ascontiguousarray(data['Z_concat'] * data['Z_std'] + data['Z_mean'], dtype=np.float32)
        offsets = data['file_offsets'].copy()
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(z) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Invalid corpus file boundaries')
    torch.set_num_threads(1)
    onnx_selection = select_decoder(selection_config(args, 'onnxruntime', 'cpu'), corpus_spec=spec)
    # Bind the native source to the resolved artifact, even for a legacy corpus.
    native_config = selection_config(args, 'pytorch', args.device)
    native_config['decoder_source_identity'] = dict(onnx_selection.artifact.source)
    native_selection = select_decoder(native_config, corpus_spec=spec)
    fixture_receipt = None
    if args.fixture:
        with np.load(args.fixture, allow_pickle=False) as fixture:
            source = json.loads(str(fixture['source_identity'].item()))
            if (str(fixture['vae_id'].item()) != spec['vae_id'] or
                    int(fixture['sample_rate']) != onnx_selection.artifact.sample_rate or
                    source != dict(onnx_selection.artifact.source)):
                raise ValueError('Fixture source/model does not match selected artifact')
            raw = fixture['raw_latents']
            if raw.ndim != 3 or raw.shape[1] != z.shape[1] or raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError('Invalid source-bound fixture latents')
            z = np.ascontiguousarray(raw.transpose(0,2,1).reshape(-1,raw.shape[1]))
            offsets = np.arange(len(raw)+1) * raw.shape[2]
            fixture_receipt = dict(path=str(args.fixture.resolve()), sha256=digest(args.fixture),
                source=source, audio_sha256=str(fixture['audio_sha256'].item()))
    policy = protocol['models'][spec['vae_id']]
    if not args.device.startswith('cpu'):
        policy = protocol['native_gpu'][spec['vae_id']]
    report = evidence_context(args, spec, onnx_selection)
    report.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        corpus=str(path.resolve()), corpus_sha256=digest(path),
        device=normalize_device(native_selection.device), native_identity=list(native_selection.identity),
        protocol=protocol, comparisons=[], synthetic=[], listening_pairs=[], error=None,
        numeric_passed=False, perceptual_qualified=False,
        validation_scope='quick_diagnostic' if args.quick else 'compact_qualification' if args.compact else 'full',
        historical_corpus_provenance='recorded' if 'weights_sha256' in spec else 'unknown; geometry matched')
    report['real_input_fixture'] = fixture_receipt
    onnx = native = None
    try:
        started = time.perf_counter()
        onnx = create_decoder(onnx_selection)
        report['onnx_preparation_ms'] = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        native = create_decoder(native_selection)
        validate_native_source(native, onnx_selection.artifact)
        report['native_preparation_ms'] = (time.perf_counter()-started)*1000
        windows = [w for w in (2,8,32) if w in protocol['windows']] if args.quick else protocol['windows']
        if not set(windows).issubset(onnx.supported_windows) or not set(windows).issubset(native.supported_windows):
            raise ValueError('Protocol windows are not supported by both decoders')
        maximum = max(windows)
        starts = [(index, int(a+(b-a-maximum)*fraction))
            for index, (a,b) in enumerate(zip(offsets[:-1], offsets[1:])) if b-a >= maximum
            for fraction in protocol['positions_per_file']]
        report['excluded_short_files'] = [i for i,(a,b) in enumerate(zip(offsets[:-1], offsets[1:])) if b-a < maximum]
        if not starts:
            raise ValueError('No complete real corpus probes')
        if args.quick or args.compact:
            starts = starts[:1]
        seed_count = 1 if (args.quick or (args.compact and spec['vae_id'] != 'same_s')) else protocol['synthetic_seeds']
        ola_frames = 32 if args.quick or args.compact else 128
        report['coverage'] = dict(windows=windows, real_positions=len(starts),
            synthetic_seeds=seed_count if args.synthetic else 0, ola_frames=ola_frames,
            ola_files=1 if args.quick or args.compact else len(offsets)-1)
        for window in windows:
            for file_index, start in starts:
                report['comparisons'].append(dict(window=window, start=start, file_index=file_index,
                    **probe(onnx, native, z[start:start+window], policy)))
            if args.synthetic:
                for seed in range(seed_count):
                    raw = np.random.default_rng(window*1000+seed).normal(0,.05,(window,z.shape[1])).astype(np.float32)
                    report['synthetic'].append(dict(window=window, seed=seed,
                        **probe(onnx, native, raw, policy, synthetic=True)))
            print(f'T{window}: comparisons complete', flush=True)
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        window, hop = protocol['ola_window'], protocol['ola_hop']
        meta = onnx.metadata_for(window)
        for index,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
            if (args.quick or args.compact) and index != starts[0][0]:
                continue
            raw = z[int(start):min(int(stop),int(start)+ola_frames)]
            if len(raw) < window:
                continue
            outputs, repeats, files = {}, {}, {}
            for name, decoder in (('onnx',onnx), ('native',native)):
                outputs[name] = assemble(decoder, raw, window, hop)
                repeats[name] = compare(outputs[name], assemble(decoder, raw, window, hop))
                filename = args.audio_dir / f'file{index}_{name}.wav'
                sf.write(filename, outputs[name], meta.sample_rate, subtype='FLOAT')
                files[name] = dict(path=str(filename.resolve()), sha256=digest(filename))
            cross = compare(outputs['onnx'], outputs['native'])
            report['listening_pairs'].append(dict(file_index=index, start_frame=int(start),
                input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), window=window, hop=hop,
                sample_rate=meta.sample_rate, samples=len(outputs['onnx']),
                tail='One emitted hop per window; unflushed OLA tail omitted',
                comparison_after_ola=cross, within_onnx=repeats['onnx'], within_native=repeats['native'],
                files=files, peaks={k:float(np.max(np.abs(v))) for k,v in outputs.items()},
                listening_review='pending',
                **numerical_gate(policy, cross, repeats['onnx'], repeats['native'])))
        report['numeric_passed'] = bool(report['listening_pairs']) and all(
            row['passed'] for group in ('comparisons','synthetic','listening_pairs') for row in report[group])
        print(f'{report["validation_scope"]}: numerical gate {"passed" if report["numeric_passed"] else "failed"}', flush=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for decoder in (native,onnx):
            if decoder is not None:
                try:
                    decoder.close()
                except Exception as exc:
                    report['error'] = f'Cleanup failed: {exc}'
                    report['numeric_passed'] = False
        write_report(args.output, report)
    if not report['numeric_passed']:
        raise SystemExit('Numerical validation failed or pending; inspect report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--compact', action='store_true', help='Qualification with all windows, one real position, one deterministic seed (eight SAME-S seeds), one 32-frame OLA clip')
    parser.add_argument('--fixture', type=Path, help='Source-bound real input fixture; preserves original navigation corpus identity')
    scope.add_argument('--quick', action='store_true', help='Diagnostic only: T2/T8/T32, one real position, one synthetic seed and one 32-frame OLA clip; does not qualify the full campaign')
    add_selection_arguments(parser)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_report(args.output, dict(schema_version=1,numeric_passed=False,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
