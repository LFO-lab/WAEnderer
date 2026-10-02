"""Compare native/ONNX on identical raw corpus windows and export listening pairs.

Cross-engine RMSE must be <= 3 times the larger within-engine repeat RMSE
(floor 1e-7). This stochastic regression gate is not a perceptual threshold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from stable_audio_wanderer.vae.onnx_decoder import validate_same_s_corpus, load_same_s_app_decoder
from stable_audio_wanderer.vae.torch_decoder import SameSTorchDecoder
from stable_audio_wanderer.runtime.overlap_add import StreamingFullOverlapAdd
from eval_scripts.validate_torch_decoder import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    args = parser.parse_args()
    path = validate_same_s_corpus(args.corpus)
    with np.load(path, allow_pickle=False) as data:
        z = (data['Z_concat'] * data['Z_std'] + data['Z_mean']).astype(np.float32)
        offsets = data['file_offsets'].copy()
    starts = [int(a + (b-a-32)*fraction) for a,b in zip(offsets[:-1], offsets[1:])
              if b-a >= 32 for fraction in (0., .25, .5, .75)]
    if not starts: raise SystemExit('Corpus needs at least one file of 32 frames')
    torch.set_num_threads(1)
    onnx = load_same_s_app_decoder()
    native = None
    rows, clips = [], []
    try:
        native = SameSTorchDecoder(device=args.device)
        for window in native.supported_windows:
            for start in starts:
                raw = z[start:start+window]
                a, b = onnx.decode(raw), native.decode(raw)
                cross = compare(a.audio, b.audio)
                within_onnx = compare(a.audio, onnx.decode(raw).audio)
                within_native = compare(b.audio, native.decode(raw).audio)
                limit = 3*max(within_onnx['rmse'], within_native['rmse'], 1e-7)
                rows.append({'window':window, 'start':start, 'cross':cross,
                    'within_onnx':within_onnx, 'within_native':within_native,
                    'rmse_limit':limit, 'passed':cross['rmse'] <= limit,
                    'onnx_ms':a.decode_time_ms, 'native_ms':b.decode_time_ms})
            print(f'T{window}: corpus comparisons complete', flush=True)
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        for file_index,(a,b) in enumerate(zip(offsets[:-1], offsets[1:])):
            start = int(a)
            stop = min(int(b), start+128)
            if stop-start < 8: continue
            outputs = {}
            for name,decoder in (('onnx',onnx), ('native',native)):
                assembler = StreamingFullOverlapAdd(4*4096, channels=2)
                output = np.concatenate([assembler.push(decoder.decode(z[i:i+8]).audio.T).T
                    for i in range(start,stop-7,4)])
                if not np.isfinite(output).all(): raise RuntimeError('Non-finite assembled PCM')
                filename = args.audio_dir / f'file{file_index}_{name}.wav'
                sf.write(filename, output, 44100, subtype='FLOAT')
                outputs[name] = output
            clips.append({'file_index':file_index, 'start_frame':start,
                'samples':len(outputs['onnx']), 'comparison_after_ola':compare(outputs['onnx'], outputs['native']),
                'peak_onnx':float(np.max(np.abs(outputs['onnx']))),
                'peak_native':float(np.max(np.abs(outputs['native']))),
                'listening_review':'pending'})
        report = {'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'corpus':str(path), 'corpus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'device':str(native.info.device), 'weights_sha256':native.info.model_sha256,
            'source_revision':native.info.source_revision, 'library_revision':native.info.library_revision,
            'protocol':'four positions per file, all even T2..32; two calls per engine; no RNG reset',
            'threshold':'cross RMSE <= 3 * max(ONNX repeat RMSE, native repeat RMSE, 1e-7)',
            'comparisons':rows, 'listening_pairs':clips, 'audio_directory':str(args.audio_dir),
            'numeric_passed':all(row['passed'] for row in rows), 'perceptual_qualified':False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    finally:
        onnx.close()
        if native is not None: native.close()
    if not report['numeric_passed']: raise SystemExit('Corpus numeric gate failed; inspect report')


if __name__ == '__main__': main()
