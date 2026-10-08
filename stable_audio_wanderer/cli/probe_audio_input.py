"""Short recorded-input comparison, no microphone or output device access.

Run: python -m stable_audio_wanderer.cli.probe_audio_input CORPUS_DIR INPUT.wav
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from ..io.corpus_io import load_corpus
from ..runtime.audio_input import create_audio_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus_dir', type=Path)
    parser.add_argument('input', type=Path)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--window', type=float, default=1.0)
    parser.add_argument('--update', type=float, default=0.1)
    parser.add_argument('--paths', nargs='+', choices=['descriptors', 'latents'],
                        default=['descriptors', 'latents'])
    parser.add_argument('--concurrent', action='store_true', help='Run analyzers concurrently')
    parser.add_argument('--decoder-backend', choices=['onnxruntime', 'pytorch'],
                        help='Also time a CPU decoder window per batch; no audio output')
    parser.add_argument('--decoder-window', type=int, default=4)
    args = parser.parse_args()
    corpus = load_corpus(str(args.corpus_dir / 'corpus.npz'))
    with np.load(args.corpus_dir / 'manual_navigation.npz', allow_pickle=False) as manual:
        navigation = create_audio_input(corpus, manual, {'audio_input_settings': {
            'window_seconds': args.window, 'update_seconds': args.update}})
    with sf.SoundFile(args.input) as source:
        rate = source.samplerate
        audio = source.read(max(1, int(args.seconds * rate)), dtype='float32', always_2d=True)
    if audio.shape[1] > 2:
        parser.error('recorded input must have one or two channels')
    size, step = int(args.window * rate), max(1, int(args.update * rate))
    if len(audio) < size:
        parser.error('recorded input is shorter than the analysis window')
    decoder = None
    pool = ThreadPoolExecutor(max_workers=3) if args.concurrent else None
    anchor = 0
    try:
        if args.decoder_backend:
            from ..vae.corpus_decoder import corpus_decoder_spec
            from ..vae.decoder_factory import select_decoder, create_decoder
            selection = select_decoder({'decoder_backend': args.decoder_backend, 'decoder_device': 'cpu'},
                                       corpus_spec=corpus_decoder_spec(args.corpus_dir))
            decoder = create_decoder(selection)
            decoder.metadata_for(args.decoder_window)

        def analyze(path, block, end):
            started = time.monotonic()
            index, distance = navigation.analyzers.analyze(path, block, rate)
            return dict(input_seconds=end/rate, path=path, index=index,
                        distance=distance, analysis_ms=(time.monotonic()-started)*1000)

        def decode(anchor, end):
            from ..runtime.manual_windows import build_file_bounded_latent_window
            started = time.monotonic()
            raw, _ = build_file_bounded_latent_window(corpus['Z_concat'], corpus['file_offsets'],
                anchor, args.decoder_window, mean=corpus['Z_mean'], std=corpus['Z_std'])
            decoded = decoder.decode(raw)
            if not np.isfinite(decoded.audio).all():
                raise RuntimeError('decoder returned nonfinite audio')
            return dict(input_seconds=end/rate, path='decoder', index=anchor,
                        analysis_ms=(time.monotonic()-started)*1000)

        for end in range(size, len(audio) + 1, step):
            block = audio[end-size:end]
            started = time.monotonic()
            jobs = [(analyze, (path, block, end)) for path in args.paths]
            if decoder is not None:
                jobs.append((decode, (anchor, end)))
            if pool is not None:
                futures = [pool.submit(function, *values) for function, values in jobs]
                results = [future.result() for future in futures]
            else:
                results = [function(*values) for function, values in jobs]
            elapsed = (time.monotonic() - started) * 1000
            for result in results:
                print(json.dumps({**result, 'batch_ms': elapsed}))
                if result['path'] in args.paths:
                    anchor = result['index']
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if decoder is not None:
            decoder.close()
        navigation.close()


if __name__ == '__main__':
    main()
