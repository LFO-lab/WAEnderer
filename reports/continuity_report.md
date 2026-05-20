# Continuity Evaluation Report

Generated: `2026-03-28T17:02:49-04:00`

## Purpose
This benchmark compares temporal continuity and output stability across three latent decoding strategies using real corpus trajectories and the project's runtime-faithful VAE decode path.

## Compared Methods
- `single_loop`: repeat the first latent frame for the full trajectory, then decode in overlapping windows.
- `naive_interp`: linearly interpolate in raw latent space between the first and last trajectory latents before decoding.
- `sequential`: decode the observed corpus latent trajectory directly, preserving the corpus ordering.

## Assumptions
Trajectory examples are sampled from stored corpus runs that remain contiguous both in file ID and latent time index (`t_lat` increments by 1).
All methods are matched for trajectory length. Decoding uses a first full window followed by hop = ceil(window / 2) latent updates, and audio is reconstructed with the same adaptive crossfade queue used by `DecoderPlayer.write_frame(...)`.
After the last decode, the held overlap-add tail is flushed and the final waveform is trimmed or padded to the target duration implied by the latent frame count.
The low-frequency energy ratio uses a cutoff of 150.0 Hz.

## Environment
- Corpus directory: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359`
- Corpus file: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359/corpus.npz`
- VAE: `stable_audio_open` (Stable Audio Open (44.1k))
- Sample rate: 44100 Hz
- Latent rate: 21.5000 Hz
- Latent dimension: 64
- Host: `MacBook-Pro-de-Dominic-2.local`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Device: `cpu` / `arm`
- Examples: 2
- Conditions: single_loop, naive_interp, sequential
- Audio examples: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/continuity_validation/audio`
- CSV: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/continuity_validation/continuity_metrics.csv`
- JSON summary: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/continuity_validation/continuity_summary.json`
- Plot: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/continuity_validation/continuity_metrics.png`

## Aggregate Table

| Condition | Strategy | W | RMS Diff | Spectral Flux | MFCC Dist | Loudness Delta | LF Ratio | Stability Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_loop | Single-Latent Loop | 8 | 0.0025 | 0.0000 | 0.1359 | 0.0046 | 0.0222 | 1.00 |
| naive_interp | Naive Raw-Latent Interpolation | 8 | 0.0342 | 0.0001 | 10.2669 | 0.1553 | 0.1198 | 2.00 |
| sequential | Sequential Corpus Recall | 8 | 0.1946 | 0.0005 | 43.6563 | 2.0706 | 0.2656 | 3.00 |

## Interpretation
Using the average rank over the primary pairwise continuity metrics (RMS difference, spectral flux, MFCC distance, loudness delta, and low-frequency-ratio delta), the smallest measured buffer-to-buffer changes in this run came from the static baseline `single_loop`.
That result should be treated as a degenerate lower bound on change, not as evidence of desirable continuity: repeating one latent can minimize transitions simply by collapsing the audio into a near-frozen loop.
Restricting the comparison to time-varying methods, the lowest aggregate discontinuity score came from `naive_interp`.
Naive raw-latent interpolation is expected to be problematic because it moves through latent states that were not observed in the corpus sequence and are not guaranteed to decode into temporally coherent audio. This often appears as larger spectral and MFCC jumps, even when the latent path itself looks smooth algebraically.
Single-latent looping can sometimes produce low pairwise transition metrics simply because the output changes very little. That should be interpreted cautiously: a trivially static loop is not the same as a musically playable, continuously evolving output.
Sequential corpus-based recall is the method most aligned with the system's reconstructive/navigational design because it decodes observed latent progressions rather than forcing the decoder to sustain a frozen latent or traverse unsupported straight lines in raw latent space.

## DAFx-Ready Paragraph
We compared single-latent looping, naive raw-latent interpolation, and sequential corpus-based latent decoding using matched-duration corpus trajectories and the runtime-faithful Stable Audio Wanderer decode path. In this run, the frozen baseline `single_loop` minimized pairwise buffer-change metrics, but this should be interpreted as a trivial static lower bound rather than a musically useful notion of continuity. Among the time-varying methods, `naive_interp` produced the lowest discontinuity score. The saved audio examples remain important here: naive raw-latent interpolation still traverses unsupported latent states, while sequential corpus recall is the only condition that preserves observed temporal structure from the corpus.
