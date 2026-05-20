# Latency Benchmark Report

Generated: `2026-03-28T16:27:03-04:00`

## Test Purpose
This benchmark measures steady-state end-to-end latent decoding latency for Stable Audio Wanderer using observed corpus latents and the same runtime batch construction path used by live playback. Each request decodes a latent batch of shape `[1, D, W]`, where `D` is the active adapter latent dimension and `W` is the evaluated window size.

## Operational Definition
`Audio buffer ready` is defined here as the point when decoded audio has completed the same overlap-add/crossfade queue write used by `DecoderPlayer.write_frame(...)`. The measurement excludes DAC, driver, and OS scheduling latency, so it reflects software-side request-to-buffer-ready time.

## Environment
- Corpus directory: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359`
- Corpus file: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359/corpus.npz`
- Source audio files: 1
- Corpus frames: 4530
- VAE: `stable_audio_open` (Stable Audio Open (44.1k))
- Sample rate: 44100 Hz
- Latent rate: 21.5000 Hz
- Latent dimension: 64
- Host: `MacBook-Pro-de-Dominic-2.local`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Device: `cpu` / `arm`
- Python / Torch: `3.12.13` / `2.10.0`
- Measured trials per window: 2
- Warmup trials per window: 1
- Modes: manual, random, reorganized
- `manual` windows: 1, 2, 4, 8, 16, 32, 64
- `random` windows: 2, 4, 8, 16, 32, 64
- `reorganized` windows: 2, 4, 8, 16, 32, 64
- CSV: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/latency_validation/latency_trials.csv`
- JSON summary: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/latency_validation/latency_summary.json`
- Figure: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/latency_validation/latency_plot.png`

## Latency Table

| Mode | W | Prepare Mean | Prepare Median | Decode Mean | Decode Median | Total Mean | Total Median | Total IQR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| manual | 1 | 0.61 ms | 0.61 ms | 107.98 ms | 107.98 ms | 108.72 ms | 108.72 ms | 28.73 ms |
| manual | 2 | 0.54 ms | 0.54 ms | 117.57 ms | 117.57 ms | 118.23 ms | 118.23 ms | 10.95 ms |
| manual | 4 | 1.15 ms | 1.15 ms | 151.03 ms | 151.03 ms | 152.47 ms | 152.47 ms | 15.98 ms |
| manual | 8 | 1.70 ms | 1.70 ms | 253.86 ms | 253.86 ms | 255.90 ms | 255.90 ms | 3.83 ms |
| manual | 16 | 2.87 ms | 2.87 ms | 398.70 ms | 398.70 ms | 402.26 ms | 402.26 ms | 14.92 ms |
| manual | 32 | 9.89 ms | 9.89 ms | 680.40 ms | 680.40 ms | 690.98 ms | 690.98 ms | 21.58 ms |
| manual | 64 | 10.64 ms | 10.64 ms | 1418.17 ms | 1418.17 ms | 1429.86 ms | 1429.86 ms | 35.03 ms |
| random | 2 | 136.05 ms | 136.05 ms | 125.87 ms | 125.87 ms | 262.04 ms | 262.04 ms | 7.40 ms |
| random | 4 | 172.38 ms | 172.38 ms | 211.80 ms | 211.80 ms | 384.64 ms | 384.64 ms | 55.71 ms |
| random | 8 | 191.59 ms | 191.59 ms | 263.62 ms | 263.62 ms | 455.68 ms | 455.68 ms | 22.86 ms |
| random | 16 | 288.82 ms | 288.82 ms | 403.51 ms | 403.51 ms | 692.65 ms | 692.65 ms | 12.87 ms |
| random | 32 | 424.49 ms | 424.49 ms | 695.36 ms | 695.36 ms | 1120.58 ms | 1120.58 ms | 22.53 ms |
| random | 64 | 801.58 ms | 801.58 ms | 1417.12 ms | 1417.12 ms | 2220.05 ms | 2220.05 ms | 19.11 ms |
| reorganized | 2 | 0.45 ms | 0.45 ms | 130.49 ms | 130.49 ms | 131.08 ms | 131.08 ms | 22.69 ms |
| reorganized | 4 | 0.62 ms | 0.62 ms | 159.69 ms | 159.69 ms | 160.59 ms | 160.59 ms | 11.97 ms |
| reorganized | 8 | 1.10 ms | 1.10 ms | 255.81 ms | 255.81 ms | 257.35 ms | 257.35 ms | 4.37 ms |
| reorganized | 16 | 1.79 ms | 1.79 ms | 395.37 ms | 395.37 ms | 397.48 ms | 397.48 ms | 5.18 ms |
| reorganized | 32 | 4.07 ms | 4.07 ms | 710.85 ms | 710.85 ms | 715.39 ms | 715.39 ms | 4.19 ms |
| reorganized | 64 | 9.25 ms | 9.25 ms | 1447.74 ms | 1447.74 ms | 1458.28 ms | 1458.28 ms | 0.47 ms |

## Interpretation
Across the evaluated settings, shorter latent windows are expected to give the lowest request-to-buffer latency because fewer latent frames are prepared and decoded per request. Longer windows increase both decode cost and queue fill time, but they also provide longer contiguous latent context and are expected to produce smoother overlap-add transitions at runtime.

In manual mode, median total latency rose from 108.72 ms at W=1 to 1429.86 ms at W=64. In random mode, median total latency rose from 262.04 ms at W=2 to 2220.05 ms at W=64. In reorganized mode, median total latency rose from 131.08 ms at W=2 to 1458.28 ms at W=64.

## DAFx-Ready Paragraph
We evaluated the Stable Audio Wanderer decoding path by constructing runtime-faithful latent windows from the observed corpus and measuring steady-state request-to-buffer-ready latency after batch preparation, VAE decode, and overlap-add buffer enqueue. manual-mode median request-to-buffer latency increased from 108.72 ms at W=1 to 1429.86 ms at W=64; random-mode latency increased from 262.04 ms at W=2 to 2220.05 ms at W=64; reorganized-mode latency increased from 131.08 ms at W=2 to 1458.28 ms at W=64. These results support the intended reconstructive/navigational framing of the system: shorter windows minimize interaction latency, while longer windows trade latency for longer latent continuity and smoother output.
