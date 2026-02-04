# Manifold-Constrained Latent Generation Refactor — Changes

Date: 2026-02-04

## Summary
- Replaced grain playback with realtime VAE decoding driven by a manifold‑constrained generator.
- Extended latent geometry to store **full‑rank PCA** for global manifold projection (2D still used for viz).
- Removed grain system code, controls, UI, and dependencies.
- Added decoder streaming output using `sounddevice` plus new OSC/WS/UI controls (`gain`, `smoothing`).

## Breaking Changes
- **Corpus format**: `geom_pca_components` is now full‑rank. Old corpora must be regenerated with `bin/preprocess.py`.
- **Runtime API**: `LatentNavigationEngine` no longer exposes grain APIs (`get_render_weights`, `grain_rate`, `grain_jitter`, scheduler hooks). Use `step()` and the `NavFrame` output.
- **Controls**: All `/grain/*` and `/scheduler/*` OSC/WS/UI controls removed. New `/decoder/*` controls added.
- **Dependency change**: `pyo` removed, `sounddevice` added. `diffusers/transformers/accelerate/safetensors` are now required for runtime decode.

## New / Updated Components
### Geometry
- `LatentGeometry` now stores `pca_components` with shape `[D, 64]` (full PCA).
- Added `pca_components_2d` accessor for visualization.

### Manifold Generator
- New `stable_audio_wanderer/runtime/manifold.py`:
  - `ManifoldConfig(k=16, n_local=8, n_global=32, sparse_quantile=0.75)`
  - `ManifoldConstrainedGenerator`:
    - Anchor = weighted centroid of kNN neighbors
    - Local PCA projection (SVD on neighbors)
    - Global PCA fallback if region is sparse or local PCA fails
    - Magnitude clamp using local sigma and exploration

### Decoder
- New `stable_audio_wanderer/vae/decoder.py`: `decode_latents(ae, z_raw)`
- New `stable_audio_wanderer/runtime/decoder_player.py`:
  - `sounddevice` streaming output
  - Optional crossfade smoothing between decoded frames
  - Decoder state: gain, smoothing, frame_samples, underruns

## Runtime Changes
### Navigation
- `LatentNavigationEngine.step()` returns `NavFrame`:
  - `z_nav`, `indices`, `distances`, `nearest_idx`, `local_sigma`, `time_gradient`, `t_lat`, `file_id`

### Perform Loop
- `bin/perform.py` now:
  - Loads corpus + geometry + VAE
  - Uses `ManifoldConstrainedGenerator` to constrain latents
  - Decodes to audio and streams via `DecoderPlayer`
  - Sleeps according to decoder frame duration

### OSC Controls
- Removed `/grain/*` and `/scheduler/*`
- Added:
  - `/decoder/gain` (0..2)
  - `/decoder/smoothing` (0..1)

### WebSocket/UI
- Removed grain/scheduler state and UI
- Added decoder controls:
  - `gain`
  - `smoothing`

## Preprocess Changes
- `bin/preprocess.py` no longer renders grains or writes manifests.
- Outputs only corpus + geometry (now full PCA).

## Dependencies
Updated `requirements.txt`:
- Added: `sounddevice` (runtime audio)
- Removed: `pyo`
- `diffusers`, `transformers`, `accelerate`, `safetensors` now required for runtime decode

## Files Added
- `stable_audio_wanderer/runtime/manifold.py`
- `stable_audio_wanderer/runtime/decoder_player.py`
- `stable_audio_wanderer/vae/decoder.py`

## Files Removed
- `stable_audio_wanderer/runtime/grain_player.py`

## How To Run
### Preprocess (regenerate corpora)
```bash
python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus
```

### Perform (realtime decode)
```bash
python bin/perform.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

### UI
Open `web/index.html` and connect to the WS server (default `ws://127.0.0.1:8765`).

## Notes
- Decoder smoothing is a crossfade between consecutive decoded frames.
- Exploration control now directly impacts perturbation magnitude through manifold constraints.
- Old corpora without full PCA will fail with a clear error; re-run preprocess.
