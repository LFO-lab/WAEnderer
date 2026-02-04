# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stable Audio Wanderer is a real-time 64-dimensional latent space navigation instrument for exploring audio corpora. It uses the Stable Audio Open VAE for encoding/decoding, a learned GRU policy for navigation, and manifold-constrained generation for real-time synthesis.

## Commands

**Preprocess audio into corpus:**
```bash
python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus
```

**Train navigation policy:**
```bash
python bin/train_policy.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

**Real-time performance:**
```bash
python bin/perform.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS \
  --policy_path corpus/.../latent_policy_*.pt \
  --osc_port 9000 --ws_port 8765
```
Open `web/index.html` in browser for visualization.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

### Three-Phase Pipeline

1. **Preprocess** (`bin/preprocess.py`): Encodes WAV files via VAE into 64D latent sequences, segments with sliding windows (200ms/50ms hop), computes kNN geometry (k=32), PCA components, and adaptive window targets based on local density.

2. **Train** (`bin/train_policy.py`): Trains a GRU policy that learns to navigate the latent space given 6 control dimensions (width, energy, gravity, memory, coherence, exploration). Outputs include Gaussian mixture displacements and window size predictions.

3. **Perform** (`bin/perform.py`): Real-time navigation with VAE decoding. Two threads: navigation thread runs policy steps, decode thread converts latents to audio with overlap-add crossfading.

### Key Components

- **LatentNavigationEngine** (`stable_audio_wanderer/runtime/player.py`): Maintains 64D position, queries kNN, runs policy, applies control modulations, and returns `NavFrame` objects.

- **ManifoldConstrainedGenerator** (`stable_audio_wanderer/runtime/manifold.py`): Projects navigation output back onto the corpus manifold using adaptive local/global PCA projection based on density.

- **DecoderPlayer** (`stable_audio_wanderer/runtime/decoder_player.py`): Dual-buffer streaming with overlap-add (50% Hann windows) or adaptive logarithmic crossfade for variable window sizes.

- **LatentGeometry** (`stable_audio_wanderer/policy/latent_geometry.py`): Precomputed kNN indices, local sigma/density, time gradients, and full-rank PCA for manifold operations.

- **LatentPolicy** (`stable_audio_wanderer/policy/latent_policy.py`): GRU model outputting 4-component Gaussian mixture (64D each), velocity updates, and 5-class window size predictions.

### Data Flow

```
Audio → VAE Encode → Segment/Pool → Normalize → Geometry → corpus.npz
                                                              ↓
                           GRU Policy Training → policy checkpoint
                                                              ↓
OSC Controls → Navigation Engine → Manifold Constraint → VAE Decode → Audio Out
                    ↓
            WebSocket → Browser Visualization (p5.js)
```

## Configuration

Global constants in `stable_audio_wanderer/config.py`:
- `SR = 44100` (sample rate)
- `LATENT_HZ = 21.5` (VAE latent frame rate)
- `DEVICE` auto-detects MPS/CUDA/CPU

Key defaults:
- Segment: 200ms window, 50ms hop
- kNN k=32, cosine distance
- Adaptive windows: 1/2/4/8/16 frames based on local density quantiles

## Control Protocol (OSC)

- `/policy/width`, `/policy/energy`, `/policy/gravity`, `/policy/memory`, `/policy/coherence`, `/policy/exploration` (0-1 float)
- `/policy/reset` - Reset navigation state
- `/decoder/gain`, `/decoder/smoothing` (0-1 float)
- `/cursor x y ...` - Set cursor position

## Corpus Requirements

Corpora must include geometry fields (`geom_*`). Legacy corpora without geometry are not supported. The `corpus.npz` contains: GG (normalized latents), metadata, geometry arrays, and Z_mean/Z_std for denormalization.
