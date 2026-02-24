# Stable Audio Wanderer

A real-time 64-dimensional latent space navigation instrument for exploring audio corpora. Audio is encoded with the Stable Audio Open VAE, segmented into a geometry-enabled latent corpus, and navigated using a learned GRU policy with manifold-constrained generation for real-time synthesis.

## Features

- **64D Latent Navigation** - Explore audio corpora in continuous latent space
- **Manual Navigation Mode** - 8-fader nearest-frame retrieval in MFCC/PCA control space
- **Learned Navigation Policy** - GRU network with 6 expressive control dimensions
- **Manifold-Constrained Generation** - Stay on the learned audio manifold with adaptive PCA projection
- **Real-Time VAE Decoding** - Overlap-add synthesis with adaptive window sizing
- **WebSocket Visualization** - p5.js interface showing trajectory, controls, and corpus structure
- **Explicit Transport** - Choose mode, then Start/Stop decoding from the web UI
- **OSC Control** - Full parameter control for integration with external controllers

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

### Dependencies

| Category | Packages |
|----------|----------|
| Core | `torch`, `torchaudio`, `numpy`, `soundfile` |
| Navigation | `scikit-learn`, `scipy`, `faiss-cpu` |
| Runtime | `sounddevice`, `python-osc`, `websockets` |
| VAE | `diffusers`, `transformers`, `accelerate`, `safetensors` |

## Pipeline

### 1. Preprocess

Encode audio files into a geometry-enabled latent corpus and manual-navigation feature space.

```bash
python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus
```

| Option | Default | Description |
|--------|---------|-------------|
| `--audio_dir` | required | Directory containing WAV files |
| `--out_prefix` | required | Output corpus name prefix |
| `--seg_sec` | 0.2 | Segment duration in seconds |
| `--hop_sec` | 0.05 | Hop duration in seconds |
| `--latent_nav_k` | 32 | k-NN neighbors for geometry |
| `--encode_chunk_sec` | 60.0 | VAE encode chunk size in seconds (0 disables chunking) |
| `--encode_chunk_overlap_sec` | 1.0 | VAE encode chunk overlap in seconds |

Output: `corpus/[prefix]_YYYYMMDD_HHMMSS/corpus.npz`

`corpus.npz` now includes manual-navigation fields:
- `manual_pca_points` `[N, 8]`
- `manual_pca_components` `[8, 20]`
- `manual_pca_mean` `[20]`
- `manual_fader_p01` `[8]`
- `manual_fader_p99` `[8]`

### 2. Train Policy

Train a GRU navigation policy, manual KD-tree artifact, or both.

```bash
python bin/train_policy.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Path to corpus directory |
| `--navigation_mode` | `policy` | `policy`, `manual`, or `both` |
| `--manual_out_path` | `<corpus_dir>/manual_navigation.npz` | Output path for manual artifact |
| `--manual_kdtree_leafsize` | 32 | Leaf size used when fitting manual `cKDTree` |
| `--epochs` | 2000 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--hidden` | 256 | GRU hidden dimension |
| `--layers` | 2 | GRU layers |
| `--seq_len` | 32 | Training sequence length |

Outputs:
- Policy mode: `corpus/.../latent_policy_*.pt`
- Manual mode: `corpus/.../manual_navigation.npz`
- Both mode: both artifacts

### 3. Perform

Real-time decoding with selectable policy/manual navigation, OSC control, and WebSocket visualization.

```bash
python bin/perform.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Path to corpus directory |
| `--manual_artifact` | `<corpus_dir>/manual_navigation.npz` | Manual navigation artifact path (required at startup) |
| `--initial_navigation_mode` | `policy` | Initial selected mode (`policy` or `manual`) |
| `--autostart` | `false` | Start transport immediately on launch |
| `--policy_path` | optional | Path to policy checkpoint |
| `--osc_port` | 9000 | OSC server port |
| `--ws_port` | 8765 | WebSocket server port |
| `--output_gain` | 1.0 | Initial output gain |
| `--smoothing` | 0.1 | Crossfade smoothing |
| `--window_size` | 2 | Initial policy decode window size (frames) |
| `--fixed_window` | false | Disable adaptive policy window sizing |
| `--boundary_window_updates` | false | Apply adaptive window changes only on boundaries |

Open `web/index.html` in a browser, choose a navigation tab, and press **Start Decode**.
`perform.py` exits early if `manual_navigation.npz` is missing or invalid.

## Architecture

```
Audio Files
    │
    ▼
┌─────────────────┐
│   Preprocess    │  VAE encode → Segment → Normalize → Geometry
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  corpus.npz     │  64D latents + kNN + PCA + metadata
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Train Policy   │  GRU learns navigation dynamics
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Policy (.pt)   │  Gaussian mixture + window predictions
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    Perform                           │
│  ┌───────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ OSC Input │───▶│  Navigation  │───▶│ Manifold │ │
│  └───────────┘    │   Engine     │    │ Constrain│ │
│                   └──────────────┘    └──────────┘ │
│                          │                  │       │
│                          ▼                  ▼       │
│                   ┌──────────────┐    ┌──────────┐ │
│                   │  WebSocket   │    │   VAE    │ │
│                   │  Broadcast   │    │  Decode  │ │
│                   └──────────────┘    └──────────┘ │
│                          │                  │       │
│                          ▼                  ▼       │
│                   ┌──────────────┐    ┌──────────┐ │
│                   │     Web      │    │  Audio   │ │
│                   │     UI       │    │  Output  │ │
│                   └──────────────┘    └──────────┘ │
└─────────────────────────────────────────────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **LatentNavigationEngine** | `runtime/player.py` | Maintains 64D position, runs policy, applies controls |
| **ManifoldConstrainedGenerator** | `runtime/manifold.py` | Projects navigation onto corpus manifold via PCA |
| **DecoderPlayer** | `runtime/decoder_player.py` | Streaming audio with overlap-add crossfade |
| **LatentPolicy** | `policy/latent_policy.py` | GRU model with Gaussian mixture output |
| **LatentGeometry** | `policy/latent_geometry.py` | kNN indices, local density, PCA components |

## Control Protocol

### OSC Messages (default port 9000)

**Navigation Controls** (0.0 - 1.0):
```
/policy/width       Temperature for sampling breadth
/policy/energy      Displacement magnitude
/policy/gravity     Forward (>0.5) / backward (<0.5) bias
/policy/memory      Attraction to recent positions
/policy/coherence   Bias toward same source file
/policy/exploration Entropy injection for diversity
/policy/reset       Reset navigation state (no value)
```

**Decoder Controls** (0.0 - 1.0):
```
/decoder/gain       Output volume
/decoder/smoothing  Crossfade smoothing
```

**Cursor**:
```
/cursor x [y [z ...]]   Set cursor position (up to 64D)
```

Manual mode faders are web-only in this release (no OSC manual-fader endpoints).

### Web UI

The web interface (`web/index.html`) provides:
- Real-time 2D projection of corpus and trajectory
- Policy/manual mode tabs with transport Start/Stop buttons
- Sliders for all 6 policy navigation controls
- 8 manual faders for KD-tree nearest-frame retrieval
- Decoder gain and smoothing controls
- Visualization options (point size, trail length, heatmap)
- Connection status and position readout

## Corpus Format

The `corpus.npz` file contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `Z_concat` | `[N, 64]` | Normalized frame-level latents |
| `file_offsets` | `[num_files + 1]` | Frame offsets per source file |
| `meta` | `[N, 3]` | (file_id, t_lat, win_lat) per segment |
| `paths` | `[M]` | Source audio file paths |
| `Z_mean`, `Z_std` | `[64]` | Denormalization statistics |
| `window_targets_log2` | `[N]` | Adaptive policy window targets in log2(frame) space |
| `geom_knn_indices` | `[N, K]` | Neighbor indices |
| `geom_knn_distances` | `[N, K]` | Neighbor distances |
| `geom_local_sigma` | `[N]` | Local density scale |
| `geom_pca_components` | `[D, 64]` | Full-rank PCA matrix |
| `geom_pca_mean` | `[64]` | PCA centering mean |
| `manual_pca_points` | `[N, 8]` | Manual navigation coordinates (MFCC -> PCA) |
| `manual_pca_components` | `[8, 20]` | PCA basis over z-scored MFCC features |
| `manual_pca_mean` | `[20]` | PCA centering mean in MFCC-z space |
| `manual_fader_p01` | `[8]` | Per-dimension 1st percentile range floor |
| `manual_fader_p99` | `[8]` | Per-dimension 99th percentile range ceiling |

Geometry fields (`geom_*`) and manual fields (`manual_*`) are required for full dual-mode performance.

## Configuration

Global constants in `stable_audio_wanderer/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `SR` | 44100 | Audio sample rate |
| `LATENT_HZ` | 21.5 | VAE latent frame rate |
| `DEVICE` | auto | MPS / CUDA / CPU |

Default preprocessing:
- Segment: 200ms window, 50ms hop
- kNN: k=32, cosine distance
- Adaptive windows: 2/4/8/16/64 frames based on latent velocity

## Key Algorithms

### Latent Geometry
- **kNN Search**: FAISS IndexFlatIP on L2-normalized latents
- **Local Density**: Median k-NN distance (local_sigma)
- **PCA Projection**: Full-rank SVD for manifold operations

### Navigation Dynamics
- **Velocity Decay**: Momentum-based movement (α=0.95)
- **Manifold Attraction**: Pull toward kNN centroid (β=0.1)
- **Policy Integration**: GRU with 4-component Gaussian mixture output

### Audio Reconstruction
- **Overlap-Add**: 50% Hann window overlap (COLA compliant)
- **Logarithmic Crossfade**: Perceptually linear loudness blending
- **Adaptive Windows**: 2-64 frames based on latent velocity or policy prediction

## Project Structure

```
stable-audio-wanderer/
├── bin/
│   ├── preprocess.py      # Audio → corpus pipeline
│   ├── train_policy.py    # Policy training
│   └── perform.py         # Real-time performance
├── stable_audio_wanderer/
│   ├── config.py          # Global constants
│   ├── io/
│   │   ├── audio_io.py    # WAV loading/saving
│   │   └── corpus_io.py   # NPZ serialization
│   ├── policy/
│   │   ├── latent_policy.py    # GRU policy network
│   │   ├── latent_geometry.py  # kNN, PCA, density
│   │   └── sequence.py         # Temporal grouping
│   ├── vae/
│   │   ├── sae.py         # VAE encoding
│   │   └── decoder.py     # VAE decoding
│   └── runtime/
│       ├── player.py          # Navigation engine
│       ├── manual_player.py   # 8-fader manual KD-tree engine
│       ├── manifold.py        # Manifold constraint
│       ├── decoder_player.py  # Audio streaming
│       ├── osc_server.py      # OSC control
│       └── ws_server.py       # WebSocket server
├── web/
│   ├── index.html         # Visualization UI
│   └── sketch.js          # p5.js rendering
├── docs/                   # Technical documentation
├── requirements.txt
└── setup.py
```

## Acknowledgments

- [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) VAE by Stability AI
- [FAISS](https://github.com/facebookresearch/faiss) for fast nearest neighbor search
- [p5.js](https://p5js.org/) for web visualization
