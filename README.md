# Stable Audio Wanderer

A real-time latent space navigation instrument for exploring audio corpora. Audio is encoded with a selectable VAE, segmented into a geometry-enabled latent corpus, and navigated using a learned GRU policy with manifold-constrained generation for real-time synthesis.

## Features

- **Latent Navigation** - Explore audio corpora in continuous latent space
- **Manual Navigation Mode** - 4-axis nearest-frame retrieval in descriptor embedding space (PCA or UMAP)
- **Random Morphologies Mode** - GRU-guided + corpus-locked timbre recomposition
- **Reorganized Morphologies Mode** - Unit-graph sequencing with optional learned transition scorer
- **Manifold-Constrained Generation** - Stay on the learned audio manifold with adaptive PCA projection
- **Real-Time Decoding** - Torch in standalone mode; CPU SAME-S ONNX with normalized full-output overlap-add in the unified Web performance path
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
| Navigation | `scikit-learn`, `scipy`, `faiss-cpu`, `umap-learn` |
| Runtime | `sounddevice`, `python-osc`, `websockets`, `onnxruntime` |
| VAE (Stable Audio Open) | `diffusers`, `transformers`, `accelerate`, `safetensors` |

### Using Other VAEs

The pipeline supports pluggable VAE backends. The default is [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) (44.1 kHz, 64D latents). Additional VAEs can be selected from the GUI dropdown or via CLI flags.

#### EAR VAE

[EAR VAE](https://huggingface.co/earlab/EAR_VAE) supports 44.1 kHz and 48 kHz encoding with 64D latents. To use it:

1. **Clone the repository** (contains model code + pretrained weights):
   ```bash
   git clone https://huggingface.co/earlab/EAR_VAE /path/to/EAR_VAE
   ```

2. **Install its dependency**:
   ```bash
   pip install descript-audio-codec
   ```

3. **Select in the GUI**: Choose "EAR VAE (48k)" or "EAR VAE (44.1k)" from the VAE dropdown in the Preprocess panel, then provide the path to the `.pyt` weight file (e.g. `/path/to/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt`).

   **Or via CLI**:
   ```bash
   python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus \
       --vae_id ear_vae_48k \
       --vae_weight_path /path/to/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt
   ```

The corpus records which VAE was used (`vae_id`), so train and perform phases automatically load the correct adapter and parameters.

[SAME-S](https://huggingface.co/stabilityai/SAME-S) is also available as a 44.1 kHz stereo adapter with 256D latents and a 4096x temporal compression ratio. Install the Stable Audio 3 library without dependency resolution so this project keeps its existing Torch/Torchaudio build:

```bash
pip install --no-deps git+https://github.com/Stability-AI/stable-audio-3.git
pip install einops-exts "huggingface-hub>=1.7.1"
```

Then select "SAME-S (44.1k)" in the GUI or use:

```bash
python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus \
    --vae_id same_s
```

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
| `--vae_id` | `stable_audio_open` | VAE to use (`stable_audio_open`, `same_s`, `ear_vae_44k`, `ear_vae_48k`) |
| `--vae_weight_path` | | Path to VAE weights (required for EAR VAE) |
| `--seg_sec` | 0.2 | Segment duration in seconds |
| `--hop_sec` | 0.05 | Hop duration in seconds |
| `--latent_nav_k` | 32 | k-NN neighbors for geometry |
| `--encode_chunk_sec` | 60.0 | VAE encode chunk size in seconds (0 disables chunking) |
| `--encode_chunk_overlap_sec` | 1.0 | VAE encode chunk overlap in seconds |
| `--trim_silence` / `--no_trim_silence` | enabled | Remove long silent runs from stored corpus frames while keeping a small amount of surrounding silence |
| `--silence_threshold_db` | -45.0 | RMS dBFS threshold used to classify frames as silent |
| `--silence_min_duration_sec` | 0.25 | Only silent runs at least this long are removed |
| `--silence_keep_sec` | 0.10 | Silence padding preserved around active regions |
| `--manual_reducer` | `pca` | Manual embedding reducer (`pca` or `umap`) |
| `--manual_embed_dim` | 4 | Manual embedding dimensionality (`3` or `4`; dim4 maps to W/color axis) |
| `--manual_umap_n_neighbors` | 30 | UMAP `n_neighbors` (when reducer is `umap`) |
| `--manual_umap_min_dist` | 0.05 | UMAP `min_dist` (when reducer is `umap`) |
| `--manual_umap_metric` | `euclidean` | UMAP metric |
| `--manual_umap_random_state` | 42 | UMAP random seed |
| `--reorg_min_sec` | 2.0 | Reorganized unit minimum duration |
| `--reorg_max_sec` | 10.0 | Reorganized unit maximum duration |
| `--reorg_target_sec` | 5.0 | Reorganized unit target duration |
| `--reorg_candidate_k` | 64 | Reorganized candidate transition pool size |
| `--reorg_graph_k` | 24 | Reorganized outgoing transitions per unit |

Outputs:
- `corpus/[prefix]_YYYYMMDD_HHMMSS/corpus.npz`
- `corpus/[prefix]_YYYYMMDD_HHMMSS/policy_v2_units.npz` (always generated)

`corpus.npz` now includes manual-navigation fields:
- `manual_embed_points` `[N, 3 or 4]`
- `manual_embed_reducer` `["pca"|"umap"]`
- `manual_pca_components` `[embed_dim, D_desc]` (PCA mode only)
- `manual_pca_mean` `[D_desc]` (PCA mode only)
- `manual_desc_weighted` `[N, D_desc]`
- `manual_fader_p01` `[embed_dim]`
- `manual_fader_p99` `[embed_dim]`

### 2. Train Policy

Train navigation artifacts/models for manual, random, reorganized, or all modes.

```bash
python bin/train_policy.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Path to corpus directory |
| `--navigation_mode` | `all` | `manual`, `random`, `reorganized`, or `all` |
| `--manual_out_path` | `<corpus_dir>/manual_navigation.npz` | Output path for manual artifact |
| `--random_out_path` | `<corpus_dir>/latent_policy_<timestamp>.pt` | Output path for random model checkpoint |
| `--reorganized_out_path` | `<corpus_dir>/policy_v2_<timestamp>.pt` | Output path for reorganized model checkpoint |
| `--reorganized_units_path` | embedded / `<corpus_dir>/policy_v2_units.npz` | Reorganized unit artifact source |
| `--manual_kdtree_leafsize` | 32 | Leaf size used when fitting manual `cKDTree` |
| `--epochs` | 2000 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--hidden` | 256 | GRU hidden dimension |
| `--layers` | 2 | GRU layers |
| `--seq_len` | 32 | Training sequence length |

Outputs:
- Random mode: `corpus/.../latent_policy_*.pt`
- Reorganized mode: `corpus/.../policy_v2_*.pt`
- Manual mode: `corpus/.../manual_navigation.npz`
- All mode: all artifacts

### 2.5 Rebuild Reorganized Unit Artifact (Optional)

`preprocess.py` now always writes `policy_v2_units.npz`.
Use this script only when you want to regenerate units with different segmentation/graph hyperparameters without re-running full preprocess.

```bash
python bin/build_policy_v2_units.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Path to corpus directory |
| `--out` | `<corpus_dir>/policy_v2_units.npz` | Output path for V2 unit artifact |
| `--min_sec` | 2.0 | Minimum unit duration |
| `--max_sec` | 10.0 | Maximum unit duration |
| `--target_sec` | 5.0 | Target unit duration |
| `--candidate_k` | 64 | Candidate transition pool size per unit |
| `--graph_k` | 24 | Saved outgoing transitions per unit |
| `--weight_entry` | 0.70 | Exit→entry timbre continuity weight |
| `--weight_delta` | 0.30 | Descriptor-trajectory compatibility weight |
| `--crossfile_penalty` | 0.10 | Cost for cross-file transitions |

Output:
- `corpus/.../policy_v2_units.npz`

Listen to extracted units:

```bash
python bin/listen_v2_units.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS --num_units 12
```

If internet is unavailable, provide a local decoder checkpoint via `--pretrained /path/to/stable-audio-open-1.0`.

Train an optional reorganized transition model:

```bash
python bin/train_policy.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS --navigation_mode reorganized
```

This creates a checkpoint like `corpus/.../policy_v2_YYYYMMDD_HHMMSS.pt` that can be used at runtime.
`bin/train_policy_v2.py` remains as a compatibility wrapper and forwards to this command.

### 3. Perform

The standalone command remains the original Torch workflow with selectable
manual/random/reorganized navigation, OSC control, and WebSocket visualization:

```bash
python bin/perform.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Path to corpus directory |
| `--manual_artifact` | auto from `<corpus_dir>` | Manual navigation artifact path |
| `--initial_navigation_mode` | `random` | Initial selected mode (`manual`, `random`, or `reorganized`) |
| `--random_model_path` | latest `latent_policy_*.pt` | Random mode model checkpoint |
| `--reorganized_units_path` | `<corpus_dir>/policy_v2_units.npz` | Reorganized units artifact (required for reorganized mode) |
| `--reorganized_model_path` | latest `policy_v2_*.pt` | Optional reorganized transition model |
| `--reorganized_temperature` | 1.0 | Reorganized unit-selection sampling temperature |
| `--manual_wander_k` | 1 | Manual timbre-neighbor wander neighborhood size (`1` disables) |
| `--manual_wander_speed` | 0.0001 | Manual wander transition speed (`0.0` instant, `1.0` slowest) |
| `--manual_coarse_k` | 96 | Coarse candidate count for manual two-stage retrieval |
| `--manual_refine_k` | 16 | Refined descriptor-nearest subset size for manual retrieval/wander |
| `--manual_desc_interp_k` | 8 | Descriptor interpolation neighbors (mainly for UMAP query rerank) |
| `--manual_window_size` | 6 | Fixed manual decode batch size (`[T,D]` per chunk) |
| `--manual_buffer_ratio` | 0.15 | Manual target buffer ratio vs current chunk duration (higher = safer, higher latency) |
| `--manual_fader_motion_threshold` | 0.01 | Max-abs fader delta treated as active motion |
| `--autostart` | `false` | Start transport immediately on launch |
| `--random_timbre_swap` | `true` | Enable corpus-locked timbre-aware seed swapping in random mode |
| `--random_recompose` | `true` | Enable non-serial short-unit recomposition in random mode |
| `--osc_port` | 9000 | OSC server port |
| `--osc_debug` | `false` | Log incoming OSC messages, including unmapped paths |
| `--ws_port` | 8765 | WebSocket server port |
| `--output_gain` | 1.0 | Initial output gain |
| `--smoothing` | 0.1 | Crossfade smoothing |
| `--window_size` | 2 | Initial random/reorganized decode window size (frames) |
| `--fixed_window` | false | Disable adaptive random/reorganized window sizing |
| `--boundary_window_updates` | false | Apply adaptive window changes only on boundaries |
| `--ctrl_phrase_scale` .. `--ctrl_crossfile` | random defaults | Random mode controls |
| `--ctrl_morph_len` .. `--ctrl_reorg_crossfile` | reorganized defaults | Reorganized mode controls |

Open `web/index.html` in a browser, choose a navigation tab, and press **Start Decode**.
`perform.py` exits early if `manual_navigation.npz` is missing or invalid.
If manual descriptor fields are unavailable, random timbre swap/recompose automatically falls back to baseline contiguous retrieval.
If reorganized units are missing/invalid, reorganized mode is unavailable while manual/random continue to work.

#### AIMC unified Web performance (SAME-S ONNX)

The unified Web server uses an app-owned SAME-S ONNX model as its realtime
decoder. SAME-S is the default encoder in the Web UI, while the other encoder
choices remain available for non-realtime experiments.

Prepare the untracked release resource once from local SAME-S weights:

```bash
uv run python bin/export_web_decoder.py
SAW_RELEASE_BUILD=1 uv run python setup.py build_py
```

The exporter passes the registered model name `same-s` to `stable-audio-3`
(a filesystem placeholder such as `/path/to/same-s` is not valid). It verifies
T2/T4/T8/T16/T32 against Torch, then writes the dynamic
model, lightweight metadata, and parity report into the package resource
directory. Release builds fail when any of these inputs is absent.

Start the unified server offline with:

```bash
python bin/serve.py
```

Open `http://localhost:8080` and select or create a corpus. After preprocessing
or training, that corpus remains selected and **Start Perform** is immediately
available. The decoder loads lazily on Start and checks the packaged model,
CPU ONNX session, I/O contract, selected window, and corpus VAE geometry. A
non-SAME-S corpus reports a visible compatibility error at Start.

T2 is selected initially, with T4, T8, T16, and T32 also available. The window
selector remains active during playback; a change is decoded
into a generation-tagged staging buffer and handed off with a 256-sample
crossfade only after its first complete hop is ready. Rapid changes are
latest-wins.

The Web runtime emits exactly one manifest-declared audio hop per decode using
the JUCE reference sine-window normalized full-output overlap-add. ONNX and OLA
run on the decode worker; the sounddevice callback only consumes prepared PCM,
applies the transition/gain, and zero-fills an underrun. There is no Torch or
CoreML fallback. A runtime ONNX failure is shown in Decoder state, fades the
current audio to silence, and requires a new transport Start.

Presentation startup never downloads or exports models. Corpora contain their
latents, geometry, metadata, and trained navigation artifacts, but no decoder
weights. `bin/perform.py` remains the standalone Torch path, and `.sawbundle`
export remains available for the JUCE application.

Before presentation use, validate a network-disabled cold start on the target
Mac, then require the displayed p99 decode time to remain at least 20 ms below
the selected audio-hop duration, with zero steady-state underruns and no audible
window-transition gap. Soak the real output device for at least 30 minutes or
twice the planned presentation duration, whichever is longer. T4 is an explicit
operator-selected contingency when T2 does not meet that gate.

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
│  corpus.npz     │  Latents + kNN + PCA + metadata
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
| **LatentNavigationEngine** | `runtime/player.py` | Maintains latent position, runs policy, applies controls |
| **ManifoldConstrainedGenerator** | `runtime/manifold.py` | Projects navigation onto corpus manifold via PCA |
| **DecoderPlayer** | `runtime/decoder_player.py` | Streaming audio with overlap-add crossfade |
| **LatentPolicy** | `policy/latent_policy.py` | GRU model with Gaussian mixture output |
| **LatentGeometry** | `policy/latent_geometry.py` | kNN indices, local density, PCA components |

## Control Protocol

### OSC Messages (default port 9000)

**Random Controls** (0.0 - 1.0):
```
/random/phrase_scale
/random/jump_rate
/random/timbre_lock
/random/drift
/random/repeat_avoid
/random/crossfile
/random/reset
```

**Reorganized Controls** (0.0 - 1.0):
```
/reorganized/morph_len
/reorganized/jump_rate
/reorganized/timbre_lock
/reorganized/evolution
/reorganized/novelty
/reorganized/crossfile
/reorganized/reset
```

Backward compatibility:
```
/policy/* aliases to /random/*
```

**Decoder Controls** (0.0 - 1.0):
```
/decoder/gain       Output volume
/decoder/smoothing  Crossfade smoothing
```

**Cursor**:
```
/cursor x [y [z ...]]   Set cursor position (up to latent dim)
```

**Manual Controls** (0.0 - 1.0):
```
/manual/x              Set manual X axis
/manual/y              Set manual Y axis
/manual/z              Set manual Z axis
/manual/w              Set manual W axis
/manual/wander_k k     Set manual wander neighborhood size (1..64)
/manual/xyz x y z      Set all three manual axes at once
/manual/xyzw x y z w   Set all four manual axes at once
```

`/cursor` routing note:
- In `random`/`reorganized` mode, `/cursor ...` controls latent cursor as before.
- In `manual` mode, `/cursor ...` is routed to manual controls (`X/Y/Z[/W]`, up to control dim).

### Web UI

The web interface (`web/index.html`) provides:
- Real-time random/reorganized 2D projection and manual 3D corpus view
- Random/Reorganized/Manual mode tabs with transport Start/Stop buttons
- Separate 6-control banks for random and reorganized modes
- 4 manual XYZW controls for timbre-space navigation, plus manual wander-k
- Manual 3D camera nudges (left/right/over/under/forward/backward)
- Decoder gain and smoothing controls
- Visualization options (point size, trail length, heatmap)
- Connection status and position readout

## Corpus Format

The `corpus.npz` file contains (where `D` = latent dim, typically 64):

| Array | Shape | Description |
|-------|-------|-------------|
| `Z_concat` | `[N, D]` | Normalized frame-level latents |
| `file_offsets` | `[num_files + 1]` | Frame offsets per source file |
| `meta` | `[N, 3]` | (file_id, t_lat, win_lat) per segment |
| `paths` | `[M]` | Source audio file paths |
| `Z_mean`, `Z_std` | `[D]` | Denormalization statistics |
| `vae_id` | scalar | VAE adapter ID used for encoding (e.g. `stable_audio_open`, `same_s`, `ear_vae_48k`) |
| `sr` | scalar | Audio sample rate used for encoding |
| `latent_hz` | scalar | Latent frame rate of the VAE |
| `window_targets_log2` | `[N]` | Adaptive policy window targets in log2(frame) space |
| `geom_knn_indices` | `[N, K]` | Neighbor indices |
| `geom_knn_distances` | `[N, K]` | Neighbor distances |
| `geom_local_sigma` | `[N]` | Local density scale |
| `geom_pca_components` | `[D, D]` | Full-rank PCA matrix |
| `geom_pca_mean` | `[D]` | PCA centering mean |
| `manual_embed_points` | `[N, 3 or 4]` | Manual navigation control coordinates (descriptor embedding) |
| `manual_embed_reducer` | `["pca"|"umap"]` | Embedding method used for manual space |
| `manual_pca_components` | `[embed_dim, D_desc]` | PCA basis over weighted descriptor space (PCA reducer only) |
| `manual_pca_mean` | `[D_desc]` | PCA centering mean in weighted descriptor space (PCA reducer only) |
| `manual_desc_weighted` | `[N, D_desc]` | Full weighted timbre descriptor vectors for two-stage reranking |
| `manual_fader_p01` | `[embed_dim]` | Per-dimension 1st percentile range floor |
| `manual_fader_p99` | `[embed_dim]` | Per-dimension 99th percentile range ceiling |

Geometry fields (`geom_*`) and manual fields (`manual_*`) are required for full dual-mode performance.

## Configuration

Global constants in `stable_audio_wanderer/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `SR` | 44100 | Default audio sample rate (overridden by VAE adapter) |
| `LATENT_HZ` | 21.5 | Default VAE latent frame rate (overridden by VAE adapter) |
| `DEVICE` | auto | MPS / CUDA / CPU |

Pipeline code reads sample rate and latent rate from the active VAE adapter's `info()`, not from these globals. The globals remain for backward compatibility with standalone scripts and tests.

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
│   ├── perform.py         # Real-time performance (standalone)
│   └── serve.py           # GUI pipeline server (web UI)
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
│   │   ├── base.py        # VAEAdapter ABC + VAEInfo
│   │   ├── registry.py    # VAE registry (list/load adapters)
│   │   ├── sae.py         # VAE encoding utilities
│   │   ├── decoder.py     # VAE decoding utilities
│   │   └── adapters/
│   │       ├── stable_audio_open.py  # Stable Audio Open adapter
│   │       ├── same_s.py             # SAME-S adapter
│   │       └── ear_vae.py            # EAR VAE adapter (44k/48k)
│   └── runtime/
│       ├── player.py          # Navigation engine
│       ├── manual_player.py   # 3-axis manual timbre embedding engine
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
- [EAR VAE](https://huggingface.co/earlab/EAR_VAE) by earlab
- [FAISS](https://github.com/facebookresearch/faiss) for fast nearest neighbor search
- [p5.js](https://p5js.org/) for web visualization
