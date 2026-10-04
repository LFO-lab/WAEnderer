# Stable Audio Wanderer

A real-time latent space navigation instrument for exploring audio corpora. Audio is encoded with a selectable VAE, segmented into a geometry-enabled latent corpus, and navigated using a learned GRU policy with manifold-constrained generation for real-time synthesis.

## Features

- **Latent Navigation** - Explore audio corpora in continuous latent space
- **Manual Navigation Mode** - 4-axis nearest-frame retrieval in descriptor embedding space (PCA or UMAP)
- **Random Morphologies Mode** - GRU-guided + corpus-locked timbre recomposition
- **Reorganized Morphologies Mode** - Unit-graph sequencing with optional learned transition scorer
- **Manifold-Constrained Generation** - Stay on the learned audio manifold with adaptive PCA projection
- **Real-Time Decoding** - Torch in standalone mode; corpus-selected ONNX/PyTorch with normalized full-output overlap-add in the unified Web performance path
- **WebSocket Visualization** - p5.js interface showing trajectory, controls, and corpus structure
- **Explicit Transport** - Choose mode, then Start/Stop decoding from the web UI
- **OSC Control** - Full parameter control for integration with external controllers

## Installation

Requires Python 3.11+. The packaged Web decoder requires ONNX Runtime 1.26 or
newer, whose supported Python range sets this project minimum.

```bash
pip install -r requirements.txt
```

For a reproducible development environment, install from the project metadata
and committed lockfile instead:

```bash
uv sync --all-extras
```

### Dependencies

The native SAME-S decoder (selectable in Perform → Decode with)
has a separately validated installation profile. See
[native decoder setup and validation](docs/DUAL_INFERENCE_PHASE2.md) for the
pinned weights, PyTorch environment, supported API and hardware test status.
See [pipeline selection and lifecycle](docs/DUAL_INFERENCE_PHASE3.md) for the
configuration fields and stop/reconfigure sequence. The
[Web selector guide](docs/DUAL_INFERENCE_PHASE4.md) explains availability, loading
and the Stop Perform → select → Start Perform workflow.

| Category | Packages |
|----------|----------|
| Core | `torch`, `torchaudio`, `numpy`, `soundfile` |
| Navigation | `scikit-learn`, `scipy`, `faiss-cpu`, `umap-learn` |
| Runtime | `sounddevice`, `python-osc`, `websockets`, `onnxruntime` |
| VAE (Stable Audio Open) | `diffusers`, `transformers`, `accelerate`, `safetensors` |
| ONNX export extra | `onnx`, `onnxscript` |

### Using Other VAEs

The pipeline supports pluggable VAE backends. The default is [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) (44.1 kHz, 64D latents). Additional VAEs can be selected from the GUI dropdown or via CLI flags.

#### EAR VAE

[EAR VAE](https://huggingface.co/earlab/EAR_VAE) supports 44.1 kHz and 48 kHz encoding with 64D latents. To use it:

1. **Clone the repository** (contains model code + pretrained weights):
   ```bash
   git clone https://huggingface.co/earlab/EAR_VAE /path/to/EAR_VAE
   ```

2. **Install the EAR profile with normal dependency resolution**:
   ```bash
   python -m pip install -r requirements-ear-native.txt
   python -m pip check
   ```

   The profile pins the audio-tools revision compatible with the ONNX protobuf
   requirements. The EAR repository and checkpoint remain explicit model assets.

3. **Select in the GUI**: Choose "EAR VAE (48k)" or "EAR VAE (44.1k)" from the VAE dropdown in the Preprocess panel, then provide the path to the `.pyt` weight file (e.g. `/path/to/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt`).

   **Or via CLI**:
   ```bash
   python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus \
       --vae_id ear_vae_48k \
       --vae_weight_path /path/to/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt
   ```

The corpus records which VAE was used (`vae_id`), so train and perform phases automatically load the correct adapter and parameters.

[SAME-S](https://huggingface.co/stabilityai/SAME-S) is also available as a 44.1 kHz stereo adapter with 256D latents and a 4096x temporal compression ratio. Install its pinned native profile, honoring upstream Torch/Torchaudio requirements:

```bash
python -m pip install -r requirements-same-s-native.txt
python -m pip check
```

For both SAME-S and EAR, use `requirements-native.txt` in a fresh Python 3.12 environment. Environment names and locations are unrestricted. The combined source profile resolves successfully for Python 3.12; it still needs clean-install/runtime validation; the existing macOS SAME-S lock and EAR export lock record separate historical reference environments.

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

#### Unified Web performance (corpus-selected VAE)

The unified Web server defaults to an app-owned SAME-S ONNX realtime decoder.
Pipeline configuration also supports the native decoder; see the
[selection and lifecycle protocol](docs/DUAL_INFERENCE_PHASE3.md).
SAME-S is the default encoder in the Web UI, while the other encoder
choices also work in Perform through their existing PyTorch adapters: Stable Audio Open, EAR 44.1 kHz and EAR 48 kHz. Stable Audio Open additionally supports a locally prepared ONNX CPU decoder. The corpus determines the VAE; the selector offers compatible execution devices. EAR requires its local `.pyt` path in Perform and the EAR repository/dependencies. These paths do not use the SAME-S ONNX graph.

Prepare the untracked release resource once from local SAME-S weights:

```bash
uv run python bin/export_web_decoder.py \
  --source-revision fbeb3dcf53a326e5682f38e22e7f740202d44232
SAW_RELEASE_BUILD=1 uv run python setup.py build_py
```

The exporter passes the registered model name `same-s` to `stable-audio-3`
(a filesystem placeholder such as `/path/to/same-s` is not valid). It verifies
every even T from T2 through T32 against Torch, then writes the dynamic
model, lightweight metadata, and parity report into the package resource
directory. Release builds fail when any of these inputs is absent.
The exporter downloads both SAME-S files from that exact revision and refuses
an unpinned or cache-substituted checkpoint. The revision is written to
`decoder.json` together with the exported ONNX SHA-256, upstream model URL,
license, and conversion description.
When updating SAME-S, verify the new Hugging Face commit and pass that full
revision explicitly rather than using a branch name.

Prepare Stable Audio Open ONNX explicitly from the installed pinned weights:

```bash
.venv/bin/python -m bin.prepare_stable_audio_open_onnx \
  --corpus corpus/Rack_20260428_181107
```

The validated dynamic graph is stored in `~/.cache/waenderer/decoders`, covers
T2–T32, and survives restart. Preparation is offline and reuses a verified existing
export. Restart the Web server after upgrading, then select **ONNX · CPU ·
stable_audio_open** for Rack. PyTorch CPU/MPS remain separate choices. The short
CPU tests showed underruns; successful export does not establish real-time
performance. See [Phase 2 evidence and reproduction](docs/MULTI_VAE_ONNX_PHASE2.md).

Start the unified server offline with:

```bash
python bin/serve.py
```

Open `http://localhost:8080` and select or create a corpus. After preprocessing
or training, that corpus remains selected. **Start Perform** is enabled after corpus-specific availability is received. The decoder loads lazily on Start and checks its model, I/O contract, windows and corpus geometry. SAME-S and prepared Stable Audio Open/EAR corpora offer ONNX CPU or native PyTorch on the explicitly selected device. There is no fallback to another VAE. Stable Audio Open uses cached Hugging Face weights in the default environment, without requiring `stable-audio-3`. See [multi-VAE restoration and measured limits](docs/MULTI_VAE_WEB.md).

All four VAEs now share model-specific decoder choices, visible unavailability reasons, and explicit **Prepare ONNX decoder / Retry** controls. A verified ONNX CPU artifact is the default for a new corpus; explicit backend choices persist across refresh/reconnect/reload. Preparation never starts playback or changes the selected backend. Use `python -m bin.prepare_decoders --all-installed` for batch preparation/reuse. See [Phase 4 configuration, lifecycle and verification](docs/MULTI_VAE_ONNX_PHASE4.md) for separate exporter environments and the server configuration file.

Phase 5 validation tools now compare identical corpus latents and assembled audio across all four VAEs, record physical runtime measurements, and check a source-bound qualification matrix. Compact qualification reduces inference work while retaining all windows and frozen tolerances; CPU listening reviews are recorded. GPU listening and physical qualification remain pending. See [the Phase 5 code walkthrough and shared testing guide](docs/MULTI_VAE_ONNX_PHASE5.md) for short diagnostics, manual checks and the relevant functions to debug.

Native playback uses a persistent decoder process so its interpreter and GPU runtime are independent of the Web server. The current interpreter is used by default. Optional source/interpreter overrides live in the user configuration directory or a file selected by `WAENDERER_NATIVE_RUNTIMES`, using [the example](docs/native_runtimes.example.json). Relative paths resolve against the configuration file’s directory. The registry is checked on discovery and Start; availability reports include the actual interpreter. See [native runtime debugging](docs/NATIVE_DECODER_RUNTIMES.md).

Prepare EAR explicitly with `python -m bin.prepare_ear_onnx --vae-id ear_vae_44k --weights /path/to/ear_vae_44k.pyt --repo /path/to/EAR_VAE` in the EAR export environment; use `ear_vae_48k` with its matching checkpoint for 48 kHz. Both support even windows T2–T32. See [EAR preparation, validation and measured limits](docs/MULTI_VAE_ONNX_PHASE3.md). Restart an existing server after updating the code.

T2 is selected initially for SAME-S; other VAE adapters start at T8 to leave more audio time per decode. Newly exported SAME-S decoders support every even T through T32; older decoder resources expose their existing validated sizes. The window
selector remains active during playback. The current T keeps generating audio
while one replacement stream is prepared. After the replacement has enough
buffered audio, playback crossfades over 256 samples. Rapid changes update a
single latest target: an in-flight replacement is allowed to finish and become
audible, then the engine pursues the newest target. Intermediate sizes that have
not started decoding are skipped. Requested T and audible T remain separate in
WebUI/OSC feedback.

The Web runtime emits exactly one manifest-declared audio hop per decode using
the JUCE reference sine-window normalized full-output overlap-add. One coordinator
owns navigation and separate per-stream overlap-add/planner state. At most two
CPU ONNX calls run concurrently, sharing the loaded model; there is no decode-job
backlog. The sounddevice callback only consumes prepared PCM,
applies the transition/gain, and zero-fills an underrun. There is no Torch or
CoreML fallback. A runtime ONNX failure is shown in Decoder state, fades the
current audio to silence, and requires a new transport Start.

The Web visualization follows the prepared-PCM playback clock rather than the
ahead-of-playback decoder producer. It advances through the exact corpus frame
indices attached to accepted hops once per 4,096 rendered samples (about
10.77 Hz at 44.1 kHz), independently of decoder window T; the 30 fps WebSocket
stream repeats each authoritative discrete position between advances.

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
│   ├── sketch.js          # p5.js rendering
│   └── vendor/p5/         # Unmodified p5.js 1.9.0 + LGPL license
├── docs/                   # Technical documentation
├── licenses/               # Model and third-party license texts
├── LICENSE                 # Apache-2.0 for original project code
├── NOTICE                  # Model attribution and modification notice
├── THIRD_PARTY_NOTICES.md  # Dependency and redistribution inventory
├── requirements.txt
├── pyproject.toml
└── setup.py                # Release-build compliance hook
```

## Citation

If you use or adapt WÆnderer (WAEnderer) in research, software, performances,
or other creative work, please cite the project in the associated publication,
documentation, or credits using this DOI:

**[10.5281/zenodo.22275276](https://doi.org/10.5281/zenodo.22275276)**

Use the author list, title, year, and version provided by the Zenodo record
when preparing a full bibliographic reference.

This citation is requested as scholarly and creative credit; it is not an
additional condition of the Apache License 2.0. When redistributing the code
or derivative works, you must comply with the license's redistribution
conditions, including preserving applicable copyright and attribution notices
and the relevant attribution notices in [`NOTICE`](NOTICE).

## License

Except where otherwise noted, WÆnderer's original source code and documentation
are licensed under the [Apache License 2.0](LICENSE).

Model weights and derived model artifacts are not covered by that license.
SAME-S, Stable Audio Open weights, and WÆnderer's derived SAME-S ONNX decoder
are governed by the [Stability AI Community License](licenses/STABILITY_AI_COMMUNITY_LICENSE.md).
The SAME-S upstream notice set also includes the
[Gemma Terms of Use](licenses/GEMMA_TERMS_OF_USE.md). A model-bearing release
must include those terms, [`NOTICE`](NOTICE), and the model-specific notice next
to the decoder resource.

**Powered by Stability AI.**

See [Third-party notices](THIRD_PARTY_NOTICES.md) for dependency obligations and
[`docs/media/RIGHTS.md`](docs/media/RIGHTS.md) for the separate clearance status
of public demo recordings, video, images, and source audio. The ignored JUCE
application is outside this repository's Apache-2.0 grant and requires a
separate JUCE licensing decision before distribution.

Maintainers should complete the [public release checklist](RELEASE_CHECKLIST.md)
for every source or conference build.

## Acknowledgments

- [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) VAE by Stability AI
- [SAME-S](https://huggingface.co/stabilityai/SAME-S) by Stability AI
- [EAR VAE](https://huggingface.co/earlab/EAR_VAE) by earlab
- [FAISS](https://github.com/facebookresearch/faiss) for fast nearest neighbor search
- [p5.js](https://p5js.org/) for web visualization

### Adaptive windows and Manual textures (Web ONNX)

After loading Perform, select **Fixed** or **Adaptive** window length. Adaptive
mode offers minimum and maximum T from the loaded decoder's supported sizes.
It shortens windows during movement and grows them gradually during stability.
Transitions retain navigation state and use the existing staged audio crossfade;
large windows still take longer to decode. The standalone Torch performer keeps
its existing controls.

Only Manual mode offers **Window content**:

- **Source sequence** (default): retain the original consecutive source-file loop.
- **Held latent**: sustain the selected latent, smoothing changes of centre.
- **Local variation**: smoothly interpolate nearby corpus latents, with adjustable
  variation amount. Zero variation behaves like Held latent. Variation continues
  across decode windows and does not itself trigger adaptive shortening.

For synthesized textures the map shows the timbral centre, not exact source
frames. Wander's source/order controls and Reorganized construction are unchanged.

Existing local decoder resources can be validated and expanded without exporting
another graph using `PYTHONPATH=. HF_HUB_OFFLINE=1 python bin/validate_decoder_windows.py`
when the pinned Torch checkpoint is cached. This writes a separate parity/timing
report and updates decoder metadata only after every size passes. The packaged
conversion has approximate Torch parity; the report records its numerical error.


### Continuous decoder-window handoff validation

The transport regression suite deliberately blocks a replacement decode while
consuming more than the current audio reserve and issuing rapid T requests. It
checks continued old-T decoding, zero buffer underruns in that scenario, audible
intermediate handoff followed by the latest target, one navigation owner, and at
most two concurrent inference calls. Hard resets still reject stale results;
stop joins in-flight calls before resetting PCM. This fixes transition starvation,
not CPU overload: the active decoder must still sustain real-time throughput.

ONNX Runtime documents concurrent `Run` support in its
[session API](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/inference_session.h).
This path uses the existing CPU execution provider with one internal thread per
call. Actual Erae/Live playback and a sustained hardware soak remain acceptance
checks; offline decoding and a simulated PCM consumer do not exercise the audio
device driver.

A local offline smoke check on 2026-09-17 used the installed dynamic ONNX model:
concurrent T2 decodes took about 70.5–71.5 ms (92.9 ms of audio per hop), while T32
took about 817 ms. A four-second simulated real-time PCM consumer observed
T2 → T32 → T8 with zero underrun callbacks after rapid requests for T32/T16/T8.
These are short measurements with synthetic latents, not a hardware soak or a
performance guarantee under other system loads.

### Dual inference release and qualification

See [installation, engine selection and platform limits](docs/DUAL_INFERENCE_RELEASE.md)
and the [phase 5 campaign](docs/DUAL_INFERENCE_PHASE5.md). Native SAME-S uses the
separate pinned environment; the default ONNX environment is unchanged.
The recorded release campaign qualifies ONNX CPU and MPS on the tested M1 Max/Core Audio scenario: ten minutes per engine without underruns, numeric parity and comparative listening. CUDA remains experimental and unqualified. Software-clock measurements alone do not qualify an audio device.
