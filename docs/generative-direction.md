# Toward True Generativity: Manifold-Aware Latent Synthesis

## The Problem

Stable Audio Wanderer currently operates as a sophisticated concatenative synthesis system. The Stable Audio VAE encodes an artist's corpus into latent space, a navigation model explores that space, and grains from the original audio are played back based on the model's trajectory.

This architecture has interesting *behavior*—the navigation model explores, stabilizes, gets stuck, breaks free—but limited *generativity*. The sonic palette is closed: no sound can emerge that wasn't already in the corpus. The VAE functions as a learned feature extractor, replacing hand-crafted descriptors (MFCCs, spectral features) with trained representations. The decoder sits idle.

The question we're addressing: **How do we achieve timbral novelty under artist-scale constraints?**

### Constraints

| Constraint | Target |
|------------|--------|
| Dataset size | ~1 hour maximum |
| Training compute | Overnight on CPU/MPS (no CUDA) |
| Inference latency | <50ms response to input |
| Latent rate | 21.5 Hz (Stable Audio VAE) |

## Why Naive Latent Perturbation Fails

Our initial attempts at generativity involved perturbing latent vectors before decoding:

1. **Adding Gaussian noise** to retrieved latents
2. **Linear interpolation** between latent points

Both failed in characteristic ways:

### The "Same Coloration" Problem

Adding isotropic noise produced movement and texture, but with identical character regardless of dataset. We were hearing the **prior being forced onto the audio**—the decoder mapping arbitrary noise toward "generic" texture.

### The "Incoherent Morph" Problem

Linear interpolation between corpus points produced noise, not coherent structural morphs. The path cut through **regions the decoder never saw during training**.

### Root Cause: Falling Off the Data Manifold

Stable Audio's VAE was trained on massive, diverse audio. Its latent space accommodates everything from speech to orchestral music. The "valid" regions—where decoding produces coherent audio—are **thin, curved manifolds** embedded in high-dimensional space, not convex blobs.

```
    Latent Space (simplified 2D view)

    ○ ○ ○           ← Your corpus points
     ╲ │ ╱
      ╲│╱           ← Valid manifold (curved, thin)
    ───●───────×
      ╱│╲        ↑
     ╱ │ ╲       Linear interpolation passes through void

    × = invalid region (decodes to noise/prior texture)
```

Your corpus occupies a tiny, specific region of this vast space. Perturbations that leave that region don't find "nearby sounds"—they find the void.

**Key insight**: Stable Audio's latent space is a *compression* space, not a *creative* space. It was optimized for faithful encode/decode, not smooth interpolation. That's why Stable Audio needs the Diffusion Transformer on top—the DiT learned where the valid trajectories are.

## The Proposed Direction

We shift from **granular playback of corpus audio** to **realtime decoding of manifold-constrained latent sequences**.

This requires two components:

1. **Corpus-aware latent generation**: A model that produces novel latent sequences while respecting the local geometry of the corpus manifold
2. **Optimized realtime decoding**: A forward pass through the VAE decoder fast enough to synthesize audio within our latency budget

### Component 1: Manifold-Constrained Latent Generation

The navigation model produces a trajectory through 64D latent space. Currently, this trajectory is used to *index* into the corpus for grain playback. In the generative architecture, we need to transform this trajectory into *decodable* latents that:

1. **Follow navigation intent**: Preserve the exploratory behavior learned by the policy
2. **Stay on-manifold**: Produce latents the decoder can render as coherent audio
3. **Enable novelty**: Generate sounds that weren't in the corpus

The approaches below are **not alternatives**—they compose into a unified pipeline:

```
Navigation Model
       │
       ▼
   z_nav (unconstrained 64D trajectory)
       │
       ├──────────────────────────────────┐
       ▼                                  │
   ANCHOR (Approach D)                    │
   Find kNN corpus points                 │
   z_anchor = weighted centroid           │
       │                                  │
       ▼                                  │
   INTENT EXTRACTION                      │
   δ = z_nav - z_anchor                   │
   (where does nav want to go?)      ◄────┘
       │
       ▼
   VALID PROJECTION (Approach B, fallback A)
   Project δ onto local PCA subspace
   δ_valid = project(δ, local_principal_dirs)
       │
       ▼
   OPTIONAL REFINEMENT (Approach C)
   δ_refined = learned_correction(z_anchor, δ_valid)
       │
       ▼
   z_decode = z_anchor + δ_valid (or δ_refined)
       │
       ▼
   VAE Decoder → Audio
```

#### Layer 1: Anchoring (Required)

The navigation model's trajectory can wander into invalid regions. We anchor it to the corpus manifold by projecting onto nearby corpus points.

```python
def compute_anchor(z_nav, corpus_latents, geometry, k=8):
    """
    Find a stable on-manifold anchor point near the navigation position.

    Uses the existing kNN infrastructure from LatentGeometry.
    """
    # Query kNN (already computed in navigation step)
    indices = geometry.knn_indices[nearest_segment][:k]
    neighbors = corpus_latents[indices]  # [k, 64]

    # Distance-weighted centroid (Gaussian kernel)
    distances = geometry.knn_distances[nearest_segment][:k]
    weights = np.exp(-distances / (2 * geometry.local_sigma[nearest_segment]**2))
    weights /= weights.sum()

    z_anchor = (neighbors * weights[:, None]).sum(axis=0)
    return z_anchor, neighbors, weights
```

**Why this works**: The anchor is a convex combination of corpus points, guaranteed to be "near" the manifold. The navigation model's high-level decisions (which region to explore) are preserved, but the exact position is regularized.

#### Layer 2: Intent Extraction

The difference between where navigation wants to go and where we anchor tells us the "creative direction":

```python
def extract_intent(z_nav, z_anchor):
    """
    Compute the navigation model's intended perturbation.

    This captures: "the nav model wanted to move THIS direction
    from the nearest corpus region."
    """
    delta = z_nav - z_anchor
    magnitude = np.linalg.norm(delta)
    direction = delta / (magnitude + 1e-8)
    return delta, magnitude, direction
```

**Key insight**: We don't discard the navigation model's output—we *interpret* it as a desired perturbation direction. The manifold constraint determines *how much* of that intent we can realize.

#### Layer 3: Valid Projection (Required)

Project the intended perturbation onto directions that exist in the local corpus neighborhood.

**Option B (Local PCA)** — Recommended default:

```python
def project_to_local_manifold(delta, neighbors, n_components=8):
    """
    Project perturbation onto local principal directions.

    Only perturbations along directions of actual variation
    in the local neighborhood are allowed.
    """
    # Local PCA of the k neighbors
    centered = neighbors - neighbors.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Keep top components (where variance exists)
    local_basis = Vt[:n_components]  # [n_components, 64]

    # Project delta onto this subspace
    coefficients = delta @ local_basis.T  # [n_components]
    delta_valid = coefficients @ local_basis  # [64]

    return delta_valid, local_basis, S[:n_components]
```

**Option A (Global PCA)** — Fallback for sparse regions:

```python
def project_to_global_manifold(delta, global_pca_components, n_components=32):
    """
    Fallback when local neighborhood is too sparse for reliable local PCA.

    Uses precomputed global corpus PCA (extended from current 2D to full rank).
    """
    global_basis = global_pca_components[:n_components]  # [n_components, 64]
    coefficients = delta @ global_basis.T
    delta_valid = coefficients @ global_basis
    return delta_valid
```

**When to use which**:
- **Local (B)**: When kNN neighborhood has sufficient variance (check `local_sigma`)
- **Global (A)**: When in sparse regions or at corpus boundaries

#### Layer 4: Magnitude Control

The projected perturbation needs magnitude control to prevent artifacts:

```python
def apply_magnitude_control(delta_valid, local_sigma, max_sigma_mult=2.0):
    """
    Clip perturbation magnitude based on local density.

    In dense regions: allow larger perturbations (more corpus support)
    In sparse regions: restrict to stay closer to known points
    """
    magnitude = np.linalg.norm(delta_valid)
    max_magnitude = local_sigma * max_sigma_mult

    if magnitude > max_magnitude:
        delta_valid = delta_valid * (max_magnitude / magnitude)

    return delta_valid
```

**Perceptual interpretation**:
- `max_sigma_mult = 1.0`: Conservative, stays very close to corpus (subtle coloration)
- `max_sigma_mult = 2.0`: Moderate exploration (recommended starting point)
- `max_sigma_mult = 3.0+`: Aggressive, may produce artifacts in sparse regions

This becomes a **user-controllable parameter** (maps to existing "exploration" control).

#### Layer 5: Optional Learned Refinement (Future)

If PCA-based projection still produces audible artifacts, train a small network to correct:

```python
class ManifoldCorrector(nn.Module):
    """
    Learns to push invalid latents back onto the manifold.

    Training signal: encode → perturb → decode → re-encode
    Valid perturbations round-trip with low reconstruction error.
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # (anchor, delta)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),  # refined delta
        )

    def forward(self, z_anchor, delta_valid):
        x = torch.cat([z_anchor, delta_valid], dim=-1)
        correction = self.net(x)
        return delta_valid + correction  # residual refinement
```

**Training procedure**:
1. Sample corpus point `z_corpus`
2. Perturb with candidate delta: `z_perturbed = z_corpus + delta`
3. Decode to audio: `audio = decode(z_perturbed)`
4. Re-encode: `z_reencoded = encode(audio)`
5. Loss: `||z_reencoded - z_perturbed||` + `||z_reencoded - z_corpus||`

**This is optional**—start with PCA-based projection (Layers 1-4) and only add learned refinement if needed.

#### Complete Generation Pipeline

```python
class ManifoldConstrainedGenerator:
    """
    Transforms navigation trajectory into decodable latents.
    """
    def __init__(self, corpus_latents, geometry, global_pca, n_local=8, n_global=32):
        self.corpus = corpus_latents
        self.geometry = geometry
        self.global_pca = global_pca
        self.n_local = n_local
        self.n_global = n_global

    def generate(self, z_nav, nearest_segment, exploration=0.5):
        """
        Args:
            z_nav: [64] navigation model's current position
            nearest_segment: int, index of nearest corpus segment
            exploration: float 0-1, maps to magnitude control

        Returns:
            z_decode: [64] manifold-constrained latent for decoding
        """
        # Layer 1: Anchor
        z_anchor, neighbors, weights = self.compute_anchor(
            z_nav, nearest_segment, k=16
        )

        # Layer 2: Intent
        delta = z_nav - z_anchor

        # Layer 3: Valid projection
        local_sigma = self.geometry.local_sigma[nearest_segment]
        if local_sigma < self.sparse_threshold:
            # Sparse region: use global PCA
            delta_valid = self.project_global(delta)
        else:
            # Dense region: use local PCA
            delta_valid = self.project_local(delta, neighbors)

        # Layer 4: Magnitude control
        max_mult = 1.0 + exploration * 3.0  # exploration=0→1x, exploration=1→4x
        delta_valid = self.apply_magnitude_control(
            delta_valid, local_sigma, max_mult
        )

        # Final output
        z_decode = z_anchor + delta_valid
        return z_decode
```

#### Relationship to Current Controls

The existing 6D control parameters map naturally to this pipeline:

| Control | Current Effect | Generative Effect |
|---------|---------------|-------------------|
| `exploration` | Entropy injection | Perturbation magnitude (`max_sigma_mult`) |
| `width` | Temperature scaling | Local PCA component count (`n_local`) |
| `coherence` | File bias | Anchor weight concentration |
| `memory` | Position history | Temporal smoothing of `z_decode` |
| `energy` | Displacement magnitude | Scales navigation velocity before anchoring |
| `gravity` | Temporal bias | Bias toward time-gradient direction in projection |

The control semantics are preserved—the underlying mechanism changes from "which grain to play" to "how to perturb the latent."

### Component 2: Realtime Decoder Optimization

With granular playback removed, audio synthesis depends entirely on the VAE decoder's forward pass. This must complete within our latency budget.

#### Optimization Strategies

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| **Quantization** | INT8/INT4 weights | Minor quality loss, significant speedup |
| **Distillation** | Train smaller decoder on corpus-specific data | Training cost, but inference wins |
| **Chunked streaming** | Decode N frames, overlap-add | Adds buffer latency, smooths compute |
| **MPS optimization** | Metal Performance Shaders on Apple Silicon | Platform-specific, but relevant for target users |
| **ONNX/CoreML export** | Framework-level optimization | One-time conversion cost |

#### Latency Budget Analysis

At 21.5 Hz latent rate, each frame represents ~46.5ms of audio.

For 50ms interaction latency:
- We can afford ~1 frame of buffering
- Decoder must complete in <50ms per frame
- Leaves room for navigation model + audio output

**Critical unknown**: Current decode cost per frame on target hardware (M1/M2 MacBook, CPU fallback). This determines feasibility.

## Architecture Sketch

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Artist Audio ──► Stable Audio VAE ──► Corpus Latents     │
│        │                  (encoder)           │             │
│        │                                      │             │
│        │                          ┌───────────┴──────────┐  │
│        │                          │                      │  │
│        │                          ▼                      ▼  │
│        │                   Navigation Model      Manifold   │
│        │                   (learns dynamics)     Geometry   │
│        │                                        (PCA/local) │
│        │                                                    │
└────────┼────────────────────────────────────────────────────┘
         │
         │
┌────────┼────────────────────────────────────────────────────┐
│        │            INFERENCE PHASE                         │
├────────┼────────────────────────────────────────────────────┤
│        │                                                    │
│        ▼                                                    │
│   User Input ──► Navigation ──► Manifold-Constrained ──►   │
│   (steering)      Model          Latent Generation          │
│                                        │                    │
│                                        ▼                    │
│                              Optimized VAE Decoder          │
│                                        │                    │
│                                        ▼                    │
│                                   Novel Audio               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Open Questions

1. **Decoder feasibility**: What is the actual per-frame decode cost on MPS/CPU? This determines whether realtime decode is viable or requires aggressive optimization.

2. **Manifold learning complexity**: How much corpus structure do we need to capture? Global PCA may suffice for some corpora; others may require local methods.

3. **Temporal coherence**: Generating frame-by-frame may produce artifacts at frame boundaries. Do we need overlap-add, or does the VAE's temporal receptive field handle this?

4. **Steering integration**: How do user controls map to manifold exploration? Current navigation model behavior (explore/stabilize/escape) should transfer, but the interface to latent generation needs design.

5. **Hybrid fallback**: Should we keep corpus playback as a fallback for when decode can't keep up? Or commit fully to the generative path?

## Existing Infrastructure (Codebase Analysis)

The current codebase already provides significant infrastructure for manifold-aware navigation. This section documents what exists and how it maps to the proposed generative architecture.

### VAE Integration

**Location**: `stable_audio_wanderer/vae/sae.py`

```python
# Current: encoder only
from diffusers import AutoencoderOobleck
vae = AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")

# Encoding (currently used)
z = vae.encode(audio_tensor).latent_dist.sample()  # [B, 64, T_lat]

# Decoding (NOT currently used - this is what we need)
audio = vae.decode(z).sample  # [B, 2, T_audio]
```

**Key constants** (`config.py`):
- `SR = 44100` (sample rate)
- `LATENT_HZ = 21.5` (latent frame rate)
- `DEVICE = "mps" | "cuda" | "cpu"` (auto-detected)

**MPS workarounds**: Chunked encoding for inputs >240s to avoid Metal shader bugs.

### Geometry Infrastructure

**Location**: `stable_audio_wanderer/policy/latent_geometry.py`

The `LatentGeometry` dataclass already computes and stores:

| Field | Shape | Description |
|-------|-------|-------------|
| `knn_indices` | `[N, 32]` | 32 nearest neighbors per corpus point |
| `knn_distances` | `[N, 32]` | Cosine distances to neighbors |
| `local_sigma` | `[N]` | Median kNN distance (local scale) |
| `local_density` | `[N]` | Inverse sigma (density estimate) |
| `time_gradients` | `[N, 64]` | Temporal direction vectors per point |
| `pca_components` | `[2, 64]` | Global PCA for 2D projection |
| `pca_mean` | `[64]` | PCA centering vector |
| `centroid` | `[64]` | Global corpus centroid |

**What's missing for manifold-constrained generation**:
- Full PCA components (currently only 2D for visualization)
- Local PCA per region (currently only global)
- Valid perturbation direction predictor (Approach C)

### Navigation Model

**Location**: `stable_audio_wanderer/policy/latent_policy.py`

The `LatentPolicy` is a GRU that predicts:
- **4-component Gaussian mixture** for displacement
- **Velocity updates** for momentum-based navigation

**Input features** (already computed per step):
- Current 64D position and velocity
- 6 control parameters (width, energy, gravity, memory, coherence, exploration)
- 16D local geometry features (density, time alignment, centroid distance, etc.)

**This model can be preserved**: The navigation model's trajectory through latent space is what we want. The change is what happens *after* we have a latent vector—decode instead of grain lookup.

### Current Playback Pipeline

**Location**: `stable_audio_wanderer/runtime/grain_player.py`

```
Navigation Engine
       │
       ▼
get_render_weights() → (kNN indices, Gaussian weights, times, file_ids)
       │
       ▼
GrainScheduler (4 streams, staggered phases)
       │
       ▼
GrainPlayer (64 pyo voices, pre-rendered grain buffers)
       │
       ▼
Audio Output
```

**What gets replaced**:
- `GrainScheduler` → Latent sequence generator
- `GrainPlayer` → Streaming VAE decoder
- Pre-rendered grain buffers → Removed entirely

**What stays**:
- `LatentNavigationEngine` (drives trajectory)
- `LatentPolicy` (learned dynamics)
- `LatentGeometry` (extended for manifold constraints)
- OSC/WebSocket control interfaces

### Integration Points

**`bin/preprocess.py`** - Add manifold geometry computation:
```python
# Current: only computes 2D PCA for visualization
# Needed: full PCA components for corpus-constrained perturbation
# Needed: local covariance structures (optional, for Approach B)
```

**`runtime/player.py`** - Navigation engine stays, output changes:
```python
# Current: returns indices into pre-rendered grains
# New: returns latent vectors for decoding
```

**`bin/perform.py`** - Main runtime loop replacement:
```python
# Current:
while running:
    indices, weights, times, file_ids = nav.get_render_weights()
    scheduler.set_latent_render_data(...)  # grain scheduling
    time.sleep(interval)

# New:
while running:
    z = nav.get_current_latent()           # 64D vector
    z_perturbed = manifold_perturb(z)      # corpus-constrained
    audio = decoder.decode_frame(z_perturbed)  # realtime decode
    audio_output.write(audio)
```

## Benchmarking Decoder Throughput

A benchmark script has been created at `bin/benchmark_decoder.py`.

**Usage**:
```bash
python bin/benchmark_decoder.py [--device mps|cpu|cuda] [--trials 50]
```

### Benchmark Results (Apple Silicon MPS)

Tested on the development machine:

| Frames | Audio Duration | Decode Time (mean) | Decode Time (P95) | Realtime Ratio |
|--------|---------------|-------------------|-------------------|----------------|
| 1 | 46.4ms | **12.31ms** | 13.91ms | 0.26x |
| 2 | 92.9ms | 14.45ms | 15.91ms | 0.16x |
| 4 | 185.8ms | 17.86ms | 19.47ms | 0.10x |
| 8 | 371.5ms | 25.32ms | 26.69ms | 0.07x |

**Result: REALTIME VIABLE**

Single-frame decoding completes in ~12ms with a budget of ~46ms. This leaves **~34ms of headroom** for:
- Navigation model inference (~2-5ms)
- Manifold perturbation computation (~1-2ms)
- Audio output and system overhead (~5-10ms)

**Recommended architecture: Path A (True Realtime Streaming)**

No chunking, double-buffering, or optimization required. Frame-by-frame decode is feasible.

> Note: CPU fallback will be significantly slower. Run benchmark with `--device cpu` to assess non-MPS performance.

## Decoder Architecture Options

Based on benchmark results, one of these paths:

### Path A: True Realtime (if decode < 30ms/frame) ✓ SELECTED

```
┌─────────────────────────────────────────────────┐
│              STREAMING DECODE                    │
├─────────────────────────────────────────────────┤
│                                                  │
│   Navigation  ──►  Manifold   ──►   VAE      ──►│──► Audio
│   Engine          Perturb         Decode         │    Output
│   (21.5 Hz)       (per frame)    (per frame)    │
│                                                  │
│   Latency: ~50ms (1 frame + overhead)           │
└─────────────────────────────────────────────────┘
```

### Path B: Chunked Streaming (if decode 30-100ms/frame)

```
┌─────────────────────────────────────────────────┐
│              DOUBLE-BUFFERED DECODE              │
├─────────────────────────────────────────────────┤
│                                                  │
│   Navigation  ──►  Latent   ──►  Async   ──►   │
│   Engine          Buffer       Decoder          │
│   (21.5 Hz)       (N frames)   (N frames)       │
│                                                  │
│        Playing Buffer A ◄──┬──► Decoding B      │
│                            │                     │
│                       (swap on complete)         │
│                                                  │
│   Latency: ~N×46ms (chunk size)                 │
└─────────────────────────────────────────────────┘
```

### Path C: Optimized Decode (if baseline too slow)

```
┌─────────────────────────────────────────────────┐
│              OPTIMIZED PIPELINE                  │
├─────────────────────────────────────────────────┤
│                                                  │
│   Options (in order of complexity):              │
│                                                  │
│   1. torch.compile() + inference_mode           │
│   2. ONNX export + ONNX Runtime                 │
│   3. CoreML export (Apple Silicon)              │
│   4. INT8 quantization                          │
│   5. Decoder distillation (corpus-specific)     │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Next Steps

1. ~~**Run decoder benchmark**~~: ✓ COMPLETE
   - Result: 12.31ms/frame on MPS (0.26x realtime)
   - Path A (true realtime streaming) is viable

2. **Extend geometry computation**: Add full PCA to preprocessing
   - Currently only 2D for visualization
   - Need all 64 components (or top-K) for perturbation

3. **Prototype decode path**: Minimal integration
   - Load VAE with decoder at runtime
   - Single-frame decode test in perform.py
   - Verify audio quality before full integration

4. **Implement manifold perturbation**: Start with global PCA (Approach A)
   - Simplest, uses existing infrastructure
   - Evaluate perceptual quality of perturbations
   - Graduate to local methods if needed

5. **Replace grain system**: Swap playback backend
   - Remove GrainPlayer/GrainScheduler
   - Implement streaming decoder with appropriate buffering
   - Preserve OSC/WebSocket control interfaces while updating the controls to match new backend

6. **Optimize as needed**: Based on benchmark results
   - Apply quantization/export if latency too high