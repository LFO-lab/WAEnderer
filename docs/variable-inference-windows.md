# Variable Length Inference Windows: Design Analysis

**Date**: 2026-02-04
**Status**: Proposal
**Related**: [generative-direction.md](generative-direction.md), [manifold-refactor-changes.md](manifold-refactor-changes.md)

---

## Problem Statement

The current frame-by-frame decoding architecture produces audio at 21.5 Hz (one latent frame per ~46.5ms). While this meets latency requirements, it prevents the VAE decoder from leveraging its temporal receptive field to create coherent, evolving audio structures.

**Observed symptoms**:
- Static timbres that are recognizable but don't evolve coherently
- Audio lacks the longer-form structure present in the training corpus
- Each frame decodes independently, missing cross-frame dependencies

**Hypothesis**: The Stable Audio VAE decoder has an implicit receptive field spanning multiple latent frames. Training optimized for full-sequence reconstruction, not frame-by-frame generation. By decoding longer windows, we allow the decoder to synthesize more coherent temporal structures.

---

## Current Architecture

```
┌─────────────────┐
│  Navigation     │ 21.5 Hz (per-frame)
│  Engine         │
└────────┬────────┘
         │ NavFrame (z_nav, local_sigma, indices, etc.)
         ▼
┌─────────────────┐
│  Manifold       │ Per-frame constraint
│  Generator      │
└────────┬────────┘
         │ z_decode [64]
         ▼
┌─────────────────┐
│  VAE Decoder    │ Single frame decode (~12ms)
│                 │
└────────┬────────┘
         │ audio [~2048 samples]
         ▼
┌─────────────────┐
│  DecoderPlayer  │ Stream with crossfade
└─────────────────┘
```

**Latency budget** (from benchmarks):
| Component | Time |
|-----------|------|
| Navigation step | ~2-5ms |
| Manifold constraint | ~1-2ms |
| VAE decode (1 frame) | ~12ms |
| Audio output | ~5-10ms |
| **Total** | ~20-30ms |
| **Frame duration** | 46.5ms |
| **Headroom** | ~16-26ms |

---

## Variable Window Architecture

### Core Concept

Instead of decoding one frame at a time, accumulate N navigation frames and decode them together. The decoder sees temporal context and can produce more coherent audio.

```
┌─────────────────┐
│  Navigation     │ 21.5 Hz (continues at full rate)
│  Engine         │
└────────┬────────┘
         │ NavFrame stream
         ▼
┌─────────────────┐
│  Frame Buffer   │ Accumulate N frames
│  [N × NavFrame] │
└────────┬────────┘
         │ When buffer full
         ▼
┌─────────────────┐
│  Manifold       │ Batch constraint (N latents)
│  Generator      │
└────────┬────────┘
         │ z_decode [N, 64]
         ▼
┌─────────────────┐
│  VAE Decoder    │ Multi-frame decode
│                 │
└────────┬────────┘
         │ audio [N × ~2048 samples]
         ▼
┌─────────────────┐
│  DecoderPlayer  │ Queue N frames for playback
└─────────────────┘
```

### Latency Trade-off

With window size N:
- **Playback latency**: (N-1) × 46.5ms additional delay before first audio
- **Control responsiveness**: User input affects audio N frames later
- **Audio coherence**: Decoder has N-frame context for temporal structure

| Window Size | Added Latency | Total Response | Audio Context |
|-------------|---------------|----------------|---------------|
| 1 (current) | 0ms | ~50ms | None |
| 2 | 46.5ms | ~97ms | 93ms |
| 4 | 139.5ms | ~190ms | 186ms |
| 8 | 325.5ms | ~376ms | 372ms |
| 16 | 697.5ms | ~748ms | 744ms |

**User acceptance threshold**: The user has indicated that control responsiveness degradation is acceptable as long as audio output remains continuous and smooth. This opens the design space to larger windows (4-16 frames).

---

## Impact on Policy Training

### Current Policy Architecture

The `LatentPolicy` (GRU with Gaussian mixture output) operates at 21.5 Hz:

```python
# Per-step policy inference
z_tensor      # [1, 1, 64] current position
v_tensor      # [1, 1, 64] current velocity
ctrl_tensor   # [1, 1, 6]  control parameters
local_tensor  # [1, 1, 16] local geometry features

# Outputs
delta_mean       # [1, 4, 64] mixture means
delta_log_std    # [1, 4, 64] mixture log-stds
delta_weights    # [1, 4]     mixture weights
vel_delta        # [1, 64]    velocity update
hidden           # GRU state (persists across steps)
```

The policy predicts **displacement** (`delta_z`) for the next frame. The GRU hidden state captures temporal context within the navigation trajectory.

### How Variable Windows Affect Policy

**Key insight**: The policy's job doesn't change—it still generates a trajectory through latent space at 21.5 Hz. Variable windows affect **how that trajectory is rendered to audio**, not how it's generated.

However, there are important interactions:

#### 1. Temporal Coherence in Trajectory

If decoding N frames together, the N latents should form a **coherent sequence**. Random noise in the policy's displacement predictions will become audible artifacts when decoded together.

**Implication**: Policy training may need to emphasize **trajectory smoothness** when window sizes increase.

```python
# Potential training loss addition
trajectory_smoothness = torch.norm(delta_z[t+1] - delta_z[t])
loss += lambda_smooth * trajectory_smoothness
```

#### 2. Control Response Delay

With N-frame windows, user control changes take N frames to affect audio. The policy sees the new control values immediately, but the audio pipeline buffers them.

**Implication**: Users may find the control → audio mapping confusing. Consider:
- Smoothing control inputs over N frames
- Predictive control anticipation (look-ahead)
- Visual feedback showing the "committed" trajectory

#### 3. Window Size as Policy Output

**Core design question**: Should the policy **predict** the optimal window size?

This is attractive because:
- Local geometry varies (dense vs sparse regions)
- Some regions benefit from longer context (evolving textures)
- Others need responsiveness (transient attacks, navigation jumps)

### Policy Extension: Window Size Prediction

Extend the policy to output a window size prediction alongside displacement:

```python
class LatentPolicyWithWindow(LatentPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Window size head: predicts log-scale window preference
        self.window_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, z, v, controls, local_features, hidden):
        # ... existing forward pass ...
        out = self.output_norm(out)

        # Existing outputs
        delta_mean = ...
        delta_log_std = ...
        delta_weights = ...
        vel_delta = ...

        # NEW: Window size logit
        window_logit = self.window_head(out)  # [B, T, 1]

        return delta_mean, delta_log_std, delta_weights, vel_delta, window_logit, h_next
```

**Window size mapping**:
```python
def compute_window_size(window_logit, min_window=1, max_window=16):
    """
    Map policy logit to discrete window size.

    Logit interpretation:
      -inf → min_window (most responsive)
      +inf → max_window (most coherent)
    """
    # Sigmoid maps to [0, 1]
    p = torch.sigmoid(window_logit)
    # Scale to window range
    window_float = min_window + p * (max_window - min_window)
    # Discretize (or keep continuous for soft decisions)
    return torch.round(window_float).int()
```

---

## Window Size Determination Strategies

### Strategy A: Fixed Window Size

Simplest approach—set window size as a hyperparameter.

```python
# In perform.py
WINDOW_SIZE = 4  # Fixed 4-frame windows

frame_buffer = []
while running:
    frame = nav.step()
    frame_buffer.append(frame)

    if len(frame_buffer) >= WINDOW_SIZE:
        z_batch = stack_and_constrain(frame_buffer, manifold)
        audio = decode_latents(vae, z_batch)
        decoder.write_frames(audio)
        frame_buffer.clear()
```

**Pros**: Simple, predictable latency
**Cons**: Doesn't adapt to local structure

### Strategy B: Geometry-Adaptive Window Size

Use local geometry features to determine window size dynamically.

**Intuition**:
- **Sparse regions** (high `local_sigma`): Need more context to find coherent structure → larger windows
- **Dense regions** (low `local_sigma`): Local structure is rich → smaller windows OK
- **High velocity** (navigator moving fast): User wants responsiveness → smaller windows
- **Low velocity** (navigator stable): Can afford longer windows for quality

```python
def compute_adaptive_window_size(
    frame: NavFrame,
    nav_velocity: float,
    sigma_stats: dict,
    min_window: int = 2,
    max_window: int = 16,
) -> int:
    """
    Compute window size from local geometry and navigation state.
    """
    sigma = frame.local_sigma
    sigma_normalized = (sigma - sigma_stats['mean']) / sigma_stats['std']

    # Sparse regions → larger windows
    sparsity_factor = np.clip(sigma_normalized, -2, 2) / 4 + 0.5  # [0, 1]

    # High velocity → smaller windows (responsiveness)
    velocity_factor = 1.0 - np.clip(nav_velocity / 0.5, 0, 1)  # [0, 1]

    # Combine factors
    window_preference = 0.6 * sparsity_factor + 0.4 * velocity_factor  # [0, 1]

    # Map to window range
    window_size = int(min_window + window_preference * (max_window - min_window))
    return np.clip(window_size, min_window, max_window)
```

**Pros**: Adapts to local structure
**Cons**: Window size changes create discontinuities; requires tuning

### Strategy C: Policy-Predicted Window Size

Train the policy to predict optimal window size based on learned features.

**Training signal options**:

1. **Reconstruction coherence**: Decode with various window sizes, measure audio quality (spectral continuity, energy smoothness). Train policy to predict window size that maximizes quality.

2. **Temporal predictability**: Larger windows should be used when the trajectory is predictable (policy can confidently predict multiple steps). Small windows when uncertain.

3. **Corpus structure alignment**: Windows should align with natural segment boundaries in the corpus (detected during preprocessing).

```python
# Training loop with window prediction
for batch in dataloader:
    z, v, controls, local_features, targets = batch

    # Policy forward
    delta_mean, delta_log_std, delta_weights, vel_delta, window_logit, hidden = policy(
        z, v, controls, local_features, hidden
    )

    # Standard displacement loss
    displacement_loss = mixture_nll(delta_mean, delta_log_std, delta_weights, targets.delta_z)

    # Window prediction loss (if ground truth available)
    # Option 1: Supervised from offline analysis
    window_loss = F.mse_loss(window_logit, targets.optimal_window)

    # Option 2: Self-supervised from trajectory confidence
    trajectory_entropy = compute_mixture_entropy(delta_mean, delta_log_std, delta_weights)
    # High entropy → want smaller window (uncertain)
    # Low entropy → can use larger window (confident)
    window_target = 1.0 - trajectory_entropy / max_entropy  # normalized
    window_loss = F.mse_loss(torch.sigmoid(window_logit), window_target)

    loss = displacement_loss + lambda_window * window_loss
```

**Pros**: Learned, potentially optimal
**Cons**: Requires training infrastructure changes; may overfit to corpus

### Strategy D: Hybrid Control-Adaptive

Let the user control window size preference, with geometry as a modifier.

```python
# New control parameter: coherence_preference (0-1)
# 0 = prioritize responsiveness (small windows)
# 1 = prioritize audio quality (large windows)

def compute_hybrid_window_size(
    coherence_preference: float,  # User control
    local_sigma: float,
    velocity: float,
    min_window: int = 1,
    max_window: int = 16,
) -> int:
    # Base window from user preference
    base_window = min_window + coherence_preference * (max_window - min_window)

    # Geometry modifier (±2 frames)
    geometry_mod = np.clip((local_sigma - 0.1) * 10, -2, 2)

    # Velocity modifier (reduce window when moving fast)
    velocity_mod = -np.clip(velocity * 4, 0, 2)

    window_size = int(base_window + geometry_mod + velocity_mod)
    return np.clip(window_size, min_window, max_window)
```

**Pros**: User agency, intuitive control
**Cons**: Adds another control parameter

---

## Implementation Components

### Target Components to Modify

| Component | File | Changes |
|-----------|------|---------|
| **NavFrame** | `runtime/player.py` | Add window_size field (optional) |
| **LatentPolicy** | `policy/latent_policy.py` | Add window prediction head |
| **ManifoldGenerator** | `runtime/manifold.py` | Batch processing for N frames |
| **DecoderPlayer** | `runtime/decoder_player.py` | Multi-frame write support |
| **decode_latents** | `vae/decoder.py` | Already supports batch decode |
| **perform.py** | `bin/perform.py` | Frame buffering, adaptive loop |

### New Components

#### 1. FrameBuffer

Manages accumulation of NavFrames until window threshold:

```python
# runtime/frame_buffer.py
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .player import NavFrame

@dataclass
class WindowConfig:
    min_window: int = 2
    max_window: int = 16
    strategy: str = "adaptive"  # "fixed", "adaptive", "policy", "hybrid"
    fixed_size: int = 4  # For strategy="fixed"

class FrameBuffer:
    """Accumulates NavFrames and determines when to decode."""

    def __init__(self, config: WindowConfig, sigma_stats: dict):
        self.config = config
        self.sigma_stats = sigma_stats
        self.buffer: List[NavFrame] = []
        self.target_window_size = config.min_window

    def push(self, frame: NavFrame, velocity: float = 0.0,
             coherence_preference: float = 0.5,
             policy_window_logit: Optional[float] = None) -> Optional[List[NavFrame]]:
        """
        Add frame to buffer. Returns frames if window complete, else None.
        """
        self.buffer.append(frame)

        # Compute target window size
        if self.config.strategy == "fixed":
            self.target_window_size = self.config.fixed_size
        elif self.config.strategy == "adaptive":
            self.target_window_size = self._compute_adaptive(frame, velocity)
        elif self.config.strategy == "policy" and policy_window_logit is not None:
            self.target_window_size = self._compute_from_policy(policy_window_logit)
        elif self.config.strategy == "hybrid":
            self.target_window_size = self._compute_hybrid(
                frame, velocity, coherence_preference
            )

        # Check if window complete
        if len(self.buffer) >= self.target_window_size:
            frames = self.buffer.copy()
            self.buffer.clear()
            return frames
        return None

    def _compute_adaptive(self, frame: NavFrame, velocity: float) -> int:
        sigma = frame.local_sigma
        sigma_norm = (sigma - self.sigma_stats['mean']) / (self.sigma_stats['std'] + 1e-6)
        sparsity = np.clip(sigma_norm, -2, 2) / 4 + 0.5
        vel_factor = 1.0 - np.clip(velocity / 0.5, 0, 1)
        preference = 0.6 * sparsity + 0.4 * vel_factor
        window = int(
            self.config.min_window +
            preference * (self.config.max_window - self.config.min_window)
        )
        return np.clip(window, self.config.min_window, self.config.max_window)

    def _compute_from_policy(self, logit: float) -> int:
        p = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        window = int(
            self.config.min_window +
            p * (self.config.max_window - self.config.min_window)
        )
        return np.clip(window, self.config.min_window, self.config.max_window)

    def _compute_hybrid(self, frame: NavFrame, velocity: float,
                        coherence_preference: float) -> int:
        base = (
            self.config.min_window +
            coherence_preference * (self.config.max_window - self.config.min_window)
        )
        geom_mod = np.clip((frame.local_sigma - 0.1) * 10, -2, 2)
        vel_mod = -np.clip(velocity * 4, 0, 2)
        window = int(base + geom_mod + vel_mod)
        return np.clip(window, self.config.min_window, self.config.max_window)
```

#### 2. BatchManifoldGenerator

Extend ManifoldConstrainedGenerator for batch processing:

```python
# In runtime/manifold.py

def generate_batch(self, frames: List[NavFrame], exploration: float = 0.5) -> np.ndarray:
    """
    Generate manifold-constrained latents for a batch of frames.

    Args:
        frames: List of NavFrame from navigation engine
        exploration: [0,1] control for perturbation magnitude

    Returns:
        z_decode: [N, 64] batch of constrained latents
    """
    z_batch = []
    for frame in frames:
        z = self.generate(frame, exploration=exploration)
        z_batch.append(z)
    return np.stack(z_batch, axis=0)
```

#### 3. MultiFrameDecoderPlayer

Extend DecoderPlayer for efficient multi-frame writes:

```python
# In runtime/decoder_player.py

def write_frames(self, audio_batch: np.ndarray):
    """
    Queue multiple decoded frames for playback.

    Args:
        audio_batch: [N * frame_samples, 2] or [N, frame_samples, 2]
    """
    audio_batch = np.asarray(audio_batch, dtype=np.float32)

    if audio_batch.ndim == 3:
        # [N, frame_samples, 2] → [N * frame_samples, 2]
        audio_batch = audio_batch.reshape(-1, audio_batch.shape[-1])

    # Determine frame count from total samples
    if self.frame_samples > 0:
        n_frames = audio_batch.shape[0] // self.frame_samples
    else:
        n_frames = 1
        self.frame_samples = audio_batch.shape[0]
        self.frame_duration = self.frame_samples / float(self.sr)

    # Apply per-frame crossfade smoothing
    for i in range(n_frames):
        start = i * self.frame_samples
        end = start + self.frame_samples
        frame_audio = audio_batch[start:end]
        self.write_frame(frame_audio)
```

---

## Policy Training Considerations

### Training Data Requirements

For policy-predicted window sizes, we need ground truth or proxy signals:

1. **Offline analysis**: Pre-compute optimal window sizes for corpus trajectories using audio quality metrics.

2. **Self-supervised**: Use policy uncertainty (mixture entropy) as proxy for window size target.

3. **Online RL**: Treat window size as action, reward based on audio quality feedback.

### Curriculum Learning

Start training with fixed small windows, gradually increase:

```python
# Training schedule
epoch_to_max_window = {
    0: 1,    # Frame-by-frame initially
    10: 2,   # Introduce 2-frame windows
    20: 4,   # Expand to 4-frame
    30: 8,   # Allow up to 8
    50: 16,  # Full range
}

for epoch in range(100):
    max_window = epoch_to_max_window.get(epoch, 16)
    train_epoch(policy, dataloader, max_window=max_window)
```

### Loss Function Extensions

```python
def compute_loss(policy_output, targets, window_size):
    # Standard displacement loss
    disp_loss = mixture_nll(
        policy_output.delta_mean,
        policy_output.delta_log_std,
        policy_output.delta_weights,
        targets.delta_z
    )

    # Trajectory smoothness (weighted by window size)
    # Larger windows need smoother trajectories
    smooth_weight = 0.1 * (window_size / 16.0)
    smooth_loss = trajectory_smoothness(policy_output.delta_mean)

    # Window prediction loss (if using policy-predicted windows)
    if policy_output.window_logit is not None:
        window_loss = window_prediction_loss(
            policy_output.window_logit,
            targets.optimal_window
        )
    else:
        window_loss = 0.0

    return disp_loss + smooth_weight * smooth_loss + 0.1 * window_loss
```

---

## Recommended Implementation Path

### Phase 1: Fixed Window Baseline (1-2 hours)

1. Add `write_frames` to DecoderPlayer
2. Modify perform.py to use fixed 4-frame windows
3. Benchmark audio quality vs baseline

```bash
# Test command
python bin/perform.py --corpus_dir corpus/my_corpus --window_size 4
```

### Phase 2: Geometry-Adaptive Windows (2-3 hours)

1. Implement FrameBuffer with adaptive strategy
2. Add sigma statistics to corpus preprocessing
3. Test with adaptive window selection

### Phase 3: Hybrid User Control (1-2 hours)

1. Add `--coherence_preference` CLI argument
2. Add `/policy/coherence_preference` OSC control
3. Implement hybrid window computation

### Phase 4: Policy Window Prediction (4-8 hours)

1. Add window head to LatentPolicy
2. Compute proxy targets (entropy-based)
3. Update training loop
4. Fine-tune on existing checkpoints

---

## Open Questions

1. **Window transition artifacts**: What happens when window size changes? Need smooth handoff strategy.

2. **Buffer underrun risk**: Longer windows mean longer gaps between audio writes. Sufficient buffering needed.

3. **Memory pressure**: Buffering N NavFrames plus N latents plus decoded audio. Monitor memory usage.

4. **Visualization sync**: WebSocket state updates should reflect buffered vs playing positions.

5. **Metric validation**: How do we objectively measure "more coherent audio"? Need evaluation framework.

---

## Summary

Variable length inference windows trade control responsiveness for audio coherence. The architecture supports windows from 2-16 frames without fundamental changes. Window size can be determined by:

- **Fixed**: Simple, predictable
- **Geometry-adaptive**: Responds to local corpus structure
- **Policy-predicted**: Learned, potentially optimal
- **Hybrid**: User control with geometric modulation

The policy training can incorporate window-aware losses (trajectory smoothness, window prediction) with curriculum learning from small to large windows.

**Recommended starting point**: Fixed 4-frame windows for baseline comparison, then geometry-adaptive for production use. Policy prediction is a future enhancement once the benefit of variable windows is validated.
