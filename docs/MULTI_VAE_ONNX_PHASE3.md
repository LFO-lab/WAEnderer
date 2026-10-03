# Multi-VAE ONNX — Phase 3

Completed on 3 October 2026 using both supplied EAR checkpoints. Phase 3 covers functional decoding and export; it does not qualify real-time performance or listening quality.

## Implementation

`vae/ear_weights.py` binds the selected sample-rate variant to its checkpoint, raw configuration, effective configuration and actual EAR model-source files. The two supplied checkpoint/config pairs are verified against their recorded hashes. Unknown checkpoints require an explicit matching configuration, and all checkpoints load with `weights_only=True` and strict state-dictionary matching. A source/config/code change invalidates expected identity and native selection. Repository paths are explicit or inferred from the checkpoint, and conflicting cached `model` imports fail instead of selecting another repository silently.

The 44.1 kHz checkpoint retains its transformer. The 48 kHz checkpoint has no transformer weights, so its effective configuration removes the unused transformer declaration through a recorded reconciliation rule. Native loading and export use the same effective configuration. The full `EAR_VAE.decode()` method is exported; no transformer or convolution implementation is substituted.

`export_common.py` shares graph export, parity checks and atomic publication with Stable Audio Open. Stable Audio Open's CLI and published artifact identity remain unchanged. `ear_export.py` supplies EAR-specific geometry, source identity, probe selection and tolerances. Both variants support all even windows T2–T32. Their actual sample ratios are **1024 at 44,100 Hz** and **960 at 48,000 Hz**; the 48 kHz registry metadata and synthetic tests now agree with the actual model.

Legacy tracing rejected the 44.1 kHz transformer's dynamic LayerNorm shape. The integrated EAR exporter uses `torch.export` through `torch.onnx.export(dynamo=True)` for dynamic time. Fixed-window legacy export remains a validated fallback. Exports run in temporary storage, include every external parameter file and a source/graph-bound parity report, then publish through Phase 1's validation and atomic current-pointer update. Failed preparation cannot replace the previous usable artifact. Reuse also checks requested audio/corpus evidence so a synthetic-only result cannot stand in for a newly requested real-audio validation.

EAR discovery now exposes ONNX CPU independently from PyTorch CPU/MPS and native dependency presence. Explicit local weight/config/repository choices bind the corresponding source bytes without importing EAR/DAC into the ONNX path. With no native path selected, a validated ONNX artifact can load independently of the native repository and weights. Optional corpus provenance now includes effective-config and source-code hashes in addition to weights/config/revision.

## Environment

The default `.venv` and SAME-S environment were not modified. EAR native/export work uses `.venv-ear`. It now includes ONNX 1.22.0 and ONNXScript 0.7.0, alongside Torch 2.14.1 and ONNX Runtime 1.26.0. The original descript-audiotools 0.7.2 protobuf pin conflicted with modern ONNX. The environment therefore uses the upstream audiotools 0.7.4 source at immutable commit `348ebf2034ce24e2a91a553e3171cb00c0c71678`, with protobuf 4.25.8. See the [upstream dependency declaration](https://github.com/descriptinc/audiotools/blob/348ebf2034ce24e2a91a553e3171cb00c0c71678/setup.py). `uv pip check` passes. `requirements-ear-export.lock` records the complete macOS validation environment, including test dependencies; it is not a new default runtime requirement.

The shared ONNX external-data scanner was made compatible with protobuf 4 as well as the newer default environment. Complete external-file validation remains enforced. Optional Flash Attention and natten are absent; the installed EAR implementation uses its supported PyTorch attention path.

The external EAR repository was left untouched. Its recorded Git HEAD is `4ce37867bc09af11e691e72708f14996eb38a85b`, but its worktree/index is not clean. Actual model-source hashes, rather than HEAD alone, bind each artifact to the code used.

## Published artifacts

Both complete artifact sets are in the normal local decoder store, with a copy in the ignored workspace `build/decoders` directory:

| Variant | Artifact identity | Complete declared files | T8 output / hop |
| --- | --- | ---: | ---: |
| EAR 44.1 kHz | `9d7e752c079831cbb75787540bc890b06cadb9f0ed57d6ab5b931aa83d7e2694` | 297,426,047 bytes | 8192 / 4096 samples |
| EAR 48 kHz | `882ad286cada0d05b61d0b6fee744233e986e67c2eb9b55258c4659b2dcf75a1` | 338,136,198 bytes | 7680 / 3840 samples |

Each lives under `~/.cache/waenderer/decoders/<vae_id>/<artifact_identity>/` and contains `decoder.json`, `decoder_dynamic.onnx`, `decoder_dynamic.onnx.data` and `parity.json`. Byte totals above exclude the manifest itself. These model files are not committed or added to distribution package data.

## Numerical evidence

EAR-specific CPU/ONNX thresholds were frozen before evaluating either graph: maximum absolute error ≤ `1e-4`, RMSE ≤ `1e-5`. Native CPU repeats were identical in the initial T2–T32 probes. GPU comparison thresholds were separately fixed at `1e-3` maximum absolute error and `1e-4` RMSE before the first GPU comparison.

| Published dynamic export | Probes | Worst absolute error | Worst RMSE |
| --- | ---: | ---: | ---: |
| EAR 44.1 kHz | 128 | 3.4739e-6 | 5.3812e-7 |
| EAR 48 kHz | 80 | 2.8313e-6 | 2.8311e-7 |

Every advertised window passed exact shape, float32, finite-PCM and numerical checks. Probes include silence, seeded random latents and three local audio excerpts encoded with each verified EAR checkpoint. The 44.1 kHz export additionally includes three windows per size from the existing harpsichord corpus. Its historical checkpoint provenance is unrecorded and is not inferred; the newly encoded probe fixtures carry explicit source identity.

The 48 kHz Web validation corpus was produced by the real preprocessing pipeline from three eight-second harpsichord excerpts. It has actual 48 kHz EAR latents, manual-navigation descriptors/embedding, geometry and unit graphs. It does not reuse or relabel another VAE's latents. Audio, fixtures and the small corpus stay local, outside tracked deliverables. The corpus receipt records source audio hash, excerpt offsets, seed and EAR identity.

## Runtime and regression results

All six recorded audits passed: [44.1 kHz runtime](multi_vae_phase3_ear_44k.json), [48 kHz runtime](multi_vae_phase3_ear_48k.json), [44.1 kHz fixed fallback](multi_vae_phase3_fixed_44k.json), [48 kHz fixed fallback](multi_vae_phase3_fixed_48k.json), [44.1 kHz Web restart](multi_vae_phase3_web_44k.json), and [48 kHz Web restart](multi_vae_phase3_web_48k.json). The [48 kHz corpus receipt](multi_vae_phase3_corpus_48k.json) records its actual preprocessing inputs and provenance.

Both variants passed all even T2–T32 on native CPU, ONNX CPU and available native MPS. Worst CPU/MPS absolute errors were 2.2688e-6 (44.1 kHz) and 2.7121e-6 (48 kHz). Eight-window overlap-add comparisons passed with RMSE 3.4346e-8 and 5.0275e-8, respectively. CUDA hardware was unavailable and remains untested.

Each fixed fallback passed all 16 windows independently. Complete fixed graph sets would occupy approximately 4.74 GB and 5.40 GB, respectively; the smaller validated dynamic graphs are published. Repeat preparation reused the existing artifact identities. Fresh processes loaded both current-store and explicit-directory selections with network and native/export imports blocked.

Production Web restart tests started isolated servers twice per variant in the default environment without DAC. Recorded responses also passed frontend DOM tests for selection, native CPU visibility and reconnect. These are protocol/DOM checks, not visual browser inspection. Muted physical transport used the MacBook Pro speakers and stopped/drained between ONNX CPU → native CPU → MPS → ONNX CPU selections.

| Variant | Backend | Short-run decode p99 | Underrun callbacks |
| --- | --- | ---: | ---: |
| 44k | ONNX CPU | 321.64 ms | 62 |
| 44k | PyTorch cpu | 178.99 ms | 16 |
| 44k | PyTorch mps:0 | 49.28 ms | 0 |
| 44k | ONNX CPU (return) | 322.57 ms | 67 |
| 48k | ONNX CPU | 388.15 ms | 81 |
| 48k | PyTorch cpu | 141.72 ms | 12 |
| 48k | PyTorch mps:0 | 37.13 ms | 0 |
| 48k | ONNX CPU (return) | 386.65 ms | 82 |

These five-decode timing samples establish functioning transport, not a stable percentile or real-time qualification. CPU modes underrun; native MPS had no underruns in these short checks. Listening approval, longer performance qualification and broader corpus coverage remain Phase 5.

The default Python suite passed **326 tests, with 1 skipped**; the focused EAR-environment suite passed **64 tests**. All three JavaScript UI contract suites passed. Existing SAME-S all-window decoding and Stable Audio Open artifact reuse passed without replacing their graphs. Dependency consistency and `git diff --check` passed.

Restart an existing server to load the updated discovery code. Prepared EAR ONNX works in the default environment; native EAR requires its dependencies in `.venv-ear`. The user's existing server was left running. Preparation UI/batch orchestration remains Phase 4, and packaging remains Phase 6.

## Reproduction

From the Python project root, use the prepared EAR environment and the supplied checkpoint repository. For example:

```sh
.venv-ear/bin/python -m bin.prepare_ear_onnx \
  --vae-id ear_vae_44k \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_44k.pyt \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --fixture build/phase3/ear_vae_44k_probes.npz \
  --corpus corpus/TC_Harpsichord_20261002_172959

.venv-ear/bin/python -m bin.prepare_ear_onnx \
  --vae-id ear_vae_48k \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --fixture build/phase3/ear_vae_48k_probes.npz
```

Preparation is offline, explicit and model-specific. Optional `--config`, `--store-dir`, `--opset`, `--force` and `--fixed-only` controls are available. Omitting fixture/corpus performs clearly labeled synthetic-only checks. It does not download weights or silently prepare during playback. The broad preparation UI and batch job flow remain Phase 4.

Generate source-bound fixture files with `eval_scripts.prepare_ear_probe_fixtures` using `--vae-id`, `--weights`, `--repo`, `--audio` and `--output`. The fixture records the source audio hash, exact excerpt frames, resampling rates, encoding seed and decoder identity. Generate a full local navigation corpus with `eval_scripts.prepare_ear_web_corpus` using the same arguments; its output is a JSON receipt containing the corpus path.

Run the per-variant numerical/offline/physical audit with:

```sh
.venv-ear/bin/python -m eval_scripts.audit_multi_vae_phase3 \
  --vae-id ear_vae_44k \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_44k.pyt \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --fixture build/phase3/ear_vae_44k_probes.npz \
  --corpus corpus/TC_Harpsichord_20261002_172959 \
  --output docs/multi_vae_phase3_ear_44k.json --physical
```

Use the 48 kHz arguments and corpus receipt for the other variant. `--physical` uses muted output on the real audio device and requires hardware access. Without it, the audit still checks parity, available GPUs and an offline child process. The child blocks network connections and native/export imports, changes working directory, and validates both current-store and explicit-directory ONNX loading.

`eval_scripts.audit_ear_fixed` validates the real fixed-window fallback and discards duplicate graphs afterward. The production WebSocket restart audit supports EAR through `eval_scripts.audit_multi_vae_phase2_web --vae-id <id> --samples-per-latent <1024-or-960> --corpus <path> --output <report>`. It starts isolated servers twice and leaves an already running user server untouched.
