# Phase 3 implementation plan — EAR 44.1 kHz and 48 kHz

Status: implemented and validated on 3 October 2026; see [completion report and evidence](MULTI_VAE_ONNX_PHASE3.md). The following preserves the original plan. This planning pass inspected code, installed package metadata and checkpoint keys; it did not instantiate EAR models, decode audio, install dependencies or export graphs.

## Verified starting point

Phase 1 already supplies artifact resolution, strict runtime geometry, dynamic/fixed graph support, complete external-file validation and atomic publication. Phase 2 supplies a working preparation/export/parity pattern, but its implementation hardcodes Stable Audio Open's source, 64 latent channels, 44,100 Hz, 2048-sample ratio and error thresholds. Those assumptions must not become EAR defaults.

Both supplied checkpoints remain present in `/Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight`. Their checkpoint and config hashes are recorded in `multi_vae_phase0_inventory.json`; implementation must reverify the bytes before loading.

| Registered VAE | Checkpoint | Configuration | Configured ratio | Checkpoint transformer keys |
| --- | --- | --- | ---: | ---: |
| `ear_vae_44k` | `ear_vae_44k.pyt` | `config/model_config.json` | 1024 | 21 |
| `ear_vae_48k` | `ear_vae_v2_48k.pyt` | `config/ear_vae_v2.json` | 960 | 0 |

Both configurations specify 64 decoder latent channels and stereo output. These are expected values to verify against actual decoded shapes, not completed geometry validation. T8 should produce 8192 samples at 44,100 Hz and 7680 samples at 48,000 Hz if those ratios are confirmed.

The 44.1 kHz decoder includes a transformer before the convolutional decoder. Both raw configs declare a transformer, but the existing adapter disables it when checkpoint keys are absent; this applies to the supplied 48 kHz checkpoint. Export must preserve the complete `EAR_VAE.decode(z)` behavior, including the 44.1 kHz transformer.

The EAR repository reports HEAD `4ce37867bc09af11e691e72708f14996eb38a85b`, but its index/worktree is not clean: paths appear as staged deletions and untracked replacements. Record actual source bytes as well as HEAD. Do not reset or otherwise alter this repository as part of export preparation.

`.venv-ear` contains Torch 2.14.1, descript-audio-codec 1.0.0, ONNX Runtime 1.26.0 and einops 0.8.2. It lacks `onnx` and `onnxscript`. Its Torch/runtime versions differ from the Phase 2 export environment; successful Stable Audio Open exports do not establish EAR compatibility.

An existing 44.1 kHz EAR corpus is `corpus/TC_Harpsichord_20261002_172959`. It records the expected latent rate but no checkpoint/config/revision provenance. No 48 kHz EAR corpus was found in the project's corpus directory.

## Increment 1 — Resolve exact EAR sources and establish native behavior

1. Add `vae/ear_weights.py` with a source descriptor shared by the adapter, native factory and exporter. Resolve explicit VAE ID, repository, checkpoint and configuration; use exact mappings for the supplied variants. Replace the filename-only config heuristic and silent v2-to-v1 fallback for this path. Reject mismatched sample-rate IDs, checkpoint/config pairs and missing files before allocating a model.
2. Preserve the existing strict checkpoint loading. Record raw config hash, checkpoint hash, canonical effective config hash after transformer reconciliation, reconciliation version and actual model-source hashes. Include repository HEAD as provenance, but bind dirty working sources through a code digest. Inspect checkpoint containers and key prefixes explicitly; do not silently strip arbitrary prefixes or relax missing/unexpected-key checks.
3. Use the same effective configuration for native and exported decoding. Hash the selected configuration rather than every unrelated JSON file. Include effective config/code identity in cache keys and expected-source checks. Carry explicit repository/config paths through native selection instead of assuming weights always live beneath the repository.
4. Validate imports originate from the selected EAR repository. The current generic `model` import plus `sys.path` mutation can reuse another repository's cached module; isolate export workers and reject conflicting native imports rather than silently loading different code.
5. Add only the missing export packages to the EAR environment, record exact resulting versions and preserve the default/SAME-S environments. Confirm `dac`, EAR modules and CPU native loading actually work. Missing optional Flash Attention/natten must not be confused with a required dependency failure.
6. Probe both actual models on CPU float32 for every even T2–T32: dimensions, finite stereo PCM, integer output ratio, repeated-input variability and load/close behavior. Probe available MPS separately. Correct the registry's 48 kHz fallback metadata and synthetic test assumptions to 960 only after confirming actual geometry.

**Gate:** each variant has a verified source descriptor and native CPU geometry report. A failed variant remains blocked explicitly; the other may proceed independently.

## Increment 2 — Parameterize preparation and implement EAR export

- Extract a small common export runner from `stable_audio_open_export.py`: supplied wrapper, source identity, geometry, sample probes, window set, numerical policy, export settings and publication target. Keep model loading and architecture decisions in model-specific modules. Keep the existing Stable Audio Open CLI compatible and its published graph unchanged.
- Add `vae/ear_export.py`. Wrap the loaded EAR model's full `decode(z)` method; do not call its convolutional decoder directly or substitute diffusers' Oobleck implementation. Encoder export remains out of scope.
- Characterize native repeatability and CPU/MPS differences, then freeze separate EAR numerical policies before inspecting ONNX results. Use exact shape/dtype/finiteness checks plus absolute and aggregate error gates, with explicit handling for silent inputs. Do not inherit Phase 2 tolerances automatically or loosen thresholds to make a failed graph pass.
- Probe the 44.1 kHz transformer early: rotary-position calculations, dynamic sequence lengths and the selected attention implementation are concrete export risks. Also test DAC weight-normalized convolutions, Snake activations and transposed-convolution output lengths in both variants. Any export-specific decomposition must preserve the native computation and pass parity; never remove the transformer to obtain a graph.
- Attempt dynamic latent time independently for each variant. Validate every advertised even window T2–T32 with zeros, seeded random inputs and encoded audio probes. A trace succeeding at T8 does not establish dynamic support.
- If dynamic export fails, retain the specific diagnostic and test fixed-window graphs under the same parity gates. Verify the complete window map, all parameter files and disk/session costs. Unsupported operations or failed parity produce an explicit failure, not a partially validated artifact.

**Gate:** each supported variant has a real graph strategy passing its frozen parity policy for all advertised windows. Stable Audio Open/SAME-S regression checks still pass after common-code extraction.

## Increment 3 — Prepare, publish and reuse both artifacts

Proposed explicit commands:

```sh
.venv-ear/bin/python -m bin.prepare_ear_onnx \
  --vae-id ear_vae_44k \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_44k.pyt

.venv-ear/bin/python -m bin.prepare_ear_onnx \
  --vae-id ear_vae_48k \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt
```

Support explicit config, optional corpus/probe audio, store directory, export settings and force/reuse controls. A corpus must not be mandatory when none exists; synthetic-only evidence must be labeled as such. Preparation stays offline and does not alter source weights.

Emit the Phase 1 manifest, complete graph/parameter hashes, source/effective-config/code identity, tool versions and a graph-bound parity report. The existing source schema accepts additional string identity fields; use that compatibility where possible. Prepare in temporary storage and publish only through `publish_artifact()`, into separate `ear_vae_44k` and `ear_vae_48k` store branches.

Reuse requires matching source, effective configuration, code, windows and export policy plus successful artifact validation. Replacing checkpoint/config/code must invalidate selection and reuse. A failed replacement leaves the previous current pointer intact. ONNX loading must remain usable without the EAR repository, checkpoint or native dependencies, unless the user explicitly supplies a source identity/weight path that requires matching.

**Gate:** two real validated artifact sets exist and survive fresh-process reload; each is bound to its own source and measured timing.

## Increment 4 — Validate audio, runtime integration and failure cases

- Create `eval_scripts/audit_multi_vae_phase3.py` with independent per-variant results. Record source identities, native CPU and available MPS comparisons, all-window parity, short assembled OLA audio, artifact sizes, runtime provider and release behavior.
- Use the existing harpsichord corpus as an additional 44.1 kHz check, explicitly noting its unproven checkpoint provenance. For stronger evidence, encode short local audio excerpts with each newly verified adapter and store the resulting fixture's checkpoint/config identity. If using the same source audio, resample it appropriately for each variant and record the source/procedure. Do not relabel Rack or 44.1 kHz latents as 48 kHz latents. Keep fixture generation separate from decoder export.
- Exercise ONNX CPU in a fresh process from a different working directory, with network and EAR/DAC/export imports blocked. Verify current-store and explicit artifact selection, correct sample rate/hop timing and normal release.
- Add EAR ONNX discovery for prepared artifacts using the existing generic runtime. Keep native CPU/GPU choices independent. For suitable EAR corpus fixtures, test stopped ONNX CPU → native CPU → available MPS → ONNX CPU transitions and offline Web restart. A full 48 kHz navigation corpus may be prepared from local source audio for this check; minimal decoder fixtures must not be represented as full Web navigation evidence.
- Test missing native dependencies, missing weights/config, wrong variant, stale source/code identity, absent/corrupt ONNX parameters, unsupported windows, failed export and checkpoint replacement. Verify an existing ONNX artifact remains available without native dependencies.

**Gate:** each variant meets Phase 3's functional artifact/native CPU exit criteria. Missing corpus/navigation or GPU evidence is recorded separately rather than inferred from another VAE.

## Verification and delivery

Add focused source/adapter/export tests, including a 44.1 kHz transformer-preservation regression, 48 kHz effective-config reconciliation, 1024-versus-960 timing, import-origin conflict, strict checkpoint mismatch, dynamic/fixed coverage and publication/reuse failures. Reuse Phase 1's artifact safety tests. Run native EAR tests in `.venv-ear`; run the full default suite and browser contracts after integration. Recheck the existing real SAME-S and Stable Audio Open artifacts without re-exporting them merely for this refactor.

Deliver four reviewable increments corresponding to the sections above. Write `docs/MULTI_VAE_ONNX_PHASE3.md` plus per-variant machine-readable native/export/runtime reports and reproducible commands. Update roadmap checkboxes only from real checkpoint evidence. If one export is blocked, preserve its failure report and leave that variant's completion open.

The principal uncertainty is the 44.1 kHz transformer's ONNX compatibility. Resolve native loading and that export probe before estimating completion. Full preparation UI/batch orchestration remains Phase 4; ten-minute real-time qualification, broad representative-corpus testing and listening approval remain Phase 5; redistribution/packaging remains Phase 6. No EAR speed or listening conclusion follows from Phase 2's Stable Audio Open results.
