# Roadmap — ONNX coverage for every registered VAE

Status: **phases 0–2 completed on 2 October 2026; phases 3–6 pending**. See [phase 0 findings](docs/MULTI_VAE_ONNX_PHASE0.md) and [recorded inventory](docs/multi_vae_phase0_inventory.json).

This extends the SAME-S dual-inference roadmap. The current uncommitted multi-VAE PyTorch restoration is the starting point, not evidence that multi-VAE ONNX is implemented. Existing SAME-S qualification remains specific to its recorded hardware/scenarios.

## Required outcome

For each registered VAE, provide these independent decoding options:

| Corpus VAE | ONNX Runtime CPU | PyTorch CPU | PyTorch MPS / CUDA |
| --- | --- | --- | --- |
| SAME-S | Preserve and verify the existing graph | Expose in the menu; retain functional support | Preserve |
| Stable Audio Open | Export, persist and integrate | Retain, even if too slow for real time | Preserve |
| EAR 44.1 kHz | Implement export; generate/validate when weights are provided | Retain, requiring its weights/dependencies | Retain where supported |
| EAR 48 kHz | Implement export; generate/validate when weights are provided | Retain, requiring its weights/dependencies | Retain where supported |

The requested ONNX artifacts are **audio decoder graphs**, including the parameters needed to reconstruct audio from corpus latents. Encoder export is outside this task. Each graph must use the same model/configuration as its PyTorch counterpart.

PyTorch CPU is a functional option for every VAE, not merely a hidden diagnostic route. Slow performance must not remove this option. Installation, successful decoding and real-time qualification are separate states.

The user supplied both EAR checkpoints, verified on disk in phase 0: `/Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_44k.pyt` and `/Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt`. Their configurations and hashes are recorded. EAR dependencies are installed in `.venv-ear`; the default `.venv` lacks `descript-audio-codec`. Actual EAR loading/export validation remains phase 3 work. Do not substitute arbitrary checkpoints or claim existing ONNX exports.

## Verified starting point

- The packaged SAME-S graph exists at `stable_audio_wanderer/resources/same_s/same_s_decoder_dynamic.onnx`. Inspection found 166 embedded initializers and no external initializer files. Its source revision and artifact digest are recorded in `decoder.json`.
- SAME-S ONNX is already returned by decoder discovery for a SAME-S corpus. It is deliberately absent for a Stable Audio Open corpus. If absent with a SAME-S corpus, the actual server response, resource discovery, frontend state/reconnect handling and asset version must be investigated; the cause is not yet reproduced.
- At baseline, the SAME-S loader accepted PyTorch CPU but discovery omitted it. Phase 0 adds the CPU entry and regression coverage.
- Stable Audio Open native config/weights are already cached. Its ONNX exporter exists in `bin/export_vst_bundle.py`, using one graph per fixed latent window. No Stable Audio Open ONNX graph was found in the checkout and the Web loader currently rejects non-SAME-S ONNX.
- The old VST export path derives sample counts from rounded corpus `latent_hz`. For Stable Audio Open, rounding `44100 / 21.5` gives 2051, while actual decoding produces **2048 samples per latent**. This must be corrected before reusing its timing metadata.
- No EAR ONNX exporter exists. Both EAR registry entries require external weights and repository configuration.
- Existing Rack evidence validates a short MPS/T8 run, not PyTorch CPU or ONNX real-time performance. Fast-change MPS tests in another environment also showed underruns; do not infer a universal backend performance ranking.

## Architecture and artifact decisions

1. **Corpus selects the VAE; the user selects backend/device.** Never decode a Stable Audio Open corpus using SAME-S. Show the current corpus VAE clearly next to “Decode with”. A separate inventory may list all models and their preparation state.
2. **One shared artifact description and resolver.** Extend the decoder metadata contract to cover VAE ID, exact source/checkpoint/config identity, channels, sample rate, latent dimension, measured samples per latent, input/output layout, supported windows, graph paths, hashes, opset, export-tool versions and validation evidence. Support dynamic-time graphs and fixed-window graph maps without requiring different transport implementations.
3. **Keep existing SAME-S resources compatible.** Read the current SAME-S manifest through a compatibility adapter. Do not require re-exporting or downloading SAME-S merely to migrate the loader. Test its identity and numerical behavior before and after the change.
4. **Persist complete artifacts.** Resolve packaged resources plus a configurable local decoder store, defaulting to `~/.cache/waenderer/decoders/<vae_id>/<artifact_identity>/`. Identity includes source/config hashes and export settings. ONNX parameters may be embedded or stored in declared external-data files; all required files must exist and be hashed. Do not resolve weights relative to the current working directory.
5. **Retain native sources independently.** Preserve cached original weights/configurations for PyTorch and reproducible export. Hugging Face sources use exact revisions. EAR uses the supplied repository/configuration/checkpoint and their recorded identity. An ONNX runtime load must work without importing that VAE's native model library or accessing the network; other application components can still depend on Torch.
6. **Publish only validated exports.** Export into a temporary directory, check the graph and output geometry, run parity tests, then atomically publish its manifest/artifacts. Failure must preserve the previous usable export. Never replace source weights or a working model with a partial export.
7. **Prefer one dynamic graph when supported.** Try dynamic latent length and verify all advertised windows. If a model needs fixed graphs, store an explicit window-to-graph map and all of their parameter files. Account for duplicated disk/session memory; load/cache fixed sessions with a bounded policy. Unsupported operations must produce a specific export error, not an unverified graph.
8. **Preparation is explicit.** Provide a model-specific CLI and a batch preparation command for installed models. Batch results distinguish prepared, already valid, missing weights, and failed. Add a visible “Prepare ONNX decoder” action backed by the same job path, with progress/errors and refresh after completion. Selecting a corpus or starting playback must not silently download or export a model.

## Phase 0 — Establish the menu and model inventory baseline

- [x] Exercise the menu state with BurntMemory/SAME-S and Rack/Stable Audio Open, including reload and reconnect, by replaying recorded production responses through the JavaScript DOM harness.
- [x] Record decoder discovery responses and confirm which corpus VAE the menu represents.
- [x] Verify the packaged SAME-S graph, manifest/hash and offline load; distinguish an absent artifact from frontend filtering.
- [x] Inventory installed native sources and exported artifacts, including the supplied EAR weights and missing-weight error cases.
- [x] Add coverage for SAME-S PyTorch CPU discovery and for the intended option matrix before changing export code.

**Exit:** the missing-option behavior is explained from evidence; baseline tests describe the desired menu for each corpus, including unavailable entries. No performance conclusion is derived from mere availability.

**Result:** SAME-S ONNX is present for a SAME-S corpus and filtered out for Rack; no SAME-S-specific disappearance reproduced. Native CPU omission corrected. SAME-S artifact identity/offline decode verified; EAR source files inventoried without model loading. Tests and limitations are documented in the phase 0 report.

## Phase 1 — Generalize ONNX metadata, loading and persistence

- [x] Introduce the shared artifact description/resolver and migration support for the existing SAME-S format.
- [x] Remove SAME-S-specific geometry from the ONNX loader/factory while retaining strict corpus/model matching and finite float32 PCM checks.
- [x] Support dynamic graphs and fixed-window graph maps with validated I/O layouts and measured timing geometry.
- [x] Include model identity, graph hashes and execution settings in decoder cache keys; preserve stop/drain/release behavior.
- [x] Implement safe temporary export publication, missing/corrupt-file errors and complete external-weight resolution.
- [x] Test missing metadata, wrong VAE, stale checkpoint/config, corrupt or missing external weights, unsupported windows, replacement and offline reload.

**Exit:** the existing SAME-S artifact works through the generalized path, including ONNX CPU and native CPU/GPU selection, without weakening validation or requiring a new export.

**Result:** shared metadata/resolution, CPU loading and atomic validated publication implemented. Existing SAME-S graph and native selection preserved. Full suite: 299 passed, 1 skipped; browser contracts passed. See [phase 1 implementation and limits](docs/MULTI_VAE_ONNX_PHASE1.md) and [real SAME-S compatibility evidence](docs/multi_vae_phase1_compatibility.json). Actual Stable Audio Open/EAR exports and their native parity reports remain phases 2–3 work.

## Phase 2 — Export and integrate Stable Audio Open

- [x] Reuse/refactor the existing `AutoencoderOobleck` decoder wrapper instead of introducing a different decoding method.
- [x] Pin config and weights to one immutable source revision, shared with the PyTorch reference.
- [x] Derive timing from model configuration and actual decoded shapes; enforce the 2048-sample latent ratio for the tested source. Fix the shared VST-export timing calculation too.
- [x] Attempt a dynamic-time decoder graph; validate all even windows T2–T32. Retain a tested fixed-window export strategy if dynamic export is unsupported.
- [x] Generate and persist the actual graph(s), parameters, manifest and parity report from the installed weights.
- [x] Load those artifacts in Web Perform on Rack, restart offline, and switch between ONNX CPU, PyTorch CPU and MPS at a stopped transport.

**Exit:** real Stable Audio Open ONNX files exist and survive restart; the Web path uses ONNX Runtime on CPU and matches the native reference numerically and geometrically. PyTorch CPU remains independently usable.

**Result:** real dynamic Stable Audio Open decoder published to the local store; all even T2–T32 passed native parity, with fixed fallback exports independently tested. Rack Web discovery, offline server restart and stopped ONNX CPU/native CPU/MPS switching passed. Short CPU runs underrun and are not real-time qualified. See [phase 2 implementation and evidence](docs/MULTI_VAE_ONNX_PHASE2.md).

## Phase 3 — EAR export implementation and checkpoint-dependent validation

- [ ] Extend the export interface to the existing EAR adapter and both sample-rate identities.
- [ ] Obtain the exact EAR checkpoint(s) and matching repository/configuration when supplied or explicitly selected by the user. Do not assume the two registered variants use identical weights or architecture.
- [ ] Probe operations, decoder output length, latent dimension and supported windows using those actual checkpoints; record any ONNX export blockers.
- [ ] Export each installed EAR variant, persist its complete artifact set and compare against its native decoder on CPU and available GPU hardware.
- [ ] Verify 44.1 kHz/48 kHz timing, restart/reload, missing dependencies, missing weights and checkpoint replacement.

**Exit:** each installed EAR variant has a real validated ONNX artifact and native CPU functionality. Without EAR weights, only interface/missing-model tests can pass: this phase's artifact and numerical validation items remain pending. Do not mark “all four VAEs exported” on simulated tests alone.

## Phase 4 — Consistent “Decode with” choices and preparation flow

- [ ] For the selected corpus, list `ONNX · CPU · <VAE>`, `PyTorch · CPU · <VAE>`, and relevant GPU choices. SAME-S gets the same explicit model labels and CPU entry.
- [ ] Keep unavailable options visible with their reason: missing native weights, missing ONNX artifact, missing runtime/library, unavailable hardware, or failed validation.
- [ ] Separate native-weight presence from ONNX-artifact presence. An existing ONNX graph may be usable even when native export dependencies are absent.
- [ ] Expose explicit preparation, progress, failure and retry through the common CLI/job implementation. Block preparation/reconfiguration that would replace a model in active use.
- [ ] Preserve the user's explicit backend choice across refresh/reconnect and for the same model identity. Never silently change PyTorch CPU to ONNX or GPU because of performance.
- [ ] For a new corpus with no saved choice, prefer an already validated ONNX CPU artifact; otherwise present available choices without starting an implicit export or playback.
- [ ] Test corpus changes, stale discovery responses, missing EAR, reload, stop/change/start, failed preparation and current-model display.

**Exit:** each supported execution option is visible for the correct VAE, availability is understandable, and SAME-S ONNX/CPU-native options do not disappear through stale frontend state. PyTorch CPU remains selectable after successful preparation even if it underruns.

## Phase 5 — Numerical, audio and runtime validation

- [ ] Freeze per-model numerical tolerances from reference behavior before evaluating exported graphs. Preserve stochastic behavior and measure within-engine variability where present; do not require SAME-S bitwise equality.
- [ ] Compare identical raw latent windows, output shapes, sample counts and assembled OLA audio on synthetic data and representative real corpora.
- [ ] Use BurntMemory for SAME-S and Rack for Stable Audio Open. EAR needs representative data encoded with the supplied checkpoint(s), in addition to synthetic contract probes.
- [ ] Keep functional decoding results separate from real-time qualification. CPU functionality can pass despite missed audio deadlines.
- [ ] Measure decode p50/p95/p99, preparation, command transitions, memory, CPU usage, buffer underruns and device underruns for every backend/device being claimed as real-time capable.
- [ ] Run at least ten minutes per claimed real-time scenario, covering Random/Manual/Reorganized, fixed/adaptive windows, restarts and engine changes on the actual audio device.
- [ ] Obtain comparative listening feedback for new ONNX outputs. Never transfer SAME-S listening approval to another VAE.
- [ ] Keep failed tests and performance limits explicit. ONNX export success alone is not proof of sufficient CPU throughput.

**Exit:** a per-VAE/backend/window matrix distinguishes functional support, measured real-time support, missing models and untested hardware. No available PyTorch CPU option is removed solely for failing the real-time gate.

## Phase 6 — Packaging and reproducible delivery

- [ ] Update package-data/distribution rules beyond the current SAME-S-only resource directory. Keep user-specific model exports in the local decoder store unless included deliberately in a model-bearing distribution.
- [ ] Preserve each distributed model's identity and applicable notices in the existing release checks; include all external ONNX data files where used.
- [ ] Keep native and export dependencies separate from runtime requirements; preserve the working default and native SAME-S environments.
- [ ] Test a clean ONNX runtime load without native VAE libraries, offline startup, export reuse and recovery from missing/corrupt artifacts.
- [ ] Document preparation commands, storage paths, cache identity, menu states, CPU functionality and measured performance limits.
- [ ] Inspect the produced source/wheel/model bundle, not only the checkout. Include manifests/reports without redistributing corpus audio.

**Exit:** installed VAEs have reproducible, persistent ONNX artifacts; native CPU options remain available; missing EAR is accurately represented rather than hidden or reported complete.

## Recommended implementation order

Work in reviewed increments: **0 → 1 → 2 → 4 → 5 → 6 for SAME-S and Stable Audio Open**. Both EAR checkpoints are now available. Phase 3 can follow phase 2 once the EAR environment is verified; then repeat phases 4–6 for those variants. EAR numerical/export validation remains a separate gate.

Do not implement the whole roadmap in one unchecked change. First verify SAME-S compatibility and CPU menu coverage; then complete Stable Audio Open end to end. Exporter complexity and CPU speed should be measured before estimating or promising complete EAR support.

The four-VAE objective remains open until actual EAR exports are validated. A SAME-S/Stable Audio Open milestone may be delivered independently with EAR explicitly pending.
