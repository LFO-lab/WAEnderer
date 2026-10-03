# Phase 2 implementation plan — Stable Audio Open ONNX

Status: implemented. See [Phase 2 results and evidence](MULTI_VAE_ONNX_PHASE2.md). This document records the implementation plan.

## Scope and starting point

Complete Phase 2 of `ROADMAP_MULTI_VAE_ONNX.md`: produce a real, validated Stable Audio Open decoder artifact from the installed weights, publish it through Phase 1, and exercise Rack through Web Perform with ONNX CPU and native CPU/MPS selection.

Phase 1 supports this VAE in artifact resolution, loading and persistence. Implementation inspection found that Web discovery still needed its Phase 2 ONNX entry. Reuse `onnx_artifacts.py`, `artifact_decoder.py`, the existing factory and transport lifecycle. No new transport implementation is needed.

The existing VST exporter wraps `AutoencoderOobleck.decode(latents).sample`, exports fixed windows, and records numerical metrics. It does not pin the source or reject exports on numerical error. Its caller derives 2051 samples per latent from rounded corpus metadata; the installed configuration has downsampling ratios `[2, 4, 4, 8, 8]`, whose product is 2048. Native `AdapterTorchDecoder` already measures output geometry rather than using that rounded rate.

## 1. Share and pin the native source

- Introduce a Stable Audio Open source helper, analogous to `same_s_weights.py`, shared by export and native loading.
- Use installed revision `f21265c1e2710b3bd2386596943f0007f55f802e`. Verify the actual config and checkpoint hashes against the Phase 0 inventory before export; do not rely solely on cache directory names.
- Resolve both files from that immutable revision, defaulting to local-only access. Preserve original source files. Missing or mismatched sources must produce a specific error.
- Update `adapters/stable_audio_open.py` and the Stable Audio Open branch in `decoder_factory.py` to use the same resolved source. The adapter currently ignores revision arguments and the factory currently resolves the default Hugging Face revision.
- Include revision and file hashes in native selection identity and enforce any supplied expected source identity. Preserve existing local-path support with explicit identity validation.

Acceptance: native reference, native Perform decoder and exporter demonstrably use identical source files; a moved upstream default cannot change the selected model.

## 2. Extract decoder export and correct timing

- Extract the existing Oobleck wrapper into an export module under `vae/`, importing export/native dependencies only within the preparation path. Keep `model.decode(latents).sample` unchanged.
- Derive expected geometry from config, then verify actual float32 native outputs for every even window T2–T32: `[1, 64, T]` input and `[1, 2, T * 2048]` output at 44,100 Hz.
- Reject inconsistent, nonfinite or unexpected outputs. Enforce the 2048 ratio for this pinned source.
- Accept both historical corpus rate `21.5` and exact rate `44100 / 2048` in the artifact. Do not rewrite Rack metadata or use either rate to round PCM lengths.
- Refactor `bin/export_vst_bundle.py` to share source loading, wrapper, geometry and parity logic while retaining its bundle schema. Forward `--decoder-revision` for Stable Audio Open too. Derive its timing entries from measured geometry; retain SAME-S regression coverage.

Acceptance: T8 produces 16,384 samples and an 8,192-sample OLA hop in both artifact and VST metadata; no Stable Audio Open path derives 2051 samples per latent.

## 3. Export with enforced parity

- Establish and record numerical tolerances before evaluating ONNX output. Use repeated native CPU float32 probes to characterize variability; freeze absolute/relative error and aggregate-error criteria in the validation policy. Do not loosen them after seeing an ONNX failure.
- First attempt a graph with dynamic latent time and dynamic output time, keeping batch, latent dimension and channels fixed. Record exporter mode, opset and tool versions.
- Validate all 16 advertised windows with zero inputs, seeded random inputs and multiple denormalized Rack windows (`Z_concat * Z_std + Z_mean`). Use identical arrays for native CPU and ORT CPU.
- Require graph checks, exact output shape/sample count, float32/finiteness and the frozen parity criteria to pass. Record max absolute error, RMSE and SNR with JSON-safe handling of zero-error/silent cases.
- If dynamic export or dynamic-window validation is unsupported, record the concrete failure and attempt fixed graphs covering every required window. Fixed export must pass the same gates; parity failure is never permission to publish.
- Enumerate and hash all graph and external parameter files. Prefer the single dynamic graph; report total disk size for a fixed fallback and use Phase 1's existing bounded session cache.

Acceptance: an actual dynamic artifact or complete fixed-window set passes native-reference parity at every advertised window. Synthetic tests alone do not satisfy this step.

## 4. Add explicit preparation and atomic publication

- Add a model-specific CLI, proposed as `bin/prepare_stable_audio_open_onnx.py`, over a callable preparation function. Accept corpus, immutable revision, store directory and export settings; report progress, reuse and concrete failures.
- Prepare in a temporary directory. Write the Phase 1 `waenderer.onnx_decoder.v1` manifest, source identity, geometry, window map, tool versions and hashed parity report. Bind the report to the actual source and graph hashes.
- Publish only through `publish_artifact()`. Its checks supplement exporter parity; they do not replace it. Failure must leave the previously selected artifact usable.
- Reuse an existing artifact only after validating its files, source, requested windows and export settings. Persist new files under the configured decoder store, not repository package resources.
- Keep loading, discovery and playback free of implicit export or download. The batch preparation command and visible preparation job UI remain Phase 4 work.

Acceptance: export failure preserves the previous current pointer; successful publication resolves normally and a second identical preparation can reuse verified files.

## 5. Validate the real Rack workflow

- Add `eval_scripts/audit_multi_vae_phase2.py` to record artifact location/identity, sources, geometry, CPU provider and parity results for `corpus/Rack_20260428_181107`.
- Compare a short assembled OLA sequence as well as individual raw windows, catching sample-count and timing mistakes across window transitions.
- Start a fresh process offline, with native model/export imports blocked for the ONNX decode path. Load the published artifact from another working directory and decode Rack windows successfully.
- Exercise Web Perform discovery and actual loading after publication, then restart the server offline. Confirm the resolved backend/provider is ONNX Runtime CPU, not a native fallback.
- At stopped transport, switch ONNX CPU → PyTorch CPU → MPS → ONNX CPU, checking source identity, stop/drain/release and audio production. Native CPU remains selectable even if it cannot meet playback deadlines.
- Retest packaged SAME-S availability/loading without re-exporting or modifying its graph. Make only integration fixes required by Phase 2; defer broad UI changes to Phase 4.

Acceptance: real persisted files survive restart, Rack loads them through Web Perform, all three backend/device choices work on available hardware, and ONNX output passes the numerical and geometry gates.

## Tests and delivery

Add focused tests for pinned resolution, stale/mismatched sources, measured VST timing, dynamic-window coverage, fixed fallback, numerical rejection, complete external files, reuse and failed-publication preservation. Extend existing factory/discovery/lifecycle tests where source pinning changes their contracts. Reuse Phase 1 artifact tests rather than duplicating them.

Run focused tests during each increment. After integration, run the complete Python suite and the three existing browser contract suites. Run the real export and audit separately; record failures and hardware limitations instead of treating mocked tests as model qualification.

Deliver in three reviewable increments:

1. Shared source identity, wrapper and timing correction, with regressions.
2. Dynamic/fixed export, parity gates, preparation CLI and actual published artifact.
3. Rack offline/Web integration evidence, regression verification and Phase 2 documentation.

Write `docs/MULTI_VAE_ONNX_PHASE2.md` and `docs/multi_vae_phase2_compatibility.json` with reproduction commands, artifact size/location/hash, parity thresholds/results, fallback decisions and limitations. Mark roadmap items complete only when their real evidence exists.

Remaining uncertainties are dynamic export compatibility, acceptable numerical tolerances established from native probes, and fixed-graph disk/session cost. Resolve these experimentally before promising artifact size or runtime speed. Ten-minute real-time qualification, comparative listening approval, EAR exports and distribution packaging remain their later roadmap phases.
