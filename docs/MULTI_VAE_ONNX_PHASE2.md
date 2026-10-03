# Multi-VAE ONNX — phase 2

Completed 2 October 2026. Stable Audio Open now has a real dynamic decoder artifact, shared pinned native/export sources, explicit preparation, and Web Perform discovery. EAR preparation, the preparation UI and real-time qualification remain later phases.

## Implementation

`vae/stable_audio_open_weights.py` resolves revision `f21265c1e2710b3bd2386596943f0007f55f802e` and verifies the exact config/checkpoint hashes inventoried in Phase 0. The adapter, native decoder factory and exporter share that source. Local snapshots remain supported when their bytes match the pinned source. Explicit corpus/source identity conflicts fail before preparation or native loading. Native weights are retained independently.

`vae/stable_audio_open_export.py` reuses the existing `AutoencoderOobleck.decode(latents).sample` wrapper. The config's downsampling product is 2048 and actual outputs must have exactly that ratio for every advertised window. Historical `21.5` Hz corpus metadata remains accepted alongside the exact rate; PCM timing never rounds it to 2051. The VST exporter shares the wrapper, source, geometry checks and enforced parity, preserving its fixed-window bundle schema.

Preparation first attempts a dynamic graph, then falls back to separately validated fixed graphs on failure. Native/export imports stay out of ONNX runtime loading. A temporary export receives a complete manifest, file hashes and a source/graph-bound parity report before Phase 1's `publish_artifact()` checks and atomically publishes it. Existing valid exports are rechecked and reused; source, windows, versions and export settings must match. Failed preparation cannot replace the current usable artifact.

The Rack discovery response now exposes `ONNX · CPU · stable_audio_open`. PyTorch CPU and available GPU choices remain independent. Selecting a corpus, discovering choices or starting Perform never prepares a graph implicitly. The visible preparation action and broader menu behavior remain Phase 4.

## Published artifact

The normal store contains:

```text
~/.cache/waenderer/decoders/stable_audio_open/
  current.json
  5b180bcfbef623248357efb0cc6e3b77bc4cef8e4b743d246f133cd106b2bd75/
    decoder.json
    decoder_dynamic.onnx
    parity.json
```

- Graph: **312,698,221 bytes**, embedded parameters, opset 18, legacy Torch ONNX tracing with dynamic time.
- Graph SHA-256: `d2abf24bb91f9dccc56395eb58568b72c0916d084094ad9efef884f071f9c57f`.
- Input `[1,64,T]`; output `[1,2,T*2048]`; float32 stereo, 44,100 Hz.
- All even T2–T32 supported; default T8; T8 output 16,384 samples and OLA hop 8,192 samples.
- A workspace copy is retained in `build/decoders/`; model files are not added to Git or package resources.

## Real validation

[Compatibility evidence](multi_vae_phase2_compatibility.json) records 80 dynamic-graph probes: silence, seeded random inputs and three denormalized Rack excerpts at every even T2–T32. Error thresholds were frozen before the first ONNX evaluation: maximum absolute error ≤ `2e-4` and RMSE ≤ `2e-5`. Native CPU repeated outputs were identical in these probes. Worst ONNX absolute error was **1.3322e-5**, worst RMSE **9.2279e-7**. Every output passed geometry, dtype and finiteness checks.

An eight-window T8 OLA sequence produced 65,536 samples with maximum absolute error **2.0862e-6**, RMSE **2.8095e-7**, and SNR **118.75 dB**. This is a short numerical check, not a listening assessment or an exhaustive window-transition campaign.

[Fixed fallback evidence](multi_vae_phase2_fixed.json) separately validates actual fixed exports for all 16 windows using three Rack excerpts each. Those temporary graphs were discarded after validation because the dynamic graph is deployed. Retaining all fixed graphs would cost approximately **5.0 GB**, versus 313 MB for the dynamic graph; runtime uses Phase 1's two-session cache if a fixed artifact is selected.

A fresh child process loaded the published artifact from a different working directory with native VAE/export imports and socket connections blocked. It decoded all 16 windows through CPUExecutionProvider. Repeating preparation reused the same verified artifact identity.

The production Web Perform setup was exercised on Rack through ONNX CPU → PyTorch CPU → MPS → ONNX CPU with stopped transport between changes. Each backend decoded at least five windows through the actual audio device, with output gain muted. Stop/drain/release and return to idle succeeded.

[Web server evidence](multi_vae_phase2_web.json) records two isolated production HTTP/WebSocket server starts with Hugging Face offline mode enabled. Each server discovered the published Rack artifact, loaded ONNX CPU, decoded through Manual transport and stopped cleanly. The recorded discovery responses also pass the browser DOM contract harness, including reconnect preservation. This is protocol/DOM evidence, not a screenshot-based browser inspection. The user's existing server was left running; restart it to load the new discovery code.

The packaged SAME-S artifact was loaded again across all 16 windows without modification or re-export; its graph hash remains `333dc948f52fa59e9a80a066ffd2ca801d3a54b2fb07c93aa87cac845c547872`.

## Performance limits

These short T8 runs on the MacBook Pro speakers are functional checks, not real-time qualification. The audit uses one native CPU compute thread; ORT uses one intra/inter-op thread and basic optimization:

| Backend | Decode p99 in short probe | Buffer underruns |
| --- | ---: | ---: |
| ONNX CPU, first run | 583 ms | 109 |
| PyTorch CPU | 264 ms | 5 |
| PyTorch MPS | 54 ms | 0 |
| ONNX CPU, return run | 575 ms | 101 |

T8 has an approximately 186 ms audio hop. Both CPU paths missed deadlines in this scenario; neither is qualified as real-time capable. Native CPU remains selectable. MPS also needs Phase 5's longer campaign and listening review before broader qualification claims. CUDA was not tested. Rack's historical metadata lacks checkpoint hashes, so its original encoding checkpoint cannot be proven from corpus metadata alone; current native/export sources are identical and verified.

## Reproduction and tests

From the Python project root, using the environment containing the installed weights and export dependencies:

```sh
.venv/bin/python -m bin.prepare_stable_audio_open_onnx \
  --corpus corpus/Rack_20260428_181107

.venv/bin/python -m eval_scripts.audit_multi_vae_phase2 \
  --corpus corpus/Rack_20260428_181107 \
  --output docs/multi_vae_phase2_compatibility.json --physical

.venv/bin/python -m eval_scripts.audit_multi_vae_phase2_web \
  --corpus corpus/Rack_20260428_181107 \
  --output docs/multi_vae_phase2_web.json

.venv/bin/python -m eval_scripts.audit_stable_audio_open_fixed \
  --corpus corpus/Rack_20260428_181107 \
  --output docs/multi_vae_phase2_fixed.json
```

Preparation accepts `--store-dir`, `--revision`, `--opset` and `--force`; only the verified source revision is accepted. Default preparation is offline. A custom store requires matching `decoder_store_dir` in Perform configuration; the current menu discovery uses the normal store. `--force` re-exports into temporary storage before publishing. The fixed-fallback audit intentionally exports large temporary files one at a time.

Final regression suite: **311 passed, 1 skipped**, with three existing WebSocket deprecation warnings. All three browser contract suites pass. New tests cover source/revision mismatch, corrupt native source files, measured VST timing, strict numerical rejection, dynamic/fixed coverage, external-file inclusion, report binding, failed parity preventing publication, stale corpus identity and verified reuse. Phase 1 continues to cover atomic publication failures and external-file safety. Local socket tests require execution outside the filesystem/network sandbox.
