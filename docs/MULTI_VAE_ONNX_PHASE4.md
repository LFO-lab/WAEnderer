# Multi-VAE ONNX — Phase 4

Completed on 3 October 2026. All four registered VAEs now use consistent discovery, explicit offline preparation jobs and persistent backend selection. Numerical export policies and real-time qualification remain separate.

Rechecked on 4 October 2026: the Phase 4 implementation is committed as `898283e` (`ONNX uniformity - Phase 4`). Existing runtime, export and browser audit reports retain passing results. A fresh read-only discovery check confirmed all four correct VAE labels, usable ONNX CPU entries and visible PyTorch CPU entries; its local receipt is `build/phase4/recheck_20261004.json`. This recheck did not repeat exports or the browser campaign.

## Delivered behavior

“Decode with” includes the corpus VAE in every ONNX CPU, PyTorch CPU and GPU label, including SAME-S. Unavailable options remain visible; their full reasons appear under “Unavailable choices and reasons” and in option help. Native source presence, verified ONNX artifacts, runtime dependencies and hardware are independent. An EAR ONNX artifact remains usable without DAC or its native checkpoint. Explicitly selected conflicting EAR sources cannot leave ONNX selectable.

Discovery validates SAME-S through the common artifact resolver too; file presence alone no longer enables a corrupt graph. Correlated discovery requests run outside the WebSocket/lifecycle dispatch lock. The browser rejects obsolete responses after corpus or source changes, including same-corpus requests. Runtime failures remain visible for the matching model/backend; Refresh availability explicitly permits retry.

For a corpus without a saved preference, a verified usable ONNX CPU artifact is preferred. Otherwise the menu asks the user to choose. Explicit preferences are stored locally by server namespace, corpus and model identity, and survive refresh, reconnect and reload. An unavailable saved backend stays selected and disables Start. Active server state takes precedence on reconnect. Export completion does not replace a CPU preference, select another model or start playback. The last corpus can be restored when the server has no active selection; EAR source paths must still be supplied for native loading when not already present in the form.

Prepare ONNX decoder and Retry use the same service as the common CLI. Controls show actual preparation stages and terminal errors. They disable conflicting operations while a job owns the pipeline. The selected corpus is checked for geometry/source compatibility; merely selecting it does not demand a new parity export of an already valid artifact. Explicit CLI `--corpus`/`--fixture` evidence still must be covered before reuse. When a new export is necessary, the selected corpus supplies real evidence where supported.

## Preparation and lifecycle

`decoder_preparation.py` defines validated requests, source/evidence snapshots, reuse and model-specific staging. `decoder_preparation_jobs.py` owns the subprocess, bounded diagnostics, monotonic job events, journal, per-VAE store lock and atomic publication. `bin.prepare_decoders` exposes single-model and serial batch operation. Existing EAR and Stable Audio Open CLI arguments remain supported through this service.

Reuse checks complete files, graph structure, source/corpus compatibility and every advertised runtime window before needing a native exporter. Workers use the explicitly configured Python environment and fixed internal module, with Hugging Face offline mode. No Web request supplies an interpreter or shell command. The service never installs packages or downloads models.

Exports stage in temporary storage. The coordinator checks unchanged inputs and job ownership before the existing publisher validates and atomically updates the current pointer. Failures preserve the previous artifact. Cooperating CLI/server jobs cannot prepare the same VAE store concurrently. Immutable older artifacts remain usable by other processes; this is not a cross-process transport ownership service.

Preparation requires an idle pipeline and successful release of cached decoder/encoder resources. Paused audio still owns its model. Preprocessing, training and Perform starts are rejected during preparation. Disconnects leave the job running; reconnects recover job ID, progress and result. Shutdown terminates and reaps the exporter before cleanup. Incomplete journal entries are reported as interrupted on restart and require explicit retry. Cancellation UI is outside this phase.

SAME-S normally reuses the existing packaged graph. Its explicit rebuild path adapts the pinned release exporter to shared metadata and the existing stochastic comparison gate (RMSE < 0.005 and SNR > 20 dB). Resolution order is explicit artifact, local current artifact, then the packaged graph if no local pointer exists. A broken selected local artifact is an error, not a silent fallback. Packaged model files were not modified.

## Commands and configuration

Check or prepare every registered model serially:

```sh
.venv/bin/python -m bin.prepare_decoders --all-installed --json
```

A missing prerequisite produces an independent per-model result. Statuses include `prepared`, `already_valid`, `missing_weights`, `missing_dependencies`, `missing_inputs`, `busy` and `failed`. Batch processing retains successes and exits nonzero on partial failure. Valid existing artifacts do not require a parity corpus or export environment.

For exports requiring another environment, provide a local JSON configuration:

```json
{
  "interpreters": {
    "ear_vae_44k": "/absolute/project/.venv-ear/bin/python",
    "ear_vae_48k": "/absolute/project/.venv-ear/bin/python",
    "same_s": "/absolute/path/to/native-same-s/bin/python"
  },
  "models": {
    "stable_audio_open": {"corpus": "/absolute/path/to/Rack_corpus"},
    "ear_vae_44k": {
      "weights": "/absolute/EAR_VAE/pretrained_weight/ear_vae_44k.pyt",
      "repo": "/absolute/EAR_VAE"
    },
    "ear_vae_48k": {
      "weights": "/absolute/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt",
      "repo": "/absolute/EAR_VAE"
    }
  }
}
```

Replace example paths with installed environments and sources; this file does not create environments. `models` supplies CLI batch requests. The Web UI supplies source fields for its selected corpus; only interpreter/store/journal settings apply to the server. Unspecified interpreters use the current Python executable. Optional `store_dir` and `journal_dir` configure local storage; model-specific request paths cannot override the server's store.

```sh
.venv/bin/python -m bin.prepare_decoders --all-installed \
  --config-file /path/to/decoder-preparation.json

.venv/bin/python -m bin.prepare_decoders --vae-id ear_vae_44k \
  --weights /path/to/EAR_VAE/pretrained_weight/ear_vae_44k.pyt \
  --repo /path/to/EAR_VAE \
  --config-file /path/to/decoder-preparation.json

.venv/bin/python -m bin.serve \
  --decoder-preparation-config /path/to/decoder-preparation.json
```

An EAR custom checkpoint requires `--config`. `--force` requests a rebuild; `--fixed-only` remains EAR-specific. Stable Audio Open requires a compatible corpus for a new export. The old `bin.prepare_ear_onnx` and `bin.prepare_stable_audio_open_onnx` entry points remain available. Runtime PyTorch choices depend on the server environment, independently of the configured exporter environment.

Restart an existing server and reload the browser after updating. An isolated server using a nondefault WebSocket port can be inspected with `http://127.0.0.1:18080/?ws_port=18765`; the default remains port 8765.

## Verification and evidence

- [Production runtime audit](multi_vae_phase4_runtime.json): all four actual artifacts, two offline server starts, correlated discovery, preparation reuse, reconnect recovery of both job and selected corpus, and short muted ONNX playback/stop. Existing user servers were not reconfigured.
- [Real export and reuse](multi_vae_phase4_export.json): a fresh EAR 44.1 kHz export ran in `.venv-ear`, staged and published through the default-environment coordinator into an isolated store, then reused. It reproduced Phase 3 artifact identity `9d7e752c079831cbb75787540bc890b06cadb9f0ed57d6ab5b931aa83d7e2694` with complete external parameters.
- [Common batch CLI](multi_vae_phase4_batch.json): actual single-command verification/reuse of the four installed artifacts.
- [SAME-S local resolution](multi_vae_phase4_same_s_local.json): the actual existing graph was copied into an isolated local store and decoded every advertised window, with the packaged identity unchanged. The new rebuild-to-shared-schema adapter is covered by staging/policy tests; a new native SAME-S export was not required for this compatibility check.
- Python regression suite: **348 passed, 1 skipped**, with three existing WebSocket deprecation warnings. A separate EAR-environment run passed **54 tests**. Tests cover real tiny external-data graphs, publication/reuse, source replacement, malformed requests, partial batch failure, locking, release failure, subprocess shutdown, interrupted journals, correlation and source conflicts.
- All three JavaScript contract suites passed. New cases cover ONNX preference over GPU, explicit CPU persistence across reload, unavailable saved choices, source identity changes, same-corpus stale responses, preparation ownership and late prior-job events.
- [Actual browser inspection](multi_vae_phase4_browser.json) confirmed EAR availability reasons, progress, reconnect/reload, explicit Stable Audio Open PyTorch CPU retained after ONNX preparation, and missing-checkpoint error/retry. This is additional evidence beyond the DOM harness.

A concurrent batch attempt correctly returned `busy` while the browser was checking the same model; the completed batch report was captured after that job ended. The new EAR export and local-store fixtures remain in ignored `build/phase4`; no model files or corpus audio are added to tracked deliverables.

Short CPU playback can underrun. Phase 4 establishes functional selection/preparation behavior, not new real-time, CUDA or listening qualification. Those remain Phase 5; distribution work remains Phase 6.
