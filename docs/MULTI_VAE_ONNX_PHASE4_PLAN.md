# Phase 4 implementation plan — decoder choices and explicit preparation

Status: planned on 3 October 2026, after completed [Phase 3](MULTI_VAE_ONNX_PHASE3.md). This pass inspected the current discovery, pipeline lifecycle, browser selection and preparation entry points. No runtime implementation or export was performed for this plan.

## Outcome and scope

For every registered VAE, the selected corpus determines the model and the user chooses the backend/device. Show ONNX CPU, PyTorch CPU and relevant GPU options consistently, including unavailable choices with actionable reasons. Provide explicit preparation, progress, retry and refresh through one service used by the CLI and Web server. Preserve deliberate selections across refresh, reconnect and browser reload. Never export, download, start playback or switch backend implicitly.

Phases 0–3 already supply all four real decoder artifacts, validated EAR/Stable Audio Open exporters, artifact hashing, external-data validation, atomic publication and native/runtime separation. Preserve those numerical policies and artifacts. Ten-minute performance qualification, listening approval and distribution packaging remain Phases 5–6.

## Verified gaps

| Area | Current behavior | Remaining change |
| --- | --- | --- |
| Discovery | Non-SAME-S ONNX artifacts are verified independently from native libraries; SAME-S checks file presence and omits VAE labels/IDs | One schema and validation path for all four VAEs |
| Availability | `weights` means ONNX artifact or native checkpoint depending on backend; `validated` is always false | Distinct native source, artifact, runtime and preparation states |
| Missing models | EAR can be selectable without weights; native detail does not enumerate every missing dependency/device | Visible, disabled unavailable choices with precise reasons and editable source inputs |
| Default selection | `web/pipeline.js:onDecoderList` prefers a GPU for a new non-SAME-S corpus | Prefer a verified usable ONNX CPU artifact when no explicit choice exists |
| Persistence | Selection is preserved in the current dropdown; there is no persistent model-bound preference | Versioned browser preferences tied to server/corpus/model identity |
| Stale responses | Corpus path guards exist, but requests have no correlation token | Reject stale responses for the same corpus after source edits and A→B→A switches |
| Preparation | Separate EAR and Stable Audio Open CLIs; SAME-S uses a release-oriented exporter | Shared request/result/job service and batch CLI |
| Lifecycle | Perform loading and stop/drain guards exist; stopped decoders may remain cached | Preparation ownership, cache release, publication and shutdown rules |
| Web controls | EAR weight path only; no preparation action or job recovery | Source fields, Prepare/Retry, progress and reconnectable job status |

Primary integration points: `vae/decoder_availability.py`, `vae/onnx_artifacts.py`, `vae/decoder_factory.py`, `runtime/pipeline_server.py`, `web/pipeline.js`, `web/index.html`, existing model exporters and their CLIs. Reuse existing lifecycle tests and `test_web_onnx_ui.js` rather than creating a second transport implementation.

## Increment 1 — Normalize discovery and identities

1. Introduce a typed discovery response with a stable choice key (`vae_id`, backend, normalized device), consistent label, `selectable`, reason codes and readable details. Include `vae_id` for SAME-S. Keep existing response fields temporarily where needed for compatibility, with tests defining their meaning.
2. Separate `native_source_present`, `artifact_present`, `artifact_verified`, runtime dependencies, hardware availability and export capability. Export dependencies must never disable an already usable ONNX decoder. Artifact verification means manifest/hash/source/geometry validation; runtime validation occurs on Start. Neither implies real-time qualification.
3. Use the common resolver for SAME-S too. Report missing files, corrupt/external parameters, incompatible source/geometry, missing runtime and unavailable GPU distinctly. Preserve unavailable CPU/MPS/CUDA entries even when Torch cannot be imported. Missing EAR weights disable native Start, while its source fields remain editable.
4. Resolve EAR weight/repository/config inputs consistently between discovery, preparation and Start. Surface repository/config errors before model allocation where possible. Preserve standalone ONNX operation when no native source was explicitly requested. Never guess missing historical corpus provenance.
5. Return a model identity token derived from VAE, corpus provenance and any explicitly selected source. Return ONNX artifact identity separately: an export-tool change alone should not discard a deliberate native backend choice. If source identity cannot yet be resolved, mark it unresolved and do not restore an identity-bound saved choice until it is known. Legacy corpora remain scoped by corpus identity and known source fields without claiming an exact checkpoint match.
6. Echo a client request token and selection fingerprint (corpus plus source fields). Move expensive hashing off the WebSocket/lifecycle dispatch path; any cache must be invalidated when files change, and Start/publication must perform authoritative verification. Scope runtime failure evidence to the exact selection/artifact identity; refreshing cannot report a known failed selection as validated.

**Gate:** a table-driven matrix covers all four VAEs, native/ONNX independence, consistent labels, unavailable devices, missing/corrupt files and source replacement. Discovery performs no model loading, export or download.

## Increment 2 — One offline preparation service and batch CLI

Add proposed modules `vae/decoder_preparation.py` for request/result models and model dispatch, and `runtime/decoder_preparation_jobs.py` for subprocess orchestration. Add `bin/prepare_decoders.py`; preserve the existing model-specific CLI arguments as wrappers over the same service.

- A request names the registered VAE, source inputs, optional evidence corpus/fixture, store and export settings. Validate model/corpus compatibility before allocating a worker. Structured results distinguish `prepared`, `already_valid`, `missing_weights`, `missing_dependencies`, `missing_inputs`, `busy` and `failed`, with artifact identity/path and diagnostic text where relevant.
- Resolve and verify a reusable artifact before requiring native/export libraries. A force/rebuild request requires the exact native source and exporter environment. Retain the Phase 3 check that stronger requested parity evidence cannot be satisfied by a synthetic-only artifact.
- Keep runtime and export environments separate. Configure exporter interpreters through server startup/local configuration, with `sys.executable` as the default and an explicit per-model override for the existing EAR and SAME-S environments. Web requests select a registered VAE, never a command or interpreter. Invoke a fixed worker module with an argument array; no shell interpolation or automatic package installation.
- Run preflight in the configured worker environment and report which capability is missing. Keep ONNX usage in the default environment independent of EAR/DAC. Do not import multiple EAR repositories into the server process.
- EAR delegates to the existing exporter for either checkpoint. Stable Audio Open retains its real-corpus requirement: use the selected compatible corpus or an explicit batch mapping; missing corpus is `missing_inputs`, not invented evidence or a silently changed parity policy.
- SAME-S normally verifies/reuses its packaged artifact. For missing/corrupt artifacts or explicit rebuild, adapt the existing pinned SAME-S export/parity implementation to stage a shared-schema artifact and publish into the local store. Preserve packaged resources and their legacy identity. Extend SAME-S resolution to prefer an explicit selection, then a verified local current artifact, then packaged resources when no local selection exists. A corrupt selected local artifact is an error, not a silent fallback. Cover this resolver change before enabling the UI action.
- Emit versioned structured events with job ID, monotonic sequence and stages: checking inputs, checking reuse, loading, exporting, validating and publishing. Use real window/probe counts where available; no invented percentage. Separate diagnostic logs from machine-readable events and cap retained logs.
- Batch requests run serially to bound model memory and report each model independently. `--all-installed` visits the registered inventory, verifies existing artifacts and reports unmet prerequisites explicitly. Return machine-readable results plus a concise console summary; partial failure produces a nonzero exit status while retaining successes. Do not require a corpus for artifact reuse.

**Gate:** CLI and Web-facing service use identical preparation/reuse decisions. Test worker environment failure, malformed requests, exact source binding, mixed batch results and complete external-data publication. Existing export CLIs remain compatible.

## Increment 3 — Job lifecycle and publication ownership

Keep preparation jobs distinct from Perform's existing `preparing` state (which means runtime load/warm-up). Expose request, status and terminal results in `pipeline_get_state` so reconnecting clients can recover the active job.

1. Permit one preparation job per server, only while pipeline idle. Reject jobs while preprocessing, training, loading Perform, performing, stopping or retaining resources after failed cleanup. A paused transport still owns its model. Reject Start, preprocessing and training while preparation owns the pipeline; enforce this server-side for all clients.
2. Before launching a worker, release the idle cached decoder and preprocessing model through existing teardown methods. If release fails, retain error ownership and refuse preparation. Snapshot request/source identity so later source-field edits cannot change a running job.
3. Reserve the job under the lifecycle lock, then launch and monitor outside it. Never hold the lock while waiting for subprocess completion or broadcast progress. Use generation/job checks to prevent a late worker result from publishing after shutdown or a superseding lifecycle event.
4. Refactor exporter staging from publication: the worker returns a validated staged artifact; the coordinator rechecks ownership and source/request identity, then uses the existing atomic publisher. CLI publication uses the same coordinator contract. Failed export/validation never changes the current pointer.
5. Add a per-store/per-VAE preparation lock shared by CLI and server jobs to prevent two cooperating writers from racing. Do not mutate immutable artifact directories. External applications holding an older immutable artifact remain valid; Phase 4 does not claim cross-process transport ownership detection.
6. Browser disconnect does not cancel a job. Record a bounded job journal outside artifact directories for progress and terminal recovery. Server shutdown terminates/reaps the worker and cleans its staging area without lock/join deadlock. After restart, unfinished jobs become interrupted/failed and require explicit retry; never auto-resume or publish orphaned output.
7. Retry creates a new job ID from a freshly validated request. Completion refreshes discovery for the matching corpus/source identity and reports the result without changing backend choice or starting playback. A job for a previously selected corpus remains visible as that job, not as the new corpus's preparation result.

Cancellation UI is not required for this phase; safe server shutdown and interruption recovery are required. A later cancellation feature must use the same publication boundary.

**Gate:** race tests exercise Prepare versus Start, two clients, duplicate submissions, stopped cached resources, failed drain/release, shutdown, restart, publication failure and stale completion. Existing transport and atomic-publication guarantees remain intact.

## Increment 4 — UI selection, preparation and persistence

- Display the corpus VAE beside “Decode with” and the actual active VAE/backend/device separately. Render availability from the new fields, with plain reasons such as “ONNX decoder missing” or “EAR weights required.” Keep performance limits informational; CPU remains available when functionally supported.
- Add Prepare ONNX and Retry actions, stage/progress output and a retained failure summary. Expose EAR checkpoint/repository/config inputs and the applicable evidence corpus; keep interpreter settings in server configuration. Disable mutation controls while the server owns an active model/job. Refresh remains read-only.
- Persist only explicit user backend/device choices in versioned local storage keyed by server namespace, corpus identity and model identity. Store artifact identity separately for diagnostics. Handle unavailable/disabled storage gracefully. Source changes invalidate the matching preference; an unavailable saved backend stays selected with its reason and Start disabled instead of silently switching.
- Selection precedence: authoritative active server decoder on reconnect; otherwise matching explicit preference; otherwise verified usable ONNX CPU; otherwise an explicit “Choose a decoder” placeholder. No automatic GPU preference and no automatic playback. Server state reconciliation must not overwrite a stored explicit preference as though the user made a new choice.
- Add monotonically increasing request tokens and source fingerprints to discovery requests. Accept only the latest matching response. Invalidate readiness immediately on corpus or source edits, including same-path checkpoint replacement once fresh identity arrives. Debounce source edits, and gate Start on current discovery.
- Preserve the existing native dropdown interaction safeguards. Keep job events scoped by job ID/sequence, and ensure old refresh responses cannot overwrite newer job or active decoder state. A local preference is never authority to reconfigure a running server.
- Refresh asset versioning as needed so an existing browser loads the new protocol/UI together. Provide a readable incompatibility message for an older server lacking preparation capability.

**Gate:** DOM tests cover each selection precedence rule, browser reload, reconnect during active playback/job, CPU preference after export, unavailable saved GPU, changed source identity, A→B→A corpus switches and out-of-order same-corpus responses.

## Verification and delivery

Run focused discovery/job/lifecycle tests after each increment, then the full Python suite and all three browser contract suites. Use temporary stores and deterministic worker fixtures for failure/race tests; never corrupt installed artifacts to simulate errors.

Run real integration checks for all four installed artifacts: discovery, reuse through the common CLI, explicit CPU retention, offline server restart and stop/change/start. Run at least one actual new export through the job service into a temporary store to verify progress, external data, publication and subsequent reuse; use an EAR checkpoint because it exercises the separate environment and external parameter file. Validate the new SAME-S local-store route and migration against its real graph without replacing packaged files. Reuse existing per-model parity gates rather than changing tolerances for job tests.

Inspect the actual browser for current-model labels, disabled-option reasons, Prepare/progress/retry, reconnect and keyboard dropdown behavior. Record protocol/DOM evidence separately from visual inspection and physical audio evidence. Repeat physical transport checks only where lifecycle changes affect ownership; these remain functional checks, not Phase 5 qualification.

Deliver `docs/MULTI_VAE_ONNX_PHASE4.md`, machine-readable preparation/discovery/restart evidence and documented CLI/environment configuration. Update README and Phase 4 roadmap checkboxes only after their gates pass. Keep source audio and model files out of tracked deliverables. Recommended implementation order is discovery → shared service/CLI → lifecycle → UI/persistence → integration evidence.
