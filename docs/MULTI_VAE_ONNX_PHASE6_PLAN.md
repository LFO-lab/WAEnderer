# Phase 6 implementation plan — packaging and reproducible delivery

Status: planned and implemented on 4 October 2026. See [delivery and verification](MULTI_VAE_ONNX_PHASE6.md) and [recorded build/install evidence](multi_vae_phase6_delivery.json). This document retains the implementation sequence and gates.

## Outcome and boundaries

Deliver source and wheel distributions that work outside the checkout, plus an explicitly assembled decoder model bundle when requested. Preserve model/source identity, complete ONNX parameter files, validation reports and applicable notices. User exports stay in the configured decoder store unless deliberately selected for distribution.

Reuse the existing artifact resolver, validated publication, preparation jobs and Phase 5 evidence. Do not re-export working graphs, change numerical tolerances, migrate the user's environments or run listening/physical campaigns for a packaging change. Functional CPU support remains distinct from real-time qualification. GPU listening and physical multi-VAE qualification remain pending; historical SAME-S qualification applies only to its recorded profile.

## Findings from the current code

| Area | Current state | Packaging implication |
| --- | --- | --- |
| Model resources | `pyproject.toml` includes only `resources.same_s` patterns; `.onnx` and decoder manifests are ignored by Git but present locally | A build can differ depending on local files; non-SAME-S external data has no distribution rule |
| Release validation | `release_compliance.py` and `SAW_RELEASE_BUILD` validate only the legacy SAME-S payload | Additional selected models need identity and notice checks; validation must also inspect delivered files |
| Installed commands | Package discovery includes only `stable_audio_wanderer*`; preparation invokes `bin.decoder_preparation_worker` | A wheel does not supply the preparation worker or the documented `bin` commands |
| Browser assets | `bin/serve.py` finds a sibling checkout `web/` directory; `MANIFEST.in` includes Web files in the source archive | Wheel delivery needs package-owned browser assets and an installed server command |
| Dependencies | `onnx` is a runtime dependency used for external-data checks; `onnxscript` is an export extra; `requirements.txt` installs that extra | Keep `onnx` at runtime, separate exporter-only installation, and preserve existing native profiles |
| Source archive | The manifest includes reports/docs but only the dual-inference roadmap; it prunes `corpus` and `eval_out` | Include the multi-VAE roadmap and explicitly exclude user audio, stores, staging and environment directories |
| Evidence | SBOM/license generation describes Python distributions, not selected model payloads | Record model inventory and final archive checksums separately |

These are source-level findings. Archive contents and fresh installed behavior still need verification during implementation.

## Increment 1 — Make installed commands and assets portable

1. Put the server, common preparation CLI and preparation worker under an application-owned package namespace. Add console entry points for serving and preparing decoders. Retain existing `bin` scripts as thin checkout wrappers so established commands keep working. Follow exporter imports transitively: include each model-specific preparation/export dependency needed by the worker, without moving unrelated utilities.
2. Replace the `bin.decoder_preparation_worker` subprocess target with the installed package module. Remove assumptions that the worker's current directory is the checkout. Each configured exporter interpreter must have a compatible application installation; report a missing or incompatible worker explicitly. Do not inject a developer checkout into its import path.
3. Package Web assets under the application namespace and resolve them through package resources. Reuse the existing HTTP server and asset versioning. Retain vendored browser notices. Avoid duplicate hand-maintained copies; choose one canonical asset location with a documented build/checkout path.
4. Extend source inclusion for the multi-VAE roadmap and preparation examples. Exclude private audio, user stores, jobs/staging, virtual environments and build outputs explicitly. Shared Web and licensing assets must survive both a direct wheel build and a wheel built from the source archive.

**Gate:** install the wheel non-editably into a temporary environment, change to an unrelated directory, and exercise CLI help, worker preflight and HTTP asset retrieval. Use a deterministic worker fixture for publication wiring; actual model reuse is checked in Increment 4. No audio device is required.

## Increment 2 — Explicit distribution contents and model compliance

Recommended default: the ordinary source/wheel build contains application code, assets, notices and tracked reports, without acquiring model binaries from local caches. A model-bearing build requires an explicit staging inventory. Preserve the existing SAME-S release command as a compatibility route that selects its complete payload and applies the same checks.

1. Define a small versioned distribution inventory naming each selected VAE, artifact identity, manifest digest, graph/parameter/report files and notice/license payload. Use the existing shared artifact schema; the inventory describes delivery and does not create a second runtime decoder schema.
2. Assemble a standalone decoder-store bundle from explicitly selected verified artifacts. Preserve `<vae_id>/<artifact_identity>/` and reconstruct only the selected `current.json` pointers. Include `decoder.json`, all declared hashed files, external tensors and validation reports; exclude original checkpoints, corpus arrays/audio, journals and staging. Legacy SAME-S needs an explicit adapter for its parity report and notices because its runtime file list only describes the graph.
3. Copy only the verified file closure into staging. Validate paths, source identity, graph/external-data declarations and hashes through existing artifact code, then validate the copied payload. A missing or corrupt file fails assembly without changing the user's store. Avoid wildcard collection of arbitrary cache contents.
4. Generalize `validate_model_release` to selected model payloads while preserving existing SAME-S provenance/notice requirements and tests. Require exact-source notice/license evidence for any additional distributed model; a VAE registry label is insufficient evidence. Record unresolved material as a release failure. Confirm applicable materials against the actual selected sources during implementation rather than inferring rights from checkpoint availability.
5. Keep distribution notices outside the immutable runtime manifest where practical, so adding delivery metadata does not change a validated artifact identity or require re-export. The distribution inventory hashes those additional files.
6. Inspect the produced source archive, wheel and model bundle. Require the complete selected payload and reject undeclared model binaries, absent external parameters, private audio and source-machine configuration. Run checks on unpacked outputs as well as staging.

**Gate:** a few tiny artifact fixtures cover dynamic and fixed graphs, external tensors, legacy SAME-S, incomplete notices and corrupted/missing files. Check both ordinary and model-bearing build profiles. Reuse existing resolver/compliance tests rather than duplicate graph validation.

## Increment 3 — Document and verify dependency profiles

- Audit imports reached by installed ONNX loading, discovery, preparation reuse and server startup before changing requirements. Native model libraries and `onnxscript` must not be required for an existing artifact's ONNX load/reuse. Torch remains an application dependency where other components need it; this phase does not promise a Torch-free application.
- Keep `onnx` as a runtime requirement because structural/external-weight validation uses it. Provide a clear base runtime installation and explicit export installation. Preserve `requirements.txt` compatibility if changing it would disrupt the working default; document its exporter-inclusive meaning and add a runtime-only profile instead.
- Preserve pinned native SAME-S and EAR profiles and their separate reference locks. Do not force a combined environment or replace the working default. Record exact Python/platform/resolved package versions and `pip check` for the clean environment used as evidence. Regenerate `uv.lock` only if metadata changes require it.
- Add explicit exporter/native interpreter configuration examples using portable placeholder paths. Explain that model libraries/checkpoints are preparation/native requirements, while validated ONNX artifacts can run independently.

**Gate:** a non-editable runtime installation passes `pip check`, and fresh decoder processes reject imports of native SAME-S/EAR/Stable Audio Open model libraries and exporter-only modules while loading existing ONNX artifacts. Do not use `--no-deps` as evidence of a successfully resolved clean environment. Native dependency resolution is checked only when its declarations change; retain earlier installation evidence otherwise.

## Increment 4 — Small installed-artifact checks and recovery

Use temporary stores, synthetic raw latents and copied existing artifacts. Never damage the user's working exports.

1. In fresh installed processes with networking denied, resolve and decode one short supported window per real VAE. Assert model identity, exact sample count, stereo finite float32 PCM and CPU provider. Use a narrow ONNX boundary to exclude native libraries; separately check application startup without optional SAME-S/EAR libraries. Record installed package location so checkout imports cannot satisfy the check.
2. Run common preparation reuse against each copied artifact and require `already_valid`, unchanged identity and no exporter invocation. Test clean restart and selection of the same artifact. Resolve from an unrelated working directory and configurable store path.
3. Exercise missing/corrupt graph and external-data files, invalid pointers, and a failed replacement using small fixtures. Verify actionable failure, preservation of the prior usable artifact and explicit retry/recovery. A selected corrupt artifact must not silently switch to another model or graph. Reuse Phase 4 lifecycle evidence for unchanged ownership rules.
4. Generate release evidence containing application/archive hashes, selected model identities, runtime environment and links/digests for reused Phase 5 reports. Do not copy listening WAVs or corpus audio. Check report text/configuration for unnecessary absolute personal paths; preserve model and evidence identities.

**Gate:** focused packaging/compliance/installed-runtime tests pass. Real decoding is a short packaging smoke check, not new parity or throughput qualification. CI uses tiny fixtures; real model bundles are validated locally or in an explicitly provisioned release job without automatic model downloads.

## Documentation, debugging and completion

Deliver `docs/MULTI_VAE_ONNX_PHASE6.md` and machine-readable build/install evidence. Update README and the release checklist with installed commands, explicit preparation, default/configurable store paths, cache/source identity, model-bundle installation, missing dependency/weight/artifact states and recovery. Correct stale dependency and EAR setup descriptions. State that native CPU remains selectable even when measured decoding misses audio deadlines.

Debugging entry points: package configuration and inventory for absent delivered files; `release_compliance.py` for notices/provenance failures; `vae/onnx_artifacts.py` for resolution/hash/external-data errors; `runtime/decoder_preparation_jobs.py` and the installed worker for interpreter/reuse/publication failures; the server asset resolver for an installed UI failure. Evidence should identify the failing layer and selected model/artifact without hiding the original exception.

Implementation order: installed commands/assets → explicit model contents/compliance → dependency profiles → installed smoke/recovery → documentation/evidence. Review each increment before broadening it. Run focused checks after changes; build and inspect final archives once the contents stabilize. The user handles browser integration, native playback and listening checks. Broaden regression testing only if shared runtime behavior changes.

Mark Phase 6 complete only after actual output inspection and clean installed checks pass. Keep model preparation, functional support and measured real-time support separate in the final matrix. An installation without EAR checkpoints/artifacts must report them missing, even though this workspace already has validated EAR exports.
