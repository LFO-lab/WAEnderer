# Phase 5 implementation plan — numerical, audio and runtime validation

Status: planned on 3 October 2026, after completed Phase 4. Validation tooling delivered on 4 October 2026; see [the implementation and shared testing guide](MULTI_VAE_ONNX_PHASE5.md). Full evidence campaigns remain pending. This document defines implementation and evidence gates; it does not mark Phase 5 complete.

## Outcome and scope

Produce a source-bound matrix for all four registered VAEs that independently reports functional decoding, numerical parity, listening review and measured real-time support. Keep ONNX CPU and PyTorch CPU selectable when functional decoding passes, even when they miss audio deadlines. Qualify only explicitly tested model/backend/device/window/navigation/hardware scenarios.

Reuse the production decoder factory, transport, navigation setup and overlap-add implementation. Extend the existing evaluation tools rather than introducing another playback path. Preserve the earlier SAME-S campaign and its failures as historical evidence; its listening approval and hardware results do not qualify Stable Audio Open or EAR.

Packaging and clean distribution validation remain Phase 6. No export, download, tolerance adjustment or production performance change is implied by running a campaign.

## Verified starting point and gaps

| Area | Existing evidence or implementation | Phase 5 work |
| --- | --- | --- |
| Artifacts | All four actual artifacts pass export validation; Phase 4 verifies offline reuse and lifecycle | Bind new reports to the exact selected source, artifact files and execution settings |
| Comparison | `compare_corpus_decoders.py` loads SAME-S directly and hardcodes 4096 samples/latent, stereo and 44.1 kHz | Generalize through `corpus_decoder_spec`, `select_decoder` and `create_decoder`; obtain geometry from decoder metadata |
| Numerical policy | SAME-S stochastic gates; Stable Audio Open/EAR deterministic export tolerances | Freeze a versioned per-model/device policy and record repeats in both engines |
| Runtime harness | `benchmark_dual_inference.py` uses the common factory and production navigation | Add EAR source/artifact configuration, explicit scenarios, engine-change/restart evidence and complete timing identity |
| Synthetic runtime input | No-corpus benchmark assumes SAME-S and 256 dimensions | Make synthetic model selection explicit and geometry-driven; synthetic/replay runs remain diagnostics |
| Qualification | `qualification.py` checks a SAME-S-oriented two-backend campaign, all even windows and a fixed 20,000-block floor | Add a versioned multi-VAE evaluator with scenario-specific coverage, identities, duration and independent statuses |
| Physical evidence | Earlier SAME-S ONNX CPU/MPS ten-minute campaign; other VAEs have short functional checks | Fresh physical qualification for claimed scenarios; short no-underrun MPS checks are candidates only |
| EAR data | Source-bound probe fixtures; existing 44k harpsichord corpus and receipt-backed three-excerpt 48k corpus | Verify provenance and navigation prerequisites; label narrow content coverage and add representative excerpts where needed |

Current EAR CPU checks underrun. Stable Audio Open CPU also has recorded performance limits. Start with functional checks and screening; do not promise a real-time CPU pass. CUDA remains untested unless an actual NVIDIA host is available.

## Increment 1 — Freeze the campaign contract and reference policy

Add proposed `eval_scripts/multi_vae_validation_common.py` for evaluation-only configuration, identities, raw-window selection and report serialization. Keep qualification decisions in `stable_audio_wanderer/qualification.py` so reports can be checked without loading models. Add a checked-in versioned policy such as `docs/multi_vae_phase5_protocol.json` before evaluating new ONNX results.

Each campaign records VAE ID, corpus and raw-input hashes, available corpus provenance, exact native checkpoint/config/code identity, artifact identity and all declared file hashes, supported windows, sample rate, channels, samples per latent, dtype/layout, library versions, OS/hardware, execution provider, thread settings and seeds. Include a policy digest and run ID in every report. Reject explicit source conflicts. Legacy corpora without checkpoint provenance remain labeled geometry-matched with unknown historical provenance; never invent hashes for them.

Freeze the existing CPU policies as the baseline:

| Comparison | Baseline gate |
| --- | --- |
| SAME-S synthetic | Eight seeds per even T2–T32; RMSE < 0.005 and SNR > 20 dB |
| SAME-S real windows | Cross-engine RMSE <= 3 × max(ONNX repeat RMSE, native repeat RMSE, 1e-7) |
| Stable Audio Open native CPU / ONNX CPU | Max absolute error <= 2e-4 and RMSE <= 2e-5 |
| EAR 44k native CPU / ONNX CPU | Max absolute error <= 1e-4 and RMSE <= 1e-5 |
| EAR 48k native CPU / ONNX CPU | Max absolute error <= 1e-4 and RMSE <= 1e-5 |
| EAR native CPU / GPU | Existing Phase 3 baseline: max absolute error <= 1e-3 and RMSE <= 1e-4 |

Confirm each policy against native reference behavior before the new ONNX campaign. Stable Audio Open GPU and any new device policy require a separately recorded native-reference characterization before cross-engine evaluation. Preserve SAME-S model stochasticity; seeds select reproducible inputs, and the comparison must not disable noise or reset RNG between repeat calls to force equality. Record at least two decodes per engine for every probe. Additional reference repetitions may characterize variability, with their count fixed in the protocol.

Specify OLA tolerances separately before evaluating assembled output: deterministic-model PCM uses the baseline error limits above; SAME-S uses the corresponding within-engine repeated OLA renders and the frozen relative-noise rule. Record silence handling without infinite JSON SNR values. A later policy revision creates a new campaign and preserves the failed original; it cannot retroactively pass an existing run.

**Gate:** policy/schema review and tests establish strict finite PCM, shape/sample checks, stochastic handling, provenance and independent status meanings before real evaluation.

## Increment 2 — Identical latent comparisons and listening renders

Generalize `compare_corpus_decoders.py` while retaining its current SAME-S CLI behavior and historical report readability. Add explicit source/config/store/artifact options matching the production factory. Extend or wrap `validate_torch_decoder.py` for model-specific synthetic inputs using the same shared helpers.

1. Use BurntMemory for SAME-S and Rack for Stable Audio Open. Verify the existing EAR 44k corpus against the supplied checkpoint; use the receipt-backed EAR 48k corpus. Reuse `prepare_ear_probe_fixtures.py` and `prepare_ear_web_corpus.py` when additional data is needed. Do not relabel another VAE's latents. Record audio hash, excerpts, resampling, encoding seed and exact EAR source identity.
2. Select at least four positions per eligible file, with no window crossing a file boundary, for every supported even T2–T32 window. Denormalize once to contiguous float32 raw latents and feed the same bytes to every compared engine. Record probe IDs/hashes, counts and excluded short files; empty coverage fails.
3. Verify output dtype, layout, channels, finiteness and exact sample count before computing RMSE/max error/SNR. Expected samples come from measured decoder metadata: SAME-S 4096, Stable Audio Open 2048, EAR 44k 1024 and EAR 48k 960 samples per latent. Never derive these ratios from rounded corpus `latent_hz`.
4. Compare native CPU against ONNX CPU, then each available native GPU against the reference under its frozen device policy. Measure repeats in both engines, including ONNX; preserve per-probe failures instead of stopping before a report is written.
5. Assemble identical sequences using production `StreamingFullOverlapAdd`, with T8/hop 4 as the listening baseline and additional declared fixed/window-change sequences for geometry and continuity checks. Record start/tail treatment, duration, peaks and post-OLA errors. Do not add normalization, clipping or engine-specific processing.
6. Write paired local float WAVs at the actual model sample rate and a receipt identifying each excerpt and engine. Include sustained/quiet/transient material where available. The current EAR harpsichord excerpts establish a limited domain, not broad audio coverage; any absent content categories remain explicit.

**Gate:** real and synthetic comparisons pass for each claimed functional backend; OLA geometry and numerical results are recorded separately. Listening files and corpus/model data stay in ignored local output directories; tracked reports contain hashes and receipts.

## Increment 3 — Runtime scenarios and physical instrumentation

Extend `benchmark_dual_inference.py` with a versioned scenario definition and common selection inputs. Preserve software-clock/replay modes for diagnostics and label them clearly. Use production navigation and the actual audio device for real-time claims.

- Record decode p50/p95/p99 and observation counts per window, warm-up and session preparation separately, command request-to-PCM-generation transitions, callback timings, buffer occupancy, process CPU usage, RSS in normalized bytes and available GPU allocation metrics. Synchronize GPU timing where required; state what each metric measures. Generation activation is a PCM boundary estimate, not measured speaker latency.
- Report negotiated device/host API, sample rate, channels, block size, latency settings and execution/thread configuration. Separate buffer underruns from device underruns. Preserve nonfinite output, timeout, cleanup and navigation errors in a terminal report, including interrupted runs.
- Measure active callback time and rendered sample counts independently of startup, stops and teardown. The existing duration includes cleanup; it must not count toward the ten-minute playback gate. Replace the fixed block-count assumption with scenario/sample-rate/block-size coverage and explicit permitted stop intervals.
- A continuous qualification run must provide at least 600 seconds of active playback per claimed scenario. Its declared schedule covers Random/Manual/Reorganized and each claimed fixed/adaptive setting. If qualifying the entire T2–T32/adaptive set in a combined run, require completed transitions and decode observations for every member; disclose time/count per member. A restricted-window claim lists its tested set explicitly and makes no broader claim.
- Exercise stop/drain/start, stopped engine changes and fresh-process reloads in a separate declared lifecycle sequence. Carry exact identity and resource ownership through each segment; time spent stopped or using another engine cannot satisfy a backend's 600-second requirement. Verify preference retention through the Phase 4 Web path and that reload resolves the same artifact offline.
- Keep runs serial and isolate them from exports or other heavy inference. Native SAME-S and EAR use their established environments; the runner dispatches fixed script arguments to configured interpreters, without installation or shell commands supplied by Web clients.

Screen all locally available CPU/GPU choices with short runs after numerical validation. Run ten-minute physical campaigns for candidates intended to carry real-time claims. A screening failure may classify a backend as functional with measured deadline failures; a later restricted-window qualification is a new explicit scenario. It must not silently change the advertised functional windows.

**Gate:** each real-time report has >=600 seconds active physical production playback, complete declared scenario coverage, zero buffer and device underruns, finite PCM and no unresolved runtime/lifecycle failure. A performance failure does not revoke a passing functional result.

## Increment 4 — Evidence checking and the qualification matrix

Extend `qualification.py` and add proposed `bin/check_multi_vae_qualification.py`. Preserve `evaluate_campaign` and `bin/check_dual_inference_qualification.py` for the historical SAME-S manifest. Introduce a new schema rather than interpreting old reports as though they contain missing Phase 5 fields.

The new checker verifies matching VAE/source/artifact/input/policy identities across numerical, OLA, runtime and listening reports, explicit required coverage and complete finite measurements. An ONNX/native pair must refer to the same model; GPU evidence must match the exact device. Missing, empty, contradictory or truncated evidence fails the relevant claim. Numerical failure can fail functional validation without requiring transport/listening evidence for an unavailable model.

Build a per-VAE/backend/device/window/scenario matrix with separate fields for installation/availability, functional result, numerical result, physical performance result and listening status. Statuses distinguish passed, failed, missing model/dependency, unavailable hardware and not tested/pending. Include report links, limitations and failure reasons. Avoid a single Boolean that hides partial success or conflates CPU decoding with real-time capability.

Listening records identify reviewer, date, exact paired files/hashes, VAE, excerpt, compared engines and explicit regression verdict. SAME-S historical feedback remains tied to its original pairs. Request user feedback on the new per-VAE pairs after they exist; until received, mark perceptual review pending. Muted physical playback cannot satisfy listening review. No automated metric can supply that verdict.

**Gate:** checker tests reject wrong checkpoint/artifact/corpus, another VAE's listening approval, another GPU's result, insufficient active duration, uncovered scenarios, nonzero separate underrun counters, missing metrics and absent listening feedback. Valid mixed matrices retain CPU functional passes and explicit performance failures.

## Increment 5 — Campaign execution and delivery

Implement and review in this order: protocol/identity → comparison/OLA/listening generation → benchmark/lifecycle instrumentation → checker/matrix → actual serial campaign. Run focused tests at each integration boundary, then the full Python suite and the three existing JavaScript contract suites after shared production interfaces change. Use deterministic decoder/clock fixtures for evidence rejection and lifecycle tests; real model/audio measurements remain separate integration evidence.

Recommended actual sequence:

1. Recheck SAME-S on BurntMemory with the current artifact/runtime and retain the previous campaign alongside the new one.
2. Evaluate Stable Audio Open on Rack: native CPU, ONNX CPU and available MPS; screen before deciding which scenarios warrant ten-minute qualification.
3. Evaluate both EAR variants separately using exact supplied sources and their matching real corpora. Complete native CPU/ONNX comparisons and MPS screening/qualification; report known CPU deadline failures.
4. Run separate restart/engine-change sequences, prepare the four-VAE listening pairs and collect feedback. If feedback is outstanding, numerical/runtime delivery can proceed with Phase 5 explicitly incomplete.
5. Check the combined evidence matrix and publish the report with all failed attempts and untested CUDA hardware retained.

Deliver `docs/MULTI_VAE_ONNX_PHASE5.md`, the frozen protocol, machine-readable per-model numerical/runtime/lifecycle reports, listening receipts and a checked qualification matrix. Keep WAVs, source audio, fixtures, models and corpora local. Update README and roadmap checkboxes only as their actual gates pass; mark the phase complete only when required model coverage and comparative listening feedback are present. Full completion may include functionally supported CPU backends that failed real-time qualification, with those limits stated plainly.

Estimated physical playback alone is at least 80 minutes if all four models qualify both ONNX CPU and native MPS; extra scenarios, retries, CPU claims and lifecycle runs add time. Screening can reduce unnecessary long runs, but cannot substitute for any claimed ten-minute qualification. Performance fixes discovered during the campaign should be separate reviewed changes, followed by new measurements under the same policy with old failures preserved.
