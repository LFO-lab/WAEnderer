# Roadmap — Audio-driven corpus navigation prototype

Status: **prototype implemented on `experiment/audio-input-navigation`; user microphone integration and listening checkpoint pending**.
Date: 2026-10-08.

## Objective

Add an Audio Input navigation mode to WÆnderer with a live switch between **Descriptors** and **VAE Latents**. Feed the same microphone audio to both analysis paths, retrieve locations in the loaded corpus, and render them through the existing decoder and window transport. The user will alternate between the paths and decide what to develop next after listening.

This is corpus-driven resynthesis. Loading a corpus does not train the VAE; direct microphone encode/decode alone would not impose that corpus's sound palette.

## Scope and initial decisions

- Implement on a dedicated experiment branch in the `WAEnderer_python` repository; proposed name: `experiment/audio-input-navigation`. Create it when implementation begins, after checking the current branch and local changes.
- Target the existing Python/Web performance path first. Erae and other integrations are deferred.
- Use the loaded corpus's VAE identity, weights, sample rate, channel convention and normalization. Initially qualify one available corpus/model pair, without hardcoding it into the analysis interface.
- Run both analyzers on the same timestamped input. The switch chooses the result driving navigation; capture and models remain loaded.
- Retrieve observed corpus locations and contiguous, file-bounded latent windows. Reuse decoding, overlap-add and output buffering.
- Start with direct matching, fixed output windows and modest selection stabilization. Defer context-guided navigation, latent blending, pitch shifting, envelope transfer and training.
- Keep analysis settings explicit and portable. Do not persist machine-specific device indices as universal defaults.

## Architecture

```text
Configured microphone → bounded timestamped rolling buffer
                          ├─ descriptors → corpus descriptor search ─┐
                          └─ matching VAE → corpus latent search ─────┤
                                                                     ↓
                                       selected path + selection stabilization
                                                                     ↓
                                        corpus location → latent window
                                                                     ↓
                                       existing decoder / overlap-add / output
```

The audio callback only copies input and records timing. Analysis and inference run outside it. Each analyzer consumes the newest available window and publishes its latest result; old work must not accumulate. Encoder and decoder scheduling/device contention must be measured rather than assuming that separate workers provide independent compute.

An analysis result should carry path, input timestamp, corpus generation/identity, selected index, distance, analysis duration and error state. Reject results from a previous corpus or expired input. Distances are diagnostic within each path and are not directly comparable across paths.

## Phase 0 — Confirm integration and establish the experiment

- [x] Check repository instructions, current branch and working tree; create the experiment branch without discarding local work.
- [x] Trace corpus loading, active model ownership, Web messages and transport mode selection.
- [x] Confirm microphone capture ownership. Prefer backend capture for a local Python prototype if compatible with the current audio device setup; confirm this before implementation. Browser capture would require audio transport and is a separate complexity.
- [x] Identify the smallest entry point for externally selected corpus indices in `runtime/decoder_transport.py` and `runtime/manual_windows.py`.
- [x] Inspect descriptor extraction for whole-clip behavior and VAE encoder context, padding and stochastic behavior before choosing rolling-window settings.
- [x] Select one corpus/model pair and record existing decoder evidence that can be reused.

Exit: document the concrete integration points, supported configuration and initial analysis window/update settings. Share findings if live encoder context or device sharing changes the scope. No low-latency claim yet.

## Phase 1 — Offline comparison harness

- [x] Reuse or minimally extract descriptor computation from `cli/preprocess.py` so offline preprocessing and input queries share the same definitions.
- [x] Apply stored descriptor center, scale, clipping, group weights and pitch gating in the same order as corpus preprocessing. Do not fit normalization to incoming audio.
- [x] Search the full weighted descriptor space, bypassing the reduced Manual control coordinates. Start with exact Euclidean matching using existing NumPy/SciPy dependencies.
- [x] Encode recorded input with the matching `VAEAdapter`. Apply stored `Z_mean` and `Z_std`, then search observed normalized corpus latent frames using exact Euclidean distance.
- [x] Use the newest valid analysis frame, excluding encoder boundary/padding frames where inspection shows that is necessary. Do not average complete gestures into one point.
- [x] Feed short recorded inputs through rolling windows to exercise the intended live analysis behavior, not only full-file encoding.
- [x] Emit compact results: input time, path, index, source file/time, distance and processing time.

Exit: both paths return valid corpus indices from the same short input. Check transformation consistency using a small stored corpus excerpt; exact self-retrieval is not required if encoding is stochastic or window context differs.

## Phase 2 — Shared microphone capture and analysis

- [x] Add configured input device, channel mapping and resampling to corpus sample rate. Reuse existing audio libraries where suitable; declare any new dependency explicitly.
- [x] Maintain a bounded rolling buffer with timestamps and explicit analysis duration/update interval.
- [x] Give each analyzer bounded pending work and latest-result publication. Slow encoding must not block descriptor analysis or the output callback.
- [x] Load the encoder once. Verify model/device compatibility with the active decoder and avoid unsafe concurrent access to a shared model.
- [x] Add configurable silence gating, result freshness and clear device/model errors. Hold the current selection during silence or temporarily missing input; expose the reason.
- [x] Stop capture and workers cleanly; invalidate pending results on corpus replacement or shutdown.

Implementation: one backend capture stream and two latest-window workers; simulated capture verifies bounded work and closure. A concurrent recorded-input probe exercised both real analyzers and ONNX decoding. **Exit remains subject to user microphone integration**; physical input age, device behavior and output underruns are not yet measured.

## Phase 3 — Audible A/B switching

- [x] Add Audio Input to transport mode selection and route selected corpus indices into the existing file-bounded window builder.
- [x] Use identical window duration and transition behavior for both query paths. Preserve corpus frame provenance.
- [x] Add a shared minimum selection interval; gate silence before applying a query. Keep stabilization parameters explicit.
- [x] Switch at the next normal playback scheduling boundary using a fresh result. Define and display behavior when the destination path has no fresh result; never silently use the other path.
- [x] Keep already queued output bounded. Measure switch-to-audible delay, since changing the selected path cannot replace audio already buffered.
- [x] Add minimal Web controls: input device, level meter, Descriptors/VAE Latents switch and start/stop. Put selected source, result age, timing and error details in a compact diagnostic view.

Implementation: Web Audio Input tab, live method switch, source windows and diagnostics are connected. **Audible exit criterion remains pending user listening**; automated routing and UI checks do not establish sound quality.

## Phase 4 — Listening checkpoint and next decision

Use one corpus, one decoder configuration and the same output window settings. Start with short examples: a sustained tone with changing brightness, a few attacks, and a noisy gesture. A recorded input replay can supplement microphone use for repeatable comparisons.

The user evaluates responsiveness, meaningful changes, repeated selection of the same locations, continuity and how much input articulation survives. Record observations beside selected source locations and query age so retrieval problems can be distinguished from playback delay.

Choose the next step from that evidence:

- Descriptor path: adjust feature relevance or streaming extraction if matching is understandable but too coarse.
- Latent path: consider causal context matching if individual frame selection is unstable or ignores gestures.
- Shared playback: adjust dwell time, candidate continuity or window timing if both paths select useful material but sound fragmented.
- Guided navigation: use input as a neighborhood target if literal following is less musically useful.

Exit: user listening feedback determines continuation. Do not expand into a production feature or broader model qualification automatically.

## Proportional verification

- A few meaningful checks for query normalization/weighting, valid index-to-window mapping, and stale/cross-corpus result rejection.
- A small switching/lifecycle check using simulated analysis results; no microphone hardware required in automated tests.
- Reuse relevant existing window, transport and decoder tests. Run affected suites once; avoid a new exhaustive decoder matrix.
- One short recorded-input timing run with both analyzers and decoding active. Report processing time, query age and existing underrun counters separately.
- User performs microphone integration and listening tests. Timing success does not establish musical quality.

## Where to look when debugging

| Symptom | First evidence to inspect |
| --- | --- |
| Surprising descriptor matches | Feature values, saved normalization/weights/gating, channel mix and input gain |
| Surprising latent matches | Model/weights identity, resampling, latent normalization, encoder context and stochasticity |
| Stuck selection | Silence gate, result freshness, dwell interval and repeated nearest indices |
| Delayed reaction or switch | Capture window duration, analysis completion time, query age and queued output |
| Clicks or broken gestures | Retrieved source boundaries, window duration and existing overlap-add transitions |
| Dropouts | Encoder/decoder device contention, worker backlog and transport underrun counters |

Reference files: `cli/preprocess.py`, `vae/base.py`, `vae/sae.py`, `runtime/player.py`, `runtime/manual_player.py`, `runtime/manual_windows.py`, `runtime/decoder_transport.py`, `runtime/pipeline_server.py`, `runtime/ws_server.py`, and the Web performance controls. The new query service is `runtime/audio_input.py`; the recorded-input harness is `cli/probe_audio_input.py`.

## Evidence status

Implemented: shared capture, descriptor queries with corpus-fitted transforms, native VAE queries with corpus normalization, exact Euclidean retrieval, latest-window workers, silence/freshness/dwell handling, source-window playback and Web A/B controls. Initial native encoder support is SAME-S and Stable Audio Open in the server interpreter; EAR and external encoder interpreters are deferred. No new package dependencies or model downloads were added.

Verified: 65 relevant Python checks across input, transport, windows, descriptors, pipeline and WebSocket handling; the focused Audio Input JavaScript check; JavaScript syntax checks. A real 92-frame SAME-S corpus and ONNX decoder were wired through Web performance setup with a simulated output device. A 1.3-second recorded-input CPU probe exercised both analyzers concurrently with real ONNX T4 decoding: warmed descriptors 196–209 ms, encoder 58–66 ms, decoder 129–130 ms; initial encoder query about 2.2 seconds including loading. These are compute observations, not end-to-end latency or underrun qualification.

Existing Web Wander and ONNX UI suites fail with the same errors when run against unchanged HEAD: respectively, a mock element without `setAttribute`, and the corpus-specific availability assertion. They were not rewritten as part of this prototype.

Pending: actual microphone/device integration, audible switching, output underruns, musical quality and Phase 4 user judgement. No physical audio device was available to the execution environment. See [setup and debugging guide](docs/AUDIO_INPUT_PROTOTYPE.md) and [portable settings example](docs/audio_input_settings.example.json).

Implementation defaults: 1-second analysis buffer, 100 ms post-analysis wait, 2-second freshness limit, 150 ms dwell, -45 dBFS silence gate, mono capture and one excluded encoder tail frame. Capture and decoding can be stopped independently; unloading closes both. The tail guard is experimental, not an established encoder receptive-field bound.
