# Audio Input prototype — setup, comparison and debugging

Implemented on `experiment/audio-input-navigation`. This is an experimental Web navigation mode. Both analyzers consume the same backend microphone buffer and select existing corpus locations; the current decoder and overlap-add transport render file-bounded source windows. Switching methods does not reload either model.

## First listening session

1. Restart the Python server from this branch using your usual environment and refresh the Web page. The existing `python bin/serve.py` entry point remains valid.
2. Load a SAME-S corpus and its usual decoder. SAME-S is the initially exercised configuration. Stable Audio Open is implemented but not qualified by this listening prototype; EAR is unavailable in this mode.
3. Stop Decode if running, then select **Audio Input**. Refresh devices and select the microphone on the **server computer**. Browser microphone permission is not used; operating-system permission for the Python/terminal host may be needed.
4. Use headphones. Click **Start Input**, make sound, and wait for a ready result. Both analyzers run even when only one drives playback. The first VAE analysis includes model loading and can briefly be stale; the next result should recover.
5. Start with **Descriptors** and click **Start Decode**. Keep the same decoder window size for both methods; T4 is a useful first comparison. Switch to **VAE Latents** while decoding continues.
6. Compare a sustained tone with changing brightness, a few attacks, and a noisy gesture. Watch the retrieved source and result age alongside what you hear.
7. **Stop Decode** stops output but leaves input analysis active for restarting and comparison. **Stop Input** closes capture and joins its workers; output, if still running, holds its last corpus location. Unload corpus closes both lifecycles and releases the encoder.

Silence, errors, stopped input or stale results hold the last rendered location. Before the first playback start, a fresh non-silent result from the selected path is required. Switching to a path without a fresh result holds the previous location rather than using the other analyzer. A switch affects newly scheduled windows; buffered output remains audible until it drains.

## Dependencies and configuration

No new package dependency was added. Capture uses existing SoundDevice/PortAudio; descriptors use existing Torch/Torchaudio, and exact Euclidean search uses SciPy. Live latent analysis requires the matching **native encoder dependencies and pinned cached weights in the server interpreter**, including when output decoding uses ONNX. The prototype does not install packages or download weights. External decoder interpreter registrations are not supported for live encoding; an explanatory VAE error is shown. Existing optional-model installation instructions still apply.

The encoder uses the existing `config.DEVICE` selection, independently of the decoder. The existing `STABLE_AUDIO_FORCE_CPU=1` setting can explicitly force encoding to CPU. GPU contention and simultaneous native encoder/decoder workloads have not been qualified.

Default analysis uses a 1-second rolling window; each worker waits 100 ms after finishing an analysis before considering the latest input. This is **not a guaranteed 10 Hz analysis rate**. At most one analysis per path is in progress. Older snapshots are skipped; work does not queue.

To override settings, copy `docs/audio_input_settings.example.json` to a local configuration and launch:

```sh
python bin/serve.py --audio-input-settings path/to/input-settings.json
```

- `window_seconds`: input context, not decoder window length. Shorter windows can change matching and encoder boundary behavior.
- `update_seconds`: minimum wait after analysis completion.
- `freshness_seconds`: maximum buffer-end age of an eligible result; defaults to 2 seconds.
- `dwell_seconds`: minimum interval between applying selections; switching paths resets this interval.
- `silence_db`: microphone block RMS gate in dBFS.
- `channels`: one or two capture channels. Mono is duplicated for stereo encoders; descriptors use a mono mix.
- `encoder_tail_frames`: number of final encoder frames excluded; defaults to one as a conservative experimental boundary guard. This is not a verified receptive-field bound.

Capture uses the selected device's default sample rate and resamples analysis windows to the corpus rate. Corpus normalization and descriptor weights/gating are reused; input statistics are never independently fitted. Recorded model provenance, when present, is checked against the pinned encoder. Older corpora without that provenance retain metadata-based compatibility; exact historical model identity cannot be established from absent records.

## Recorded-input probe

No microphone or output audio device is opened:

```sh
python -m stable_audio_wanderer.cli.probe_audio_input CORPUS_DIR INPUT.wav --seconds 3
```

Add `--concurrent --decoder-backend onnxruntime --decoder-window 4` to compare analysis compute with CPU decoding active. `--decoder-backend pytorch` is also available. Use `--window`, `--update` and `--paths descriptors` or `--paths latents` for focused comparisons.

The probe reports input time, selected index, distance, analysis duration and batch duration. Initial timings include lazy loading. In concurrent mode it waits for each batch before issuing another; this is a short compute-contention check, not a realtime capture/output simulation or an underrun measurement. Model libraries may print initialization messages alongside the JSON records.

## Debugging

- **Matches:** inspect each path's selected index, source file/time and distance. The distances live in different spaces and cannot be compared numerically between methods.
- **Delay:** input age measures time since the captured buffer ended, using backend receipt time. Encoder tail exclusion and analysis context can make the actual feature older. Add existing decoder buffer duration to understand playback lag; these are not calibrated ADC-to-DAC measurements.
- **Holding:** check silence, errors, result age, selected method and render anchor. The render anchor is scheduled content, not necessarily the location currently reaching the output device.
- **Descriptor behavior:** see `cli/preprocess.py` for shared extraction and weighting. Pitch smoothing and MFCC dB behavior depend on the analysis window; short-window descriptors need listening validation against whole-file preprocessing.
- **Latent behavior:** see `runtime/audio_input.py` for encoding, normalization, tail exclusion and search. VAE proximity does not guarantee timbral similarity. No temporal context embedding or guided navigation is used.
- **Playback:** see `runtime/decoder_transport.py` and `runtime/manual_windows.py` for selection and source windows. Crossfades/overlap-add reuse existing transport behavior.
- **Lifecycle/UI:** `cli/serve.py` attaches a corpus-specific input service, `runtime/ws_server.py` keeps device operations off the WebSocket event loop, and `web/audio_input.js` displays both paths' state.

## Verification and remaining listening work

Focused checks cover corpus-fitted normalization, valid source windows, switching/dwell/silence/stale generations, latest-window workers, closure, transport routing and UI method switching. The short CPU SAME-S probe used a 92-frame test corpus and 1.3 seconds of recorded input, with both analyzers and the real ONNX T4 decoder concurrently active. Warmed timings were approximately descriptors 196–209 ms, encoder 58–66 ms, decoder 129–130 ms. Encoder initialization took about 2.2 seconds. These figures are representative observations from this environment, not portable performance guarantees.

Actual microphone permission/device integration, output underruns, audible switching delay and musical quality remain for the user. Start with the defaults and report what each method follows, where it sticks or jumps, and whether response delay prevents a fair comparison. That feedback determines whether matching, analysis settings or playback continuity should change next.
