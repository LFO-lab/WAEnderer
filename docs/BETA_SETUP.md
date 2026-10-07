# Beta setup and debugging

Start with the [README quick start](../README.md#beta-quick-start). Python 3.12
is the reference version. The macOS M1 Max/Core Audio evidence in
[the qualification guide](DUAL_INFERENCE_RELEASE.md) does not establish Windows,
Linux or CUDA audio performance. Other platforms remain beta validation targets.

## What the tester needs

- Python and pip, a browser, sufficient disk space for Python dependencies,
  model files and corpus data, and an available audio output device.
- Either the source project or an application wheel, plus a matching corpus and
  verified decoder bundle for prepared ONNX playback. Ordinary wheels intentionally
  exclude model binaries, private audio, corpora and local configuration.
- For encoding: the selected native profile, model weights, and input WAV files.
  The default install includes Stable Audio Open and SAME-S libraries.
  Git is required by the SAME-S source dependency. EAR requires its repository
  and matching checkpoint. Installation and first downloads need internet access.
- On Linux, PortAudio may need installation through the system package manager
  for `sounddevice`. The folder picker needs Tk and a desktop display; typing a
  folder path or source CLI preprocessing avoids depending on the picker.

No Erae hardware, Ableton, ONNX exporter, or native SAME-S library is required
for prepared ONNX playback. Model licenses are separate from the application
license; supplied bundles include their notices and terms.

## Structure and ownership

`WAEnderer_python` and `WAEnderer_erae` are independent repositories in the local
workspace. Install the Python application from its own `pyproject.toml` directory.
Install the bridge separately only when testing Erae.

| Location in the application | Purpose |
| --- | --- |
| `stable_audio_wanderer/cli/` | Installed command implementations |
| `bin/` | Source command wrappers and maintenance tools |
| `stable_audio_wanderer/runtime/` | Pipeline lifecycle, transport and audio |
| `stable_audio_wanderer/vae/` | Model adapters, availability, preparation and native workers |
| `web/` | Canonical browser assets, staged into wheels by `setup.py` |
| `docs/`, `eval_scripts/`, `tests/` | Guides, diagnostic tools and regression checks |
| `corpus/`, `user_audio/`, `eval_out/`, `.venv*/`, `build/` | Local data or generated work, excluded from releases |

The default decoder store is `~/.cache/waenderer/decoders`. Native runtime
configuration uses the platform user configuration directory; see
[NATIVE_DECODER_RUNTIMES.md](NATIVE_DECODER_RUNTIMES.md). Avoid editing package
code to set machine-specific model paths. Use documented configuration overrides.

## When setup fails

| Symptom | First check | Where to debug |
| --- | --- | --- |
| Install fails / no matching distribution | Python version, platform wheel availability, full pip error; use a fresh environment and normal dependency resolution | `pyproject.toml`, selected `requirements*.txt` |
| Command not found | Activate the same environment used for installation; run `python -m pip show stable-audio-wanderer` | Project scripts in `pyproject.toml` |
| Browser cannot connect | Terminal startup error, HTTP 8080 and WebSocket 8765 port conflicts, another running server | `cli/serve.py`, `runtime/ws_server.py` |
| Missing corpus in selector | Start from the intended working folder; put complete corpus folders under its `corpus/` | `runtime/pipeline_server.py` corpus listing |
| Decoder unavailable | Read the displayed reason; corpus VAE must match decoder, model store and selected interpreter | `vae/decoder_availability.py`, `vae/onnx_artifacts.py`, native runtime guide |
| ONNX preparation fails | Exporter environment and model/checkpoint configuration; runtime-only install does not provide export tooling | `vae/decoder_preparation.py`, [preparation guide](MULTI_VAE_ONNX_PHASE4.md) |
| Folder picker fails | Tk and desktop availability; enter the path instead | Pipeline folder-picker handling |
| No sound / gaps | Selected output device, transport, gain and underrun/decode timing indicators | `runtime/decoder_player.py`, `runtime/decoder_transport.py` |

Default ports are simplest for the beta. If changing the WebSocket port, also
check the browser connection settings; changing the HTTP port alone does not
change the WebSocket port. Keep the startup terminal output when reporting errors.
Never force an install with `--no-deps` to bypass an incompatible profile.

A useful bug report includes application version/revision, OS and architecture,
Python version, installation profile, `python -m pip check` output, corpus VAE,
decoder backend/device, exact action, displayed error and relevant terminal log.
Redact personal file paths and credentials before sharing. Do not send source
audio or model weights just to report a setup problem.

## Verification scope

The existing [portability receipt](installation_portability_checks.json) confirms
packaged worker inclusion and discovery outside a checkout. It explicitly leaves
the combined native clean install unverified. Historical locks and qualification
reports describe particular environments; they are not universal guarantees.

The beta handoff still needs a new-machine dependency installation, the supplied
corpus/bundle pair, browser workflow, output-device integration and listening.
A package build and short HTTP startup check verify distribution structure and
launch only. They do not verify encoding, model inference or audible performance.
