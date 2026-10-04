# Phase 6 — portable packaging and decoder delivery

Implemented on 4 October 2026. Build/install evidence is recorded separately in
`multi_vae_phase6_delivery.json`. These checks establish functional delivery;
listening, physical integration and uncompleted Phase 5 qualification remain with
the user. No models, original checkpoints or corpus audio are committed or published.

Verification: 49 focused tests passed. Ordinary and SAME-S model-bearing source
and wheel outputs were inspected, along with the four-real-VAE decoder bundle.
A fresh resolved Python 3.12.13/macOS runtime installation passed `pip check` and
the dependency license policy, eight offline initial/restart probes, explicit
CLI reuse, missing-native EAR discovery, installed worker preflight and actual
HTTP/WebSocket server startup/asset retrieval/clean shutdown. New playback or
listening tests were not run. The existing development environment's pre-existing
native SAME-S/Torch requirement conflicts were observed and left unchanged.

Final checksum evidence accompanies releases separately and is excluded from the
source archive to avoid hashing a report that contains its own archive checksum.
Other tracked manifests and validation reports remain included in the source
archive; each selected decoder bundle includes its own immutable parity reports.

## Install and run

```bash
python -m pip install /path/to/stable_audio_wanderer-0.1.0-py3-none-any.whl
python -m pip check
waenderer-serve
```

The ordinary wheel contains the server, preparation CLI/worker, model exporters,
Web UI and license texts. It does not inherit local ONNX exports from the builder.
The source archive includes documentation, reports and canonical Web assets.
`uv build` builds the wheel from that source archive. For an explicitly selected
legacy SAME-S payload, use `SAW_RELEASE_BUILD=1 uv build`; its provenance, graph
digest and notice checks remain required. Other models use a separate explicit
decoder bundle instead of entering a wheel through package-data wildcards.

The canonical command implementations live in `stable_audio_wanderer/cli`.
Existing `python bin/serve.py` and `python -m bin.prepare_decoders` commands remain
checkout wrappers. Installed workers use the package namespace and run in a
temporary working directory. Each configured exporter/native interpreter must
have the application installed, plus its own required libraries; the server does
not inject a developer checkout into another interpreter's import path.

## Runtime and native/export profiles

Use the base wheel or `requirements-runtime.txt` for runtime. `requirements.txt`
retains its previous exporter-inclusive behavior. Install `.[export]` explicitly
when exporting. `onnx` remains a runtime dependency for structural and external
parameter validation; `onnxscript` is exporter-only. Torch remains an application
dependency for navigation and other components. The clean ONNX load boundary does
not import native VAE model libraries.

Keep `requirements-same-s-native.txt`, `requirements-ear-native.txt` and their
reference locks in separate environments. The source/native profiles explicitly
include `native-stable-audio-open` to retain their previous effective dependency
sets; pinned native requirements and installed working environments are unchanged.
The combined native profile is not newly qualified. The base wheel excludes all
native model libraries; install `.[native-stable-audio-open]` for Stable Audio Open
preprocessing/native playback/export. Runtime checks also block native imports to
catch accidental coupling.

```bash
waenderer-prepare --all-installed --config-file /path/to/preparation.json --json
waenderer-prepare --vae-id ear_vae_44k --weights /path/to/ear_vae_44k.pyt \
  --repo /path/to/EAR_VAE --store-dir /path/to/decoders
```

The common command reuses a verified artifact before requiring an exporter.
Missing weights/dependencies/evidence inputs stay explicit. Select a compatible
corpus for a new Stable Audio Open export; existing artifacts need no corpus for
reuse. Interpreter and source configuration are described in the
[Phase 4 guide](MULTI_VAE_ONNX_PHASE4.md) and
[native runtime guide](NATIVE_DECODER_RUNTIMES.md).

## Assemble and install a decoder bundle

Start from [the selection example](model_distribution.example.json). Pin each
artifact identity and its exact `source` object from `decoder.json` (legacy SAME-S
source identity is obtained through `read_artifact`). Supply reviewed model
notices, full license texts and upstream notices with file hashes. Paths are
relative to the selection file unless absolute; they are not copied into the
distribution inventory. The example is a template, not a usable model receipt.

```bash
waenderer-model-bundle assemble --selection /path/to/selection.json --output /path/to/bundle
waenderer-model-bundle check /path/to/bundle
waenderer-model-bundle install /path/to/bundle --store-dir /path/to/decoders
```

Assembly copies only the selected manifests, graphs, declared external tensors,
validation reports and notice files. Legacy SAME-S also retains its graph-bound
`even_window_validation.json` when that report supplies additional advertised
windows. It stages, hashes and checks the copied
payload before publishing a new destination. Existing destinations are refused.
`distribution.json` describes complete contents; extra files, missing reports,
changed identities and undeclared graph parameters fail validation.

Installation validates the entire bundle first, then uses per-VAE store locks,
immutable artifact directories and atomic pointers. Models are installed
independently; a later model failure does not roll back earlier successful
models. Notices/receipts are retained under `delivery-notices` outside the
immutable runtime identities. Installation is explicit and should happen while
the application is stopped; this CLI does not detect another process's playback.

The default runtime store is `~/.cache/waenderer/decoders`, configurable through
preparation/server configuration. Its layout is
`<vae_id>/<artifact_identity>/decoder.json`, with `<vae_id>/current.json` selecting
the artifact. Identity covers source/configuration, export settings, manifest and
declared file hashes. Runtime cache keys also include execution settings. An
explicit selected corrupt artifact never silently falls back to another graph.

For missing files, install a trusted bundle or prepare explicitly in the matching
export environment. To repair an existing corrupt copy, stop the application and
run the install command with `--repair`. The old directory is retained as
`.corrupt-<identity>-<unique_id>`; the verified replacement is staged first, and a
failed rename restores the old directory. Failed preparation preserves the
previous usable pointer. Missing native EAR checkpoints still disable native
choices while a validated EAR ONNX decoder remains usable independently.

## Checks and evidence

```bash
waenderer-model-bundle inspect /path/to/application.whl
waenderer-model-bundle inspect /path/to/source.tar.gz
# Add --model-bearing for the legacy SAME-S release profile.
waenderer-check-delivery --bundle /path/to/bundle --output /path/to/delivery.json
```

The delivery check requires a non-editable installation without optional native
SAME-S/EAR/export libraries. It installs into a temporary store and launches
fresh processes from an unrelated directory with networking and native model
imports denied. Each selected model decodes a short synthetic window, checks
geometry/finite float32 stereo PCM/CPU provider, reuses its artifact and survives
a fresh restart. The installed common CLI checks reuse, and application startup
imports are checked offline. These checks do not allocate an audio device.

Tiny dynamic/fixed/external-data fixtures test delivery failure, notice/source
binding, corruption, explicit repair and preservation of prior artifacts. CI
inspects actual ordinary source and wheel archives. Real model smoke checks use
previously validated exports; no numerical tolerances or listening approvals are
changed. CPU functionality remains available when measured audio deadlines fail.
Multi-VAE GPU/physical qualification remains pending; historical SAME-S claims
retain their original hardware and scenario limits.

Some immutable parity reports retain absolute local evidence paths. They contain
no audio payload, but those paths remain visible in a model bundle. Changing a
hashed report changes artifact identity: inspect this before any public release,
and do not silently redact a validated artifact. New release evidence avoids
embedding private selection/interpreter configuration.

Debug absent wheel files in `setup.py`/`pyproject.toml` and `inspect_archive`;
license/source failures in `release_compliance.py`/`validate_materials`;
runtime hash/geometry/external-weight failures in `vae/onnx_artifacts.py`;
reuse/worker failures in `runtime/decoder_preparation_jobs.py` and the installed
worker; Web delivery failures in `assets.web_directory`. Job errors retain the
worker's diagnostic log and identify missing application installation. Keep
browser/manual integration and listening observations separate from these checks.
