# Multi-VAE ONNX — phase 1

Completed 2 October 2026. This phase implements the shared artifact, loading and publication infrastructure. It does **not** generate Stable Audio Open or EAR exports or add their preparation actions to the menu; those remain phases 2–4.

## Runtime and compatibility

`vae/onnx_artifacts.py` resolves and validates artifacts; `vae/artifact_decoder.py` implements the shared CPU decoder. The factory now accepts ONNX artifacts for every registered VAE, with strict corpus VAE, latent dimension and sample-rate matching. PCM must be finite float32 stereo with exactly `T * samples_per_latent` samples. Layouts BDT/BTD and BCT/BTC are supported. The measured integer sample ratio controls output and overlap-add timing; explicitly declared corpus rates can accommodate historical rounded metadata without deriving sample counts from it.

The existing `same_s.web_decoder.v1` manifest is adapted in memory. Its graph is neither rewritten nor exported. Loading still uses `SameSAppOnnxDecoder`, with CPUExecutionProvider, one intra/inter-op thread, basic optimization and the existing startup behavior. Native CPU/MPS/CUDA selection remains independent.

New artifacts support a dynamic graph covering multiple even windows or separate fixed-window graphs. Every window must have exactly one graph. Every graph's I/O is checked on load. Sessions use the same CPU/thread/optimization policy and a two-session LRU limit; decode and close are serialized. Evicted sessions are rebuilt and files revalidated on reuse, so fixed-graph switching can be expensive. These checks establish functionality, not real-time qualification.

Cache keys include the resolved directory, artifact identity, manifest digest and execution settings. Selection is checked again before loading; changed files cannot silently substitute a previously selected artifact. Existing pipeline stop/drain/release behavior is preserved.

## Artifact contract

New `decoder.json` files use `format_version: waenderer.onnx_decoder.v1` and contain:

| Field | Contract |
| --- | --- |
| `vae_id`, `backend`, `provider` | Registered VAE, `onnxruntime`, `CPUExecutionProvider` |
| `source` | `model`, exact `config_sha256` and `weights_sha256`; `revision` when applicable |
| `export` | Positive `opset` and nonempty `tool_versions` map |
| Geometry | `sample_rate`, `channels: 2`, `latent_dim`, measured `samples_per_latent`, accepted `corpus_latent_hz` list |
| Windows | Sorted distinct even `supported_windows`, member `default_window` |
| I/O | `input_name`, `output_name`, `input_layout`, `output_layout`, `ola_mode: full_overlap_add` |
| `graphs` | Entries with relative `path`, boolean `dynamic`, and covered `windows`; fixed graphs cover one window |
| `files` | Relative filename to lowercase SHA-256 map, including every graph, external parameter file and validation report |
| `validation_report` | Name of a hashed JSON report |

The report must contain `passed: true`, the same `source` object and a `graphs` map binding graph paths to their hashes. Exporters must supply the actual native-reference parity results and tolerances in this report. The publisher checks that this evidence is bound to the selected files; it does not independently perform native parity or establish that a supplied report is truthful. Real per-model exporter/parity work remains in phases 2 and 3. Tests use clearly synthetic models, not substitute VAE exports.

Artifact identity is SHA-256 of canonical JSON containing the manifest and file-hash map. Manifest edits, export settings, source identities or file changes therefore change identity. Runtime also checks the manifest's actual byte digest.

Corpus metadata may include `vae_weights_sha256`, `vae_config_sha256` and `vae_source_revision`; when present these must match. Historical corpora without these fields can establish VAE/geometry compatibility but cannot prove checkpoint identity. Callers can additionally supply `decoder_source_identity`; an explicit EAR `vae_weight_path` binds its actual checkpoint hash to the selected ONNX artifact.

## Resolution and publication

Factory configuration supports:

- `decoder_backend: onnxruntime`, `decoder_device: cpu`.
- `decoder_artifact_dir`: explicit directory containing `decoder.json`; takes precedence over store lookup.
- `decoder_store_dir`: optional store root, default `~/.cache/waenderer/decoders`.
- `decoder_artifact_id`: explicit immutable identity within the store.
- `decoder_source_identity`: expected source fields, rejecting stale checkpoint/configuration identities.

Store layout is `<store>/<vae_id>/<artifact_identity>/`, with an atomic `<vae_id>/current.json` pointer. For compatibility, SAME-S defaults to its packaged resource rather than the store pointer; use an explicit artifact ID or directory to select a managed SAME-S export. Other VAEs resolve the current store pointer unless an ID/directory is supplied. Missing or invalid artifacts fail explicitly; resolution never downloads, exports or silently switches engines.

The Python publication API for later exporters is:

```python
from stable_audio_wanderer.vae.onnx_artifacts import publish_artifact

# prepared_dir already contains complete graphs, parameters, manifest and parity report.
published_dir = publish_artifact(prepared_dir, store_dir=decoder_store)
```

Publication copies declared files to a temporary directory inside the store, verifies hashes/report bindings, parses external references, runs the ONNX checker, and executes every advertised window through ORT with finite PCM/shape checks. Only successful artifacts are renamed into their immutable directory and selected through an atomic pointer replacement. Failures clean staging and preserve the previous selection. Existing identical exports are reusable; older identities remain available for explicit selection.

All external tensor references, including nested graph attributes, must stay within the artifact, be declared/hashed, and use valid file ranges. Traversal, escaping symlinks, absolute paths and undeclared external files are rejected before ORT opens a session. Paths resolve from the graph directory, never the process working directory. Exporters should keep shape-control constants embedded where required by ORT; the external-weight test uses external learned parameters and embedded shape constants.

The `onnx` parser is now a runtime dependency for this validation; `onnxscript` remains an export extra. The offline lock update changed dependency classification without upgrading versions. Shared artifact loading does not import native VAE libraries, Hugging Face or export libraries. Native source files and both existing virtual environments are unchanged.

## Verification

- Full Python suite: **299 passed, 1 skipped**, with three existing WebSocket deprecation warnings.
- All three browser contract suites pass: ONNX/dual-decoder UI, Wander UI and Erae UI, including lifecycle/reconnect checks.
- **32 artifact tests** use real small ONNX graphs: dynamic/fixed windows, all four layout combinations, external parameters, bounded sessions, serialized calls, close, corrupt/missing metadata/files, stale sources, changed selections, unsafe external references, publication failure/replacement/reuse and offline reload without native VAE/export libraries.
- The real packaged SAME-S graph was loaded through the generalized factory against BurntMemory metadata and decoded all 16 windows T2–T32. PCM shape/type/finiteness, CPU provider, thread settings, release and native CPU/MPS selection passed. See [recorded compatibility evidence](multi_vae_phase1_compatibility.json).
- SAME-S graph SHA-256 remains `333dc948f52fa59e9a80a066ffd2ca801d3a54b2fb07c93aa87cac845c547872`. Compatibility preserves the actual legacy decoding implementation; this probe does not claim bitwise equality for its stochastic output or new native/performance qualification.

Reproduce the real compatibility probe from the project root:

```sh
.venv/bin/python -m eval_scripts.audit_multi_vae_phase1 \
  --corpus corpus/BurntMemory_20260915_211501 \
  --output docs/multi_vae_phase1_compatibility.json
```

No Stable Audio Open/EAR graph was produced in this phase. The supplied EAR checkpoints remain available for phase 3; their installation is not evidence of ONNX support.
