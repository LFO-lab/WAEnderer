# SAME-S Web decoder resource

Release builds must inject `same_s_decoder_dynamic.onnx`, `decoder.json`, and
`decoder_parity.json` into this directory with `bin/export_web_decoder.py`.
The generated files are intentionally excluded from Git.

The exporter requires the full Git revision of the upstream
[`stabilityai/SAME-S`](https://huggingface.co/stabilityai/SAME-S) repository.
It downloads the model configuration and weights from that exact revision,
rejects an unpinned/cache-substituted checkpoint, and records the revision, the
upstream license, the conversion description, and the generated ONNX SHA-256 in
`decoder.json`.

`MODEL_NOTICE.md` and `UPSTREAM_NOTICE.txt` are tracked release inputs. The
complete Stability AI Community License and Gemma Terms of Use are stored in the
repository-level `licenses/` directory and are included in built distributions.
The derived ONNX decoder is not covered by WÆnderer's Apache-2.0 license.

The web exporter now validates every even window from T2 to T32. For an existing
packaged graph, `bin/validate_decoder_windows.py` validates the additional sizes
against the pinned Torch checkpoint and updates local metadata without replacing
the graph. `even_window_validation.json` records numerical parity and CPU timing;
these checks do not replace listening or output-device soak tests.
