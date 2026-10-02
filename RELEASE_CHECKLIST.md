# Public release checklist

This checklist separates the Apache-licensed source repository from the
multi-license conference build that contains a derived SAME-S decoder.

## One-time approval gates

- [ ] All project contributors and applicable institutional rights holders have
  approved Apache-2.0 distribution of the original WÆnderer code and docs.
- [ ] University research/IP requirements have been reviewed.
- [ ] The release qualifies for the intended Stability AI Community License use;
  any required commercial registration or enterprise license is complete.
- [ ] The release and demonstrations comply with the current Stability AI AUP.
- [ ] The ignored JUCE application is not distributed, or has completed a
  separate AGPL/commercial JUCE licensing review.

## Repository gate

```bash
uv lock --check
uv run python bin/check_release_compliance.py
uv run python -m pytest
uv build
```

- [ ] `smalley-spectromorphology.pdf` is absent from the working tree.
- [ ] The same PDF is absent from every Git ref:

  ```bash
  uv run python bin/check_release_compliance.py --check-history
  ```

  If this fails, coordinate with every collaborator, create a recoverable mirror
  backup, then use `git filter-repo` or an equivalent reviewed history-rewrite
  procedure before making the remote public. Re-clone and rerun the check after
  the rewritten refs are pushed. Also remove any affected release attachments.

## Demo-media gate

- [ ] Every public recording, video, image, and source corpus has an `approved`
  entry in `docs/media/RIGHTS.md`.
- [ ] Written permission records are retained in controlled institutional
  storage.
- [ ] The website displays every required media credit and license.

## Model-bearing conference build

Use the reviewed upstream revision rather than a branch name:

```bash
uv sync --locked --all-extras
uv run python bin/export_web_decoder.py \
  --source-revision fbeb3dcf53a326e5682f38e22e7f740202d44232
SAW_RELEASE_BUILD=1 uv build
```

The exporter must download the model configuration and weights through the
specified revision. The release build then verifies the ONNX digest, provenance,
model notice, upstream notices, and full license texts.

- [ ] `decoder.json` names `stabilityai/SAME-S`, contains a full source commit,
  describes the ONNX conversion, and matches the packaged ONNX SHA-256.
- [ ] The application and project website visibly show “Powered by Stability AI.”
- [ ] `NOTICE`, `THIRD_PARTY_NOTICES.md`, the Stability AI Community License,
  the Gemma terms, and the model notice are present in the delivered package.
- [ ] Network-disabled cold start and the performance gates in `README.md` pass
  on the actual conference machine.

## Release evidence

Generate evidence inside the exact locked environment used to build the
artifact:

```bash
uv run python bin/generate_release_evidence.py
uv run python bin/check_release_compliance.py \
  --license-report build/release-evidence/licenses.json
```

- [ ] Archive `sbom.cdx.json`, `licenses.json`, wheel/source checksums, and test
  output with the release record.
- [ ] Review any new, unknown, GPL, or AGPL result before distribution.
- [ ] Inspect the final wheel/source archive and conference bundle—not only the
  source checkout—for the complete license payload.

## Dual inference qualification

Use [DUAL_INFERENCE_RELEASE.md](docs/DUAL_INFERENCE_RELEASE.md) for installation,
engine selection and the supported-platform matrix. Keep native SAME-S in its
separate pinned environment; do not replace the default environment to enable it.

- [x] Full Python regression and the three Web suites pass.
- [x] Default ONNX startup works without native `stable-audio-3`; standalone CLI and playback are checked.
- [x] Native weights/library revisions and exported ONNX identity are recorded.
- [x] Synthetic parity and corpus parity reports include within-engine stochastic repeats.
- [x] At least ten minutes on the actual audio device pass without underruns for every advertised scenario.
- [x] Window/mode changes, adaptive windows, stop/reconfigure/retry and memory release pass.
- [x] Comparative listening of aligned OLA renders is recorded; no new-engine degradation reported (ONNX slightly noisier); raw PCM has numeric parity checks.
- [x] Each advertised GPU has its own hardware report. MPS results never qualify CUDA.
- [x] Unqualified devices and failed scenarios remain explicitly identified in release notes.
- [x] Reports contain no redistributed private corpus or recordings without permission.

The software-clock benchmark is diagnostic evidence, not an audio-device gate.

Completed 2026-10-02 for the recorded ONNX CPU / Apple MPS scenario only: 252 Python tests passed, one optional skip, three Web suites passed. CUDA remains experimental and excluded. See [machine-readable qualification result](docs/dual_inference_qualification_result.json). These checks do not approve the unrelated distribution gates above.
