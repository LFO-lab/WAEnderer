# Third-party notices

This document covers declared dependencies and the notable bundled/transitive
components found in the release audit. The release SBOM and license report are
the authoritative version-specific inventory for each built artifact.

## Model materials

### SAME-S

Source: <https://huggingface.co/stabilityai/SAME-S>

License: Stability AI Community License. The complete agreement is in
`licenses/STABILITY_AI_COMMUNITY_LICENSE.md`. The derived ONNX decoder remains
under those terms and is excluded from WÆnderer's Apache-2.0 grant.

Required notice:

> This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved.

Powered by Stability AI.

The upstream SAME-S repository also provides Gemma terms and a Gemma notice.
They are preserved in `licenses/GEMMA_TERMS_OF_USE.md` and `NOTICE`.

### Stable Audio Open 1.0

Source: <https://huggingface.co/stabilityai/stable-audio-open-1.0>

The model weights are governed by the Stability AI Community License. The
Diffusers/Transformers integration code is separately Apache-2.0.

### EAR VAE and Descript Audio Codec

EAR VAE is made available under Apache-2.0. Descript Audio Codec is MIT.
Their pretrained files are optional and are not stored in this repository.
Because its current protobuf constraint conflicts with the ONNX exporter,
EAR VAE is installed in a separate environment rather than in the release lock.

## Browser component

### p5.js 1.9.0

Source: <https://github.com/processing/p5.js/tree/v1.9.0>

License: GNU Lesser General Public License 2.1. The unmodified minified release
file is stored at `web/vendor/p5/p5.min.js`; its upstream license is stored
beside it and in `licenses/p5.js-LGPL-2.1.txt`.

Vendored SHA-256:
`726ac96626b93f5bcaff83a910b6c60d3a9728f063e0eb73b5d0819ffc356915`

## Python dependencies

| Component | License family | Redistribution note |
| --- | --- | --- |
| PyTorch | BSD/composite permissive | Retain upstream notices for bundled binaries. |
| torchaudio | BSD-2-Clause | Retain notice. |
| NumPy, SciPy, scikit-learn, UMAP | BSD-3-Clause/composite permissive | Wheels may carry additional BLAS/runtime notices. |
| FAISS / faiss-cpu | MIT; wheels may include BSD OpenBLAS | Retain notices. |
| SoundFile | BSD-3-Clause wrapper | Wheels may bundle LGPL-2.1-or-later libsndfile. |
| SoundDevice | MIT | Retain notice. |
| python-osc | Public-domain style | Preserve upstream attribution where provided. |
| websockets | BSD-3-Clause | Retain notice. |
| tqdm | MIT and MPL-2.0-covered files | Modified MPL-covered files remain MPL-2.0. |
| diffusers, transformers, accelerate, safetensors, huggingface-hub | Apache-2.0 | Retain license and NOTICE material. Model weights have separate terms. |
| ONNX | Apache-2.0 | Retain license and NOTICE material. |
| ONNX Runtime, ONNX Script | MIT | Retain notices. |
| stable-audio-3, einops, einops-exts | MIT | Applies to code, not model weights. |
| descript-audio-codec | MIT | Optional EAR VAE dependency. |
| NVIDIA CUDA Toolkit packages (Linux, transitive through PyTorch) | [NVIDIA Software License Agreement and CUDA Supplement](https://docs.nvidia.com/cuda/eula/index.html) | The `cuda-toolkit==13.0.3.0` metapackage omits license metadata and is covered by the reviewed override. CUDA is not bundled in WÆnderer's wheel; review NVIDIA's redistribution terms before shipping an environment or container containing its libraries. |
| certifi (transitive) | MPL-2.0 | Preserve MPL-covered files and notice if bundled. |
| soxr (transitive) | LGPL-2.1-or-later | Preserve LGPL rights and notices if bundled. |

Most remaining audited transitive distributions use MIT, BSD, Apache, ISC,
PSF, Zlib, or CC0-style terms. Dependency ranges can resolve differently over
time and by platform; never reuse this document as a substitute for generating
the release SBOM and license report.

## Fonts

The project website currently requests DM Mono and Manrope from Google Fonts.
Both font families use the SIL Open Font License 1.1. No font files are stored
in this repository. If the fonts are self-hosted later, include the OFL text
and applicable reserved-font-name notices with them.
