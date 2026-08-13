# License scope

WÆnderer is a multi-license distribution. A public repository and a packaged
conference build do not have identical contents, so their scopes differ.

| Material | Governing terms |
| --- | --- |
| Original WÆnderer source code and documentation | Apache License 2.0 (`../LICENSE`) |
| SAME-S weights and the derived ONNX decoder | Stability AI Community License (`STABILITY_AI_COMMUNITY_LICENSE.md`) |
| Gemma-derived material carried by SAME-S | Gemma Terms of Use (`GEMMA_TERMS_OF_USE.md`) |
| Vendored p5.js 1.9.0 | GNU LGPL 2.1 (`p5.js-LGPL-2.1.txt`) |
| Third-party Python packages and optional model backends | Their respective upstream licenses, summarized in `../THIRD_PARTY_NOTICES.md` |
| Demo audio, video, images, and source recordings | The per-item rights record in `../docs/media/RIGHTS.md`; not covered by Apache-2.0 unless expressly stated |
| JUCE application or plug-in | Not included in this repository or license grant; requires a separate JUCE licensing review before distribution |

The Apache-2.0 grant does not override any third-party terms. A model-bearing
release must carry `NOTICE`, the two model terms files, and the model-specific
notice stored beside the decoder resource.
