# SAME-S decoder notice

This Stability AI Model is licensed under the Stability AI Community License,
Copyright © Stability AI Ltd. All Rights Reserved.

**Powered by Stability AI.**

Upstream model: `stabilityai/SAME-S`

Source: <https://huggingface.co/stabilityai/SAME-S>

Governing terms:

- `licenses/STABILITY_AI_COMMUNITY_LICENSE.md`
- `licenses/GEMMA_TERMS_OF_USE.md`
- Stability AI Acceptable Use Policy: <https://stability.ai/use-policy>

## Modifications

WÆnderer exports the SAME-S decoder to a dynamic-time ONNX graph, packages the
decoder separately from the upstream checkpoint, and adds application-specific
metadata, CPU runtime validation, and Torch/ONNX parity evidence. The precise
upstream revision and SHA-256 digest of the generated ONNX file are recorded in
the adjacent `decoder.json` generated for each release.

The ONNX decoder is a derived model artifact. It is not licensed under the
Apache License 2.0 that covers WÆnderer's original source code.
