#!/usr/bin/env python3
"""Validate even windows against the pinned Torch model before enabling them.

Run from the repository root with PYTHONPATH=. HF_HUB_OFFLINE=1 to use cached
weights only. The packaged graph has approximate, not sample-exact, Torch parity.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bin.export_vst_bundle import _load_pinned_same_s_autoencoder, _parity_metrics
from stable_audio_wanderer.vae.onnx_decoder import ALLOWED_WINDOWS


def main():
    root = Path(__file__).resolve().parents[1] / "stable_audio_wanderer/resources/same_s"
    metadata = json.loads((root / "decoder.json").read_text())
    model_path = root / metadata["model"]
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise RuntimeError("Packaged model hash does not match metadata")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    model = _load_pinned_same_s_autoencoder(metadata["source_revision"])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    results = []
    for window in ALLOWED_WINDOWS:
        latents = np.random.default_rng(window).normal(0, .05, (1, 256, window)).astype("float32")
        start = time.monotonic()
        audio = session.run(None, {metadata["input_name"]: latents})[0]
        elapsed_ms = (time.monotonic() - start) * 1000
        with torch.inference_mode():
            reference = model.decode_audio(torch.from_numpy(latents), chunked=False).numpy()
        if audio.shape != (1, 2, window * 4096) or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid ONNX output for T{window}")
        metrics = _parity_metrics(reference, audio)
        # The original packaged conversion measures around 24 dB SNR / .002 RMSE.
        # This gate detects a regression, rather than claiming sample-exact parity.
        if metrics["snr_db"] <= 20 or metrics["rmse"] >= .005:
            raise RuntimeError(f"T{window} parity regression: {metrics}")
        results.append(dict(window=window, decode_ms=elapsed_ms, **metrics))
        print(f"T{window}: {elapsed_ms:.1f} ms, SNR {metrics['snr_db']:.2f} dB", flush=True)
    report = {
        "model_sha256": digest,
        "source_revision": metadata["source_revision"],
        "validation": {
            "minimum_snr_db": 20,
            "maximum_rmse": .005,
            "input": "seeded normal, std=0.05",
            "torch_seed": 0,
        },
        "results": results,
    }
    (root / "even_window_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata["supported_windows"] = list(ALLOWED_WINDOWS)
    (root / "decoder.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
