#!/usr/bin/env python3
"""Prepare the app-owned SAME-S ONNX decoder for a release build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

try:
    from bin.export_vst_bundle import export_same_s_decoder_onnx
except ModuleNotFoundError:  # Direct execution puts bin/ on sys.path.
    from export_vst_bundle import export_same_s_decoder_onnx


WINDOWS = (2, 4, 8, 16, 32)
SAMPLES_PER_LATENT = 4096


def export_web_decoder(output_dir: Path, repo_or_path: str = "same-s", opset: int = 20):
    """Export, parity-check, and stage the dynamic Web decoder resource."""

    if repo_or_path != "same-s":
        raise ValueError(
            "The Web decoder export requires stable-audio-3's registered model "
            "name 'same-s'. Run without --model, or pass --model same-s; filesystem "
            "paths such as '/path/to/same-s' are not accepted by this loader."
        )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0x5A17)
    latent_samples = {
        window: rng.standard_normal((1, 256, window), dtype=np.float32) * 0.05
        for window in WINDOWS
    }

    with tempfile.TemporaryDirectory(prefix="saw-web-decoder-") as temp_name:
        temp = Path(temp_name)
        (temp / "reports").mkdir(parents=True)
        decoder = export_same_s_decoder_onnx(
            models_dir=temp / "models",
            reports_dir=temp / "reports",
            latent_samples=latent_samples,
            samples_per_latent=SAMPLES_PER_LATENT,
            repo_or_path=repo_or_path,
            opset=opset,
        )
        model_name = "same_s_decoder_dynamic.onnx"
        shutil.copy2(temp / "models" / "decoder" / model_name, output_dir / model_name)
        shutil.copy2(
            temp / "reports" / "decoder_parity.json",
            output_dir / "decoder_parity.json",
        )

    metadata = {
        "format_version": "same_s.web_decoder.v1",
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "vae_id": "same_s",
        "model": model_name,
        "input_name": decoder["input_name"],
        "output_name": decoder["output_name"],
        "sample_rate": 44100,
        "channels": 2,
        "latent_dim": 256,
        "samples_per_latent": SAMPLES_PER_LATENT,
        "supported_windows": list(WINDOWS),
        "default_window": 2,
        "ola_mode": "full_overlap_add",
    }
    (output_dir / "decoder.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="stable_audio_wanderer/resources/same_s",
        help="Release resource staging directory.",
    )
    parser.add_argument(
        "--model", "--repo-or-path",
        dest="repo_or_path",
        default="same-s",
        help="stable-audio-3 registered model name (must be 'same-s').",
    )
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()
    metadata = export_web_decoder(Path(args.output_dir), args.repo_or_path, args.opset)
    print(
        f"Exported {metadata['model']} with windows "
        f"{metadata['supported_windows']} to {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
