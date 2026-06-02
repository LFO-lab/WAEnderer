#!/usr/bin/env python3
"""Smoke-test a .sawbundle ONNX decoder model with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest(bundle: Path) -> Dict[str, Any]:
    manifest_path = bundle / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_array(bundle: Path, manifest: Dict[str, Any], key: str) -> np.ndarray:
    entry = manifest.get("arrays", {}).get(key)

    if not entry:
        raise KeyError(f"Manifest is missing arrays.{key}")

    path = bundle / entry["path"]

    if not path.exists():
        raise FileNotFoundError(f"Missing array file for {key}: {path}")

    expected_hash = entry.get("sha256")

    if expected_hash and sha256(path).lower() != str(expected_hash).lower():
        raise RuntimeError(f"Hash mismatch for {key}: {path}")

    return np.load(path, allow_pickle=False)


def choose_decoder_window(manifest: Dict[str, Any], requested: Optional[int]) -> Dict[str, Any]:
    decoder = manifest.get("models", {}).get("decoder", {})

    if decoder.get("backend") != "onnxruntime":
        raise RuntimeError("Bundle does not declare an ONNX Runtime decoder")

    windows = decoder.get("windows", {})

    if not windows:
        raise RuntimeError("Bundle decoder has no fixed-window models")

    if requested is not None:
        entry = windows.get(str(requested))

        if entry is None:
            raise RuntimeError(f"Bundle does not contain decoder window T={requested}")

        return entry

    first_key = sorted(windows, key=lambda item: int(item))[0]
    return windows[first_key]


def run_decoder(bundle: Path, window_entry: Dict[str, Any], start_frame: int) -> np.ndarray:
    import onnxruntime as ort

    manifest = load_manifest(bundle)
    z = np.asarray(load_array(bundle, manifest, "Z_concat"), dtype=np.float32)
    z_mean = np.asarray(load_array(bundle, manifest, "Z_mean"), dtype=np.float32).reshape(1, -1)
    z_std = np.asarray(load_array(bundle, manifest, "Z_std"), dtype=np.float32).reshape(1, -1)

    latent_window = int(window_entry["latent_window"])
    latent_dim = z.shape[1]

    if z_mean.shape[1] != latent_dim or z_std.shape[1] != latent_dim:
        raise RuntimeError("Z_mean/Z_std latent dimensions do not match Z_concat")

    start = max(0, min(start_frame, max(0, z.shape[0] - 1)))
    end = min(z.shape[0], start + latent_window)
    z_window = z[start:end]

    if z_window.shape[0] < latent_window:
        pad = np.repeat(z_window[-1:, :], latent_window - z_window.shape[0], axis=0)
        z_window = np.concatenate([z_window, pad], axis=0)

    latents = np.ascontiguousarray((z_window * z_std + z_mean).T[None, :, :], dtype=np.float32)
    model_path = bundle / window_entry["path"]

    expected_hash = window_entry.get("sha256")

    if expected_hash and sha256(model_path).lower() != str(expected_hash).lower():
        raise RuntimeError(f"Hash mismatch for decoder model: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_name = window_entry.get("output_name", "audio")
    input_name = window_entry.get("input_name", "latents")
    audio = session.run([output_name], {input_name: latents})[0].astype(np.float32)

    if audio.ndim != 3:
        raise RuntimeError(f"Expected decoder output [1, channels, samples], got {audio.shape}")

    return audio[0].T.copy()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Bundle directory containing manifest.json.")
    parser.add_argument("--window", type=int, default=None, help="Decoder latent window size to run.")
    parser.add_argument("--start-frame", type=int, default=0, help="First corpus frame for the latent window.")
    parser.add_argument("--out-wav", type=Path, default=None, help="Optional decoded WAV output path.")
    parser.add_argument("--min-energy", type=float, default=1.0e-9, help="Fail if decoded audio energy is below this.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle
    manifest = load_manifest(bundle)
    window_entry = choose_decoder_window(manifest, args.window)
    audio = run_decoder(bundle, window_entry, args.start_frame)
    energy = float(np.sum(np.square(audio, dtype=np.float64)))

    if energy <= args.min_energy:
        raise RuntimeError(f"Decoded audio is silent or too small: energy={energy}")

    sample_rate = int(manifest.get("vae", {}).get("sample_rate", 44100))

    if args.out_wav is not None:
        write_wav(args.out_wav, audio, sample_rate)

    print(f"OK: bundle={bundle}")
    print(f"OK: window={window_entry['latent_window']}")
    print(f"OK: output_shape={list(audio.shape)}")
    print(f"OK: energy={energy:.9g}")

    if args.out_wav is not None:
        print(f"OK: wav={args.out_wav}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
