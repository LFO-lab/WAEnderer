"""Offline feasibility probe, not a runtime backend or a real-time benchmark.

Run from the repository root:
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python eval_scripts/audit_dual_inference_phase0.py \
      --device cpu --output docs/dual_inference_phase0_cpu.json
Use --device mps on a host with GPU access. Cached pinned weights are required.
Only the requested report is written; packaged models/reports are never modified.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from stable_audio_3.model import AutoencoderModel

from bin.export_vst_bundle import _load_pinned_same_s_autoencoder


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compare(reference, actual):
    assert reference.shape == actual.shape
    error = reference.astype(np.float64) - actual.astype(np.float64)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    signal = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": rmse,
        "snr_db": float(20 * np.log10(signal / rmse)) if signal and rmse else None,
        "exact_equal": bool(np.array_equal(reference, actual)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback allowed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback allowed")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable MPS CPU fallback for this capability probe")

    root = Path(__file__).resolve().parents[1]
    resources = root / "stable_audio_wanderer/resources/same_s"
    config = json.loads((resources / "decoder.json").read_text())
    model_path = resources / config["model"]
    assert digest(model_path) == config["model_sha256"]
    weights = {}
    for name in ("model_config.json", "model.safetensors"):
        path = hf_hub_download(config["source_model"], name,
                               revision=config["source_revision"], local_files_only=True)
        weights[name] = {"sha256": digest(path), "bytes": Path(path).stat().st_size}

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "device": args.device, "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in
                     ("torch", "torchaudio", "numpy", "onnx", "onnxruntime",
                      "stable-audio-3", "huggingface-hub", "safetensors")},
        "stable_audio_3_install": json.loads(metadata.distribution("stable-audio-3").read_text("direct_url.json") or "null"),
        "model_sha256": digest(model_path), "source_revision": config["source_revision"],
        "cached_sources": weights,
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "protocol": "one seeded normal input std=.05 per even T; one warmup, three timed calls; CPU PCM conversion included; not a real-time qualification",
        "threads": 1, "windows": [],
    }
    graph = onnx.load(str(model_path))
    report["onnx_random_ops_top_level"] = dict(Counter(
        node.op_type for node in graph.graph.node if "Random" in node.op_type))
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    torch.set_num_threads(1)
    model = _load_pinned_same_s_autoencoder(config["source_revision"])
    report["parameter_bytes_by_component"] = {
        name: sum(p.numel() * p.element_size() for p in child.parameters())
        for name, child in model.named_children()
    }
    report["noise_regularize"] = model.bottleneck.noise_regularize
    probe = torch.from_numpy(np.random.default_rng(2).normal(0, .05, (1, 256, 2)).astype("float32"))
    with torch.inference_mode():
        torch.manual_seed(0)
        complete = model.decode_audio(probe, chunked=False).numpy()
        torch.manual_seed(0)
        public = AutoencoderModel(model, 44100, "cpu").decode(probe).numpy()
        report["public_vs_export_path_same_seed_cpu"] = compare(complete, public)
        model.encoder = None
        gc.collect()
        torch.manual_seed(0)
        pruned = model.decode_audio(probe, chunked=False).numpy()
        report["without_encoder_same_seed_cpu"] = compare(complete, pruned)
    model.to(args.device)
    report["resident_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    if args.device == "mps":
        report["mps_allocated_before_decode"] = torch.mps.current_allocated_memory()

    def native(array):
        with torch.inference_mode():
            return model.decode_audio(torch.from_numpy(array).to(args.device), chunked=False).cpu().numpy()

    for window in config["supported_windows"]:
        array = np.random.default_rng(window).normal(0, .05, (1, 256, window)).astype("float32")
        torch.manual_seed(0)
        reference = native(array)
        onnx_audio = session.run(None, {config["input_name"]: array})[0]
        assert reference.shape == onnx_audio.shape == (1, 2, window * 4096)
        assert reference.dtype == onnx_audio.dtype == np.float32
        assert np.isfinite(reference).all() and np.isfinite(onnx_audio).all()
        torch.manual_seed(0)
        reseeded = native(array)
        repeated = native(array)
        onnx_repeated = session.run(None, {config["input_name"]: array})[0]
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            audio = native(array)
            assert np.isfinite(audio).all()
            timings.append((time.perf_counter() - start) * 1000)
        row = {
            "window": window, "shape": list(reference.shape),
            "native_vs_onnx": compare(reference, onnx_audio),
            "native_reseeded": compare(reference, reseeded),
            "native_repeat": compare(reseeded, repeated),
            "onnx_repeat": compare(onnx_audio, onnx_repeated),
            "native_ms_three_calls": timings,
            "audio_hop_ms": (window // 2) * 4096 / 44100 * 1000,
        }
        report["windows"].append(row)
        print(f"{args.device} T{window}: {row['native_vs_onnx']}, native median {np.median(timings):.1f} ms", flush=True)
    if args.device == "mps":
        report["mps_allocated_after_decode"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_after_decode"] = torch.mps.driver_allocated_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
