"""Offline phase-2 native/ONNX comparison, separate from the Web pipeline.

Run with PYTHONPATH=. and --device cpu|mps|cuda:0 --output report.json.
An optional --audio-dir writes aligned listening samples from a synthetic
harmonic/percussive source, encoded by the pinned original SAME-S model.
This is not a sustained real-time/underrun benchmark or a listening verdict.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

from stable_audio_wanderer.runtime.overlap_add import StreamingFullOverlapAdd
from stable_audio_wanderer.vae.onnx_decoder import load_same_s_app_decoder
from stable_audio_wanderer.vae.torch_decoder import SameSTorchDecoder
from stable_audio_wanderer.vae.same_s_weights import resolve_same_s_weights


def compare(reference, actual):
    delta = reference.astype(np.float64) - actual.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return {"rmse": rmse, "max_abs_error": float(np.max(np.abs(delta))),
            "snr_db": float(20*np.log10(rms/rmse)) if rmse and rms else None}


def listening_samples(native, onnx, destination):
    import soundfile as sf
    from safetensors.torch import load_file
    from stable_audio_3.factory import create_autoencoder_from_config

    files = resolve_same_s_weights()
    config = json.loads(files.config_path.read_text())
    model = create_autoencoder_from_config(config["model"], config["sample_rate"])
    model.load_state_dict(load_file(str(files.model_path), device="cpu"), strict=True)
    model.eval().requires_grad_(False)
    sr = 44100
    # Exactly 64 latent frames; generated source contains no external audio.
    t = np.arange(64*4096, dtype=np.float32) / sr
    envelope = np.exp(-4*(t % .5))
    left = .18 * envelope * (np.sin(2*np.pi*220*t) + .3*np.sin(2*np.pi*440*t))
    right = .18 * envelope * (np.sin(2*np.pi*277.18*t) + .3*np.sin(2*np.pi*554.36*t))
    source = np.stack((left, right), axis=0).astype(np.float32)
    with torch.inference_mode():
        raw = model.encode_audio(torch.from_numpy(source[None]), chunked=False)[0].T.numpy().copy()
    del model
    destination.mkdir(parents=True, exist_ok=True)
    sf.write(destination / "source.wav", source.T, sr, subtype="FLOAT")
    outputs = {}
    for name, decoder in (("onnx", onnx), ("native", native)):
        ola = StreamingFullOverlapAdd(4*4096, channels=2)
        hops = []
        for start in range(0, len(raw)-8+1, 4):
            decoded = decoder.decode(raw[start:start+8])
            hops.append(ola.push(decoded.audio.T).T)
        audio = np.concatenate(hops)
        if not np.isfinite(audio).all():
            raise RuntimeError("Non-finite listening sample")
        sf.write(destination / f"{name}.wav", audio, sr, subtype="FLOAT")
        outputs[name] = audio
    # No normalization or clipping: level differences remain audible/measurable.
    return {"source": "synthetic stereo decaying harmonics; SAME-S encoded",
            "window": 8, "hop": 4, "samples": len(outputs["native"]),
            "comparison": compare(outputs["onnx"], outputs["native"]),
            "listening_review": "pending human review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--audio-dir", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    torch.set_num_threads(1)
    started = time.perf_counter()
    native = SameSTorchDecoder(device=args.device)
    startup = (time.perf_counter()-started)*1000
    onnx = load_same_s_app_decoder()
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in
                     ("torch", "torchaudio", "stable-audio-3", "onnxruntime", "numpy", "safetensors")},
        "device": native.info.device, "source_revision": native.info.source_revision,
        "library_revision": native.info.library_revision,
        "weights_sha256": native.info.model_sha256, "config_sha256": native.info.config_sha256,
        "parameter_bytes": native.info.parameter_bytes, "startup_ms": startup,
        "warmup_decode_ms": dict(native.info.warmup_decode_ms),
        "protocol": "normal std=.05, NumPy seed=T*1000+seed; sequential calls; 1 Torch thread; packaged ORT runtime defaults",
        "threshold": {"snr_db_greater_than": 20, "rmse_less_than": .005},
        "windows": [],
    }
    for window in native.supported_windows:
        rows = []
        for seed in range(args.seeds):
            raw = np.random.default_rng(window*1000+seed).normal(0,.05,(window,256)).astype(np.float32)
            decoded = native.decode(raw)
            baseline = onnx.decode(raw)
            metrics = compare(baseline.audio, decoded.audio)
            passed = metrics["snr_db"] is not None and metrics["snr_db"] > 20 and metrics["rmse"] < .005
            rows.append({"seed": seed, **metrics, "passed": passed,
                         "native_ms": decoded.decode_time_ms, "onnx_ms": baseline.decode_time_ms})
        repeat = compare(decoded.audio, native.decode(raw).audio)
        report["windows"].append({"window": window, "comparisons": rows, "native_repeat": repeat})
        print(f"{args.device} T{window}: {sum(r['passed'] for r in rows)}/{len(rows)} parity checks", flush=True)
    # Actual per-instance concurrency path, beyond the simulated unit test.
    raw = np.zeros((2,256),dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(native.decode, [raw, raw]))
    report["concurrent_calls_valid"] = all(r.audio.shape == (8192,2) and np.isfinite(r.audio).all() for r in results)
    if args.audio_dir:
        report["audio"] = listening_samples(native, onnx, args.audio_dir)
    report["passed"] = report["concurrent_calls_valid"] and all(r["passed"] for w in report["windows"] for r in w["comparisons"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("Native decoder comparison failed; see report")


if __name__ == "__main__":
    main()
