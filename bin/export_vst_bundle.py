#!/usr/bin/env python3
"""Export a minimal Stable Audio Wanderer bundle for the JUCE MVP runtime.

This is intentionally narrower than the full TOUCH.md design. It exports the
arrays the current C++ MVP can consume: normalized latents and manual embedding
points. With --export-decoder-onnx, it also exports decoder ONNX artifacts and a
CPU parity report for the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


FLOAT_ARRAYS = {
    "Z_concat": "Z_concat.f32.npy",
    "Z_mean": "Z_mean.f32.npy",
    "Z_std": "Z_std.f32.npy",
    "manual_embed_points": "manual_embed_points.f32.npy",
    "manual_fader_p01": "manual_fader_p01.f32.npy",
    "manual_fader_p99": "manual_fader_p99.f32.npy",
    "manual_desc_weighted": "manual_desc_weighted.f32.npy",
    "manual_pca_components": "manual_pca_components.f32.npy",
    "manual_pca_mean": "manual_pca_mean.f32.npy",
    "geom_embeddings_l2": "geom_embeddings_l2.f32.npy",
    "geom_ctx_pca_components": "geom_ctx_pca_components.f32.npy",
    "geom_ctx_pca_mean": "geom_ctx_pca_mean.f32.npy",
    "geom_local_sigma": "geom_local_sigma.f32.npy",
    "geom_time_gradients": "geom_time_gradients.f32.npy",
    "unit_graph_scores": "unit_graph_scores.f32.npy",
    "unit_entry_desc": "unit_entry_desc.f32.npy",
    "unit_exit_desc": "unit_exit_desc.f32.npy",
    "unit_delta_desc": "unit_delta_desc.f32.npy",
}

INT_ARRAYS = {
    "file_offsets": "file_offsets.i64.npy",
    "frame_file_ids": "frame_file_ids.i32.npy",
    "frame_t": "frame_t.i32.npy",
    "unit_start_idx": "unit_start_idx.i32.npy",
    "unit_end_idx": "unit_end_idx.i32.npy",
    "unit_file_id": "unit_file_id.i32.npy",
    "unit_start_t": "unit_start_t.i32.npy",
    "unit_end_t": "unit_end_t.i32.npy",
    "frame_to_unit": "frame_to_unit.i32.npy",
    "unit_graph_neighbors": "unit_graph_neighbors.i32.npy",
}

DEFAULT_DECODER_WINDOWS = (2, 4, 8, 16, 32)
STABLE_AUDIO_OPEN_REPO = "stabilityai/stable-audio-open-1.0"
SAME_S_REPO = "same-s"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default

    value = np.asarray(data[key]).reshape(-1)

    if value.size == 0:
        return default

    item = value[0]

    if hasattr(item, "item"):
        item = item.item()

    if isinstance(item, bytes):
        return item.decode("utf-8")

    return item


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def _write_array(path: Path, array: np.ndarray, dtype: np.dtype) -> Dict[str, Any]:
    typed = np.ascontiguousarray(array, dtype=dtype)
    np.save(path, typed)

    return {
        "path": f"{path.parent.name}/{path.name}",
        "dtype": str(typed.dtype),
        "shape": list(typed.shape),
        "sha256": _sha256(path),
    }


def _copy_arrays(
    arrays_dir: Path,
    corpus: np.lib.npyio.NpzFile,
    manual: Optional[np.lib.npyio.NpzFile],
) -> Dict[str, Dict[str, Any]]:
    manifest_arrays: Dict[str, Dict[str, Any]] = {}

    for key, filename in FLOAT_ARRAYS.items():
        source = corpus[key] if key in corpus else None

        if source is None and manual is not None and key in manual:
            source = manual[key]

        if source is not None:
            manifest_arrays[key] = _write_array(arrays_dir / filename, source, np.float32)

    for key, filename in INT_ARRAYS.items():
        if key not in corpus:
            continue

        dtype = np.int64 if filename.endswith(".i64.npy") else np.int32
        manifest_arrays[key] = _write_array(arrays_dir / filename, corpus[key], dtype)

    if "Z_concat" not in manifest_arrays:
        raise RuntimeError("corpus.npz is missing required array Z_concat")

    if "manual_embed_points" not in manifest_arrays:
        raise RuntimeError(
            "manual_embed_points is missing. Re-run preprocess.py or provide manual_navigation.npz."
        )

    return manifest_arrays


def _parse_decoder_windows(value: str) -> Sequence[int]:
    windows = []

    for raw_item in value.split(","):
        item = raw_item.strip()

        if not item:
            continue

        window = int(item)

        if window <= 0:
            raise ValueError("Decoder windows must be positive integers")

        windows.append(window)

    if not windows:
        raise ValueError("At least one decoder window is required")

    return tuple(dict.fromkeys(windows))


def _decoder_samples_from_corpus(
    corpus: np.lib.npyio.NpzFile,
    windows: Sequence[int],
) -> Dict[int, np.ndarray]:
    z_norm = np.asarray(corpus["Z_concat"], dtype=np.float32)
    z_mean = np.asarray(corpus["Z_mean"], dtype=np.float32).reshape(1, -1)
    z_std = np.asarray(corpus["Z_std"], dtype=np.float32).reshape(1, -1)

    if z_norm.ndim != 2:
        raise RuntimeError(f"Z_concat must be 2D for decoder export, got shape {z_norm.shape}")

    if z_mean.shape[1] != z_norm.shape[1] or z_std.shape[1] != z_norm.shape[1]:
        raise RuntimeError(
            "Z_mean and Z_std must match Z_concat latent dimension for decoder export"
        )

    samples: Dict[int, np.ndarray] = {}

    for window in windows:
        if z_norm.shape[0] >= window:
            z_window = z_norm[:window]
        else:
            pad = np.repeat(z_norm[-1:, :], window - z_norm.shape[0], axis=0)
            z_window = np.concatenate([z_norm, pad], axis=0)

        z_raw = z_window * z_std + z_mean
        samples[window] = np.ascontiguousarray(z_raw.T[None, :, :], dtype=np.float32)

    return samples


def _decoder_timing_entry(window: int, samples_per_latent: int) -> Dict[str, Any]:
    latent_hop = max(1, (int(window) + 1) // 2)
    return {
        "samples_per_latent": int(samples_per_latent),
        "latent_hop": int(latent_hop),
        "audio_hop_samples": int(latent_hop * int(samples_per_latent)),
        "ola_mode": "full_overlap_add",
    }


def _snr_db(reference: np.ndarray, actual: np.ndarray) -> float:
    error = reference - actual
    signal_rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
    error_rms = float(np.sqrt(np.mean(np.square(error), dtype=np.float64)))

    if error_rms == 0.0:
        return math.inf

    return 20.0 * math.log10(signal_rms / error_rms) if signal_rms > 0.0 else -math.inf


def _parity_metrics(reference: np.ndarray, actual: np.ndarray) -> Dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "shape_match": False,
            "torch_shape": list(reference.shape),
            "onnx_shape": list(actual.shape),
        }

    error = reference - actual
    return {
        "shape_match": True,
        "torch_shape": list(reference.shape),
        "onnx_shape": list(actual.shape),
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "snr_db": _snr_db(reference, actual),
    }


def export_stable_audio_open_decoder_onnx(
    *,
    models_dir: Path,
    reports_dir: Path,
    latent_samples: Mapping[int, np.ndarray],
    samples_per_latent: int,
    repo_or_path: str,
    opset: int,
    source_revision: str = "f21265c1e2710b3bd2386596943f0007f55f802e",
) -> Dict[str, Any]:
    from stable_audio_wanderer.vae.stable_audio_open_export import load_wrapper, export_graph, validate_graph, TOLERANCES
    # Retain the legacy argument for callers; measured geometry overrides it.
    wrapper, source, samples_per_latent = load_wrapper(repo_or_path, source_revision)
    decoder_dir = models_dir / "decoder"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    windows: Dict[str, Any] = {}
    parity: Dict[str, Any] = {
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "repo_or_path": repo_or_path,
        "opset": opset,
        "source": source,
        "passed": True,
        "tolerances": TOLERANCES,
        "windows": {},
    }

    for window, latents_np in latent_samples.items():
        onnx_path = decoder_dir / f"stable_audio_open_decoder_T{window}.onnx"
        export_graph(wrapper, onnx_path, latents_np, dynamic=False, opset=opset)
        metrics = validate_graph(wrapper, onnx_path, {window: [latents_np]})[str(window)][0]
        output_shape = [1, 2, int(window) * samples_per_latent]

        entry = {
            "path": f"models/decoder/{onnx_path.name}",
            "sha256": _sha256(onnx_path),
            "latent_window": int(window),
            "input_name": "latents",
            "output_name": "audio",
            "input_shape": list(latents_np.shape),
            "output_shape": output_shape,
            "output_samples": output_shape[-1],
            "opset": opset,
            "source": source,
            **_decoder_timing_entry(window, samples_per_latent),
        }
        windows[str(window)] = entry
        parity["windows"][str(window)] = {
            **entry,
            **metrics,
            "shape_match": True,
            "torch_shape": output_shape,
            "onnx_shape": output_shape,
        }

    parity_path = reports_dir / "decoder_parity.json"
    parity_path.write_text(json.dumps(parity, indent=2, default=_jsonable) + "\n", encoding="utf-8")

    return {
        "backend": "onnxruntime",
        "provider_baseline": "CPUExecutionProvider",
        "vae_id": "stable_audio_open",
        "repo_or_path": repo_or_path,
        "opset": opset,
        "source": source,
        "input_name": "latents",
        "output_name": "audio",
        "parity_report": "reports/decoder_parity.json",
        "parity_report_sha256": _sha256(parity_path),
        "windows": windows,
    }


def _load_pinned_same_s_autoencoder(source_revision: str):
    """Load only the explicitly identified SAME-S checkpoint revision."""

    source_revision = str(source_revision).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("SAME-S source_revision must be a full 40-character Git commit SHA")

    try:
        from huggingface_hub import hf_hub_download
        from stable_audio_3.loading_utils import load_autoencoder
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Pinned SAME-S export requires huggingface_hub and stable_audio_3"
        ) from exc

    pinned_files = {
        filename: Path(
            hf_hub_download(
                repo_id="stabilityai/SAME-S",
                filename=filename,
                revision=source_revision,
            )
        )
        for filename in ("model_config.json", "model.safetensors")
    }
    for filename, path in pinned_files.items():
        if not path.is_file():
            raise RuntimeError(f"Pinned SAME-S {filename} is missing: {path}")
        if source_revision not in path.parts:
            raise RuntimeError(
                f"Pinned SAME-S {filename} did not resolve through revision {source_revision}"
            )
    autoencoder = load_autoencoder(
        pinned_files["model_config.json"],
        pinned_files["model.safetensors"],
        device="cpu",
    )
    return autoencoder.eval().requires_grad_(False)


def export_same_s_decoder_onnx(
    *,
    models_dir: Path,
    reports_dir: Path,
    latent_samples: Mapping[int, np.ndarray],
    samples_per_latent: int,
    repo_or_path: str,
    opset: int,
    source_revision: str,
) -> Dict[str, Any]:
    if repo_or_path != SAME_S_REPO:
        raise ValueError(
            "Pinned SAME-S export requires stable-audio-3's registered model name "
            f"{SAME_S_REPO!r}; alternate repositories and local paths are not accepted"
        )
    source_revision = str(source_revision).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("SAME-S source_revision must be a full 40-character Git commit SHA")

    try:
        import onnx
        import onnxruntime as ort
        import torch
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "SAME-S decoder ONNX export requires torch, stable_audio_3, onnx, "
            "onnxscript, and onnxruntime"
        ) from exc

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, latents):  # noqa: ANN001 - torch export signature
            return self.autoencoder.decode_audio(latents, chunked=False)

    decoder_dir = models_dir / "decoder"
    decoder_dir.mkdir(parents=True, exist_ok=True)

    autoencoder = _load_pinned_same_s_autoencoder(source_revision)
    wrapper = DecoderWrapper(autoencoder).eval()
    validated_windows = tuple(sorted(int(window) for window in latent_samples.keys()))

    if not validated_windows:
        raise RuntimeError("At least one latent window is required for SAME-S ONNX export")

    if any(window < 2 for window in validated_windows):
        raise RuntimeError("SAME-S dynamic ONNX export requires latent windows >= 2")

    if any(window % 2 != 0 for window in validated_windows):
        raise RuntimeError(
            "SAME-S dynamic ONNX export currently requires even latent windows. "
            "ONNX Runtime fails SAME-S decoder reshapes for odd T values."
        )

    export_window = 8 if 8 in latent_samples else validated_windows[min(len(validated_windows) // 2, len(validated_windows) - 1)]
    onnx_path = decoder_dir / "same_s_decoder_dynamic.onnx"
    dummy = torch.from_numpy(latent_samples[export_window]).to("cpu")

    windows: Dict[str, Any] = {}
    parity: Dict[str, Any] = {
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "repo_or_path": repo_or_path,
        "source_model": "stabilityai/SAME-S",
        "source_revision": source_revision,
        "opset": opset,
        "dynamic_latent_window": True,
        "validated_latent_windows": list(validated_windows),
        "min_latent_window": int(min(validated_windows)),
        "max_latent_window": int(max(validated_windows)),
        "windows": {},
    }

    latent_time_dim = torch.export.Dim(
        "latent_time",
        min=int(min(validated_windows)),
        max=int(max(validated_windows)),
    )
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["latents"],
        output_names=["audio"],
        opset_version=opset,
        dynamo=True,
        dynamic_shapes={"latents": {2: latent_time_dim}},
        external_data=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for window in validated_windows:
        latents_np = latent_samples[window]
        torch_latents = torch.from_numpy(latents_np).to("cpu")

        with torch.inference_mode():
            torch_audio = wrapper(torch_latents).detach().cpu().numpy().astype(np.float32)

        onnx_audio = session.run(["audio"], {"latents": latents_np})[0].astype(np.float32)
        metrics = _parity_metrics(torch_audio, onnx_audio)

        entry = {
            "path": f"models/decoder/{onnx_path.name}",
            "sha256": _sha256(onnx_path),
            "latent_window": int(window),
            "dynamic_latent_window": True,
            "input_name": "latents",
            "output_name": "audio",
            "input_shape": list(latents_np.shape),
            "output_shape": list(torch_audio.shape),
            "output_samples": int(torch_audio.shape[-1]) if torch_audio.ndim >= 3 else None,
            "opset": opset,
            **_decoder_timing_entry(window, samples_per_latent),
        }
        windows[str(window)] = entry
        parity["windows"][str(window)] = {
            **entry,
            **metrics,
        }

    parity_path = reports_dir / "decoder_parity.json"
    parity_path.write_text(json.dumps(parity, indent=2, default=_jsonable) + "\n", encoding="utf-8")

    return {
        "backend": "onnxruntime",
        "provider_baseline": "CPUExecutionProvider",
        "vae_id": "same_s",
        "repo_or_path": repo_or_path,
        "opset": opset,
        "input_name": "latents",
        "output_name": "audio",
        "path": f"models/decoder/{onnx_path.name}",
        "sha256": _sha256(onnx_path),
        "dynamic_latent_window": True,
        "validated_latent_windows": list(validated_windows),
        "min_latent_window": int(min(validated_windows)),
        "max_latent_window": int(max(validated_windows)),
        "samples_per_latent": int(samples_per_latent),
        "ola_mode": "full_overlap_add",
        "parity_report": "reports/decoder_parity.json",
        "parity_report_sha256": _sha256(parity_path),
        "windows": windows,
    }


def _validate_manifest_file(
    *,
    bundle_dir: Path,
    label: str,
    rel_path: Optional[str],
    expected_sha256: Optional[str],
    errors: list[str],
) -> Optional[Path]:
    if not rel_path:
        errors.append(f"{label}: missing path")
        return None

    file_path = bundle_dir / rel_path

    if not file_path.exists():
        errors.append(f"{label}: missing file {rel_path}")
        return None

    if expected_sha256 and _sha256(file_path).lower() != str(expected_sha256).lower():
        errors.append(f"{label}: sha256 mismatch")

    return file_path


def validate_bundle(bundle_dir: Path) -> Dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"

    if not manifest_path.exists():
        return {"ok": False, "errors": [f"Missing manifest: {manifest_path}"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    for key, entry in manifest.get("arrays", {}).items():
        array_path = _validate_manifest_file(
            bundle_dir=bundle_dir,
            label=key,
            rel_path=entry.get("path"),
            expected_sha256=entry.get("sha256"),
            errors=errors,
        )

        if array_path is None:
            continue

        try:
            array = np.load(array_path, allow_pickle=False)
        except Exception as exc:  # pragma: no cover - defensive CLI error path
            errors.append(f"{key}: could not reload array: {exc}")
            continue

        expected_dtype = entry.get("dtype")

        if expected_dtype and str(array.dtype) != str(expected_dtype):
            errors.append(f"{key}: dtype {array.dtype} != {expected_dtype}")

        expected_shape = entry.get("shape")

        if expected_shape and list(array.shape) != list(expected_shape):
            errors.append(f"{key}: shape {list(array.shape)} != {expected_shape}")

    decoder = manifest.get("models", {}).get("decoder", {})

    if isinstance(decoder, dict):
        if decoder.get("backend") == "onnxruntime" and decoder.get("dynamic_latent_window"):
            _validate_manifest_file(
                bundle_dir=bundle_dir,
                label="dynamic decoder",
                rel_path=decoder.get("path"),
                expected_sha256=decoder.get("sha256"),
                errors=errors,
            )

            if int(decoder.get("samples_per_latent", 0) or 0) <= 0:
                errors.append("dynamic decoder: missing positive samples_per_latent")

            if decoder.get("ola_mode") != "full_overlap_add":
                errors.append("dynamic decoder: ola_mode must be full_overlap_add")

        for window, entry in decoder.get("windows", {}).items():
            _validate_manifest_file(
                bundle_dir=bundle_dir,
                label=f"decoder window {window}",
                rel_path=entry.get("path"),
                expected_sha256=entry.get("sha256"),
                errors=errors,
            )

            if decoder.get("backend") == "onnxruntime":
                for key in ("samples_per_latent", "latent_hop", "audio_hop_samples"):
                    if int(entry.get(key, 0) or 0) <= 0:
                        errors.append(f"decoder window {window}: missing positive {key}")

                if entry.get("ola_mode") != "full_overlap_add":
                    errors.append(f"decoder window {window}: ola_mode must be full_overlap_add")

        if decoder.get("parity_report"):
            _validate_manifest_file(
                bundle_dir=bundle_dir,
                label="decoder parity report",
                rel_path=decoder.get("parity_report"),
                expected_sha256=decoder.get("parity_report_sha256"),
                errors=errors,
            )

    return {
        "ok": not errors,
        "errors": errors,
        "array_count": len(manifest.get("arrays", {})),
    }


def _resolve_corpus_npz(corpus_arg: Path) -> Path:
    if corpus_arg.is_dir():
        return corpus_arg / "corpus.npz"

    return corpus_arg


def _resolve_manual_npz(corpus_npz: Path, manual_arg: Optional[Path]) -> Optional[Path]:
    if manual_arg is not None:
        return manual_arg

    candidate = corpus_npz.parent / "manual_navigation.npz"
    return candidate if candidate.exists() else None


def _build_manifest(
    *,
    name: str,
    out_dir: Path,
    corpus_npz: Path,
    manual_npz: Optional[Path],
    corpus: np.lib.npyio.NpzFile,
    arrays: Dict[str, Dict[str, Any]],
    decoder: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    z_shape = list(np.asarray(corpus["Z_concat"]).shape)
    manual_shape = arrays["manual_embed_points"]["shape"]

    paths = corpus["paths"].tolist() if "paths" in corpus else []
    paths = [str(p) for p in paths]

    sample_rate = int(_scalar(corpus, "sr", _scalar(corpus, "sample_rate", 44100)))
    latent_hz = float(_scalar(corpus, "latent_hz", 0.0))
    vae_id = str(_scalar(corpus, "vae_id", "unknown"))
    decoder_descriptor = decoder or {
        "backend": "procedural_mvp",
        "note": "Current JUCE MVP sonifies latents procedurally; ONNX decoder export is not included yet.",
    }

    return {
        "bundle_format_version": "sawbundle.v0.mvp",
        "minimum_runtime_version": "0.1.0",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "corpus_npz": str(corpus_npz),
            "manual_npz": str(manual_npz) if manual_npz is not None else None,
        },
        "vae": {
            "vae_id": vae_id,
            "sample_rate": sample_rate,
            "latent_hz": latent_hz,
            "latent_dim": int(z_shape[1]) if len(z_shape) > 1 else 1,
            "channels": int(_scalar(corpus, "channels", 2)),
            "decoder": decoder_descriptor,
        },
        "corpus": {
            "frame_count": int(z_shape[0]) if z_shape else 0,
            "file_count": len(paths),
            "paths": paths,
            "manual_embed_dim": int(manual_shape[1]) if len(manual_shape) > 1 else 1,
            "normalized_latents": True,
        },
        "models": {"decoder": decoder_descriptor} if decoder is not None else {},
        "arrays": arrays,
    }


def export_bundle(
    corpus_arg: Path,
    out_dir: Path,
    *,
    manual_arg: Optional[Path],
    name: Optional[str],
    overwrite: bool,
    verify: bool,
    export_decoder_onnx: bool,
    decoder_windows: Sequence[int],
    decoder_repo: Optional[str],
    decoder_revision: Optional[str],
    decoder_opset: int,
) -> Path:
    corpus_npz = _resolve_corpus_npz(corpus_arg)

    if not corpus_npz.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_npz}")

    manual_npz = _resolve_manual_npz(corpus_npz, manual_arg)

    if manual_arg is not None and not manual_arg.exists():
        raise FileNotFoundError(f"Missing manual artifact: {manual_arg}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_dir} (pass --overwrite to replace it)")

        shutil.rmtree(out_dir)

    arrays_dir = out_dir / "arrays"
    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    arrays_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    decoder_manifest: Optional[Dict[str, Any]] = None

    with np.load(corpus_npz, allow_pickle=False) as corpus:
        if export_decoder_onnx:
            vae_id = str(_scalar(corpus, "vae_id", "unknown"))
            sample_rate = int(_scalar(corpus, "sr", _scalar(corpus, "sample_rate", 44100)))
            latent_hz = float(_scalar(corpus, "latent_hz", 0.0))

            if latent_hz <= 0.0:
                raise RuntimeError("corpus.npz must include a positive latent_hz for ONNX OLA export")

            # Stable Audio Open timing is measured by its shared exporter.
            samples_per_latent = (max(1, int(round(float(sample_rate) / latent_hz)))
                                  if vae_id == "same_s" else None)
            latent_samples = _decoder_samples_from_corpus(corpus, decoder_windows)

            if vae_id == "stable_audio_open":
                decoder_manifest = export_stable_audio_open_decoder_onnx(
                    models_dir=models_dir,
                    reports_dir=reports_dir,
                    latent_samples=latent_samples,
                    samples_per_latent=samples_per_latent,
                    repo_or_path=decoder_repo or STABLE_AUDIO_OPEN_REPO,
                    source_revision=decoder_revision or "f21265c1e2710b3bd2386596943f0007f55f802e",
                    opset=decoder_opset,
                )
            elif vae_id == "same_s":
                if decoder_revision is None:
                    raise RuntimeError(
                        "SAME-S decoder export requires --decoder-revision with a full "
                        "Hugging Face commit SHA"
                    )
                decoder_manifest = export_same_s_decoder_onnx(
                    models_dir=models_dir,
                    reports_dir=reports_dir,
                    latent_samples=latent_samples,
                    samples_per_latent=samples_per_latent,
                    repo_or_path=decoder_repo or SAME_S_REPO,
                    opset=decoder_opset,
                    source_revision=decoder_revision,
                )
            else:
                raise RuntimeError(
                    "Decoder ONNX export currently supports stable_audio_open "
                    f"and same_s only, got {vae_id!r}"
                )

        with np.load(manual_npz, allow_pickle=False) if manual_npz is not None else _null_npz() as manual:
            arrays = _copy_arrays(arrays_dir, corpus, manual)
            manifest = _build_manifest(
                name=name or out_dir.stem,
                out_dir=out_dir,
                corpus_npz=corpus_npz,
                manual_npz=manual_npz,
                corpus=corpus,
                arrays=arrays,
                decoder=decoder_manifest,
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_jsonable) + "\n", encoding="utf-8")

    validation = validate_bundle(out_dir) if verify else {"ok": None, "errors": [], "array_count": len(arrays)}

    if validation["ok"] is False:
        raise RuntimeError("Bundle verification failed: " + "; ".join(validation["errors"]))

    limitations = ["manual_mode_only"]

    if decoder_manifest is None:
        limitations.extend(["procedural_mvp_decoder", "no_onnx_decoder_export"])
    else:
        limitations.append("onnx_plugin_runtime_requires_opt_in_build_and_ort_packaging")

    report = {
        "status": "ok",
        "bundle": str(out_dir),
        "manifest": str(manifest_path),
        "array_count": len(arrays),
        "verified": bool(validation["ok"]) if validation["ok"] is not None else False,
        "validation_errors": validation["errors"],
        "decoder": decoder_manifest,
        "limitations": limitations,
    }
    (reports_dir / "export_report.json").write_text(
        json.dumps(report, indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )

    return out_dir


class _null_npz:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Iterable[Any]) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Corpus directory or corpus.npz path.")
    parser.add_argument("--out", type=Path, required=True, help="Output .sawbundle directory.")
    parser.add_argument("--manual", type=Path, default=None, help="Optional manual_navigation.npz path.")
    parser.add_argument("--name", default=None, help="Display name written to manifest.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output bundle.")
    parser.add_argument("--verify", action="store_true", help="Reload the bundle and validate hashes, dtypes, and shapes.")
    parser.add_argument(
        "--export-decoder-onnx",
        action="store_true",
        help="Export decoder ONNX model(s) and CPU parity report for supported VAEs.",
    )
    parser.add_argument(
        "--decoder-revision",
        default=None,
        help=(
            "Immutable Hugging Face revision (Stable Audio Open defaults to its verified pin). Required "
            "when the corpus vae_id is same_s."
        ),
    )
    parser.add_argument(
        "--decoder-windows",
        default=",".join(str(w) for w in DEFAULT_DECODER_WINDOWS),
        help="Comma-separated latent window sizes for decoder ONNX export.",
    )
    parser.add_argument(
        "--decoder-repo",
        default=None,
        help=(
            "Model repo or local model path for decoder ONNX export. Defaults to "
            f"{STABLE_AUDIO_OPEN_REPO!r} for stable_audio_open and {SAME_S_REPO!r} for same_s."
        ),
    )
    parser.add_argument("--decoder-opset", type=int, default=18, help="ONNX opset for decoder export.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = export_bundle(
        args.corpus,
        args.out,
        manual_arg=args.manual,
        name=args.name,
        overwrite=args.overwrite,
        verify=args.verify,
        export_decoder_onnx=args.export_decoder_onnx,
        decoder_windows=_parse_decoder_windows(args.decoder_windows),
        decoder_repo=args.decoder_repo,
        decoder_revision=args.decoder_revision,
        decoder_opset=args.decoder_opset,
    )
    print(f"Exported {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
