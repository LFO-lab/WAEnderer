#!/usr/bin/env python3
"""Render ONNX decoder trajectory diagnostics from a Stable Audio Wanderer bundle.

The diagnostic compares windows decoded from the true corpus latent order against
deliberately invalid latent orders. It uses the same manifest timing metadata and
full-output OLA convention as the current JUCE ONNX path.

The frame-order pass mirrors the plugin's Frame Order slider: 0.0 preserves each
window's current frame order, while 1.0 keeps the endpoints fixed and shuffles
only interior frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_MODES = ("ordered", "shuffled_windows", "shuffled_frames", "static_start")
SUPPORTED_MODES = (
    *DEFAULT_MODES,
    "static_frame",
    "reverse",
    "vst_static_xy",
    "vst_default_xy",
    "vst_chunk_xy",
)


@dataclass(frozen=True)
class DecoderWindow:
    latent_window: int
    latent_hop: int
    samples_per_latent: int
    audio_hop_samples: int
    output_samples: int
    path: Path
    input_name: str
    output_name: str


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


def load_array(bundle: Path, manifest: Mapping[str, Any], key: str) -> np.ndarray:
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


def load_raw_latents(bundle: Path, manifest: Mapping[str, Any]) -> np.ndarray:
    z_norm = np.asarray(load_array(bundle, manifest, "Z_concat"), dtype=np.float32)
    z_mean = np.asarray(load_array(bundle, manifest, "Z_mean"), dtype=np.float32).reshape(1, -1)
    z_std = np.asarray(load_array(bundle, manifest, "Z_std"), dtype=np.float32).reshape(1, -1)

    if z_norm.ndim != 2:
        raise RuntimeError(f"Z_concat must be 2D, got shape {z_norm.shape}")

    if z_mean.shape[1] != z_norm.shape[1] or z_std.shape[1] != z_norm.shape[1]:
        raise RuntimeError("Z_mean/Z_std latent dimensions do not match Z_concat")

    if manifest.get("corpus", {}).get("normalized_latents", True):
        return np.ascontiguousarray(z_norm * z_std + z_mean, dtype=np.float32)

    return np.ascontiguousarray(z_norm, dtype=np.float32)


def load_manual_points(bundle: Path, manifest: Mapping[str, Any], frame_count: int) -> np.ndarray:
    entry = manifest.get("arrays", {}).get("manual_embed_points")

    if not entry:
        if frame_count <= 1:
            return np.asarray([[0.5, 0.5]], dtype=np.float32)

        x = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
        y = np.full((frame_count,), 0.5, dtype=np.float32)
        return np.column_stack([x, y]).astype(np.float32)

    points = np.asarray(load_array(bundle, manifest, "manual_embed_points"), dtype=np.float32)

    if points.ndim != 2 or points.shape[0] != frame_count or points.shape[1] < 2:
        raise RuntimeError(
            "manual_embed_points must have shape [frames, at least 2] "
            f"for VST-like modes, got {points.shape}"
        )

    return np.ascontiguousarray(points[:, :2], dtype=np.float32)


def load_file_offsets(bundle: Path, manifest: Mapping[str, Any], frame_count: int) -> np.ndarray:
    entry = manifest.get("arrays", {}).get("file_offsets")

    if not entry:
        return np.asarray([0, frame_count], dtype=np.int64)

    offsets = np.asarray(load_array(bundle, manifest, "file_offsets"), dtype=np.int64).reshape(-1)

    if (
        offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] > frame_count
        or np.any(offsets < 0)
        or np.any(np.diff(offsets) < 0)
    ):
        return np.asarray([0, frame_count], dtype=np.int64)

    return offsets


def load_windows(bundle: Path, manifest: Mapping[str, Any], requested: Optional[Sequence[int]]) -> List[DecoderWindow]:
    decoder = manifest.get("models", {}).get("decoder", {})

    if decoder.get("backend") != "onnxruntime":
        raise RuntimeError("Bundle does not declare an ONNX Runtime decoder")

    window_entries = decoder.get("windows", {})

    if not window_entries:
        raise RuntimeError("Bundle decoder has no ONNX window entries")

    requested_set = set(requested) if requested else None
    windows: List[DecoderWindow] = []

    for key, entry in sorted(window_entries.items(), key=lambda item: int(item[0])):
        latent_window = int(entry.get("latent_window", key))

        if requested_set is not None and latent_window not in requested_set:
            continue

        for timing_key in ("samples_per_latent", "latent_hop", "audio_hop_samples"):
            if int(entry.get(timing_key, 0)) <= 0:
                raise RuntimeError(f"Decoder T{latent_window} is missing positive {timing_key}")

        if entry.get("ola_mode") != "full_overlap_add":
            raise RuntimeError(f"Decoder T{latent_window} does not declare ola_mode=full_overlap_add")

        model_path = bundle / entry["path"]

        if not model_path.exists():
            raise FileNotFoundError(f"Missing decoder model: {model_path}")

        expected_hash = entry.get("sha256")

        if expected_hash and sha256(model_path).lower() != str(expected_hash).lower():
            raise RuntimeError(f"Hash mismatch for decoder model: {model_path}")

        output_shape = entry.get("output_shape")
        output_samples = int(entry.get("output_samples", 0))

        if output_samples <= 0 and isinstance(output_shape, Sequence) and output_shape:
            output_samples = int(output_shape[-1])

        if output_samples <= 0:
            raise RuntimeError(f"Decoder T{latent_window} is missing output sample count")

        windows.append(
            DecoderWindow(
                latent_window=latent_window,
                latent_hop=int(entry["latent_hop"]),
                samples_per_latent=int(entry["samples_per_latent"]),
                audio_hop_samples=int(entry["audio_hop_samples"]),
                output_samples=output_samples,
                path=model_path,
                input_name=str(entry.get("input_name", decoder.get("input_name", "latents"))),
                output_name=str(entry.get("output_name", decoder.get("output_name", "audio"))),
            )
        )

    if requested_set is not None:
        found = {window.latent_window for window in windows}
        missing = sorted(requested_set - found)

        if missing:
            raise RuntimeError(f"Bundle is missing requested decoder windows: {missing}")

    if not windows:
        raise RuntimeError("No decoder windows selected")

    return windows


def parse_int_list(value: str) -> Sequence[int]:
    items = []

    for raw_item in value.split(","):
        item = raw_item.strip()

        if item:
            items.append(int(item))

    return tuple(dict.fromkeys(items))


def parse_str_list(value: str) -> Sequence[str]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> Sequence[float]:
    items = []

    for raw_item in value.split(","):
        item = raw_item.strip()

        if item:
            items.append(float(item))

    return tuple(items)


def ordered_windows(frame_count: int, latent_window: int, latent_hop: int) -> List[np.ndarray]:
    starts = list(range(0, max(1, frame_count), latent_hop))
    windows = []

    for start in starts:
        indices = np.arange(start, start + latent_window, dtype=np.int64)
        indices = np.clip(indices, 0, frame_count - 1)
        windows.append(indices)

    return windows


def mode_windows(
    mode: str,
    frame_count: int,
    latent_window: int,
    latent_hop: int,
    rng: np.random.Generator,
    manual_points: Optional[np.ndarray] = None,
    file_offsets: Optional[np.ndarray] = None,
    sample_rate: int = 44100,
    samples_per_latent: int = 4096,
) -> List[np.ndarray]:
    ordered = ordered_windows(frame_count, latent_window, latent_hop)

    if mode == "ordered":
        return ordered

    if mode == "shuffled_windows":
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        return shuffled

    if mode == "shuffled_frames":
        sequence = np.arange(frame_count, dtype=np.int64)
        rng.shuffle(sequence)
        windows = []

        for start in range(0, max(1, frame_count), latent_hop):
            positions = np.arange(start, start + latent_window, dtype=np.int64)
            positions = np.clip(positions, 0, frame_count - 1)
            windows.append(sequence[positions])

        return windows

    if mode == "static_start":
        return [ordered[0].copy() for _ in ordered]

    if mode == "static_frame":
        static = np.full((latent_window,), int(ordered[0][0]), dtype=np.int64)
        return [static.copy() for _ in ordered]

    if mode == "reverse":
        sequence = np.arange(frame_count - 1, -1, -1, dtype=np.int64)
        windows = []

        for start in range(0, max(1, frame_count), latent_hop):
            positions = np.arange(start, start + latent_window, dtype=np.int64)
            positions = np.clip(positions, 0, frame_count - 1)
            windows.append(sequence[positions])

        return windows

    if mode in ("vst_static_xy", "vst_default_xy"):
        if manual_points is None:
            raise RuntimeError(f"{mode} requires manual_embed_points")

        controls = {
            "manual_x": 0.5,
            "manual_y": 0.5,
            "wander": 0.0 if mode == "vst_static_xy" else 0.12,
            "motion_rate": 0.25,
        }
        return vst_like_windows(
            frame_count=frame_count,
            latent_window=latent_window,
            latent_hop=latent_hop,
            window_count=len(ordered),
            manual_points=manual_points,
            sample_rate=sample_rate,
            samples_per_latent=samples_per_latent,
            **controls,
        )

    if mode == "vst_chunk_xy":
        if manual_points is None:
            raise RuntimeError(f"{mode} requires manual_embed_points")

        if file_offsets is None:
            file_offsets = np.asarray([0, frame_count], dtype=np.int64)

        return vst_chunk_windows(
            frame_count=frame_count,
            latent_window=latent_window,
            latent_hop=latent_hop,
            window_count=len(ordered),
            manual_points=manual_points,
            file_offsets=file_offsets,
            sample_rate=sample_rate,
            samples_per_latent=samples_per_latent,
            manual_x=0.5,
            manual_y=0.5,
            wander=0.12,
            motion_rate=0.25,
        )

    raise ValueError(f"Unknown mode: {mode}")


def nearest_manual_frame(manual_points: np.ndarray, x: float, y: float) -> int:
    dx = manual_points[:, 0].astype(np.float64) - float(x)
    dy = manual_points[:, 1].astype(np.float64) - float(y)
    return int(np.argmin(dx * dx + dy * dy))


def sequence_bounds_for_frame(file_offsets: np.ndarray, frame_count: int, frame: int) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0

    frame = int(np.clip(frame, 0, frame_count - 1))

    for start, end in zip(file_offsets[:-1], file_offsets[1:]):
        start_i = int(np.clip(start, 0, frame_count))
        end_i = int(np.clip(end, 0, frame_count))

        if start_i < end_i and start_i <= frame < end_i:
            return start_i, end_i

    return 0, frame_count


def chunk_start_for_frame(file_offsets: np.ndarray, frame_count: int, frame: int, latent_window: int) -> int:
    if frame_count <= 0:
        return 0

    frame = int(np.clip(frame, 0, frame_count - 1))
    latent_window = max(1, int(latent_window))
    start, end = sequence_bounds_for_frame(file_offsets, frame_count, frame)

    if end <= start:
        return frame

    latest_start = max(start, end - latent_window)
    return int(np.clip(frame, start, latest_start))


def frame_for_chunk_offset(file_offsets: np.ndarray, frame_count: int, chunk_start: int, offset: int) -> int:
    if frame_count <= 0:
        return 0

    chunk_start = int(np.clip(chunk_start, 0, frame_count - 1))
    start, end = sequence_bounds_for_frame(file_offsets, frame_count, chunk_start)
    end = end if end > start else frame_count
    return int(np.clip(chunk_start + max(0, int(offset)), start, end - 1))


def vst_like_windows(
    frame_count: int,
    latent_window: int,
    latent_hop: int,
    window_count: int,
    manual_points: np.ndarray,
    sample_rate: int,
    samples_per_latent: int,
    manual_x: float,
    manual_y: float,
    wander: float,
    motion_rate: float,
) -> List[np.ndarray]:
    tail_frames = max(0, latent_window - latent_hop)
    cursor_x = 0.5
    cursor_y = 0.5
    wander_phase = 0.0
    latent_tail: List[int] = []
    windows: List[np.ndarray] = []

    wander_amount = float(wander) * 0.28
    rate = 0.05 + float(motion_rate) * 1.95
    smoothing = min(0.95, max(0.02, 0.04 + float(motion_rate) * 0.22))
    phase_step = float(samples_per_latent) / float(sample_rate) * rate

    for _ in range(window_count):
        current: List[int] = []

        if latent_tail:
            current.extend(latent_tail)

        while len(current) < latent_window:
            mod_x = math.sin(wander_phase * 0.71) * wander_amount
            mod_y = math.cos(wander_phase * 0.47) * wander_amount
            target_x = min(1.0, max(0.0, float(manual_x) + mod_x))
            target_y = min(1.0, max(0.0, float(manual_y) + mod_y))
            cursor_x += (target_x - cursor_x) * smoothing
            cursor_y += (target_y - cursor_y) * smoothing
            wander_phase += phase_step
            current.append(nearest_manual_frame(manual_points, cursor_x, cursor_y))

        clipped = np.clip(np.asarray(current, dtype=np.int64), 0, frame_count - 1)
        windows.append(clipped)
        latent_tail = list(clipped[latent_hop : latent_hop + tail_frames])

    return windows


def vst_chunk_windows(
    frame_count: int,
    latent_window: int,
    latent_hop: int,
    window_count: int,
    manual_points: np.ndarray,
    file_offsets: np.ndarray,
    sample_rate: int,
    samples_per_latent: int,
    manual_x: float,
    manual_y: float,
    wander: float,
    motion_rate: float,
) -> List[np.ndarray]:
    cursor_x = 0.5
    cursor_y = 0.5
    wander_phase = 0.0
    windows: List[np.ndarray] = []

    wander_amount = float(wander) * 0.28
    rate = 0.05 + float(motion_rate) * 1.95
    smoothing = min(0.95, max(0.02, 0.04 + float(motion_rate) * 0.22))
    phase_step = float(samples_per_latent * max(1, latent_hop)) / float(sample_rate) * rate

    for _ in range(window_count):
        mod_x = math.sin(wander_phase * 0.71) * wander_amount
        mod_y = math.cos(wander_phase * 0.47) * wander_amount
        target_x = min(1.0, max(0.0, float(manual_x) + mod_x))
        target_y = min(1.0, max(0.0, float(manual_y) + mod_y))
        cursor_x += (target_x - cursor_x) * smoothing
        cursor_y += (target_y - cursor_y) * smoothing
        wander_phase += phase_step

        anchor = nearest_manual_frame(manual_points, cursor_x, cursor_y)
        chunk_start = chunk_start_for_frame(file_offsets, frame_count, anchor, latent_window)
        windows.append(
            np.asarray(
                [
                    frame_for_chunk_offset(file_offsets, frame_count, chunk_start, offset)
                    for offset in range(latent_window)
                ],
                dtype=np.int64,
            )
        )

    return windows


def u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def mix_frame_order_seed(seed: int, chunk_start_frame: int, latent_window: int, decode_window_index: int) -> int:
    value = u32(seed) if seed != 0 else 1
    value ^= u32(u32(chunk_start_frame) + 0x9E3779B9)
    value = u32(value * 0x85EBCA6B)
    value ^= u32(latent_window * 0xC2B2AE35)
    value = u32(value * 0x27D4EB2D)
    value ^= u32(decode_window_index) + 0x165667B1
    value ^= value >> 16
    return value if value != 0 else 1


def next_frame_order_random(state: int) -> tuple[int, int]:
    state = u32(state * 1664525 + 1013904223)
    return state, state


def is_strictly_stepwise_frame_order(indices: np.ndarray) -> bool:
    if indices.size < 2:
        return True

    for previous, current in zip(indices[:-1], indices[1:]):
        if int(current) != int(previous) + 1 and int(current) != int(previous):
            return False

    return True


def apply_frame_order_disorder(
    indices: np.ndarray,
    frame_order: float,
    seed: int,
    chunk_start_frame: int,
    latent_window: int,
    decode_window_index: int,
) -> np.ndarray:
    amount = min(1.0, max(0.0, float(frame_order)))
    ordered = np.asarray(indices, dtype=np.int64).copy()

    if amount <= 0.0 or ordered.size <= 3:
        return ordered

    interior_count = int(ordered.size) - 2
    maximum_swaps = max(0, interior_count - 1)
    swap_count = int(round(amount * maximum_swaps))

    if amount > 0.0 and maximum_swaps > 0:
        swap_count = max(1, swap_count)

    random_state = mix_frame_order_seed(seed, chunk_start_frame, latent_window, decode_window_index)

    for step in range(swap_count):
        target_interior = 1 + (interior_count - 1 - step)
        candidate_count = interior_count - step
        random_state, random_value = next_frame_order_random(random_state)
        random_interior = 1 + int(random_value % candidate_count)
        ordered[target_interior], ordered[random_interior] = (
            ordered[random_interior],
            ordered[target_interior],
        )

    if amount > 0.0 and is_strictly_stepwise_frame_order(ordered) and interior_count > 1:
        ordered[1], ordered[2] = ordered[2], ordered[1]

    return ordered


def latent_input(raw_latents: np.ndarray, indices: np.ndarray) -> np.ndarray:
    z_window = raw_latents[indices]
    return np.ascontiguousarray(z_window.T[None, :, :], dtype=np.float32)


def synthesis_gain(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(0, length),), dtype=np.float32)

    samples = np.arange(length, dtype=np.float32)
    phase = (samples + np.float32(0.5)) / np.float32(length)
    return np.sin(np.float32(math.pi) * phase).astype(np.float32)


def full_overlap_add(decoded_windows: Sequence[np.ndarray], audio_hop_samples: int) -> np.ndarray:
    if not decoded_windows:
        return np.zeros((0, 0), dtype=np.float32)

    channel_count = int(decoded_windows[0].shape[1])
    output_samples = int(decoded_windows[0].shape[0])
    total_samples = (len(decoded_windows) - 1) * audio_hop_samples + output_samples
    emitted_samples = len(decoded_windows) * audio_hop_samples
    accumulator = np.zeros((total_samples, channel_count), dtype=np.float64)
    weights = np.zeros((total_samples,), dtype=np.float64)
    gain = synthesis_gain(output_samples).astype(np.float64)

    for index, decoded in enumerate(decoded_windows):
        if decoded.shape != (output_samples, channel_count):
            raise RuntimeError(
                f"Decoded window shape changed from {(output_samples, channel_count)} to {decoded.shape}"
            )

        start = index * audio_hop_samples
        end = start + output_samples
        accumulator[start:end] += decoded.astype(np.float64) * gain[:, None]
        weights[start:end] += gain

    emitted = accumulator[:emitted_samples].copy()
    emitted_weights = weights[:emitted_samples]
    nonzero = emitted_weights > 1.0e-8
    emitted[nonzero] /= emitted_weights[nonzero, None]
    emitted[~nonzero] = 0.0
    return np.asarray(emitted, dtype=np.float32)


def run_decodes(session: Any, window: DecoderWindow, raw_latents: np.ndarray, windows: Sequence[np.ndarray]) -> List[np.ndarray]:
    decoded = []

    for indices in windows:
        audio = session.run(
            [window.output_name],
            {window.input_name: latent_input(raw_latents, indices)},
        )[0].astype(np.float32)

        if audio.ndim != 3:
            raise RuntimeError(f"Expected decoder output [1, channels, samples], got {audio.shape}")

        decoded.append(np.ascontiguousarray(audio[0].T, dtype=np.float32))

    return decoded


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def envelope_metrics(audio: np.ndarray, sample_rate: int, cadence_hz: float) -> Dict[str, Any]:
    if audio.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "envelope_relative_std": 0.0,
            "cadence_hz": cadence_hz,
            "cadence_envelope_ratio": 0.0,
            "half_cadence_envelope_ratio": 0.0,
            "top_envelope_peaks_hz": [],
        }

    mono = np.mean(audio.astype(np.float64), axis=1)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    frame_size = max(256, min(4096, audio.shape[0]))
    hop = max(64, frame_size // 8)

    if audio.shape[0] < frame_size:
        envelope = np.array([float(np.sqrt(np.mean(np.square(mono))))], dtype=np.float64)
        envelope_rate = float(sample_rate)
    else:
        starts = np.arange(0, audio.shape[0] - frame_size + 1, hop, dtype=np.int64)
        envelope = np.empty((len(starts),), dtype=np.float64)

        for i, start in enumerate(starts):
            frame = mono[start : start + frame_size]
            envelope[i] = math.sqrt(float(np.mean(np.square(frame))))

        envelope_rate = sample_rate / hop

    mean_envelope = float(np.mean(envelope)) if envelope.size else 0.0
    relative_std = float(np.std(envelope) / mean_envelope) if mean_envelope > 0.0 else 0.0
    centered = envelope - mean_envelope

    if centered.size < 4 or np.max(np.abs(centered)) <= 0.0:
        return {
            "duration_seconds": float(audio.shape[0] / sample_rate),
            "rms": rms,
            "peak": peak,
            "envelope_relative_std": relative_std,
            "cadence_hz": cadence_hz,
            "cadence_envelope_ratio": 0.0,
            "half_cadence_envelope_ratio": 0.0,
            "top_envelope_peaks_hz": [],
        }

    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / envelope_rate)
    valid = (freqs >= 0.25) & (freqs <= min(20.0, envelope_rate * 0.5))

    if not np.any(valid):
        top_peaks: List[float] = []
    else:
        valid_indices = np.flatnonzero(valid)
        top_count = min(5, valid_indices.size)
        ranked = valid_indices[np.argsort(spectrum[valid_indices])[-top_count:]][::-1]
        top_peaks = [float(freqs[index]) for index in ranked]

    total = float(np.sum(spectrum[valid])) if np.any(valid) else 0.0

    def band_ratio(target_hz: float) -> float:
        if target_hz <= 0.0 or total <= 0.0:
            return 0.0

        band = np.abs(freqs - target_hz) <= max(0.15, target_hz * 0.04)

        if not np.any(band):
            return 0.0

        return float(np.sum(spectrum[band]) / total)

    return {
        "duration_seconds": float(audio.shape[0] / sample_rate),
        "rms": rms,
        "peak": peak,
        "envelope_relative_std": relative_std,
        "cadence_hz": cadence_hz,
        "cadence_envelope_ratio": band_ratio(cadence_hz),
        "half_cadence_envelope_ratio": band_ratio(cadence_hz * 0.5),
        "top_envelope_peaks_hz": top_peaks,
    }


def render_window(
    bundle: Path,
    output_dir: Path,
    sample_rate: int,
    raw_latents: np.ndarray,
    manual_points: np.ndarray,
    file_offsets: np.ndarray,
    decoder_window: DecoderWindow,
    modes: Sequence[str],
    frame_orders: Sequence[float],
    seed: int,
    max_windows: Optional[int],
) -> List[Dict[str, Any]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(decoder_window.path), providers=["CPUExecutionProvider"])
    rows = []

    for mode_index, mode in enumerate(modes):
        rng = np.random.default_rng(seed + decoder_window.latent_window * 1009 + mode_index * 9176)
        windows = mode_windows(
            mode=mode,
            frame_count=raw_latents.shape[0],
            latent_window=decoder_window.latent_window,
            latent_hop=decoder_window.latent_hop,
            rng=rng,
            manual_points=manual_points,
            file_offsets=file_offsets,
            sample_rate=sample_rate,
            samples_per_latent=decoder_window.samples_per_latent,
        )

        if max_windows is not None:
            windows = windows[:max_windows]

        for frame_order in frame_orders:
            ordered_windows_for_mode = [
                apply_frame_order_disorder(
                    indices=indices,
                    frame_order=frame_order,
                    seed=seed,
                    chunk_start_frame=int(indices[0]) if len(indices) else 0,
                    latent_window=decoder_window.latent_window,
                    decode_window_index=window_index,
                )
                for window_index, indices in enumerate(windows)
            ]
            decoded = run_decodes(session, decoder_window, raw_latents, ordered_windows_for_mode)
            audio = full_overlap_add(decoded, decoder_window.audio_hop_samples)
            cadence_hz = sample_rate / decoder_window.audio_hop_samples
            metrics = envelope_metrics(audio, sample_rate, cadence_hz)
            wav_path = output_dir / (
                f"T{decoder_window.latent_window:02d}_{mode}_fo{frame_order:.2f}_ola.wav"
            )
            write_wav(wav_path, audio, sample_rate)

            rows.append(
                {
                    "bundle": str(bundle),
                    "wav": str(wav_path),
                    "mode": mode,
                    "frame_order": float(frame_order),
                    "latent_window": decoder_window.latent_window,
                    "latent_hop": decoder_window.latent_hop,
                    "samples_per_latent": decoder_window.samples_per_latent,
                    "audio_hop_samples": decoder_window.audio_hop_samples,
                    "output_samples": decoder_window.output_samples,
                    "decode_windows": len(ordered_windows_for_mode),
                    **metrics,
                }
            )

    return rows


def write_reports(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "trajectory_diagnostics_report.json"
    csv_path = output_dir / "trajectory_diagnostics_report.csv"

    json_path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")

    fieldnames = [
        "mode",
        "frame_order",
        "latent_window",
        "latent_hop",
        "samples_per_latent",
        "audio_hop_samples",
        "output_samples",
        "decode_windows",
        "duration_seconds",
        "rms",
        "peak",
        "envelope_relative_std",
        "cadence_hz",
        "cadence_envelope_ratio",
        "half_cadence_envelope_ratio",
        "top_envelope_peaks_hz",
        "wav",
        "bundle",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writable = dict(row)
            writable["top_envelope_peaks_hz"] = json.dumps(writable["top_envelope_peaks_hz"])
            writer.writerow(writable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Bundle directory containing manifest.json.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <bundle>/diagnostics/trajectory_renders.",
    )
    parser.add_argument(
        "--windows",
        default="all",
        help="Comma-separated decoder windows to render, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated modes. Supported: {', '.join(SUPPORTED_MODES)}.",
    )
    parser.add_argument(
        "--frame-orders",
        default="0.0,1.0",
        help="Comma-separated Frame Order values to render. Default: 0.0,1.0.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffled modes.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on decoded windows per file, useful for quick test renders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    output_dir = (args.out_dir or (bundle / "diagnostics" / "trajectory_renders")).resolve()
    manifest = load_manifest(bundle)
    sample_rate = int(manifest.get("vae", {}).get("sample_rate", 44100))
    requested_windows = None if args.windows == "all" else parse_int_list(args.windows)
    windows = load_windows(bundle, manifest, requested_windows)
    modes = parse_str_list(args.modes)
    frame_orders = parse_float_list(args.frame_orders)
    raw_latents = load_raw_latents(bundle, manifest)
    manual_points = load_manual_points(bundle, manifest, raw_latents.shape[0])
    file_offsets = load_file_offsets(bundle, manifest, raw_latents.shape[0])
    rows: List[Dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for decoder_window in windows:
        print(
            "rendering "
            f"T{decoder_window.latent_window} "
            f"hop={decoder_window.latent_hop} "
            f"audio_hop={decoder_window.audio_hop_samples}",
            flush=True,
        )
        rows.extend(
            render_window(
                bundle=bundle,
                output_dir=output_dir,
                sample_rate=sample_rate,
                raw_latents=raw_latents,
                manual_points=manual_points,
                file_offsets=file_offsets,
                decoder_window=decoder_window,
                modes=modes,
                frame_orders=frame_orders,
                seed=args.seed,
                max_windows=args.max_windows,
            )
        )

    write_reports(output_dir, rows)

    print(f"OK: rendered={len(rows)}")
    print(f"OK: out_dir={output_dir}")
    print(f"OK: report={output_dir / 'trajectory_diagnostics_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
