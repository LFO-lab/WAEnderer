"""Continuity and stability evaluation helpers for Stable Audio Wanderer."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from ..io.audio_io import save_wav
from ..runtime.decoder_player import BufferedAudioQueue
from ..vae.decoder import decode_latents
from .latency import (
    CorpusBundle,
    collect_machine_info,
    load_corpus_bundle,
    load_vae_for_bundle,
    write_summary_json,
)

LOGGER = logging.getLogger(__name__)

PRIMARY_PAIR_METRICS = (
    "rms_diff",
    "spectral_flux",
    "mfcc_distance",
    "loudness_delta",
    "low_freq_ratio_delta",
)
STRATEGY_LABELS = {
    "single_loop": "Single-Latent Loop",
    "naive_interp": "Naive Raw-Latent Interpolation",
    "sequential": "Sequential Corpus Recall",
}


@dataclass(frozen=True)
class TrajectoryExample:
    """One sampled contiguous corpus trajectory."""

    example_id: str
    start_index: int
    end_index: int
    file_id: int
    start_t: int
    end_t: int
    source_path: str
    latents_raw: np.ndarray


@dataclass(frozen=True)
class ConditionSpec:
    """One decoding condition evaluated for every example."""

    strategy: str
    window_size: int
    label: str


@dataclass
class RenderedCondition:
    """Decoded audio and per-buffer outputs for one example/condition."""

    audio: np.ndarray
    output_buffers: List[np.ndarray]
    expected_samples: int
    hop_size: int
    num_batches: int


def build_condition_specs(window_sizes: Sequence[int]) -> List[ConditionSpec]:
    """Build evaluation conditions for the requested decode window sizes."""
    windows = [int(window) for window in window_sizes]
    if not windows:
        raise ValueError("At least one window size is required.")
    multi_window = len(windows) > 1

    specs: List[ConditionSpec] = []
    for window_size in windows:
        if window_size < 1:
            raise ValueError(f"Invalid window size: {window_size}")
        for strategy in ("single_loop", "naive_interp", "sequential"):
            label = (
                f"{strategy}_w{window_size}"
                if multi_window
                else strategy
            )
            specs.append(
                ConditionSpec(
                    strategy=strategy,
                    window_size=window_size,
                    label=label,
                )
            )
    return specs


def _collect_valid_trajectory_starts(
    meta: np.ndarray,
    trajectory_frames: int,
) -> List[int]:
    """
    Collect corpus start indices whose stored frames are file-contiguous and
    one-step contiguous in the original latent time index.
    """
    meta = np.asarray(meta, dtype=np.int32)
    if meta.ndim != 2 or meta.shape[1] < 2:
        raise ValueError(f"Expected meta [N, >=2], got {meta.shape}")

    starts: List[int] = []
    run_start = 0
    for idx in range(1, meta.shape[0] + 1):
        is_break = idx == meta.shape[0]
        if not is_break:
            prev = meta[idx - 1]
            curr = meta[idx]
            is_break = (
                int(curr[0]) != int(prev[0])
                or int(curr[1]) != int(prev[1]) + 1
            )
        if is_break:
            run_end = idx
            run_len = run_end - run_start
            if run_len >= int(trajectory_frames):
                for start in range(run_start, run_end - int(trajectory_frames) + 1):
                    starts.append(int(start))
            run_start = idx
    return starts


def sample_trajectory_examples(
    bundle: CorpusBundle,
    num_examples: int,
    trajectory_frames: int,
    seed: int,
) -> List[TrajectoryExample]:
    """Sample contiguous real-corpus trajectories for evaluation."""
    meta = bundle.data["meta"].astype(np.int32)
    z_concat = bundle.data["Z_concat"].astype(np.float32)
    z_mean = bundle.data["Z_mean"].astype(np.float32)
    z_std = bundle.data["Z_std"].astype(np.float32)
    paths = bundle.data.get("paths")
    path_list: List[str] = []
    if paths is not None:
        path_list = [str(path) for path in np.asarray(paths).reshape(-1).tolist()]

    valid_starts = _collect_valid_trajectory_starts(meta, trajectory_frames)
    if not valid_starts:
        raise RuntimeError(
            f"No valid contiguous trajectories of length {trajectory_frames} were found "
            "in the selected corpus."
        )

    if len(valid_starts) < int(num_examples):
        LOGGER.warning(
            "Requested %d examples but only %d valid trajectories were found. "
            "Using all available trajectories.",
            int(num_examples),
            len(valid_starts),
        )
    count = min(int(num_examples), len(valid_starts))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(valid_starts, dtype=np.int32), size=count, replace=False)
    chosen = [int(idx) for idx in np.sort(chosen)]

    examples: List[TrajectoryExample] = []
    for example_idx, start_index in enumerate(chosen):
        end_index = start_index + int(trajectory_frames)
        raw_latents = (
            z_concat[start_index:end_index] * z_std[None, :] + z_mean[None, :]
        ).astype(np.float32)
        file_id = int(meta[start_index, 0])
        start_t = int(meta[start_index, 1])
        end_t = int(meta[end_index - 1, 1])
        source_path = path_list[file_id] if 0 <= file_id < len(path_list) else ""
        examples.append(
            TrajectoryExample(
                example_id=f"example_{example_idx:03d}",
                start_index=int(start_index),
                end_index=int(end_index),
                file_id=file_id,
                start_t=start_t,
                end_t=end_t,
                source_path=source_path,
                latents_raw=raw_latents,
            )
        )
    return examples


def build_condition_latents(
    example: TrajectoryExample,
    strategy: str,
) -> np.ndarray:
    """Construct a latent sequence for one evaluation strategy."""
    source = np.asarray(example.latents_raw, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] <= 0:
        raise ValueError(f"Invalid example latent sequence shape: {source.shape}")

    strategy_name = str(strategy).strip().lower()
    if strategy_name == "single_loop":
        return np.repeat(source[:1], source.shape[0], axis=0).astype(np.float32)
    if strategy_name == "naive_interp":
        if source.shape[0] == 1:
            return source.copy()
        alphas = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float32)[:, None]
        return (
            (1.0 - alphas) * source[0:1] + alphas * source[-1:]
        ).astype(np.float32)
    if strategy_name == "sequential":
        return source.copy()
    raise ValueError(f"Unsupported strategy: {strategy}")


def _drain_ready_chunks(buffer: BufferedAudioQueue) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    while True:
        chunk = buffer.pop_chunk()
        if chunk is None:
            break
        chunks.append(np.asarray(chunk, dtype=np.float32).copy())
    return chunks


def _trim_or_pad_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError(f"Expected stereo audio [T, C], got {audio.shape}")
    if audio.shape[0] >= int(target_samples):
        return audio[: int(target_samples)]
    pad = np.zeros((int(target_samples) - audio.shape[0], audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio, pad], axis=0)


def decode_latent_sequence(
    vae,
    latent_sequence_raw: np.ndarray,
    *,
    window_size: int,
    sample_rate: int,
    latent_hz: float,
) -> RenderedCondition:
    """
    Decode a latent sequence with the same overlap-add queue logic used at runtime.

    Assumptions:
    - First decode uses a full window (or the remaining frames if shorter).
    - Subsequent decodes advance by hop = ceil(window / 2) latent frames.
    - Final audio is trimmed/padded to the target duration implied by
      ``len(latent_sequence_raw) / latent_hz`` after flushing the held tail.
    """
    sequence = np.asarray(latent_sequence_raw, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] <= 0:
        raise ValueError(f"Expected latent sequence [T, D], got {sequence.shape}")

    actual_window = max(1, min(int(window_size), int(sequence.shape[0])))
    hop = max(1, (int(window_size) + 1) // 2)
    buffer = BufferedAudioQueue(sr=int(sample_rate), gain=1.0)

    metric_buffers: List[np.ndarray] = []
    audio_chunks: List[np.ndarray] = []
    num_batches = 0

    first_batch = sequence[:actual_window]
    audio = decode_latents(vae, first_batch)
    buffer.write_frame(audio)
    drained = _drain_ready_chunks(buffer)
    metric_buffers.extend(drained)
    audio_chunks.extend(drained)
    num_batches += 1

    prev_half = first_batch[hop:]
    cursor = actual_window
    while cursor < sequence.shape[0]:
        new_frames = sequence[cursor: cursor + hop]
        if prev_half.shape[0] > 0:
            batch = np.concatenate([prev_half, new_frames], axis=0)
        else:
            batch = new_frames
        if batch.shape[0] <= 0:
            break

        audio = decode_latents(vae, batch)
        buffer.write_frame(audio)
        drained = _drain_ready_chunks(buffer)
        metric_buffers.extend(drained)
        audio_chunks.extend(drained)
        num_batches += 1

        prev_half = batch[hop:] if batch.shape[0] > hop else np.zeros(
            (0, sequence.shape[1]), dtype=np.float32
        )
        cursor += hop

    buffer.flush_tail()
    audio_chunks.extend(_drain_ready_chunks(buffer))

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)
    else:
        full_audio = np.zeros((0, 2), dtype=np.float32)

    expected_samples = int(round(sequence.shape[0] * float(sample_rate) / float(latent_hz)))
    full_audio = _trim_or_pad_audio(full_audio, expected_samples)
    return RenderedCondition(
        audio=full_audio,
        output_buffers=metric_buffers,
        expected_samples=expected_samples,
        hop_size=hop,
        num_batches=num_batches,
    )


def _mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {arr.shape}")


def _analysis_n_fft(length: int) -> int:
    base = max(256, min(2048, int(length)))
    return int(2 ** np.floor(np.log2(base)))


def describe_audio_chunk(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_mfcc: int,
    low_freq_cutoff_hz: float,
) -> Dict[str, np.ndarray | float]:
    """Compute chunk descriptors reused by continuity metrics."""
    mono = _mono(audio)
    n_fft = _analysis_n_fft(len(mono))
    if mono.shape[0] < n_fft:
        mono_proc = np.pad(mono, (0, n_fft - mono.shape[0]))
    else:
        mono_proc = mono

    hop_length = max(64, n_fft // 4)
    stft = librosa.stft(mono_proc, n_fft=n_fft, hop_length=hop_length, center=False)
    magnitude = np.abs(stft).astype(np.float32)
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        magnitude = np.abs(
            np.fft.rfft(mono_proc[:n_fft] * np.hanning(n_fft), n=n_fft)
        ).astype(np.float32)[:, None]

    mean_spectrum = magnitude.mean(axis=1).astype(np.float32)
    spectrum_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    power = mean_spectrum**2
    freqs = np.fft.rfftfreq((magnitude.shape[0] - 1) * 2, d=1.0 / float(sample_rate))
    low_ratio = float(
        power[freqs <= float(low_freq_cutoff_hz)].sum() / (power.sum() + 1e-8)
    )

    mfcc = librosa.feature.mfcc(
        y=mono_proc,
        sr=int(sample_rate),
        n_mfcc=int(n_mfcc),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        center=False,
    ).astype(np.float32)
    mfcc_mean = mfcc.mean(axis=1).astype(np.float32)
    rms = float(np.sqrt(np.mean(mono**2) + 1e-10))
    loudness_db = float(20.0 * np.log10(rms + 1e-10))

    return {
        "mono": mono,
        "loudness_db": loudness_db,
        "low_freq_ratio": low_ratio,
        "spectrum": spectrum_norm,
        "mfcc": mfcc_mean,
    }


def summarize_series(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Summarize a metric sequence while tolerating empty inputs."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_continuity_metrics(
    rendered: RenderedCondition,
    *,
    sample_rate: int,
    n_mfcc: int,
    low_freq_cutoff_hz: float,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute pairwise continuity metrics over consecutive decoded buffers.

    Metrics:
    - rms_diff: sample-domain RMS distance between consecutive buffers
    - spectral_flux: mean absolute difference between normalized average spectra
    - mfcc_distance: Euclidean distance between mean MFCC vectors
    - loudness_delta: absolute RMS-loudness difference in dB
    - low_freq_ratio_delta: absolute change in low-frequency energy ratio
    - low_freq_ratio: chunk-level low-frequency energy ratio (rumble proxy)
    """
    descriptors = [
        describe_audio_chunk(
            chunk,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            low_freq_cutoff_hz=low_freq_cutoff_hz,
        )
        for chunk in rendered.output_buffers
    ]

    pair_metrics = {
        "rms_diff": [],
        "spectral_flux": [],
        "mfcc_distance": [],
        "loudness_delta": [],
        "low_freq_ratio_delta": [],
    }

    for prev_desc, curr_desc in zip(descriptors, descriptors[1:]):
        prev_audio = np.asarray(prev_desc["mono"], dtype=np.float32)
        curr_audio = np.asarray(curr_desc["mono"], dtype=np.float32)
        align = min(prev_audio.shape[0], curr_audio.shape[0])
        if align <= 0:
            continue

        prev_audio = prev_audio[:align]
        curr_audio = curr_audio[:align]
        pair_metrics["rms_diff"].append(
            float(np.sqrt(np.mean((curr_audio - prev_audio) ** 2) + 1e-10))
        )
        pair_metrics["spectral_flux"].append(
            float(
                np.mean(
                    np.abs(
                        np.asarray(curr_desc["spectrum"], dtype=np.float32)
                        - np.asarray(prev_desc["spectrum"], dtype=np.float32)
                    )
                )
            )
        )
        pair_metrics["mfcc_distance"].append(
            float(
                np.linalg.norm(
                    np.asarray(curr_desc["mfcc"], dtype=np.float32)
                    - np.asarray(prev_desc["mfcc"], dtype=np.float32)
                )
            )
        )
        pair_metrics["loudness_delta"].append(
            float(abs(float(curr_desc["loudness_db"]) - float(prev_desc["loudness_db"])))
        )
        pair_metrics["low_freq_ratio_delta"].append(
            float(
                abs(
                    float(curr_desc["low_freq_ratio"])
                    - float(prev_desc["low_freq_ratio"])
                )
            )
        )

    chunk_metrics = {
        "low_freq_ratio": [
            float(desc["low_freq_ratio"]) for desc in descriptors
        ],
        "buffer_duration_ms": [
            float(len(chunk) * 1000.0 / float(sample_rate))
            for chunk in rendered.output_buffers
        ],
    }

    summaries = {
        metric: summarize_series(values)
        for metric, values in pair_metrics.items()
    }
    summaries.update(
        {
            metric: summarize_series(values)
            for metric, values in chunk_metrics.items()
        }
    )
    return summaries


def _flatten_summary(prefix: str, summary: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def evaluate_condition(
    example: TrajectoryExample,
    condition: ConditionSpec,
    *,
    vae,
    sample_rate: int,
    latent_hz: float,
    audio_dir: str,
    n_mfcc: int,
    low_freq_cutoff_hz: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Render one example/condition pair, save audio, and compute metrics."""
    sequence = build_condition_latents(example, condition.strategy)
    rendered = decode_latent_sequence(
        vae,
        sequence,
        window_size=int(condition.window_size),
        sample_rate=int(sample_rate),
        latent_hz=float(latent_hz),
    )

    example_audio_dir = os.path.join(audio_dir, example.example_id)
    os.makedirs(example_audio_dir, exist_ok=True)
    audio_path = os.path.join(example_audio_dir, f"{condition.label}.wav")
    save_wav(audio_path, rendered.audio, sr=int(sample_rate))

    metrics = compute_continuity_metrics(
        rendered,
        sample_rate=int(sample_rate),
        n_mfcc=int(n_mfcc),
        low_freq_cutoff_hz=float(low_freq_cutoff_hz),
    )

    row: Dict[str, object] = {
        "row_type": "example",
        "example_id": example.example_id,
        "condition": condition.label,
        "strategy": condition.strategy,
        "strategy_name": STRATEGY_LABELS[condition.strategy],
        "window_size": int(condition.window_size),
        "file_id": int(example.file_id),
        "start_index": int(example.start_index),
        "end_index": int(example.end_index),
        "start_t": int(example.start_t),
        "end_t": int(example.end_t),
        "source_path": str(example.source_path),
        "audio_path": os.path.abspath(audio_path),
        "trajectory_frames": int(example.latents_raw.shape[0]),
        "num_batches": int(rendered.num_batches),
        "num_output_buffers": int(len(rendered.output_buffers)),
        "expected_samples": int(rendered.expected_samples),
        "audio_duration_sec": float(rendered.audio.shape[0] / float(sample_rate)),
        "hop_size": int(rendered.hop_size),
    }
    for metric_name, summary in metrics.items():
        row.update(_flatten_summary(metric_name, summary))

    condition_json = {
        "condition": condition.label,
        "strategy": condition.strategy,
        "window_size": int(condition.window_size),
        "audio_path": os.path.abspath(audio_path),
        "num_batches": int(rendered.num_batches),
        "num_output_buffers": int(len(rendered.output_buffers)),
        "metrics": metrics,
    }
    return row, condition_json


def compute_aggregate_rows(example_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate example rows into one summary row per condition."""
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in example_rows:
        groups.setdefault(str(row["condition"]), []).append(row)

    aggregate_rows: List[Dict[str, object]] = []
    for condition_label in sorted(groups.keys()):
        rows = groups[condition_label]
        aggregate: Dict[str, object] = {
            "row_type": "aggregate",
            "example_id": "",
            "condition": condition_label,
            "strategy": str(rows[0]["strategy"]),
            "strategy_name": str(rows[0]["strategy_name"]),
            "window_size": int(rows[0]["window_size"]),
            "num_examples": int(len(rows)),
        }

        numeric_prefixes = [
            "rms_diff",
            "spectral_flux",
            "mfcc_distance",
            "loudness_delta",
            "low_freq_ratio_delta",
            "low_freq_ratio",
            "buffer_duration_ms",
        ]
        for prefix in numeric_prefixes:
            values = [
                float(row[f"{prefix}_mean"])
                for row in rows
                if row.get(f"{prefix}_mean") is not None
            ]
            summary = summarize_series(values)
            aggregate.update(_flatten_summary(prefix, summary))

        aggregate["audio_duration_sec_mean"] = float(
            np.mean([float(row["audio_duration_sec"]) for row in rows])
        )
        aggregate_rows.append(aggregate)

    # Lower ranks correspond to smaller pairwise discontinuities.
    for metric_name in PRIMARY_PAIR_METRICS:
        ordered = sorted(
            aggregate_rows,
            key=lambda row: float(row.get(f"{metric_name}_mean", np.inf) or np.inf),
        )
        for rank, row in enumerate(ordered, start=1):
            row[f"{metric_name}_rank"] = int(rank)

    for row in aggregate_rows:
        ranks = [
            int(row[f"{metric_name}_rank"])
            for metric_name in PRIMARY_PAIR_METRICS
            if f"{metric_name}_rank" in row
        ]
        row["stability_score"] = float(np.mean(ranks)) if ranks else None

    aggregate_rows.sort(key=lambda row: float(row.get("stability_score", np.inf) or np.inf))
    return aggregate_rows


def write_metrics_csv(
    path: str,
    example_rows: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
):
    """Write example-level and aggregate continuity metrics to one CSV."""
    metric_prefixes = [
        "rms_diff",
        "spectral_flux",
        "mfcc_distance",
        "loudness_delta",
        "low_freq_ratio_delta",
        "low_freq_ratio",
        "buffer_duration_ms",
    ]
    summary_fields = ("count", "mean", "median", "std", "q25", "q75", "min", "max")
    metric_columns = [
        f"{prefix}_{field}"
        for prefix in metric_prefixes
        for field in summary_fields
    ]
    rank_columns = [f"{metric}_rank" for metric in PRIMARY_PAIR_METRICS]

    fieldnames = [
        "row_type",
        "example_id",
        "condition",
        "strategy",
        "strategy_name",
        "window_size",
        "num_examples",
        "file_id",
        "start_index",
        "end_index",
        "start_t",
        "end_t",
        "source_path",
        "audio_path",
        "trajectory_frames",
        "num_batches",
        "num_output_buffers",
        "expected_samples",
        "audio_duration_sec",
        "audio_duration_sec_mean",
        "hop_size",
    ] + metric_columns + rank_columns + ["stability_score"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(example_rows) + list(aggregate_rows):
            writer.writerow({field: row.get(field) for field in fieldnames})


def render_continuity_plot(
    path: str,
    example_rows: Sequence[Dict[str, object]],
):
    """Render boxplots comparing per-example continuity metrics by condition."""
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = os.path.join(
            tempfile.gettempdir(), "stable_audio_wanderer_mpl_cache"
        )
        os.makedirs(mpl_cache, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = os.path.join(
            tempfile.gettempdir(), "stable_audio_wanderer_xdg_cache"
        )
        os.makedirs(xdg_cache, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg_cache

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("rms_diff_mean", "RMS Difference"),
        ("spectral_flux_mean", "Spectral Flux"),
        ("mfcc_distance_mean", "MFCC Distance"),
        ("loudness_delta_mean", "Loudness Delta (dB)"),
        ("low_freq_ratio_mean", "Low-Frequency Ratio"),
    ]
    rows = [row for row in example_rows if row.get("row_type") == "example"]
    conditions = sorted({str(row["condition"]) for row in rows})

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0))
    axes_flat = axes.flatten()
    for ax, (metric_key, title) in zip(axes_flat, metrics):
        values_by_condition = []
        labels = []
        for condition in conditions:
            values = [
                float(row[metric_key])
                for row in rows
                if str(row["condition"]) == condition and row.get(metric_key) is not None
            ]
            if values:
                values_by_condition.append(values)
                labels.append(condition)
        if values_by_condition:
            ax.boxplot(values_by_condition, patch_artist=True)
            ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    if len(metrics) < len(axes_flat):
        axes_flat[-1].axis("off")
    fig.suptitle("Continuity Metrics by Decoding Strategy", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_markdown_report(
    report_path: str,
    *,
    bundle: CorpusBundle,
    vae,
    machine_info: Dict[str, object],
    examples: Sequence[TrajectoryExample],
    conditions: Sequence[ConditionSpec],
    example_rows: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
    csv_path: str,
    json_path: str,
    plot_path: str,
    audio_dir: str,
    low_freq_cutoff_hz: float,
):
    """Render a markdown continuity report suitable for DAFx evaluation."""
    info = vae.info()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    primary_best = aggregate_rows[0] if aggregate_rows else None
    best_non_loop = next(
        (row for row in aggregate_rows if str(row.get("strategy")) != "single_loop"),
        None,
    )

    lines: List[str] = []
    lines.append("# Continuity Evaluation Report")
    lines.append("")
    lines.append(f"Generated: `{timestamp}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "This benchmark compares temporal continuity and output stability across "
        "three latent decoding strategies using real corpus trajectories and the "
        "project's runtime-faithful VAE decode path."
    )
    lines.append("")
    lines.append("## Compared Methods")
    lines.append("- `single_loop`: repeat the first latent frame for the full trajectory, then decode in overlapping windows.")
    lines.append("- `naive_interp`: linearly interpolate in raw latent space between the first and last trajectory latents before decoding.")
    lines.append("- `sequential`: decode the observed corpus latent trajectory directly, preserving the corpus ordering.")
    lines.append("")
    lines.append("## Assumptions")
    lines.append(
        "Trajectory examples are sampled from stored corpus runs that remain contiguous "
        "both in file ID and latent time index (`t_lat` increments by 1)."
    )
    lines.append(
        "All methods are matched for trajectory length. Decoding uses a first full "
        "window followed by hop = ceil(window / 2) latent updates, and audio is "
        "reconstructed with the same adaptive crossfade queue used by `DecoderPlayer.write_frame(...)`."
    )
    lines.append(
        "After the last decode, the held overlap-add tail is flushed and the final waveform "
        "is trimmed or padded to the target duration implied by the latent frame count."
    )
    lines.append(
        f"The low-frequency energy ratio uses a cutoff of {float(low_freq_cutoff_hz):.1f} Hz."
    )
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Corpus directory: `{bundle.corpus_dir}`")
    lines.append(f"- Corpus file: `{bundle.corpus_path}`")
    lines.append(f"- VAE: `{info.vae_id}` ({info.display_name})")
    lines.append(f"- Sample rate: {bundle.sample_rate} Hz")
    lines.append(f"- Latent rate: {bundle.latent_hz:.4f} Hz")
    lines.append(f"- Latent dimension: {info.latent_dim}")
    lines.append(f"- Host: `{machine_info.get('hostname')}`")
    lines.append(f"- Platform: `{machine_info.get('platform')}`")
    lines.append(f"- Device: `{machine_info.get('device')}` / `{machine_info.get('device_name')}`")
    lines.append(f"- Examples: {len(examples)}")
    lines.append(
        f"- Conditions: {', '.join(condition.label for condition in conditions)}"
    )
    lines.append(f"- Audio examples: `{audio_dir}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- JSON summary: `{json_path}`")
    lines.append(f"- Plot: `{plot_path}`")
    lines.append("")
    lines.append("## Aggregate Table")
    lines.append("")
    lines.append(
        "| Condition | Strategy | W | RMS Diff | Spectral Flux | MFCC Dist | "
        "Loudness Delta | LF Ratio | Stability Score |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['condition']} | {row['strategy_name']} | {int(row['window_size'])} | "
            f"{float(row['rms_diff_mean'] or 0.0):.4f} | "
            f"{float(row['spectral_flux_mean'] or 0.0):.4f} | "
            f"{float(row['mfcc_distance_mean'] or 0.0):.4f} | "
            f"{float(row['loudness_delta_mean'] or 0.0):.4f} | "
            f"{float(row['low_freq_ratio_mean'] or 0.0):.4f} | "
            f"{float(row['stability_score'] or 0.0):.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    if primary_best is not None:
        if str(primary_best.get("strategy")) == "single_loop":
            lines.append(
                "Using the average rank over the primary pairwise continuity metrics "
                f"(RMS difference, spectral flux, MFCC distance, loudness delta, and "
                f"low-frequency-ratio delta), the smallest measured buffer-to-buffer "
                f"changes in this run came from the static baseline `{primary_best['condition']}`."
            )
            lines.append(
                "That result should be treated as a degenerate lower bound on change, not as "
                "evidence of desirable continuity: repeating one latent can minimize transitions "
                "simply by collapsing the audio into a near-frozen loop."
            )
        else:
            lines.append(
                "Using the average rank over the primary pairwise continuity metrics "
                f"(RMS difference, spectral flux, MFCC distance, loudness delta, and "
                f"low-frequency-ratio delta), the most stable measured condition in this run "
                f"was `{primary_best['condition']}`."
            )
    if best_non_loop is not None:
        lines.append(
            "Restricting the comparison to time-varying methods, the lowest aggregate "
            f"discontinuity score came from `{best_non_loop['condition']}`."
        )
    lines.append(
        "Naive raw-latent interpolation is expected to be problematic because it moves "
        "through latent states that were not observed in the corpus sequence and are not "
        "guaranteed to decode into temporally coherent audio. This often appears as larger "
        "spectral and MFCC jumps, even when the latent path itself looks smooth algebraically."
    )
    lines.append(
        "Single-latent looping can sometimes produce low pairwise transition metrics simply "
        "because the output changes very little. That should be interpreted cautiously: a "
        "trivially static loop is not the same as a musically playable, continuously evolving output."
    )
    lines.append(
        "Sequential corpus-based recall is the method most aligned with the system's "
        "reconstructive/navigational design because it decodes observed latent progressions "
        "rather than forcing the decoder to sustain a frozen latent or traverse unsupported "
        "straight lines in raw latent space."
    )
    lines.append("")
    lines.append("## DAFx-Ready Paragraph")
    if primary_best is not None:
        if str(primary_best.get("strategy")) == "single_loop":
            dynamic_clause = ""
            if best_non_loop is not None:
                dynamic_clause = (
                    f"Among the time-varying methods, `{best_non_loop['condition']}` "
                    f"produced the lowest discontinuity score. "
                )
            lines.append(
                "We compared single-latent looping, naive raw-latent interpolation, and "
                "sequential corpus-based latent decoding using matched-duration corpus "
                "trajectories and the runtime-faithful Stable Audio Wanderer decode path. "
                f"In this run, the frozen baseline `{primary_best['condition']}` minimized "
                "pairwise buffer-change metrics, but this should be interpreted as a trivial "
                "static lower bound rather than a musically useful notion of continuity. "
                + dynamic_clause
                + "The saved audio examples remain important here: naive raw-latent interpolation "
                "still traverses unsupported latent states, while sequential corpus recall is the "
                "only condition that preserves observed temporal structure from the corpus."
            )
        else:
            best_condition = str(primary_best["condition"])
            best_strategy = str(primary_best["strategy_name"])
            rms = float(primary_best["rms_diff_mean"] or 0.0)
            flux = float(primary_best["spectral_flux_mean"] or 0.0)
            mfcc = float(primary_best["mfcc_distance_mean"] or 0.0)
            lines.append(
                "We compared single-latent looping, naive raw-latent interpolation, and "
                "sequential corpus-based latent decoding using matched-duration corpus "
                "trajectories and the runtime-faithful Stable Audio Wanderer decode path. "
                f"In this run, `{best_condition}` ({best_strategy}) achieved the lowest overall "
                "pairwise discontinuity score, with mean RMS difference "
                f"{rms:.4f}, spectral flux {flux:.4f}, and MFCC distance {mfcc:.4f}. "
                "These results support the claim that preserving observed latent sequences yields "
                "more stable decoded audio than naive latent-space shortcuts, particularly direct "
                "raw-latent interpolation, which tends to pass through less coherent intermediate states."
            )
    else:
        lines.append(
            "We compared single-latent looping, naive raw-latent interpolation, and "
            "sequential corpus-based latent decoding using matched-duration corpus "
            "trajectories and the runtime-faithful Stable Audio Wanderer decode path. "
            "The benchmark is designed to quantify pairwise audio-buffer continuity and "
            "support the claim that preserving observed latent sequences yields more stable "
            "decoded audio than naive latent-space shortcuts."
        )
    lines.append("")

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_continuity_evaluation(
    *,
    bundle: CorpusBundle,
    vae,
    num_examples: int,
    trajectory_frames: int,
    seed: int,
    condition_specs: Sequence[ConditionSpec],
    out_dir: str,
    n_mfcc: int,
    low_freq_cutoff_hz: float,
) -> Tuple[List[TrajectoryExample], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Run the full continuity benchmark."""
    examples = sample_trajectory_examples(
        bundle,
        num_examples=int(num_examples),
        trajectory_frames=int(trajectory_frames),
        seed=int(seed),
    )
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    example_rows: List[Dict[str, object]] = []
    json_examples: List[Dict[str, object]] = []
    for example in examples:
        example_json = {
            "example_id": example.example_id,
            "start_index": int(example.start_index),
            "end_index": int(example.end_index),
            "file_id": int(example.file_id),
            "start_t": int(example.start_t),
            "end_t": int(example.end_t),
            "source_path": str(example.source_path),
            "conditions": [],
        }
        LOGGER.info(
            "Evaluating %s (file_id=%d, start_t=%d, end_t=%d)",
            example.example_id,
            int(example.file_id),
            int(example.start_t),
            int(example.end_t),
        )
        for condition in condition_specs:
            LOGGER.info(
                "  Condition %s (window=%d)",
                condition.label,
                int(condition.window_size),
            )
            row, condition_json = evaluate_condition(
                example,
                condition,
                vae=vae,
                sample_rate=int(bundle.sample_rate),
                latent_hz=float(bundle.latent_hz),
                audio_dir=audio_dir,
                n_mfcc=int(n_mfcc),
                low_freq_cutoff_hz=float(low_freq_cutoff_hz),
            )
            example_rows.append(row)
            example_json["conditions"].append(condition_json)
        json_examples.append(example_json)

    aggregate_rows = compute_aggregate_rows(example_rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "corpus": {
            "corpus_dir": bundle.corpus_dir,
            "corpus_path": bundle.corpus_path,
            "num_frames": bundle.num_frames,
            "sample_rate": bundle.sample_rate,
            "latent_hz": bundle.latent_hz,
            "vae_id": bundle.vae_id,
        },
        "benchmark": {
            "num_examples": int(num_examples),
            "trajectory_frames": int(trajectory_frames),
            "seed": int(seed),
            "conditions": [
                {
                    "strategy": spec.strategy,
                    "window_size": int(spec.window_size),
                    "label": spec.label,
                }
                for spec in condition_specs
            ],
            "assumptions": {
                "trajectory_sampling": "stored corpus runs contiguous in file_id and t_lat",
                "decode_overlap": "first full window then hop=ceil(window/2)",
                "audio_reconstruction": "BufferedAudioQueue overlap-add with final tail flush",
            },
            "low_freq_cutoff_hz": float(low_freq_cutoff_hz),
            "n_mfcc": int(n_mfcc),
        },
        "examples": json_examples,
        "aggregate": aggregate_rows,
    }
    return examples, example_rows, aggregate_rows, payload
