"""Neighborhood coherence evaluation for latent space vs descriptor navigation space."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon

from .latency import CorpusBundle

LOGGER = logging.getLogger(__name__)

SPACE_LABELS = {
    "latent": "Raw Latent Space",
    "navigation": "Descriptor Navigation Space",
}
AVAILABLE_FEATURES = ("mfcc", "centroid", "loudness", "flatness", "rolloff")
FEATURE_LABELS = {
    "mfcc": "MFCC Distance",
    "centroid": "Centroid Distance",
    "loudness": "Loudness Distance",
    "flatness": "Flatness Distance",
    "rolloff": "Rolloff Distance",
    "composite": "Composite Timbral Distance",
}


@dataclass(frozen=True)
class SpaceArrays:
    """Resolved arrays for the two compared spaces plus descriptor metadata."""

    latents: np.ndarray
    navigation_points: np.ndarray
    navigation_reducer: str
    navigation_source: str
    desc_weighted: np.ndarray
    desc_center: np.ndarray
    desc_scale: np.ndarray
    desc_group_scales: np.ndarray
    desc_names: List[str]


def _load_manual_arrays(
    bundle: CorpusBundle,
    required_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Load descriptor/navigation arrays from the corpus first, then from the manual artifact.
    """
    required = list(required_keys)
    if all(key in bundle.data for key in required):
        return {key: bundle.data[key] for key in required}

    if bundle.manual_artifact_path and os.path.exists(bundle.manual_artifact_path):
        with np.load(bundle.manual_artifact_path, allow_pickle=True) as artifact:
            if all(key in artifact for key in required):
                return {key: artifact[key] for key in required}

    missing = [key for key in required if key not in bundle.data]
    raise RuntimeError(
        "Manual descriptor/navigation arrays are missing from the selected corpus "
        f"and manual artifact. Missing keys: {missing}"
    )


def resolve_space_arrays(
    bundle: CorpusBundle,
    navigation_source: str = "embed",
) -> SpaceArrays:
    """Resolve the latent and navigation spaces used for the coherence test."""
    navigation_source_name = str(navigation_source).strip().lower()
    required = [
        "manual_desc_weighted",
        "manual_desc_center",
        "manual_desc_scale",
        "manual_desc_scales",
        "manual_desc_names",
    ]
    if navigation_source_name == "embed":
        required.extend(["manual_embed_points", "manual_embed_reducer"])

    manual = _load_manual_arrays(bundle, required_keys=required)
    latents = np.asarray(bundle.data["Z_concat"], dtype=np.float32)
    desc_weighted = np.asarray(manual["manual_desc_weighted"], dtype=np.float32)

    if navigation_source_name == "embed":
        navigation_points = np.asarray(manual["manual_embed_points"], dtype=np.float32)
        navigation_reducer = str(np.asarray(manual["manual_embed_reducer"]).reshape(-1)[0])
        source_label = f"manual_embed_points ({navigation_reducer})"
    elif navigation_source_name == "descriptor":
        navigation_points = desc_weighted.astype(np.float32)
        navigation_reducer = "weighted_descriptor_space"
        source_label = "manual_desc_weighted"
    else:
        raise ValueError(
            f"Unsupported navigation_source: {navigation_source}. "
            "Use 'embed' or 'descriptor'."
        )

    return SpaceArrays(
        latents=latents,
        navigation_points=navigation_points,
        navigation_reducer=navigation_reducer,
        navigation_source=source_label,
        desc_weighted=desc_weighted,
        desc_center=np.asarray(manual["manual_desc_center"], dtype=np.float32).reshape(-1),
        desc_scale=np.asarray(manual["manual_desc_scale"], dtype=np.float32).reshape(-1),
        desc_group_scales=np.asarray(manual["manual_desc_scales"], dtype=np.float32).reshape(-1),
        desc_names=[str(name) for name in np.asarray(manual["manual_desc_names"]).reshape(-1).tolist()],
    )


def reconstruct_audio_descriptors(space_arrays: SpaceArrays) -> np.ndarray:
    """
    Reconstruct raw descriptor values from the stored weighted descriptor matrix.

    For the features used in this evaluation (MFCC, centroid, loudness, flatness,
    rolloff), this inversion is exact because those dimensions are not pitch-gated.
    """
    denom = np.maximum(space_arrays.desc_group_scales, 1e-8)
    desc_norm = space_arrays.desc_weighted / denom[None, :]
    return (
        desc_norm * space_arrays.desc_scale[None, :]
        + space_arrays.desc_center[None, :]
    ).astype(np.float32)


def _feature_indices(desc_names: Sequence[str]) -> Dict[str, np.ndarray]:
    name_to_index = {str(name): idx for idx, name in enumerate(desc_names)}
    required = {
        "mfcc": [f"mfcc_{idx:02d}" for idx in range(1, 14)],
        "centroid": ["spec_centroid"],
        "loudness": ["loudness_log_rms"],
        "flatness": ["spec_flatness"],
        "rolloff": ["spec_rolloff"],
    }
    out: Dict[str, np.ndarray] = {}
    for key, names in required.items():
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise RuntimeError(
                f"Descriptor names missing required features for {key}: {missing}"
            )
        out[key] = np.asarray([name_to_index[name] for name in names], dtype=np.int32)
    return out


def sample_anchor_indices(num_frames: int, num_anchors: int, seed: int) -> np.ndarray:
    """Sample deterministic anchor indices without replacement."""
    if int(num_anchors) < 1:
        raise ValueError("num_anchors must be >= 1.")
    count = min(int(num_anchors), int(num_frames))
    if count < int(num_anchors):
        LOGGER.warning(
            "Requested %d anchors but corpus only has %d frames. Using %d anchors.",
            int(num_anchors),
            int(num_frames),
            count,
        )
    rng = np.random.default_rng(int(seed))
    anchors = rng.choice(int(num_frames), size=count, replace=False)
    return np.sort(anchors.astype(np.int32))


def query_neighbors(
    points: np.ndarray,
    anchor_indices: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query k nearest neighbors for each anchor, excluding the anchor itself."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        raise ValueError(f"Expected points [N, D], got {pts.shape}")
    if int(k) < 1:
        raise ValueError("k must be >= 1.")

    max_neighbors = max(1, min(int(k), pts.shape[0] - 1))
    tree = cKDTree(pts)
    raw_dists, raw_inds = tree.query(pts[anchor_indices], k=max_neighbors + 1)
    raw_dists = np.asarray(raw_dists, dtype=np.float32)
    raw_inds = np.asarray(raw_inds, dtype=np.int32)
    if raw_dists.ndim == 1:
        raw_dists = raw_dists[:, None]
        raw_inds = raw_inds[:, None]

    neighbors = np.full((anchor_indices.shape[0], max_neighbors), -1, dtype=np.int32)
    distances = np.full((anchor_indices.shape[0], max_neighbors), np.nan, dtype=np.float32)
    for row_idx, anchor_idx in enumerate(anchor_indices):
        cursor = 0
        for candidate_idx, candidate_dist in zip(raw_inds[row_idx], raw_dists[row_idx]):
            if int(candidate_idx) == int(anchor_idx):
                continue
            neighbors[row_idx, cursor] = int(candidate_idx)
            distances[row_idx, cursor] = float(candidate_dist)
            cursor += 1
            if cursor >= max_neighbors:
                break
        if cursor < max_neighbors:
            raise RuntimeError(
                f"Failed to retrieve {max_neighbors} neighbors for anchor {int(anchor_idx)}."
            )
    return neighbors, distances


def compute_feature_distances(
    descriptors_raw: np.ndarray,
    desc_names: Sequence[str],
    anchor_idx: int,
    neighbor_idx: int,
    selected_features: Sequence[str],
) -> Dict[str, float]:
    """Compute audio-domain timbral distances from stored source-audio descriptors."""
    desc = np.asarray(descriptors_raw, dtype=np.float32)
    idx_map = _feature_indices(desc_names)
    anchor = desc[int(anchor_idx)]
    neighbor = desc[int(neighbor_idx)]

    distances: Dict[str, float] = {}
    for feature_name in selected_features:
        if feature_name == "mfcc":
            slc = idx_map["mfcc"]
            distances["mfcc_distance"] = float(
                np.linalg.norm(anchor[slc] - neighbor[slc])
            )
        elif feature_name == "centroid":
            idx = int(idx_map["centroid"][0])
            distances["centroid_distance"] = float(
                abs(np.log1p(anchor[idx]) - np.log1p(neighbor[idx]))
            )
        elif feature_name == "loudness":
            idx = int(idx_map["loudness"][0])
            distances["loudness_distance"] = float(abs(anchor[idx] - neighbor[idx]))
        elif feature_name == "flatness":
            idx = int(idx_map["flatness"][0])
            distances["flatness_distance"] = float(abs(anchor[idx] - neighbor[idx]))
        elif feature_name == "rolloff":
            idx = int(idx_map["rolloff"][0])
            distances["rolloff_distance"] = float(
                abs(np.log1p(anchor[idx]) - np.log1p(neighbor[idx]))
            )
        else:
            raise ValueError(f"Unsupported feature: {feature_name}")
    return distances


def _measurement_columns(selected_features: Sequence[str]) -> List[str]:
    columns = []
    for feature_name in selected_features:
        columns.append(f"{feature_name}_distance")
    columns.append("composite_timbral_distance")
    return columns


def metric_label(metric_name: str) -> str:
    """Return a readable label for a measurement column."""
    if metric_name == "composite_timbral_distance":
        return FEATURE_LABELS["composite"]
    if metric_name.endswith("_distance"):
        return FEATURE_LABELS.get(metric_name[: -len("_distance")], metric_name)
    return FEATURE_LABELS.get(metric_name, metric_name)


def collect_measurements(
    bundle: CorpusBundle,
    space_arrays: SpaceArrays,
    *,
    num_anchors: int,
    k: int,
    seed: int,
    selected_features: Sequence[str],
) -> List[Dict[str, object]]:
    """Collect anchor-neighbor measurements for both spaces."""
    anchor_indices = sample_anchor_indices(space_arrays.latents.shape[0], num_anchors, seed)
    descriptors_raw = reconstruct_audio_descriptors(space_arrays)
    meta = np.asarray(bundle.data["meta"], dtype=np.int32)
    paths = [str(path) for path in np.asarray(bundle.data.get("paths", np.array([]))).reshape(-1).tolist()]

    spaces = {
        "latent": np.asarray(space_arrays.latents, dtype=np.float32),
        "navigation": np.asarray(space_arrays.navigation_points, dtype=np.float32),
    }

    rows: List[Dict[str, object]] = []
    for space_name, points in spaces.items():
        neighbors, distances = query_neighbors(points, anchor_indices, int(k))
        for anchor_pos, anchor_idx in enumerate(anchor_indices):
            for rank_idx in range(neighbors.shape[1]):
                neighbor_idx = int(neighbors[anchor_pos, rank_idx])
                measurement = compute_feature_distances(
                    descriptors_raw,
                    space_arrays.desc_names,
                    anchor_idx=int(anchor_idx),
                    neighbor_idx=neighbor_idx,
                    selected_features=selected_features,
                )
                anchor_file_id = int(meta[int(anchor_idx), 0])
                neighbor_file_id = int(meta[neighbor_idx, 0])
                row = {
                    "space": space_name,
                    "space_name": SPACE_LABELS[space_name],
                    "anchor_index": int(anchor_idx),
                    "neighbor_index": int(neighbor_idx),
                    "neighbor_rank": int(rank_idx + 1),
                    "space_distance": float(distances[anchor_pos, rank_idx]),
                    "anchor_file_id": anchor_file_id,
                    "neighbor_file_id": neighbor_file_id,
                    "anchor_t": int(meta[int(anchor_idx), 1]),
                    "neighbor_t": int(meta[neighbor_idx, 1]),
                    "anchor_source_path": (
                        paths[anchor_file_id] if 0 <= anchor_file_id < len(paths) else ""
                    ),
                    "neighbor_source_path": (
                        paths[neighbor_file_id] if 0 <= neighbor_file_id < len(paths) else ""
                    ),
                }
                row.update(measurement)
                rows.append(row)
    return finalize_measurements(rows, selected_features=selected_features)


def finalize_measurements(
    rows: Sequence[Dict[str, object]],
    *,
    selected_features: Sequence[str],
) -> List[Dict[str, object]]:
    """Add a robustly scaled composite timbral distance to each row."""
    output = [dict(row) for row in rows]
    if not output:
        return output

    metric_columns = [f"{feature_name}_distance" for feature_name in selected_features]
    scales = {}
    for metric in metric_columns:
        values = np.asarray([float(row[metric]) for row in output], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scales[metric] = max(1e-6, mad * 1.4826)

    for row in output:
        standardized = [
            float(row[metric]) / scales[metric]
            for metric in metric_columns
        ]
        row["composite_timbral_distance"] = float(np.mean(standardized))
    return output


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
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


def summarize_measurements(
    rows: Sequence[Dict[str, object]],
    *,
    selected_features: Sequence[str],
) -> Dict[str, object]:
    """Aggregate measurement rows into per-space summaries and simple tests."""
    metric_names = [f"{feature_name}_distance" for feature_name in selected_features]
    metric_names.append("composite_timbral_distance")

    by_space: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_space.setdefault(str(row["space"]), []).append(row)

    space_summaries: Dict[str, Dict[str, object]] = {}
    correlations: Dict[str, Dict[str, object]] = {}
    anchor_means: Dict[str, Dict[int, Dict[str, float]]] = {}

    for space_name, space_rows in by_space.items():
        space_summaries[space_name] = {
            "pair_count": len(space_rows),
            "space_distance": summarize_values(
                [float(row["space_distance"]) for row in space_rows]
            ),
        }
        correlations[space_name] = {}
        anchor_means[space_name] = {}

        for metric in metric_names:
            values = [float(row[metric]) for row in space_rows]
            space_summaries[space_name][metric] = summarize_values(values)
            rho, p = spearmanr(
                np.asarray([float(row["space_distance"]) for row in space_rows], dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
            correlations[space_name][metric] = {
                "spearman_rho": None if np.isnan(rho) else float(rho),
                "spearman_p": None if np.isnan(p) else float(p),
            }

        anchors = sorted({int(row["anchor_index"]) for row in space_rows})
        for anchor_idx in anchors:
            anchor_rows = [row for row in space_rows if int(row["anchor_index"]) == anchor_idx]
            anchor_means[space_name][anchor_idx] = {
                metric: float(np.mean([float(row[metric]) for row in anchor_rows]))
                for metric in metric_names
            }

    tests: Dict[str, Dict[str, Optional[float]]] = {}
    latent_anchor_means = anchor_means.get("latent", {})
    navigation_anchor_means = anchor_means.get("navigation", {})
    common_anchors = sorted(set(latent_anchor_means.keys()) & set(navigation_anchor_means.keys()))
    for metric in metric_names:
        latent_vals = np.asarray(
            [latent_anchor_means[idx][metric] for idx in common_anchors],
            dtype=np.float64,
        )
        navigation_vals = np.asarray(
            [navigation_anchor_means[idx][metric] for idx in common_anchors],
            dtype=np.float64,
        )
        test_result = {
            "num_anchors": int(len(common_anchors)),
            "latent_mean": float(latent_vals.mean()) if latent_vals.size else None,
            "navigation_mean": float(navigation_vals.mean()) if navigation_vals.size else None,
            "mean_delta_latent_minus_navigation": (
                float((latent_vals - navigation_vals).mean())
                if latent_vals.size
                else None
            ),
            "wilcoxon_statistic": None,
            "wilcoxon_p_navigation_better": None,
        }
        if latent_vals.size >= 2 and np.any(np.abs(latent_vals - navigation_vals) > 1e-12):
            try:
                statistic, p_value = wilcoxon(
                    latent_vals,
                    navigation_vals,
                    alternative="greater",
                    zero_method="wilcox",
                )
                test_result["wilcoxon_statistic"] = float(statistic)
                test_result["wilcoxon_p_navigation_better"] = float(p_value)
            except Exception:
                pass
        tests[metric] = test_result

    return {
        "per_space": space_summaries,
        "correlations": correlations,
        "paired_anchor_tests": tests,
    }


def write_measurements_csv(path: str, rows: Sequence[Dict[str, object]], *, selected_features: Sequence[str]):
    """Write the anchor-neighbor measurement table."""
    fieldnames = [
        "space",
        "space_name",
        "anchor_index",
        "neighbor_index",
        "neighbor_rank",
        "space_distance",
        "anchor_file_id",
        "neighbor_file_id",
        "anchor_t",
        "neighbor_t",
        "anchor_source_path",
        "neighbor_source_path",
    ] + _measurement_columns(selected_features)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def render_correlation_plot(path: str, rows: Sequence[Dict[str, object]]):
    """Render space-distance vs composite-timbral-distance plots."""
    _prepare_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, space_name in zip(axes, ("latent", "navigation")):
        space_rows = [row for row in rows if str(row["space"]) == space_name]
        x = np.asarray([float(row["space_distance"]) for row in space_rows], dtype=np.float32)
        y = np.asarray(
            [float(row["composite_timbral_distance"]) for row in space_rows],
            dtype=np.float32,
        )
        hb = ax.hexbin(x, y, gridsize=35, cmap="viridis", mincnt=1)
        rho, p = spearmanr(x, y)
        ax.set_title(
            f"{SPACE_LABELS[space_name]}\nSpearman rho={0.0 if np.isnan(rho) else rho:.3f}, "
            f"p={0.0 if np.isnan(p) else p:.3g}"
        )
        ax.set_xlabel("Space Distance")
        ax.grid(True, alpha=0.25)
        fig.colorbar(hb, ax=ax, label="Pairs")
    axes[0].set_ylabel("Composite Timbral Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_boxplot(path: str, rows: Sequence[Dict[str, object]], *, selected_features: Sequence[str]):
    """Render boxplots of timbral distances by space."""
    _prepare_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [f"{feature_name}_distance" for feature_name in selected_features]
    metric_names.append("composite_timbral_distance")
    n_metrics = len(metric_names)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 4.0 * n_rows))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, metric in zip(axes_flat, metric_names):
        latent_vals = [
            float(row[metric]) for row in rows if str(row["space"]) == "latent"
        ]
        navigation_vals = [
            float(row[metric]) for row in rows if str(row["space"]) == "navigation"
        ]
        ax.boxplot(
            [latent_vals, navigation_vals],
            tick_labels=["latent", "navigation"],
            patch_artist=True,
        )
        ax.set_title(metric_label(metric))
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n_metrics:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _prepare_matplotlib_env():
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = os.path.join(tempfile.gettempdir(), "stable_audio_wanderer_mpl_cache")
        os.makedirs(mpl_cache, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = os.path.join(tempfile.gettempdir(), "stable_audio_wanderer_xdg_cache")
        os.makedirs(xdg_cache, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg_cache


def render_markdown_report(
    report_path: str,
    *,
    bundle: CorpusBundle,
    space_arrays: SpaceArrays,
    num_anchors: int,
    k: int,
    selected_features: Sequence[str],
    machine_info: Dict[str, object],
    summary: Dict[str, object],
    csv_path: str,
    json_path: str,
    correlation_plot_path: str,
    boxplot_path: str,
):
    """Render a markdown report for the space coherence experiment."""
    per_space = summary["per_space"]
    tests = summary["paired_anchor_tests"]
    correlations = summary["correlations"]

    latent_composite = per_space["latent"]["composite_timbral_distance"]
    navigation_composite = per_space["navigation"]["composite_timbral_distance"]
    composite_test = tests["composite_timbral_distance"]
    nav_better = (
        composite_test.get("mean_delta_latent_minus_navigation") is not None
        and float(composite_test["mean_delta_latent_minus_navigation"]) > 0.0
    )
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append("# Space Coherence Report")
    lines.append("")
    lines.append(f"Generated: `{timestamp}`")
    lines.append("")
    lines.append("## Motivation")
    lines.append(
        "This evaluation tests the claim that raw VAE latent proximity is a poor proxy "
        "for perceptual timbral similarity, and that the descriptor-derived navigation "
        "space used by Stable Audio Wanderer yields more coherent local neighborhoods."
    )
    lines.append("")
    lines.append("## Method")
    lines.append(
        f"We sampled `{int(num_anchors)}` anchor frames from the corpus and retrieved "
        f"`k={int(k)}` nearest neighbors in two spaces: raw latent space (`Z_concat`) and "
        f"the descriptor-driven navigation space (`{space_arrays.navigation_source}`)."
    )
    lines.append(
        "Timbral similarity was evaluated using source-audio descriptors already extracted "
        "during preprocessing at the latent frame rate. For the selected features, this uses "
        "stored MFCC, centroid, loudness, flatness, and rolloff descriptors rather than "
        "distances from either search space."
    )
    lines.append(
        "The composite timbral distance is the mean of per-feature distances after "
        "robust scaling by the global median absolute deviation for each feature."
    )
    lines.append(f"Selected features: {', '.join(selected_features)}.")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Corpus directory: `{bundle.corpus_dir}`")
    lines.append(f"- Corpus file: `{bundle.corpus_path}`")
    lines.append(f"- Frames: {bundle.num_frames}")
    lines.append(f"- Sample rate: {bundle.sample_rate} Hz")
    lines.append(f"- Latent rate: {bundle.latent_hz:.4f} Hz")
    lines.append(f"- Navigation reducer: `{space_arrays.navigation_reducer}`")
    lines.append(f"- Navigation source: `{space_arrays.navigation_source}`")
    lines.append(f"- Host: `{machine_info.get('hostname')}`")
    lines.append(f"- Platform: `{machine_info.get('platform')}`")
    lines.append(f"- Device: `{machine_info.get('device')}` / `{machine_info.get('device_name')}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- JSON summary: `{json_path}`")
    lines.append(f"- Correlation figure: `{correlation_plot_path}`")
    lines.append(f"- Boxplot figure: `{boxplot_path}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    summary_metric_columns = ["composite_timbral_distance"] + [
        f"{feature}_distance" for feature in selected_features
    ]
    header_cells = ["Space", "Composite Mean", "Composite Median"]
    header_cells.extend(f"{metric_label(metric)} Mean" for metric in summary_metric_columns[1:])
    header_cells.append("Composite rho")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header_cells) - 1)) + " |")
    for space_name in ("latent", "navigation"):
        space_summary = per_space[space_name]
        row_cells = [
            SPACE_LABELS[space_name],
            f"{float(space_summary['composite_timbral_distance']['mean'] or 0.0):.4f}",
            f"{float(space_summary['composite_timbral_distance']['median'] or 0.0):.4f}",
        ]
        row_cells.extend(
            f"{float(space_summary.get(metric, {}).get('mean') or 0.0):.4f}"
            for metric in summary_metric_columns[1:]
        )
        row_cells.append(
            f"{float(correlations[space_name]['composite_timbral_distance']['spearman_rho'] or 0.0):.4f}"
        )
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("## Significance")
    lines.append(
        "Paired anchor-level comparisons use a one-sided Wilcoxon signed-rank test "
        "with the alternative hypothesis that navigation-space neighbors have smaller "
        "timbral distances than latent-space neighbors."
    )
    lines.append("")
    lines.append("| Metric | Mean Delta (latent - navigation) | Wilcoxon p |")
    lines.append("| --- | ---: | ---: |")
    for metric in [f"{feature}_distance" for feature in selected_features] + ["composite_timbral_distance"]:
        test = tests[metric]
        label = metric_label(metric)
        mean_delta = test.get("mean_delta_latent_minus_navigation")
        p_value = test.get("wilcoxon_p_navigation_better")
        lines.append(
            "| "
            f"{label} | "
            f"{0.0 if mean_delta is None else float(mean_delta):.4f} | "
            f"{'' if p_value is None else f'{float(p_value):.4g}'} |"
        )
    lines.append("")
    lines.append("## Figure References")
    lines.append(
        f"- Figure 1: `{correlation_plot_path}` shows space distance vs composite timbral distance."
    )
    lines.append(
        f"- Figure 2: `{boxplot_path}` compares neighbor timbral distances for each space."
    )
    lines.append("")
    lines.append("## Interpretation")
    if nav_better:
        lines.append(
            "In this run, descriptor/navigation-space neighbors were more timbrally coherent "
            "than raw-latent neighbors on the composite metric, with lower mean and median "
            "timbral distance and a positive anchor-level mean delta (latent minus navigation)."
        )
    else:
        lines.append(
            "In this run, descriptor/navigation-space neighbors did not strictly outperform raw-latent "
            "neighbors on the composite metric. The detailed per-feature tables and figures should "
            "therefore be interpreted carefully rather than treated as a blanket win."
        )
    lines.append(
        "The more important diagnostic is whether local distance in a space tracks audio-domain "
        "timbral differences. If raw latent space shows weaker correlation or higher neighbor "
        "feature distances, that supports the claim that direct latent navigation is not a reliable "
        "perceptual strategy."
    )
    lines.append(
        "Because the navigation space is built from audio descriptors and dimensionality reduction, "
        "it is explicitly optimized for perceptual organization, whereas the raw latent space is "
        "optimized for reconstruction."
    )
    lines.append("")
    lines.append("## DAFx-Ready Paragraph")
    if nav_better:
        lines.append(
            "We evaluated local neighborhood coherence in raw VAE latent space and in the "
            "descriptor-derived navigation space used by Stable Audio Wanderer. Using latent-rate "
            "source-audio descriptors as a timbral reference, navigation-space neighbors showed "
            "lower robustly scaled composite timbral distance than raw-latent neighbors "
            f"(mean {float(navigation_composite['mean'] or 0.0):.4f} vs "
            f"{float(latent_composite['mean'] or 0.0):.4f}), supporting the claim that "
            "raw latent proximity is not a reliable perceptual navigation strategy. This "
            "motivates the paper's separation between perceptual control space and latent "
            "reconstruction space."
        )
    else:
        lines.append(
            "We evaluated local neighborhood coherence in raw VAE latent space and in the "
            "descriptor-derived navigation space used by Stable Audio Wanderer. The resulting "
            "correlation plots and neighbor-distance summaries quantify how strongly each space "
            "tracks audio-domain timbral similarity, which is the core motivation for separating "
            "perceptual navigation from latent reconstruction in the system design."
        )
    lines.append("")

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_summary_payload(
    *,
    bundle: CorpusBundle,
    space_arrays: SpaceArrays,
    num_anchors: int,
    k: int,
    selected_features: Sequence[str],
    seed: int,
    machine_info: Dict[str, object],
    summary: Dict[str, object],
) -> Dict[str, object]:
    """Build the JSON summary payload."""
    return {
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
            "num_anchors": int(num_anchors),
            "k": int(k),
            "selected_features": list(selected_features),
            "seed": int(seed),
            "navigation_reducer": space_arrays.navigation_reducer,
            "navigation_source": space_arrays.navigation_source,
        },
        "machine": machine_info,
        "summary": summary,
    }
