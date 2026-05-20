#!/usr/bin/env python3
"""Compare raw-latent neighborhood coherence against descriptor-navigation coherence."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from stable_audio_wanderer.eval.latency import (
    collect_machine_info,
    load_corpus_bundle,
    resolve_latest_corpus_dir,
    write_summary_json,
)
from stable_audio_wanderer.eval.space_coherence import (
    AVAILABLE_FEATURES,
    build_summary_payload,
    collect_measurements,
    render_boxplot,
    render_correlation_plot,
    render_markdown_report,
    resolve_space_arrays,
    summarize_measurements,
    write_measurements_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the space-coherence benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether local neighbors in the descriptor-derived navigation "
            "space are more timbrally coherent than local neighbors in raw latent space."
        )
    )
    parser.add_argument(
        "--corpus_dir",
        default=None,
        help="Corpus directory containing corpus.npz. Defaults to the newest corpus/* directory.",
    )
    parser.add_argument(
        "--manual_artifact",
        default=None,
        help="Optional override path to manual_navigation.npz.",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=256,
        help="Number of anchor frames sampled from the corpus.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of nearest neighbors retrieved per anchor in each space.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(AVAILABLE_FEATURES),
        choices=list(AVAILABLE_FEATURES),
        help="Timbral feature distances used for the coherence comparison.",
    )
    parser.add_argument(
        "--navigation_source",
        default="embed",
        choices=["embed", "descriptor"],
        help=(
            "Navigation representation to compare against raw latent space: "
            "'embed' uses the stored reduced navigation coordinates, while "
            "'descriptor' uses weighted descriptor vectors directly."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic anchor sampling.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for CSV, JSON, and plots. Defaults to eval_out/space_coherence_<timestamp>.",
    )
    parser.add_argument(
        "--report_path",
        default=os.path.join("reports", "space_coherence_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    """Validate user-supplied CLI arguments."""
    if int(args.num_anchors) < 1:
        raise ValueError("--num_anchors must be >= 1.")
    if int(args.k) < 1:
        raise ValueError("--k must be >= 1.")
    if not args.features:
        raise ValueError("At least one --features value is required.")


def main():
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    corpus_dir = (
        os.path.abspath(args.corpus_dir)
        if args.corpus_dir is not None
        else resolve_latest_corpus_dir()
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.abspath(os.path.join("eval_out", f"space_coherence_{timestamp}"))
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Using corpus directory: %s", corpus_dir)
    logging.info("Navigation source: %s", args.navigation_source)
    logging.info("Requested anchors: %d", int(args.num_anchors))
    logging.info("Requested neighbors per anchor: %d", int(args.k))
    logging.info("Selected features: %s", ", ".join(str(name) for name in args.features))

    bundle = load_corpus_bundle(
        corpus_dir=corpus_dir,
        manual_artifact_path=args.manual_artifact,
    )
    if bundle.num_frames < 2:
        raise RuntimeError("Space coherence evaluation requires at least two corpus frames.")

    effective_num_anchors = min(int(args.num_anchors), int(bundle.num_frames))
    effective_k = min(int(args.k), int(bundle.num_frames) - 1)
    if effective_num_anchors != int(args.num_anchors):
        logging.warning(
            "Clipping num_anchors from %d to %d because the corpus has %d frames.",
            int(args.num_anchors),
            effective_num_anchors,
            int(bundle.num_frames),
        )
    if effective_k != int(args.k):
        logging.warning(
            "Clipping k from %d to %d because the corpus has %d frames.",
            int(args.k),
            effective_k,
            int(bundle.num_frames),
        )

    space_arrays = resolve_space_arrays(
        bundle,
        navigation_source=args.navigation_source,
    )
    rows = collect_measurements(
        bundle,
        space_arrays,
        num_anchors=effective_num_anchors,
        k=effective_k,
        seed=int(args.seed),
        selected_features=args.features,
    )
    summary = summarize_measurements(rows, selected_features=args.features)
    machine_info = collect_machine_info()

    csv_path = os.path.join(out_dir, "space_coherence_pairs.csv")
    json_path = os.path.join(out_dir, "space_coherence_summary.json")
    correlation_plot_path = os.path.join(out_dir, "space_distance_vs_timbral_distance.png")
    boxplot_path = os.path.join(out_dir, "space_neighbor_timbral_boxplots.png")
    report_path = os.path.abspath(args.report_path)

    write_measurements_csv(csv_path, rows, selected_features=args.features)
    render_correlation_plot(correlation_plot_path, rows)
    render_boxplot(boxplot_path, rows, selected_features=args.features)

    payload = build_summary_payload(
        bundle=bundle,
        space_arrays=space_arrays,
        num_anchors=effective_num_anchors,
        k=effective_k,
        selected_features=args.features,
        seed=int(args.seed),
        machine_info=machine_info,
        summary=summary,
    )
    write_summary_json(json_path, payload)
    render_markdown_report(
        report_path,
        bundle=bundle,
        space_arrays=space_arrays,
        num_anchors=effective_num_anchors,
        k=effective_k,
        selected_features=args.features,
        machine_info=machine_info,
        summary=summary,
        csv_path=csv_path,
        json_path=json_path,
        correlation_plot_path=correlation_plot_path,
        boxplot_path=boxplot_path,
    )

    logging.info("Space coherence CSV: %s", csv_path)
    logging.info("Space coherence JSON: %s", json_path)
    logging.info("Space coherence correlation plot: %s", correlation_plot_path)
    logging.info("Space coherence boxplots: %s", boxplot_path)
    logging.info("Space coherence report: %s", report_path)


if __name__ == "__main__":
    main()
