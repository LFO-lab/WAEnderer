#!/usr/bin/env python3
"""Compare temporal continuity across latent decoding strategies."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from stable_audio_wanderer.eval.continuity import (
    build_condition_specs,
    render_continuity_plot,
    render_markdown_report,
    run_continuity_evaluation,
    write_metrics_csv,
)
from stable_audio_wanderer.eval.latency import (
    collect_machine_info,
    load_corpus_bundle,
    load_vae_for_bundle,
    resolve_latest_corpus_dir,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuity and output stability for single-latent looping, "
            "naive raw-latent interpolation, and sequential corpus-based decoding."
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
        "--reorganized_units_path",
        default=None,
        help="Optional override path to policy_v2_units.npz.",
    )
    parser.add_argument(
        "--random_model_path",
        default=None,
        help="Optional override path to latent_policy_*.pt.",
    )
    parser.add_argument(
        "--reorganized_model_path",
        default=None,
        help="Optional override path to policy_v2_*.pt.",
    )
    parser.add_argument(
        "--vae_id",
        default="",
        help="Optional VAE adapter override. If empty, the corpus vae_id is used.",
    )
    parser.add_argument(
        "--vae_weight_path",
        default="",
        help="Optional local weight file for adapters that require one.",
    )
    parser.add_argument(
        "--vae_repo_or_path",
        default="",
        help="Optional local path or repo ID for Stable Audio Open.",
    )
    parser.add_argument(
        "--pretrained",
        default="stabilityai/stable-audio-open-1.0",
        help="Legacy Stable Audio Open fallback if no vae_id is available.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=6,
        help="Number of real corpus trajectories to evaluate.",
    )
    parser.add_argument(
        "--trajectory_frames",
        type=int,
        default=192,
        help="Length of each sampled trajectory in latent frames.",
    )
    parser.add_argument(
        "--window_sizes",
        nargs="+",
        type=int,
        default=[16],
        help=(
            "Decode window sizes in latent frames. Each strategy is evaluated at "
            "every supplied window size."
        ),
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients used for chunk descriptors.",
    )
    parser.add_argument(
        "--low_freq_cutoff_hz",
        type=float,
        default=150.0,
        help="Cutoff frequency for the low-frequency energy ratio metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for trajectory sampling.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for audio, CSV, JSON, and plots. Defaults to eval_out/continuity_<timestamp>.",
    )
    parser.add_argument(
        "--report_path",
        default=os.path.join("reports", "continuity_report.md"),
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
    if int(args.num_examples) < 1:
        raise ValueError("--num_examples must be >= 1.")
    if int(args.trajectory_frames) < 2:
        raise ValueError("--trajectory_frames must be >= 2.")
    if int(args.n_mfcc) < 1:
        raise ValueError("--n_mfcc must be >= 1.")
    if float(args.low_freq_cutoff_hz) <= 0.0:
        raise ValueError("--low_freq_cutoff_hz must be > 0.")
    if not args.window_sizes:
        raise ValueError("At least one --window_sizes value is required.")
    if any(int(window) < 1 for window in args.window_sizes):
        raise ValueError("All --window_sizes values must be >= 1.")

    max_window = max(int(window) for window in args.window_sizes)
    min_required = max_window + ((max_window + 1) // 2)
    if int(args.trajectory_frames) < int(min_required):
        raise ValueError(
            "--trajectory_frames must be at least max(window) + ceil(max(window)/2) "
            f"to yield two decoded buffers. Required minimum: {min_required}"
        )


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
        else os.path.abspath(os.path.join("eval_out", f"continuity_{timestamp}"))
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Using corpus directory: %s", corpus_dir)
    logging.info("Trajectory frames: %d", int(args.trajectory_frames))
    logging.info("Examples: %d", int(args.num_examples))
    logging.info(
        "Window sizes: %s",
        ", ".join(str(int(window)) for window in args.window_sizes),
    )

    bundle = load_corpus_bundle(
        corpus_dir=corpus_dir,
        manual_artifact_path=args.manual_artifact,
        reorganized_units_path=args.reorganized_units_path,
        random_model_path=args.random_model_path,
        reorganized_model_path=args.reorganized_model_path,
    )
    vae = load_vae_for_bundle(
        bundle,
        vae_id_override=args.vae_id,
        vae_weight_path=args.vae_weight_path,
        vae_repo_or_path=args.vae_repo_or_path,
        pretrained=args.pretrained,
    )

    condition_specs = build_condition_specs(args.window_sizes)
    examples, example_rows, aggregate_rows, payload = run_continuity_evaluation(
        bundle=bundle,
        vae=vae,
        num_examples=int(args.num_examples),
        trajectory_frames=int(args.trajectory_frames),
        seed=int(args.seed),
        condition_specs=condition_specs,
        out_dir=out_dir,
        n_mfcc=int(args.n_mfcc),
        low_freq_cutoff_hz=float(args.low_freq_cutoff_hz),
    )

    csv_path = os.path.join(out_dir, "continuity_metrics.csv")
    json_path = os.path.join(out_dir, "continuity_summary.json")
    plot_path = os.path.join(out_dir, "continuity_metrics.png")
    report_path = os.path.abspath(args.report_path)
    audio_dir = os.path.join(out_dir, "audio")

    write_metrics_csv(csv_path, example_rows, aggregate_rows)
    write_summary_json(json_path, payload)
    render_continuity_plot(plot_path, example_rows)
    render_markdown_report(
        report_path,
        bundle=bundle,
        vae=vae,
        machine_info=collect_machine_info(),
        examples=examples,
        conditions=condition_specs,
        example_rows=example_rows,
        aggregate_rows=aggregate_rows,
        csv_path=csv_path,
        json_path=json_path,
        plot_path=plot_path,
        audio_dir=audio_dir,
        low_freq_cutoff_hz=float(args.low_freq_cutoff_hz),
    )

    logging.info("Continuity CSV: %s", csv_path)
    logging.info("Continuity JSON: %s", json_path)
    logging.info("Continuity plot: %s", plot_path)
    logging.info("Continuity report: %s", report_path)
    logging.info("Audio examples: %s", audio_dir)


if __name__ == "__main__":
    main()
