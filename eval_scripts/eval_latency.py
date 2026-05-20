#!/usr/bin/env python3
"""Benchmark variable-length latent decoding latency for Stable Audio Wanderer."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from stable_audio_wanderer.eval.latency import (
    DEFAULT_WINDOW_SIZES,
    BenchmarkRuntimeConfig,
    benchmark_latency,
    build_summary_payload,
    collect_machine_info,
    load_corpus_bundle,
    load_vae_for_bundle,
    render_latency_plot,
    render_markdown_report,
    resolve_latest_corpus_dir,
    write_summary_json,
    write_trial_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark steady-state request-to-buffer latency for variable-length "
            "latent decoding in Stable Audio Wanderer."
        )
    )
    parser.add_argument(
        "--corpus_dir",
        default=None,
        help="Corpus directory containing corpus.npz and sidecar artifacts. Defaults to the latest corpus/* directory.",
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
        help="Path to local VAE weight file for adapters that require one (for example EAR VAE).",
    )
    parser.add_argument(
        "--vae_repo_or_path",
        default="",
        help="Optional local path or repo ID for Stable Audio Open when overriding the adapter source.",
    )
    parser.add_argument(
        "--pretrained",
        default="stabilityai/stable-audio-open-1.0",
        help="Legacy Stable Audio Open repo/path fallback if no vae_id is available.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["manual", "random", "reorganized"],
        choices=["manual", "random", "reorganized"],
        help="Benchmark modes to run.",
    )
    parser.add_argument(
        "--window_sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional explicit window sizes applied to every selected mode. "
            "If omitted, mode-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Measured trials per window size.",
    )
    parser.add_argument(
        "--warmup_trials",
        type=int,
        default=10,
        help="Warmup trials per window size before timing begins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for reproducible request sampling.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for CSV/JSON/PNG artifacts. Defaults to eval_out/latency_<timestamp>.",
    )
    parser.add_argument(
        "--report_path",
        default=os.path.join("reports", "latency_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_window_map(args: argparse.Namespace) -> dict:
    modes = [str(mode).strip().lower() for mode in args.modes]
    if args.window_sizes:
        windows = [int(window) for window in args.window_sizes]
        return {mode: windows for mode in modes}
    return {mode: list(DEFAULT_WINDOW_SIZES[mode]) for mode in modes}


def main():
    args = parse_args()
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
        else os.path.abspath(os.path.join("eval_out", f"latency_{timestamp}"))
    )
    os.makedirs(out_dir, exist_ok=True)

    window_sizes = build_window_map(args)
    logging.info("Using corpus directory: %s", corpus_dir)
    logging.info("Selected modes: %s", ", ".join(args.modes))
    for mode, windows in window_sizes.items():
        logging.info("%s windows: %s", mode, ", ".join(str(w) for w in windows))

    bundle = load_corpus_bundle(
        corpus_dir=corpus_dir,
        manual_artifact_path=args.manual_artifact,
        reorganized_units_path=args.reorganized_units_path,
        random_model_path=args.random_model_path,
        reorganized_model_path=args.reorganized_model_path,
    )
    if bundle.random_model_path is None and "random" in args.modes:
        logging.warning(
            "No random model checkpoint found; random mode will use heuristic fallback."
        )
    if bundle.reorganized_model_path is None and "reorganized" in args.modes:
        logging.warning(
            "No reorganized model checkpoint found; reorganized mode will use heuristic scoring."
        )

    vae = load_vae_for_bundle(
        bundle,
        vae_id_override=args.vae_id,
        vae_weight_path=args.vae_weight_path,
        vae_repo_or_path=args.vae_repo_or_path,
        pretrained=args.pretrained,
    )

    runtime_cfg = BenchmarkRuntimeConfig()
    trial_rows, summary = benchmark_latency(
        bundle=bundle,
        vae=vae,
        modes=[str(mode).strip().lower() for mode in args.modes],
        window_sizes=window_sizes,
        trials=int(args.trials),
        warmup_trials=int(args.warmup_trials),
        seed=int(args.seed),
        runtime_cfg=runtime_cfg,
    )

    csv_path = os.path.join(out_dir, "latency_trials.csv")
    json_path = os.path.join(out_dir, "latency_summary.json")
    figure_path = os.path.join(out_dir, "latency_plot.png")
    report_path = os.path.abspath(args.report_path)

    machine_info = collect_machine_info()
    payload = build_summary_payload(
        bundle=bundle,
        vae=vae,
        machine_info=machine_info,
        modes=[str(mode).strip().lower() for mode in args.modes],
        window_sizes=window_sizes,
        trials=int(args.trials),
        warmup_trials=int(args.warmup_trials),
        seed=int(args.seed),
        summary=summary,
    )

    write_trial_csv(csv_path, trial_rows)
    write_summary_json(json_path, payload)
    render_latency_plot(summary, figure_path)
    render_markdown_report(
        report_path,
        bundle=bundle,
        vae=vae,
        machine_info=machine_info,
        modes=[str(mode).strip().lower() for mode in args.modes],
        trials=int(args.trials),
        warmup_trials=int(args.warmup_trials),
        window_sizes=window_sizes,
        summary=summary,
        csv_path=csv_path,
        json_path=json_path,
        figure_path=figure_path,
    )

    logging.info("Latency CSV: %s", csv_path)
    logging.info("Latency JSON: %s", json_path)
    logging.info("Latency figure: %s", figure_path)
    logging.info("Latency report: %s", report_path)


if __name__ == "__main__":
    main()
