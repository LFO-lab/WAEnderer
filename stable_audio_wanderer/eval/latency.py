"""Latency benchmark helpers for Stable Audio Wanderer."""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import platform
import random
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import DEVICE
from ..io.corpus_io import find_latest, load_corpus, read_scalar, resolve_corpus_path
from ..policy.latent_geometry import load_geometry_from_dict
from ..runtime.decoder_player import BufferedAudioQueue
from ..runtime.manual_player import ManualNavigationEngine
from ..runtime.manifold import ManifoldConfig, ManifoldConstrainedGenerator
from ..runtime.player import LatentNavigationEngine
from ..vae import load_vae_adapter
from ..vae.decoder import decode_latents
from ..vae.sae import load_vae

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZES: Dict[str, List[int]] = {
    "manual": [1, 2, 4, 8, 16, 32, 64],
    "random": [2, 4, 8, 16, 32, 64],
    "reorganized": [2, 4, 8, 16, 32, 64],
}
TIMING_FIELDS = ("prepare_ms", "decode_ms", "total_ms")
MODE_SEED_OFFSETS = {
    "manual": 11_000,
    "random": 22_000,
    "reorganized": 33_000,
}


@dataclass(frozen=True)
class BenchmarkRuntimeConfig:
    """Defaults chosen to match the live runtime as closely as practical."""

    manifold_k: int = 16
    manifold_n_local: int = 8
    manifold_n_global: int = 32
    manifold_sparse_quantile: float = 0.75
    random_temperature: float = 1.0
    random_sample: bool = True
    random_timbre_swap: bool = True
    random_recompose: bool = True
    ctrl_phrase_scale: float = 0.40
    ctrl_jump_rate: float = 0.55
    ctrl_timbre_lock: float = 0.55
    ctrl_drift: float = 0.50
    ctrl_repeat_avoid: float = 0.75
    ctrl_crossfile: float = 0.70
    reorganized_temperature: float = 1.0
    ctrl_morph_len: float = 0.50
    ctrl_reorg_jump_rate: float = 0.60
    ctrl_reorg_timbre_lock: float = 0.45
    ctrl_evolution: float = 0.60
    ctrl_novelty: float = 0.50
    ctrl_reorg_crossfile: float = 0.70
    manual_wander_k: int = 1
    manual_wander_speed: float = 0.0001
    manual_coarse_k: int = 96
    manual_refine_k: int = 16
    manual_desc_interp_k: int = 8
    manual_dither: float = 0.0


@dataclass(frozen=True)
class CorpusBundle:
    """Resolved corpus + sidecar artifacts needed to recreate runtime batches."""

    corpus_dir: str
    corpus_path: str
    data: Dict[str, np.ndarray]
    manual_artifact_path: Optional[str]
    manual_data: Optional[Dict[str, np.ndarray]]
    reorganized_units_path: Optional[str]
    reorganized_data: Optional[Dict[str, np.ndarray]]
    random_model_path: Optional[str]
    reorganized_model_path: Optional[str]
    vae_id: str
    sample_rate: int
    latent_hz: float
    num_frames: int


@dataclass(frozen=True)
class TrialRequest:
    """A reproducible latent-window request specification."""

    mode: str
    window_size: int
    trial_index: int
    seed: int
    start_index: int
    mid_index: Optional[int] = None
    target_index: Optional[int] = None


@dataclass(frozen=True)
class PreparedBatch:
    """Runtime-faithful latent batch and compact provenance metadata."""

    latents: np.ndarray
    selected_start_index: int
    selected_end_index: int


def _load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def resolve_latest_corpus_dir(corpus_root: str = "corpus") -> str:
    """Resolve the newest corpus directory containing ``corpus.npz``."""
    pattern = os.path.join(os.path.abspath(corpus_root), "*", "corpus.npz")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No corpus directories found under {os.path.abspath(corpus_root)}"
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    return os.path.dirname(matches[0])


def _resolve_manual_artifact_path(
    corpus_dir: str, arg_path: Optional[str]
) -> Optional[str]:
    if arg_path is not None:
        resolved = os.path.abspath(arg_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Manual navigation artifact not found: {resolved}")
        return resolved

    default_path = os.path.join(corpus_dir, "manual_navigation.npz")
    if os.path.exists(default_path):
        return default_path
    try:
        return find_latest(corpus_dir, "manual_navigation*.npz")
    except FileNotFoundError:
        return None


def _resolve_optional_model_path(
    corpus_dir: str, arg_path: Optional[str], pattern: str
) -> Optional[str]:
    if arg_path is not None:
        resolved = os.path.abspath(arg_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
        return resolved
    try:
        return find_latest(corpus_dir, pattern)
    except FileNotFoundError:
        return None


def _resolve_reorganized_units_path(
    corpus_dir: str, arg_path: Optional[str]
) -> Optional[str]:
    if arg_path is not None:
        resolved = os.path.abspath(arg_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Reorganized units artifact not found: {resolved}"
            )
        return resolved

    default_path = os.path.join(corpus_dir, "policy_v2_units.npz")
    if os.path.exists(default_path):
        return default_path
    try:
        return find_latest(corpus_dir, "policy_v2_units*.npz")
    except FileNotFoundError:
        return None


def load_manual_artifact(path: str, expected_frames: int) -> Dict[str, np.ndarray]:
    """Validate and load the manual-navigation sidecar artifact."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manual navigation artifact not found: {path}")

    artifact = _load_npz_dict(path)
    required = [
        "manual_fader_p01",
        "manual_fader_p99",
        "frame_file_ids",
        "frame_t",
        "manual_embed_points",
        "manual_desc_weighted",
    ]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise RuntimeError(f"Manual artifact missing keys: {missing}")

    points = artifact["manual_embed_points"].astype(np.float32)
    p01 = artifact["manual_fader_p01"].astype(np.float32).reshape(-1)
    p99 = artifact["manual_fader_p99"].astype(np.float32).reshape(-1)
    frame_file_ids = artifact["frame_file_ids"].astype(np.int32).reshape(-1)
    frame_t = artifact["frame_t"].astype(np.int32).reshape(-1)
    leaf = int(read_scalar(artifact, "kdtree_leafsize", 32))
    version = int(read_scalar(artifact, "version", 0))
    reducer_name = str(read_scalar(artifact, "manual_embed_reducer", "pca")).lower()
    if reducer_name not in ("pca", "umap"):
        reducer_name = "pca"

    desc_weighted = artifact["manual_desc_weighted"].astype(np.float32)
    pca_components = None
    pca_mean = None
    if "manual_pca_components" in artifact and "manual_pca_mean" in artifact:
        pca_components = artifact["manual_pca_components"].astype(np.float32)
        pca_mean = artifact["manual_pca_mean"].astype(np.float32).reshape(-1)

    if version != 3:
        raise RuntimeError(
            f"Unsupported manual artifact version: {version}. Re-run preprocess.py."
        )
    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError(f"manual_embed_points must be [N, D>=3], got {points.shape}")
    if points.shape[0] != expected_frames:
        raise RuntimeError(
            "manual_embed_points frame count mismatch: "
            f"{points.shape[0]} vs expected {expected_frames}"
        )
    if frame_file_ids.shape[0] != points.shape[0] or frame_t.shape[0] != points.shape[0]:
        raise RuntimeError("Manual artifact frame metadata length mismatch.")
    if p01.shape[0] != points.shape[1] or p99.shape[0] != points.shape[1]:
        raise RuntimeError(
            f"manual_fader_p01/manual_fader_p99 must both be shape [{points.shape[1]}]."
        )
    if desc_weighted.shape[0] != points.shape[0]:
        raise RuntimeError(
            "manual_desc_weighted must be aligned with manual_embed_points."
        )
    if reducer_name == "pca" and (
        pca_components is None or pca_mean is None
    ):
        raise RuntimeError(
            "PCA reducer requires manual_pca_components/manual_pca_mean in artifact."
        )
    if pca_components is not None:
        if pca_components.ndim != 2 or pca_components.shape[0] != points.shape[1]:
            raise RuntimeError(
                f"manual_pca_components must be [{points.shape[1]}, D], got {pca_components.shape}"
            )
        if pca_mean is None or pca_mean.shape[0] != pca_components.shape[1]:
            raise RuntimeError(
                "manual_pca_mean mismatch with manual_pca_components."
            )

    return {
        "version": np.array(int(version), dtype=np.int32),
        "manual_embed_points": points,
        "manual_embed_reducer": np.array(reducer_name),
        "manual_pca_components": pca_components,
        "manual_pca_mean": pca_mean,
        "manual_desc_weighted": desc_weighted,
        "manual_fader_p01": p01,
        "manual_fader_p99": p99,
        "frame_file_ids": frame_file_ids,
        "frame_t": frame_t,
        "kdtree_leafsize": np.array(max(1, leaf), dtype=np.int32),
    }


def load_policy_v2_artifact(path: str, expected_frames: int) -> Dict[str, np.ndarray]:
    """Validate and load the reorganized-mode units artifact."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy V2 artifact not found: {path}")

    artifact = _load_npz_dict(path)
    required = [
        "unit_start_idx",
        "unit_end_idx",
        "unit_file_id",
        "unit_len",
        "unit_entry_desc",
        "unit_exit_desc",
        "unit_delta_desc",
        "frame_to_unit",
        "unit_graph_neighbors",
        "unit_graph_scores",
    ]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise RuntimeError(f"Policy V2 artifact missing keys: {missing}")

    unit_start = np.asarray(artifact["unit_start_idx"], dtype=np.int32).reshape(-1)
    unit_end = np.asarray(artifact["unit_end_idx"], dtype=np.int32).reshape(-1)
    unit_file = np.asarray(artifact["unit_file_id"], dtype=np.int32).reshape(-1)
    unit_len = np.asarray(artifact["unit_len"], dtype=np.int32).reshape(-1)
    unit_entry = np.asarray(artifact["unit_entry_desc"], dtype=np.float32)
    unit_exit = np.asarray(artifact["unit_exit_desc"], dtype=np.float32)
    unit_delta = np.asarray(artifact["unit_delta_desc"], dtype=np.float32)
    frame_to_unit = np.asarray(artifact["frame_to_unit"], dtype=np.int32).reshape(-1)
    graph_neighbors = np.asarray(artifact["unit_graph_neighbors"], dtype=np.int32)
    graph_scores = np.asarray(artifact["unit_graph_scores"], dtype=np.float32)

    n_units = int(unit_start.shape[0])
    if n_units <= 0:
        raise RuntimeError("Policy V2 artifact has zero units.")
    if (
        unit_end.shape[0] != n_units
        or unit_file.shape[0] != n_units
        or unit_len.shape[0] != n_units
        or unit_entry.shape[0] != n_units
        or unit_exit.shape[0] != n_units
        or unit_delta.shape[0] != n_units
    ):
        raise RuntimeError("Policy V2 artifact unit arrays are misaligned.")
    if frame_to_unit.shape[0] != expected_frames:
        raise RuntimeError(
            "Policy V2 artifact frame_to_unit length mismatch: "
            f"{frame_to_unit.shape[0]} vs expected {expected_frames}"
        )
    if graph_neighbors.shape[0] != n_units or graph_scores.shape != graph_neighbors.shape:
        raise RuntimeError("Policy V2 artifact graph arrays are invalid.")

    return artifact


def load_corpus_bundle(
    corpus_dir: str,
    manual_artifact_path: Optional[str] = None,
    reorganized_units_path: Optional[str] = None,
    random_model_path: Optional[str] = None,
    reorganized_model_path: Optional[str] = None,
) -> CorpusBundle:
    """Load corpus/runtime artifacts needed for latency benchmarking."""
    corpus_dir = os.path.abspath(corpus_dir)
    corpus_path = os.path.abspath(resolve_corpus_path(corpus_dir))
    raw_data = load_corpus(corpus_path)
    data = {key: raw_data[key] for key in raw_data.files}
    raw_data.close()

    z_concat = np.asarray(data["Z_concat"], dtype=np.float32)
    manual_path = _resolve_manual_artifact_path(corpus_dir, manual_artifact_path)
    manual_data = None
    if manual_path is not None:
        manual_data = load_manual_artifact(manual_path, expected_frames=z_concat.shape[0])

    v2_path = _resolve_reorganized_units_path(corpus_dir, reorganized_units_path)
    v2_data = None
    if v2_path is not None:
        v2_data = load_policy_v2_artifact(v2_path, expected_frames=z_concat.shape[0])

    return CorpusBundle(
        corpus_dir=corpus_dir,
        corpus_path=corpus_path,
        data=data,
        manual_artifact_path=manual_path,
        manual_data=manual_data,
        reorganized_units_path=v2_path,
        reorganized_data=v2_data,
        random_model_path=_resolve_optional_model_path(
            corpus_dir, random_model_path, "latent_policy_*.pt"
        ),
        reorganized_model_path=_resolve_optional_model_path(
            corpus_dir, reorganized_model_path, "policy_v2_*.pt"
        ),
        vae_id=str(read_scalar(data, "vae_id", "")),
        sample_rate=int(read_scalar(data, "sr", 44100)),
        latent_hz=float(read_scalar(data, "latent_hz", 21.5)),
        num_frames=int(z_concat.shape[0]),
    )


def load_vae_for_bundle(
    bundle: CorpusBundle,
    vae_id_override: str = "",
    vae_weight_path: str = "",
    vae_repo_or_path: str = "",
    pretrained: str = "stabilityai/stable-audio-open-1.0",
):
    """Load the VAE decoder used for the benchmark."""
    vae_id = (vae_id_override or bundle.vae_id).strip()
    if vae_id:
        kwargs = {}
        if vae_weight_path:
            kwargs["weight_path"] = vae_weight_path
        if vae_repo_or_path:
            kwargs["repo_or_path"] = vae_repo_or_path
        LOGGER.info("Loading VAE adapter %s", vae_id)
        try:
            return load_vae_adapter(vae_id, **kwargs)
        except Exception as exc:
            hint = ""
            if vae_id == "stable_audio_open":
                hint = (
                    " Provide --vae_repo_or_path with a local Stable Audio Open "
                    "checkpoint if network access or cache lookup is unavailable."
                )
            elif vae_id.startswith("ear_vae") and not vae_weight_path:
                hint = (
                    " Provide --vae_weight_path pointing to the local EAR VAE "
                    "checkpoint file."
                )
            raise RuntimeError(
                f"Failed to load VAE adapter '{vae_id}'.{hint} Original error: {exc}"
            ) from exc

    repo_or_path = vae_repo_or_path or pretrained
    LOGGER.info("Loading legacy VAE decoder from %s", repo_or_path)
    try:
        return load_vae(repo_or_path=repo_or_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the legacy Stable Audio Open decoder. "
            "Provide --vae_repo_or_path with a local checkpoint if needed. "
            f"Original error: {exc}"
        ) from exc


def _load_navigation_engine(
    bundle: CorpusBundle,
    cfg: BenchmarkRuntimeConfig,
    policy_variant: str,
) -> LatentNavigationEngine:
    geometry = load_geometry_from_dict(bundle.data)
    if geometry is None:
        raise RuntimeError(
            "Corpus is missing latent geometry. Re-run preprocess.py."
        )

    z_concat = bundle.data["Z_concat"].astype(np.float32)
    meta = bundle.data["meta"].astype(np.int32)
    file_offsets = bundle.data["file_offsets"].astype(np.int64)
    desc_weighted = None
    if bundle.manual_data is not None:
        desc_weighted = bundle.manual_data["manual_desc_weighted"].astype(np.float32)

    return LatentNavigationEngine(
        GG=z_concat,
        meta=meta,
        geometry=geometry,
        file_offsets=file_offsets,
        desc_weighted=desc_weighted,
        policy_path=bundle.random_model_path,
        policy_temperature=cfg.random_temperature,
        policy_sample=cfg.random_sample,
        policy_timbre_swap_enabled=cfg.random_timbre_swap,
        policy_recompose_enabled=cfg.random_recompose,
        control_phrase_scale=cfg.ctrl_phrase_scale,
        control_jump_rate=cfg.ctrl_jump_rate,
        control_timbre_lock=cfg.ctrl_timbre_lock,
        control_drift=cfg.ctrl_drift,
        control_repeat_avoid=cfg.ctrl_repeat_avoid,
        control_crossfile=cfg.ctrl_crossfile,
        control_morph_len=cfg.ctrl_morph_len,
        control_reorg_jump_rate=cfg.ctrl_reorg_jump_rate,
        control_reorg_timbre_lock=cfg.ctrl_reorg_timbre_lock,
        control_evolution=cfg.ctrl_evolution,
        control_novelty=cfg.ctrl_novelty,
        control_reorg_crossfile=cfg.ctrl_reorg_crossfile,
        policy_v2_enabled=bundle.reorganized_data is not None,
        v2_artifact=bundle.reorganized_data,
        policy_v2_model_path=bundle.reorganized_model_path,
        policy_v2_temperature=cfg.reorganized_temperature,
        policy_variant=policy_variant,
        latent_frame_seconds=1.0 / bundle.latent_hz,
    )


class ManualSteadyStateBatchBuilder:
    """Recreate manual-mode latent batching with the same fader interpolation."""

    def __init__(self, bundle: CorpusBundle, cfg: BenchmarkRuntimeConfig):
        if bundle.manual_data is None:
            raise RuntimeError(
                "Manual latency benchmarking requires manual_navigation.npz."
            )

        manual_data = bundle.manual_data
        self.z_concat = bundle.data["Z_concat"].astype(np.float32)
        self.z_mean = bundle.data["Z_mean"].astype(np.float32)
        self.z_std = bundle.data["Z_std"].astype(np.float32)
        self.manual_points = manual_data["manual_embed_points"].astype(np.float32)
        self.fader_p01 = manual_data["manual_fader_p01"].astype(np.float32)
        self.fader_p99 = manual_data["manual_fader_p99"].astype(np.float32)
        self._fader_span = np.maximum(self.fader_p99 - self.fader_p01, 1e-6)
        self._render_faders = np.full(self.manual_points.shape[1], 0.5, dtype=np.float32)

        self.engine = ManualNavigationEngine(
            manual_points=self.manual_points,
            fader_p01=self.fader_p01,
            fader_p99=self.fader_p99,
            desc_weighted=manual_data["manual_desc_weighted"].astype(np.float32),
            pca_components=manual_data["manual_pca_components"],
            pca_mean=manual_data["manual_pca_mean"],
            reducer=str(np.asarray(manual_data["manual_embed_reducer"]).reshape(-1)[0]),
            desc_interp_k=int(cfg.manual_desc_interp_k),
            coarse_k=int(cfg.manual_coarse_k),
            refine_k=int(cfg.manual_refine_k),
            leafsize=int(np.asarray(manual_data["kdtree_leafsize"]).reshape(-1)[0]),
            wander_k=int(cfg.manual_wander_k),
            wander_speed=float(cfg.manual_wander_speed),
        )
        self.dither = float(np.clip(cfg.manual_dither, 0.0, 1.0))

    def make_request(self, trial_index: int, seed: int, window_size: int) -> TrialRequest:
        rng = np.random.default_rng(seed)
        start_index = int(rng.integers(0, self.z_concat.shape[0]))
        mid_index = int(rng.integers(0, self.z_concat.shape[0]))
        target_index = int(rng.integers(0, self.z_concat.shape[0]))
        return TrialRequest(
            mode="manual",
            window_size=int(window_size),
            trial_index=int(trial_index),
            seed=int(seed),
            start_index=start_index,
            mid_index=mid_index,
            target_index=target_index,
        )

    def _index_to_faders(self, frame_index: int) -> np.ndarray:
        point = self.manual_points[int(np.clip(frame_index, 0, self.manual_points.shape[0] - 1))]
        return np.clip((point - self.fader_p01) / self._fader_span, 0.0, 1.0).astype(
            np.float32
        )

    def reset(self, frame_index: int):
        start_faders = self._index_to_faders(frame_index)
        self.engine.set_faders(start_faders)
        self._render_faders = start_faders.copy()

    def _step_to_target(self, target_faders: np.ndarray, window_size: int) -> PreparedBatch:
        window = max(1, int(window_size))
        start_faders = self._render_faders.copy()
        target_faders = np.asarray(target_faders, dtype=np.float32).reshape(-1)

        if window == 1:
            fader_path = target_faders[None, :]
        else:
            alphas = np.linspace(0.0, 1.0, window, dtype=np.float32)[:, None]
            fader_path = (
                (1.0 - alphas) * start_faders[None, :]
                + alphas * target_faders[None, :]
            )

        batch = np.empty((window, self.z_concat.shape[1]), dtype=np.float32)
        selected_indices = np.empty(window, dtype=np.int32)
        for i in range(window):
            frame = self.engine.step_with_faders(fader_path[i])
            idx = int(np.clip(frame.nearest_index, 0, self.z_concat.shape[0] - 1))
            selected_indices[i] = idx
            batch[i] = self.z_concat[idx] * self.z_std + self.z_mean

        self._render_faders = target_faders.copy()
        if self.dither > 0.0:
            noise = np.random.randn(*batch.shape).astype(np.float32)
            batch += noise * (self.dither * self.z_std)

        return PreparedBatch(
            latents=batch,
            selected_start_index=int(selected_indices[0]),
            selected_end_index=int(selected_indices[-1]),
        )

    def prepare_batch(self, request: TrialRequest) -> PreparedBatch:
        self.reset(request.start_index)
        target_faders_prime = self._index_to_faders(int(request.mid_index))
        target_faders_eval = self._index_to_faders(int(request.target_index))

        hop = (int(request.window_size) + 1) // 2
        prime_window = self._step_to_target(target_faders_prime, int(request.window_size))
        prev_half = prime_window.latents[hop:]
        new_frames = self._step_to_target(target_faders_eval, hop)

        if prev_half.shape[0] > 0:
            full_window = np.concatenate([prev_half, new_frames.latents], axis=0)
            selected_start = int(prime_window.selected_end_index)
        else:
            full_window = new_frames.latents
            selected_start = int(new_frames.selected_start_index)

        return PreparedBatch(
            latents=full_window.astype(np.float32),
            selected_start_index=selected_start,
            selected_end_index=int(new_frames.selected_end_index),
        )


class PolicySteadyStateBatchBuilder:
    """Recreate policy-mode latent batching with manifold projection."""

    def __init__(
        self,
        bundle: CorpusBundle,
        cfg: BenchmarkRuntimeConfig,
        variant: str,
    ):
        variant = str(variant).strip().lower()
        if variant not in ("random", "reorganized"):
            raise ValueError(f"Unsupported policy variant: {variant}")
        if variant == "reorganized" and bundle.reorganized_data is None:
            raise RuntimeError(
                "Reorganized latency benchmarking requires policy_v2_units.npz."
            )

        geometry = load_geometry_from_dict(bundle.data)
        if geometry is None:
            raise RuntimeError(
                "Corpus is missing latent geometry. Re-run preprocess.py."
            )

        self.variant = variant
        self.nav = _load_navigation_engine(bundle, cfg, policy_variant=variant)
        self.nav.set_policy_variant(variant)
        self.manifold = ManifoldConstrainedGenerator(
            self.nav.GG,
            geometry,
            ManifoldConfig(
                k=int(cfg.manifold_k),
                n_local=int(cfg.manifold_n_local),
                n_global=int(cfg.manifold_n_global),
                sparse_quantile=float(cfg.manifold_sparse_quantile),
            ),
        )
        self.z_mean = bundle.data["Z_mean"].astype(np.float32)
        self.z_std = bundle.data["Z_std"].astype(np.float32)

    def make_request(self, trial_index: int, seed: int, window_size: int) -> TrialRequest:
        rng = np.random.default_rng(seed)
        start_index = int(rng.integers(0, self.nav.N))
        return TrialRequest(
            mode=self.variant,
            window_size=int(window_size),
            trial_index=int(trial_index),
            seed=int(seed),
            start_index=start_index,
        )

    def prepare_batch(self, request: TrialRequest) -> PreparedBatch:
        window_size = int(request.window_size)
        hop = (window_size + 1) // 2

        self.nav.set_cursor_index(int(request.start_index))
        self.nav.set_policy_variant(self.variant)

        prime_frames = [self.nav.step() for _ in range(window_size)]
        prev_half = prime_frames[hop:]
        new_frames = [self.nav.step() for _ in range(hop)]
        full_window = list(prev_half) + new_frames
        z_batch_norm = self.manifold.generate_batch(
            full_window,
            exploration=self.nav.get_active_jump_rate(variant=self.variant),
        )
        z_batch_raw = z_batch_norm * self.z_std[None, :] + self.z_mean[None, :]
        return PreparedBatch(
            latents=z_batch_raw.astype(np.float32),
            selected_start_index=int(full_window[0].nearest_idx),
            selected_end_index=int(full_window[-1].nearest_idx),
        )


def _seed_everything(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def synchronize_device():
    """Synchronize asynchronous device work for accurate timing."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _check_window_size(mode: str, window_size: int):
    if mode == "manual" and int(window_size) < 1:
        raise ValueError("Manual mode requires window_size >= 1.")
    if mode in ("random", "reorganized") and int(window_size) < 2:
        raise ValueError(f"{mode} mode requires window_size >= 2.")


def make_requests(
    builder,
    mode: str,
    window_size: int,
    count: int,
    seed_base: int,
) -> List[TrialRequest]:
    """Precompute reproducible trial requests so sampling is not timed."""
    _check_window_size(mode, window_size)
    requests: List[TrialRequest] = []
    for trial_index in range(int(count)):
        request_seed = int(seed_base + window_size * 100_000 + trial_index)
        requests.append(
            builder.make_request(
                trial_index=trial_index,
                seed=request_seed,
                window_size=window_size,
            )
        )
    return requests


def _run_single_trial(
    builder,
    vae,
    buffer: BufferedAudioQueue,
    request: TrialRequest,
) -> Dict[str, object]:
    _seed_everything(int(request.seed))
    synchronize_device()

    total_t0 = time.perf_counter_ns()
    prepared = builder.prepare_batch(request)
    synchronize_device()
    batch_t1 = time.perf_counter_ns()

    if prepared.latents.ndim != 2:
        raise RuntimeError(
            f"Expected latent batch [W, D], got {prepared.latents.shape}"
        )

    audio = decode_latents(vae, prepared.latents)
    synchronize_device()
    decode_t1 = time.perf_counter_ns()

    buffer.write_frame(audio)
    total_t1 = time.perf_counter_ns()
    buffer_duration_sec = float(buffer.buffer_duration())
    buffer.clear_output(preserve_crossfade=True)

    return {
        "mode": str(request.mode),
        "window_size": int(request.window_size),
        "trial_index": int(request.trial_index),
        "seed": int(request.seed),
        "request_start_index": int(request.start_index),
        "request_mid_index": (
            None if request.mid_index is None else int(request.mid_index)
        ),
        "request_target_index": (
            None if request.target_index is None else int(request.target_index)
        ),
        "selected_start_index": int(prepared.selected_start_index),
        "selected_end_index": int(prepared.selected_end_index),
        "prepare_ms": (batch_t1 - total_t0) / 1e6,
        "decode_ms": (decode_t1 - batch_t1) / 1e6,
        "total_ms": (total_t1 - total_t0) / 1e6,
        "audio_samples": int(audio.shape[0]),
        "audio_channels": int(audio.shape[1] if audio.ndim == 2 else 1),
        "buffer_duration_sec": buffer_duration_sec,
        "latent_batch_frames": int(prepared.latents.shape[0]),
        "latent_batch_dim": int(prepared.latents.shape[1]),
    }


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    """Summarize a latency metric with mean/median and IQR statistics."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty metric.")
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_trials(trial_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, dict]]:
    """Aggregate trial rows by mode and window size."""
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in trial_rows:
        key = (str(row["mode"]), int(row["window_size"]))
        grouped.setdefault(key, []).append(row)

    summary: Dict[str, Dict[str, dict]] = {}
    for (mode, window_size), rows in sorted(grouped.items()):
        mode_summary = summary.setdefault(mode, {})
        mode_summary[str(window_size)] = {
            "num_trials": len(rows),
            "prepare_ms": summarize_metric([float(r["prepare_ms"]) for r in rows]),
            "decode_ms": summarize_metric([float(r["decode_ms"]) for r in rows]),
            "total_ms": summarize_metric([float(r["total_ms"]) for r in rows]),
            "audio_samples": summarize_metric([float(r["audio_samples"]) for r in rows]),
        }
    return summary


def collect_machine_info() -> Dict[str, object]:
    """Collect host/device metadata for the markdown report."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": DEVICE,
        "cpu_count": os.cpu_count(),
    }
    if DEVICE == "cuda" and torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    elif DEVICE == "mps":
        info["device_name"] = "Apple Metal Performance Shaders"
    else:
        info["device_name"] = info["processor"]
    return info


def _as_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_trial_csv(path: str, trial_rows: Sequence[Dict[str, object]]):
    """Write one CSV row per measured trial."""
    if not trial_rows:
        raise ValueError("No trial rows to write.")

    fieldnames = [
        "mode",
        "window_size",
        "trial_index",
        "seed",
        "request_start_index",
        "request_mid_index",
        "request_target_index",
        "selected_start_index",
        "selected_end_index",
        "prepare_ms",
        "decode_ms",
        "total_ms",
        "audio_samples",
        "audio_channels",
        "buffer_duration_sec",
        "latent_batch_frames",
        "latent_batch_dim",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trial_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_summary_json(path: str, payload: Dict[str, object]):
    """Write a JSON summary payload."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_as_jsonable(payload), handle, indent=2, sort_keys=True)


def render_latency_plot(
    summary: Dict[str, Dict[str, dict]],
    figure_path: str,
):
    """Render a simple latency figure with mean, median, and IQR bars."""
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render the latency figure."
        ) from exc

    metrics = [
        ("prepare_ms", "Batch Prep"),
        ("decode_ms", "Decode"),
        ("total_ms", "Request -> Buffer Ready"),
    ]
    colors = {
        "manual": "#1d3557",
        "random": "#457b9d",
        "reorganized": "#e76f51",
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9.0, 10.5), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, title) in zip(axes, metrics):
        for mode in sorted(summary.keys()):
            window_items = sorted(
                ((int(window), stats) for window, stats in summary[mode].items()),
                key=lambda item: item[0],
            )
            xs = np.asarray([item[0] for item in window_items], dtype=np.float32)
            means = np.asarray(
                [item[1][metric_key]["mean"] for item in window_items], dtype=np.float32
            )
            medians = np.asarray(
                [item[1][metric_key]["median"] for item in window_items],
                dtype=np.float32,
            )
            q25 = np.asarray(
                [item[1][metric_key]["q25"] for item in window_items], dtype=np.float32
            )
            q75 = np.asarray(
                [item[1][metric_key]["q75"] for item in window_items], dtype=np.float32
            )
            yerr = np.vstack([means - q25, q75 - means])
            color = colors.get(mode, "#264653")
            ax.errorbar(
                xs,
                means,
                yerr=yerr,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=6.0,
                capsize=4.0,
                label=f"{mode} mean",
            )
            ax.plot(
                xs,
                medians,
                color=color,
                linestyle="--",
                linewidth=1.5,
                marker="x",
                markersize=6.0,
                label=f"{mode} median",
            )

        ax.set_title(title)
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, alpha=0.25)

    all_windows = sorted(
        {int(window) for mode_summary in summary.values() for window in mode_summary.keys()}
    )
    axes[-1].set_xlabel("Window Size (latent frames)")
    axes[-1].set_xticks(all_windows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _find_mode_window_bounds(
    summary: Dict[str, Dict[str, dict]],
    mode: str,
) -> Optional[Tuple[int, dict, int, dict]]:
    if mode not in summary or not summary[mode]:
        return None
    items = sorted((int(window), stats) for window, stats in summary[mode].items())
    first_window, first_stats = items[0]
    last_window, last_stats = items[-1]
    return first_window, first_stats, last_window, last_stats


def render_markdown_report(
    report_path: str,
    *,
    bundle: CorpusBundle,
    vae,
    machine_info: Dict[str, object],
    modes: Sequence[str],
    trials: int,
    warmup_trials: int,
    window_sizes: Dict[str, Sequence[int]],
    summary: Dict[str, Dict[str, dict]],
    csv_path: str,
    json_path: str,
    figure_path: str,
):
    """Render a markdown report suitable for the DAFx evaluation section."""
    info = vae.info()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    corpus_paths = bundle.data.get("paths")
    source_paths = []
    if corpus_paths is not None:
        source_paths = [str(path) for path in np.asarray(corpus_paths).reshape(-1).tolist()]

    lines: List[str] = []
    lines.append("# Latency Benchmark Report")
    lines.append("")
    lines.append(f"Generated: `{timestamp}`")
    lines.append("")
    lines.append("## Test Purpose")
    lines.append(
        "This benchmark measures steady-state end-to-end latent decoding latency for "
        "Stable Audio Wanderer using observed corpus latents and the same runtime "
        "batch construction path used by live playback. Each request decodes a latent "
        "batch of shape `[1, D, W]`, where `D` is the active adapter latent dimension "
        "and `W` is the evaluated window size."
    )
    lines.append("")
    lines.append("## Operational Definition")
    lines.append(
        "`Audio buffer ready` is defined here as the point when decoded audio has "
        "completed the same overlap-add/crossfade queue write used by "
        "`DecoderPlayer.write_frame(...)`. The measurement excludes DAC, driver, and "
        "OS scheduling latency, so it reflects software-side request-to-buffer-ready time."
    )
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Corpus directory: `{bundle.corpus_dir}`")
    lines.append(f"- Corpus file: `{bundle.corpus_path}`")
    lines.append(f"- Source audio files: {len(source_paths)}")
    lines.append(f"- Corpus frames: {bundle.num_frames}")
    lines.append(f"- VAE: `{info.vae_id}` ({info.display_name})")
    lines.append(f"- Sample rate: {bundle.sample_rate} Hz")
    lines.append(f"- Latent rate: {bundle.latent_hz:.4f} Hz")
    lines.append(f"- Latent dimension: {info.latent_dim}")
    lines.append(f"- Host: `{machine_info.get('hostname')}`")
    lines.append(f"- Platform: `{machine_info.get('platform')}`")
    lines.append(f"- Device: `{machine_info.get('device')}` / `{machine_info.get('device_name')}`")
    lines.append(f"- Python / Torch: `{machine_info.get('python')}` / `{machine_info.get('torch')}`")
    lines.append(f"- Measured trials per window: {int(trials)}")
    lines.append(f"- Warmup trials per window: {int(warmup_trials)}")
    lines.append(
        f"- Modes: {', '.join(str(mode) for mode in modes)}"
    )
    for mode in modes:
        if mode in window_sizes:
            lines.append(
                f"- `{mode}` windows: {', '.join(str(int(w)) for w in window_sizes[mode])}"
            )
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- JSON summary: `{json_path}`")
    lines.append(f"- Figure: `{figure_path}`")
    lines.append("")
    lines.append("## Latency Table")
    lines.append("")
    lines.append(
        "| Mode | W | Prepare Mean | Prepare Median | Decode Mean | Decode Median | "
        "Total Mean | Total Median | Total IQR |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode in modes:
        mode_summary = summary.get(mode, {})
        for window_size in sorted(int(window) for window in mode_summary.keys()):
            stats = mode_summary[str(window_size)]
            lines.append(
                "| "
                f"{mode} | {window_size} | "
                f"{stats['prepare_ms']['mean']:.2f} ms | {stats['prepare_ms']['median']:.2f} ms | "
                f"{stats['decode_ms']['mean']:.2f} ms | {stats['decode_ms']['median']:.2f} ms | "
                f"{stats['total_ms']['mean']:.2f} ms | {stats['total_ms']['median']:.2f} ms | "
                f"{stats['total_ms']['iqr']:.2f} ms |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Across the evaluated settings, shorter latent windows are expected to give "
        "the lowest request-to-buffer latency because fewer latent frames are prepared "
        "and decoded per request. Longer windows increase both decode cost and queue "
        "fill time, but they also provide longer contiguous latent context and are "
        "expected to produce smoother overlap-add transitions at runtime."
    )
    lines.append("")

    interpretation_lines = []
    manual_bounds = _find_mode_window_bounds(summary, "manual")
    if manual_bounds is not None:
        low_w, low_stats, high_w, high_stats = manual_bounds
        interpretation_lines.append(
            "In manual mode, median total latency rose from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} to "
            f"{high_stats['total_ms']['median']:.2f} ms at W={high_w}."
        )
    random_bounds = _find_mode_window_bounds(summary, "random")
    if random_bounds is not None:
        low_w, low_stats, high_w, high_stats = random_bounds
        interpretation_lines.append(
            "In random mode, median total latency rose from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} to "
            f"{high_stats['total_ms']['median']:.2f} ms at W={high_w}."
        )
    reorg_bounds = _find_mode_window_bounds(summary, "reorganized")
    if reorg_bounds is not None:
        low_w, low_stats, high_w, high_stats = reorg_bounds
        interpretation_lines.append(
            "In reorganized mode, median total latency rose from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} to "
            f"{high_stats['total_ms']['median']:.2f} ms at W={high_w}."
        )
    if interpretation_lines:
        lines.append(" ".join(interpretation_lines))
        lines.append("")

    dafx_sentence_parts = []
    if manual_bounds is not None:
        low_w, low_stats, high_w, high_stats = manual_bounds
        dafx_sentence_parts.append(
            "manual-mode median request-to-buffer latency increased from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} "
            f"to {high_stats['total_ms']['median']:.2f} ms at W={high_w}"
        )
    if random_bounds is not None:
        low_w, low_stats, high_w, high_stats = random_bounds
        dafx_sentence_parts.append(
            "random-mode latency increased from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} "
            f"to {high_stats['total_ms']['median']:.2f} ms at W={high_w}"
        )
    if reorg_bounds is not None:
        low_w, low_stats, high_w, high_stats = reorg_bounds
        dafx_sentence_parts.append(
            "reorganized-mode latency increased from "
            f"{low_stats['total_ms']['median']:.2f} ms at W={low_w} "
            f"to {high_stats['total_ms']['median']:.2f} ms at W={high_w}"
        )

    lines.append("## DAFx-Ready Paragraph")
    lines.append(
        "We evaluated the Stable Audio Wanderer decoding path by constructing "
        "runtime-faithful latent windows from the observed corpus and measuring "
        "steady-state request-to-buffer-ready latency after batch preparation, VAE "
        "decode, and overlap-add buffer enqueue. "
        + (
            "; ".join(dafx_sentence_parts) + ". "
            if dafx_sentence_parts
            else ""
        )
        + "These results support the intended reconstructive/navigational framing "
        "of the system: shorter windows minimize interaction latency, while longer "
        "windows trade latency for longer latent continuity and smoother output."
    )
    lines.append("")

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def benchmark_latency(
    *,
    bundle: CorpusBundle,
    vae,
    modes: Sequence[str],
    window_sizes: Dict[str, Sequence[int]],
    trials: int = 200,
    warmup_trials: int = 10,
    seed: int = 1234,
    runtime_cfg: BenchmarkRuntimeConfig = BenchmarkRuntimeConfig(),
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, dict]]]:
    """Run the latency benchmark and return per-trial rows plus grouped summary."""
    latent_dim = int(vae.info().latent_dim)
    corpus_dim = int(np.asarray(bundle.data["Z_concat"]).shape[1])
    if corpus_dim != latent_dim:
        raise RuntimeError(
            f"Corpus latent dim {corpus_dim} does not match adapter dim {latent_dim}."
        )

    builders = {}
    for mode in modes:
        mode_name = str(mode).strip().lower()
        if mode_name == "manual":
            builders[mode_name] = ManualSteadyStateBatchBuilder(bundle, runtime_cfg)
        elif mode_name in ("random", "reorganized"):
            builders[mode_name] = PolicySteadyStateBatchBuilder(
                bundle,
                runtime_cfg,
                variant=mode_name,
            )
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")

    all_rows: List[Dict[str, object]] = []
    for mode in modes:
        mode_name = str(mode).strip().lower()
        builder = builders[mode_name]
        for window_size in window_sizes[mode_name]:
            _check_window_size(mode_name, int(window_size))
            LOGGER.info(
                "Benchmarking %s mode at W=%d (%d warmup + %d measured)",
                mode_name,
                int(window_size),
                int(warmup_trials),
                int(trials),
            )

            buffer = BufferedAudioQueue(sr=int(bundle.sample_rate), gain=1.0)
            requests = make_requests(
                builder,
                mode=mode_name,
                window_size=int(window_size),
                count=int(warmup_trials) + int(trials),
                seed_base=int(seed) + int(MODE_SEED_OFFSETS[mode_name]),
            )

            for request in requests[: int(warmup_trials)]:
                _run_single_trial(builder, vae, buffer, request)

            for trial_offset, request in enumerate(requests[int(warmup_trials) :]):
                row = _run_single_trial(builder, vae, buffer, request)
                row["trial_index"] = int(trial_offset)
                all_rows.append(row)

    return all_rows, summarize_trials(all_rows)


def build_summary_payload(
    *,
    bundle: CorpusBundle,
    vae,
    machine_info: Dict[str, object],
    modes: Sequence[str],
    window_sizes: Dict[str, Sequence[int]],
    trials: int,
    warmup_trials: int,
    seed: int,
    summary: Dict[str, Dict[str, dict]],
) -> Dict[str, object]:
    """Build the top-level JSON summary payload."""
    corpus_paths = bundle.data.get("paths")
    source_paths = []
    if corpus_paths is not None:
        source_paths = [str(path) for path in np.asarray(corpus_paths).reshape(-1).tolist()]

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "corpus": {
            "corpus_dir": bundle.corpus_dir,
            "corpus_path": bundle.corpus_path,
            "source_audio_paths": source_paths,
            "num_frames": bundle.num_frames,
            "sample_rate": bundle.sample_rate,
            "latent_hz": bundle.latent_hz,
            "vae_id": bundle.vae_id,
        },
        "vae": {
            "vae_id": vae.info().vae_id,
            "display_name": vae.info().display_name,
            "sample_rate": vae.info().sample_rate,
            "latent_hz": vae.info().latent_hz,
            "latent_dim": vae.info().latent_dim,
        },
        "machine": machine_info,
        "benchmark": {
            "modes": list(modes),
            "window_sizes": {mode: list(window_sizes[mode]) for mode in modes},
            "trials": int(trials),
            "warmup_trials": int(warmup_trials),
            "seed": int(seed),
            "audio_buffer_ready_definition": (
                "Decoded audio has been written through the same overlap-add "
                "queue path used by DecoderPlayer.write_frame(...), excluding "
                "hardware playback latency."
            ),
        },
        "summary": summary,
    }
