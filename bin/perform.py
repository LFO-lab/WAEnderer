#!/usr/bin/env python3
"""
Real-time decoding with selectable manual/random/reorganized navigation modes.
"""

import os

# Fix OpenMP duplicate library issue on macOS (must be set before imports that use OpenMP).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import queue
import signal
import threading
import time
from typing import Optional, Tuple

import numpy as np

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus, read_scalar as _read_scalar
from stable_audio_wanderer.policy.latent_geometry import load_geometry_from_dict
from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer
from stable_audio_wanderer.runtime.manifold import (
    ManifoldConfig,
    ManifoldConstrainedGenerator,
)
from stable_audio_wanderer.runtime.manual_player import ManualNavigationEngine
from stable_audio_wanderer.runtime.osc_server import run_server
from stable_audio_wanderer.runtime.player import LatentNavigationEngine
from stable_audio_wanderer.vae import load_vae_adapter
from stable_audio_wanderer.vae.decoder import decode_latents
from stable_audio_wanderer.vae.sae import load_vae


def find_corpus_file(corpus_dir: str):
    """Find corpus in directory."""
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        corpus_npz = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(corpus_npz):
            raise FileNotFoundError(
                f"No corpus file found in {corpus_dir}. Expected 'corpus.npz'."
            )
        return corpus_npz


def smooth_window_log2(
    current_log2: float, target_log2: float, alpha: float = 0.3
) -> float:
    """Smooth window size transitions in log2 space (perceptually uniform)."""
    return (1.0 - alpha) * current_log2 + alpha * target_log2


def load_navigation_engine(
    data: dict,
    random_model_path: str = None,
    random_temperature: float = 1.0,
    random_sample: bool = True,
    desc_weighted: np.ndarray = None,
    random_timbre_swap: bool = True,
    random_recompose: bool = True,
    random_phrase_scale: float = 0.40,
    random_jump_rate: float = 0.55,
    random_timbre_lock: float = 0.55,
    random_drift: float = 0.5,
    random_repeat_avoid: float = 0.75,
    random_crossfile: float = 0.70,
    reorganized_enabled: bool = False,
    reorganized_artifact: dict = None,
    reorganized_model_path: str = None,
    reorganized_temperature: float = 1.0,
    reorganized_morph_len: float = 0.50,
    reorganized_jump_rate: float = 0.60,
    reorganized_timbre_lock: float = 0.45,
    reorganized_evolution: float = 0.60,
    reorganized_novelty: float = 0.50,
    reorganized_crossfile: float = 0.70,
    policy_variant: str = "random",
    latent_frame_seconds: float = 0.0465,
):
    """Factory function to create the LatentNavigationEngine."""
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError(
            "Corpus is missing latent geometry. Re-run preprocess.py to generate geometry-enabled corpora."
        )

    print("[info] Using LatentNavigationEngine (64D latent navigation)")
    Z_concat = data["Z_concat"].astype(np.float32)
    meta = data["meta"].astype(np.int32)
    file_offsets = data["file_offsets"].astype(np.int64)

    return LatentNavigationEngine(
        GG=Z_concat,
        meta=meta,
        geometry=geometry,
        file_offsets=file_offsets,
        desc_weighted=desc_weighted,
        policy_path=random_model_path,
        policy_temperature=random_temperature,
        policy_sample=random_sample,
        policy_timbre_swap_enabled=random_timbre_swap,
        policy_recompose_enabled=random_recompose,
        control_phrase_scale=random_phrase_scale,
        control_jump_rate=random_jump_rate,
        control_timbre_lock=random_timbre_lock,
        control_drift=random_drift,
        control_repeat_avoid=random_repeat_avoid,
        control_crossfile=random_crossfile,
        control_morph_len=reorganized_morph_len,
        control_reorg_jump_rate=reorganized_jump_rate,
        control_reorg_timbre_lock=reorganized_timbre_lock,
        control_evolution=reorganized_evolution,
        control_novelty=reorganized_novelty,
        control_reorg_crossfile=reorganized_crossfile,
        policy_v2_enabled=reorganized_enabled,
        v2_artifact=reorganized_artifact,
        policy_v2_model_path=reorganized_model_path,
        policy_v2_temperature=reorganized_temperature,
        policy_variant=policy_variant,
        latent_frame_seconds=latent_frame_seconds,
    )


def load_manual_artifact(path: str, expected_frames: int) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manual navigation artifact not found: {path}")

    artifact = np.load(path, allow_pickle=True)
    required = [
        "manual_fader_p01",
        "manual_fader_p99",
        "frame_file_ids",
        "frame_t",
    ]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise RuntimeError(f"Manual artifact missing keys: {missing}")

    if "manual_embed_points" not in artifact:
        raise RuntimeError("Manual artifact missing manual_embed_points.")
    points = artifact["manual_embed_points"].astype(np.float32)

    p01 = artifact["manual_fader_p01"].astype(np.float32).reshape(-1)
    p99 = artifact["manual_fader_p99"].astype(np.float32).reshape(-1)
    frame_file_ids = artifact["frame_file_ids"].astype(np.int32).reshape(-1)
    frame_t = artifact["frame_t"].astype(np.int32).reshape(-1)
    leaf = int(_read_scalar(artifact, "kdtree_leafsize", 32))
    version = int(_read_scalar(artifact, "version", 0))
    reducer_name = str(_read_scalar(artifact, "manual_embed_reducer", "pca")).lower()
    if reducer_name not in ("pca", "umap"):
        reducer_name = "pca"
    if version != 3:
        raise RuntimeError(f"Unsupported manual artifact version: {version}. Re-run preprocess.py.")
    if "manual_desc_weighted" not in artifact:
        raise RuntimeError("Manual artifact v3 missing manual_desc_weighted.")
    desc_weighted = artifact["manual_desc_weighted"].astype(np.float32)
    pca_components = None
    pca_mean = None
    if "manual_pca_components" in artifact and "manual_pca_mean" in artifact:
        pca_components = artifact["manual_pca_components"].astype(np.float32)
        pca_mean = artifact["manual_pca_mean"].astype(np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError(f"manual_embed_points must be [N, D>=3], got {points.shape}")
    if points.shape[0] != expected_frames:
        raise RuntimeError(
            f"manual_embed_points frame count mismatch: {points.shape[0]} vs expected {expected_frames}"
        )
    if (
        frame_file_ids.shape[0] != points.shape[0]
        or frame_t.shape[0] != points.shape[0]
    ):
        raise RuntimeError("Manual artifact frame metadata length mismatch.")
    if p01.shape[0] != points.shape[1] or p99.shape[0] != points.shape[1]:
        raise RuntimeError(
            f"manual_fader_p01/manual_fader_p99 must both be shape [{points.shape[1]}]."
        )
    if reducer_name == "pca" and pca_components is None:
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
                f"manual_pca_mean mismatch: {None if pca_mean is None else pca_mean.shape[0]} "
                f"vs {pca_components.shape[1]}"
            )
        if (
            desc_weighted is not None
            and desc_weighted.shape != (points.shape[0], pca_components.shape[1])
        ):
            raise RuntimeError(
                "manual_desc_weighted must be [N, D] aligned with manual_embed_points and "
                f"manual_pca_components, got {None if desc_weighted is None else desc_weighted.shape}"
            )

    return {
        "version": int(version),
        "manual_embed_points": points,
        "manual_embed_reducer": reducer_name,
        "manual_pca_components": pca_components,
        "manual_pca_mean": pca_mean,
        "manual_desc_weighted": desc_weighted,
        "manual_fader_p01": p01,
        "manual_fader_p99": p99,
        "frame_file_ids": frame_file_ids,
        "frame_t": frame_t,
        "kdtree_leafsize": max(1, leaf),
    }


def load_policy_v2_artifact(path: str, expected_frames: int) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy V2 artifact not found: {path}")

    artifact = np.load(path, allow_pickle=True)
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
    missing = [k for k in required if k not in artifact]
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
        raise RuntimeError("Policy V2 unit arrays are not aligned.")
    if frame_to_unit.shape[0] != int(expected_frames):
        raise RuntimeError(
            "Policy V2 frame_to_unit length mismatch: "
            f"{frame_to_unit.shape[0]} vs expected {expected_frames}"
        )
    if graph_neighbors.shape[0] != n_units or graph_scores.shape != graph_neighbors.shape:
        raise RuntimeError("Policy V2 graph arrays are invalid.")
    if np.any(unit_start < 0) or np.any(unit_end <= unit_start):
        raise RuntimeError("Policy V2 unit boundaries contain invalid ranges.")
    if np.any(unit_end > int(expected_frames)):
        raise RuntimeError("Policy V2 unit_end_idx exceeds corpus frame count.")
    if np.any(frame_to_unit < 0) or np.any(frame_to_unit >= n_units):
        raise RuntimeError("Policy V2 frame_to_unit contains invalid unit ids.")
    if unit_entry.ndim != 2 or unit_exit.ndim != 2 or unit_delta.ndim != 2:
        raise RuntimeError("Policy V2 unit descriptor arrays must be rank-2.")
    if not (
        unit_entry.shape[1] == unit_exit.shape[1] == unit_delta.shape[1]
        and unit_entry.shape[1] > 0
    ):
        raise RuntimeError("Policy V2 descriptor dimensionality is invalid.")

    out = {
        "unit_start_idx": unit_start.astype(np.int32),
        "unit_end_idx": unit_end.astype(np.int32),
        "unit_file_id": unit_file.astype(np.int32),
        "unit_len": unit_len.astype(np.int32),
        "unit_entry_desc": unit_entry.astype(np.float32),
        "unit_exit_desc": unit_exit.astype(np.float32),
        "unit_delta_desc": unit_delta.astype(np.float32),
        "frame_to_unit": frame_to_unit.astype(np.int32),
        "unit_graph_neighbors": graph_neighbors.astype(np.int32),
        "unit_graph_scores": graph_scores.astype(np.float32),
    }
    for key in ("unit_min_frames", "unit_max_frames", "unit_target_frames"):
        if key in artifact:
            out[key] = np.asarray(artifact[key], dtype=np.int32).reshape(-1)
    return out


def _resolve_manual_artifact_path(corpus_dir: str, arg_path: Optional[str]) -> str:
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
        raise FileNotFoundError(
            "Manual navigation artifact not found. Expected "
            f"'{default_path}' or a matching manual_navigation*.npz in {corpus_dir}."
        )


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
            raise FileNotFoundError(f"Reorganized units artifact not found: {resolved}")
        return resolved

    default_path = os.path.join(corpus_dir, "policy_v2_units.npz")
    if os.path.exists(default_path):
        return default_path
    try:
        return find_latest(corpus_dir, "policy_v2_units*.npz")
    except FileNotFoundError:
        return None


class TransportController:
    """Owns runtime transport state for start/stop and mode selection."""

    POLICY_QUEUE_SIZE = 4
    MANUAL_QUEUE_SIZE = 2
    MANUAL_WINDOW_MIN = 1
    MANUAL_WINDOW_MAX = 64
    MANUAL_WINDOW_SLEW_MAX_STEP = 4
    TARGET_BUFFER_HIGH_POLICY = 1.0
    TARGET_BUFFER_HIGH_MANUAL = 0.22
    MANUAL_PREBUFFER_SEC = 0.22

    def __init__(
        self,
        args,
        nav: LatentNavigationEngine,
        manual_engine: ManualNavigationEngine,
        manifold: ManifoldConstrainedGenerator,
        decoder: DecoderPlayer,
        vae,
        Z_concat: np.ndarray,
        frame_file_ids: np.ndarray,
        Z_mean: np.ndarray,
        Z_std: np.ndarray,
        initial_mode: str,
        latent_frame_sec: float = 0.0465,
    ):
        self.args = args
        self.nav = nav
        self.manual = manual_engine
        self.manifold = manifold
        self.decoder = decoder
        self.vae = vae
        self.Z_concat = np.asarray(Z_concat, dtype=np.float32)
        self.frame_file_ids = np.asarray(frame_file_ids, dtype=np.int32).reshape(-1)
        self.Z_mean = np.asarray(Z_mean, dtype=np.float32)
        self.Z_std = np.asarray(Z_std, dtype=np.float32)

        self.MANUAL_FRAME_SEC = float(latent_frame_sec)
        self.selected_mode = str(initial_mode)
        self._active_mode = str(initial_mode)
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._latent_queue: Optional[queue.Queue] = None
        self._nav_thread: Optional[threading.Thread] = None
        self._decode_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None

        self._decoder_started = False
        self._policy_state = {}
        self._manual_last_index = 0
        self._manual_last_distance = 0.0
        self._manual_window_size = max(
            self.MANUAL_WINDOW_MIN,
            min(self.MANUAL_WINDOW_MAX, int(args.manual_window_size)),
        )
        self._manual_window_runtime_size = int(self._manual_window_size)
        self._manual_buffer_ratio = float(
            np.clip(float(getattr(args, "manual_buffer_ratio", 0.15)), 0.1, 2.0)
        )
        self._manual_fader_motion_threshold = float(
            np.clip(float(args.manual_fader_motion_threshold), 0.0, 1.0)
        )
        self._manual_dither_amount = float(
            np.clip(float(getattr(args, "manual_dither", 0.0)), 0.0, 1.0)
        )
        self._manual_prev_half: Optional[np.ndarray] = None
        self._manual_live_faders = np.full(self.manual.control_dim, 0.5, dtype=np.float32)
        self._manual_render_faders = np.full(
            self.manual.control_dim, 0.5, dtype=np.float32
        )

        self._runtime_stats = {
            "current_window": int(max(2, min(args.window_size, 64))),
            "current_hop": int((max(2, min(args.window_size, 64)) + 1) // 2),
            "window_changes": 0,
            "decode_ms_last": 0.0,
            "decode_ms_sum": 0.0,
            "decode_count": 0,
        }

    def is_running(self) -> bool:
        return bool(self._running.is_set())

    def set_mode(self, mode: str) -> Tuple[bool, str]:
        mode = str(mode)
        if mode not in ("random", "reorganized", "manual"):
            return False, f"invalid mode '{mode}'"
        if mode == "reorganized" and not self.nav.has_variant("reorganized"):
            return False, "reorganized mode unavailable (missing units artifact)"
        with self._lock:
            if self._running.is_set():
                return False, "cannot change mode while transport is running"
            self.selected_mode = mode
        return True, f"mode set to {mode}"

    def set_manual_faders(self, faders) -> Tuple[bool, str]:
        try:
            next_faders = np.asarray(faders, dtype=np.float32).reshape(-1)
            if next_faders.shape[0] != self.manual.control_dim:
                raise ValueError(
                    f"Expected {self.manual.control_dim} control values, got {next_faders.shape[0]}"
                )
            next_faders = np.clip(next_faders, 0.0, 1.0)
        except Exception as exc:
            return False, str(exc)

        dropped = 0
        with self._lock:
            prev_faders = self._manual_live_faders.copy()
            fader_delta = float(np.max(np.abs(next_faders - prev_faders)))
            self._manual_live_faders = next_faders.copy()
            running_manual = self._running.is_set() and self._active_mode == "manual"
            if not running_manual:
                self._manual_render_faders = next_faders.copy()

        if not running_manual:
            self.manual.set_faders(next_faders)

        self._set_manual_window_stats(
            int(self._manual_window_size),
            changed=False,
        )
        if running_manual:
            motion_detected = fader_delta >= self._manual_fader_motion_threshold
            if motion_detected:
                # Drop queued stale chunks so next decoded chunk includes recent controls.
                dropped = self._drop_pending_latents()

        msg_parts = [f"manual faders updated (delta={fader_delta:.3f})"]
        if dropped > 0:
            msg_parts.append(f"dropped {dropped} queued batches")
        return True, ", ".join(msg_parts)

    def set_manual_axis(self, axis: int, value) -> Tuple[bool, str]:
        try:
            axis_i = int(axis)
            value_f = float(value)
        except Exception:
            return False, f"invalid manual axis/value: axis={axis}, value={value}"

        if axis_i < 0 or axis_i >= int(self.manual.control_dim):
            return False, f"manual axis out of range: {axis_i}"

        with self._lock:
            controls = self._manual_live_faders.copy()
        controls[axis_i] = float(np.clip(value_f, 0.0, 1.0))
        ok, msg = self.set_manual_faders(controls.tolist())
        if ok:
            return True, f"manual axis {axis_i} set to {controls[axis_i]:.3f}"
        return False, msg

    def set_manual_wander_params(
        self, k: int = None, speed: float = None
    ) -> Tuple[bool, str]:
        """Set manual navigation wandering parameters."""
        try:
            self.manual.set_wander_params(k=k, speed=speed)
        except Exception as exc:
            return False, str(exc)
        return True, f"manual wander params updated (k={k}, speed={speed})"

    def set_manual_window_size(self, window_size) -> Tuple[bool, str]:
        try:
            requested = int(window_size)
        except Exception:
            return False, f"invalid manual window size: {window_size}"

        clamped = max(self.MANUAL_WINDOW_MIN, min(self.MANUAL_WINDOW_MAX, requested))
        with self._lock:
            changed = clamped != self._manual_window_size
            self._manual_window_size = int(clamped)

        if changed:
            return True, f"manual window target set to {clamped} (slewed for smooth audio)"
        return True, f"manual window size set to {clamped}"

    def set_manual_buffer_ratio(self, ratio) -> Tuple[bool, str]:
        try:
            ratio_f = float(ratio)
        except Exception:
            return False, f"invalid manual buffer ratio: {ratio}"
        ratio_f = float(np.clip(ratio_f, 0.1, 2.0))
        with self._lock:
            self._manual_buffer_ratio = ratio_f
        return True, f"manual buffer ratio set to {ratio_f:.2f}"

    def set_manual_dither(self, amount) -> Tuple[bool, str]:
        try:
            amount_f = float(amount)
        except Exception:
            return False, f"invalid manual dither amount: {amount}"
        amount_f = float(np.clip(amount_f, 0.0, 1.0))
        with self._lock:
            self._manual_dither_amount = amount_f
        return True, f"manual dither set to {amount_f:.3f}"

    def stop(self) -> Tuple[bool, str]:
        if not self._running.is_set():
            return True, "already stopped"

        self._running.clear()

        if self._nav_thread is not None:
            self._nav_thread.join(timeout=1.0)
            self._nav_thread = None
        if self._decode_thread is not None:
            self._decode_thread.join(timeout=1.0)
            self._decode_thread = None
        if self._stats_thread is not None:
            self._stats_thread.join(timeout=1.0)
            self._stats_thread = None

        if self._decoder_started:
            try:
                self.decoder.stop()
            except Exception:
                pass
            self._decoder_started = False

        self.decoder.reset_buffers()
        self._latent_queue = None
        return True, "transport stopped"

    def close(self):
        self.stop()
        self.decoder.close()

    def start(self) -> Tuple[bool, str]:
        with self._lock:
            if self._running.is_set():
                return False, "transport already running"
            mode = self.selected_mode
            self._active_mode = mode

        self.decoder.reset_buffers()
        if mode == "manual":
            self._start_manual()
        else:
            if not self.nav.has_variant(mode):
                return False, f"{mode} mode unavailable (missing artifact/model inputs)"
            self._start_navigation_variant(mode)

        return True, f"transport started ({mode})"

    def _start_navigation_variant(self, variant: str):
        variant_name = str(variant)
        if not self.nav.set_policy_variant(variant_name):
            raise RuntimeError(f"Unknown navigation variant: {variant_name}")
        print(f"[info] Starting transport ({variant_name} mode)")
        self._latent_queue = queue.Queue(maxsize=self.POLICY_QUEUE_SIZE)
        self._runtime_stats["decode_ms_last"] = 0.0
        self._runtime_stats["decode_ms_sum"] = 0.0
        self._runtime_stats["decode_count"] = 0

        window_size = int(max(2, min(self.args.window_size, 64)))
        hop_size = (window_size + 1) // 2

        if self.args.fixed_window:
            print(
                f"[info] Fixed window sizing enabled (window={window_size}, hop={hop_size})"
            )
        elif self.args.boundary_window_updates:
            print(
                f"[info] Adaptive window sizing with boundary-only updates (initial={window_size})"
            )
        else:
            print(
                f"[info] Adaptive window sizing (initial={window_size}, model-predicted)"
            )

        # Pre-buffer: fill audio buffer with ~1 second of audio before starting stream.
        print("[info] Pre-buffering audio...")
        audio_per_hop = hop_size * self.MANUAL_FRAME_SEC
        num_hops = max(4, int(1.0 / audio_per_hop) + 1)

        frame_buffer = [self.nav.step() for _ in range(window_size)]
        z_batch_norm = self.manifold.generate_batch(
            frame_buffer, exploration=self.nav.get_active_jump_rate(variant=variant_name)
        )
        z_batch_raw = z_batch_norm * self.Z_std + self.Z_mean
        audio = decode_latents(self.vae, z_batch_raw)
        self.decoder.write_frame(audio)
        prev_half = frame_buffer[hop_size:]

        for _ in range(num_hops - 1):
            new_frames = [self.nav.step() for _ in range(hop_size)]
            full_window = prev_half + new_frames
            z_batch_norm = self.manifold.generate_batch(
                full_window, exploration=self.nav.get_active_jump_rate(variant=variant_name)
            )
            z_batch_raw = z_batch_norm * self.Z_std + self.Z_mean
            audio = decode_latents(self.vae, z_batch_raw)
            self.decoder.write_frame(audio)
            prev_half = full_window[hop_size:]

        print(
            f"[info] Pre-buffered {num_hops} windows ({self.decoder.buffer_duration():.2f}s)"
        )
        print("[info] Pre-filling latent queue...")
        for _ in range(self._latent_queue.maxsize):
            new_frames = [self.nav.step() for _ in range(hop_size)]
            full_window = prev_half + new_frames
            z_batch_norm = self.manifold.generate_batch(
                full_window, exploration=self.nav.get_active_jump_rate(variant=variant_name)
            )
            z_batch_raw = z_batch_norm * self.Z_std + self.Z_mean
            self._latent_queue.put(z_batch_raw)
            prev_half = full_window[hop_size:]
        print(f"[info] Latent queue filled ({self._latent_queue.qsize()} batches)")

        self._policy_state = {
            "prev_half": prev_half,
            "current_window_log2": math.log2(max(2, window_size)),
            "window_size": int(window_size),
            "hop_size": int(hop_size),
            "variant": variant_name,
        }

        self._start_threads(nav_loop=self._policy_nav_loop)

    def _start_manual(self):
        print("[info] Starting transport (manual mode)")
        self._latent_queue = queue.Queue(maxsize=self.MANUAL_QUEUE_SIZE)
        self._runtime_stats["decode_ms_last"] = 0.0
        self._runtime_stats["decode_ms_sum"] = 0.0
        self._runtime_stats["decode_count"] = 0
        with self._lock:
            self._manual_render_faders = self._manual_live_faders.copy()
            self._manual_window_runtime_size = int(self._manual_window_size)
            start_window = int(self._manual_window_runtime_size)
        self._set_manual_window_stats(start_window, changed=False)
        manual_target_buffer_high = self._manual_target_buffer_high()
        manual_prebuffer_sec = max(self.MANUAL_PREBUFFER_SEC, manual_target_buffer_high)
        hop_size = (start_window + 1) // 2
        print(
            "[info] Manual decode window config: "
            f"window={start_window}, hop={hop_size}, queue={self.MANUAL_QUEUE_SIZE}, "
            f"target_buf={manual_target_buffer_high:.2f}s"
        )

        # Scale prebuffer with window size so large decode batches do not underrun.
        # Use latent-level overlap-add: first batch is full window, subsequent
        # batches generate hop_size new frames prepended with prev_half.
        print("[info] Pre-buffering manual audio (with latent OLA)...")
        pre_hops = max(4, int(manual_prebuffer_sec / (hop_size * self.MANUAL_FRAME_SEC)) + 1)

        # First batch: full window
        z_raw = self._next_manual_latent_batch(start_window)
        audio = decode_latents(self.vae, z_raw)
        self.decoder.write_frame(audio)
        prev_half = z_raw[hop_size:]

        for _ in range(pre_hops - 1):
            new_frames = self._next_manual_latent_batch(hop_size)
            full_window = np.concatenate([prev_half, new_frames], axis=0)
            audio = decode_latents(self.vae, full_window)
            self.decoder.write_frame(audio)
            prev_half = full_window[hop_size:]

        print(
            f"[info] Pre-buffered {pre_hops} manual windows ({self.decoder.buffer_duration():.2f}s)"
        )

        # Fill latent queue with overlap
        for _ in range(self._latent_queue.maxsize):
            window_size = self._manual_batch_window_size()
            hop = (window_size + 1) // 2
            new_frames = self._next_manual_latent_batch(hop)
            full_window = np.concatenate([prev_half, new_frames], axis=0)
            self._latent_queue.put(full_window)
            prev_half = full_window[hop:]

        self._manual_prev_half = prev_half
        print(
            f"[info] Manual latent queue filled ({self._latent_queue.qsize()} batches)"
        )

        self._start_threads(nav_loop=self._manual_nav_loop)

    def _drop_pending_latents(self) -> int:
        if self._latent_queue is None:
            return 0
        dropped = 0
        while True:
            try:
                self._latent_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def _start_threads(self, nav_loop):
        if not self._decoder_started:
            self.decoder.start()
            self._decoder_started = True

        self._running.set()
        self._nav_thread = threading.Thread(target=nav_loop, daemon=True)
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._nav_thread.start()
        self._decode_thread.start()

        if self.args.audio_stats:
            self._stats_thread = threading.Thread(
                target=self._audio_stats_loop, daemon=True
            )
            self._stats_thread.start()

    def _next_manual_latent(self) -> np.ndarray:
        frame = self.manual.step()
        idx = int(np.clip(frame.nearest_index, 0, self.Z_concat.shape[0] - 1))
        dist = float(frame.distance)
        with self._stats_lock:
            self._manual_last_index = idx
            self._manual_last_distance = dist
        z_norm = self.Z_concat[idx]
        z_raw = z_norm[None, :] * self.Z_std[None, :] + self.Z_mean[None, :]
        return z_raw.astype(np.float32)

    def _next_manual_latent_batch(self, window_size: int) -> np.ndarray:
        window = max(1, int(window_size))
        with self._lock:
            start_faders = self._manual_render_faders.copy()
            target_faders = self._manual_live_faders.copy()

        if window == 1:
            fader_path = target_faders[None, :]
        else:
            alphas = np.linspace(0.0, 1.0, window, dtype=np.float32)[:, None]
            fader_path = (1.0 - alphas) * start_faders[None, :] + alphas * target_faders[None, :]

        batch = np.empty((window, self.Z_concat.shape[1]), dtype=np.float32)
        for i in range(window):
            frame = self.manual.step_with_faders(fader_path[i])
            idx = int(np.clip(frame.nearest_index, 0, self.Z_concat.shape[0] - 1))
            dist = float(frame.distance)
            with self._stats_lock:
                self._manual_last_index = idx
                self._manual_last_distance = dist
            z_norm = self.Z_concat[idx]
            batch[i] = z_norm * self.Z_std + self.Z_mean

        with self._lock:
            self._manual_render_faders = target_faders.copy()
            dither = float(self._manual_dither_amount)

        if dither > 0.0:
            noise = np.random.randn(*batch.shape).astype(np.float32)
            batch += noise * (dither * self.Z_std)

        return batch

    def _set_manual_window_stats(self, window_size: int, changed: bool) -> None:
        with self._stats_lock:
            if changed:
                self._runtime_stats["window_changes"] += 1
            self._runtime_stats["current_window"] = int(window_size)
            self._runtime_stats["current_hop"] = (int(window_size) + 1) // 2

    def _manual_batch_window_size(self) -> int:
        with self._lock:
            target = int(self._manual_window_size)
            current = int(self._manual_window_runtime_size)
            changed = False
            if current != target:
                delta = target - current
                step = min(
                    self.MANUAL_WINDOW_SLEW_MAX_STEP,
                    max(1, (abs(delta) // 8) + 1),
                )
                current += int(np.sign(delta)) * min(abs(delta), int(step))
                self._manual_window_runtime_size = int(current)
                changed = True
        self._set_manual_window_stats(int(current), changed=changed)
        return int(current)

    def _manual_target_buffer_high(self) -> float:
        with self._lock:
            window = int(max(self.MANUAL_WINDOW_MIN, self._manual_window_runtime_size))
            ratio = float(self._manual_buffer_ratio)
        chunk_sec = float(window) * float(self.MANUAL_FRAME_SEC)
        return float(
            max(
                self.TARGET_BUFFER_HIGH_MANUAL,
                min(3.0, ratio * chunk_sec),
            )
        )

    def _policy_nav_loop(self):
        nav_mode = "fixed" if self.args.fixed_window else "adaptive"
        if (not self.args.fixed_window) and self.args.boundary_window_updates:
            nav_mode = "adaptive-boundary"
        variant = str(self._policy_state.get("variant", self._active_mode))
        print(f"[info] Navigation loop started ({variant}, {nav_mode})")

        frame_buffer = []
        prev_half = list(self._policy_state["prev_half"])
        current_window_log2 = float(self._policy_state["current_window_log2"])
        pending_window_log2 = current_window_log2
        active_window = int(self._policy_state["window_size"])
        active_hop = int(self._policy_state["hop_size"])
        prev_window_for_stats = int(active_window)

        try:
            while self._running.is_set():
                if self.args.fixed_window:
                    current_window = int(self._policy_state["window_size"])
                    current_hop = (current_window + 1) // 2
                elif self.args.boundary_window_updates:
                    current_window = int(active_window)
                    current_hop = int(active_hop)
                else:
                    current_window = max(2, min(64, round(2**current_window_log2)))
                    current_hop = (current_window + 1) // 2

                frame = self.nav.step()
                frame_buffer.append(frame)

                if not self.args.fixed_window:
                    target_log2 = math.log2(max(2, frame.predicted_window_size))
                    if self.args.boundary_window_updates:
                        pending_window_log2 = smooth_window_log2(
                            pending_window_log2, target_log2
                        )
                    else:
                        current_window_log2 = smooth_window_log2(
                            current_window_log2, target_log2
                        )

                if len(frame_buffer) >= current_hop:
                    target_prev_len = current_window - current_hop
                    if len(prev_half) < target_prev_len:
                        while len(prev_half) < target_prev_len:
                            prev_half.append(
                                prev_half[-1] if prev_half else frame_buffer[0]
                            )
                    elif len(prev_half) > target_prev_len:
                        prev_half = prev_half[-target_prev_len:]

                    full_window = prev_half + frame_buffer[:current_hop]
                    z_batch_norm = self.manifold.generate_batch(
                        full_window,
                        exploration=self.nav.get_active_jump_rate(variant=variant),
                    )
                    z_batch_raw = z_batch_norm * self.Z_std + self.Z_mean

                    try:
                        self._latent_queue.put(z_batch_raw, timeout=1.0)
                    except queue.Full:
                        pass

                    prev_half = full_window[current_hop:]
                    frame_buffer = frame_buffer[current_hop:]

                    with self._stats_lock:
                        if current_window != prev_window_for_stats:
                            self._runtime_stats["window_changes"] += 1
                            prev_window_for_stats = current_window
                        self._runtime_stats["current_window"] = int(current_window)
                        self._runtime_stats["current_hop"] = int(current_hop)

                    if (
                        not self.args.fixed_window
                    ) and self.args.boundary_window_updates:
                        active_window = max(2, min(64, round(2**pending_window_log2)))
                        active_hop = (active_window + 1) // 2
                        current_window_log2 = pending_window_log2
        except Exception as e:
            print(f"[error] Navigation loop exception: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("[info] Navigation loop stopped")

    def _manual_nav_loop(self):
        print("[info] Manual navigation loop started (with latent OLA)")
        prev_half = self._manual_prev_half
        try:
            while self._running.is_set():
                if self._latent_queue.full():
                    time.sleep(0.005)
                    continue
                window_size = self._manual_batch_window_size()
                hop_size = (window_size + 1) // 2
                new_frames = self._next_manual_latent_batch(hop_size)
                if prev_half is not None and len(prev_half) > 0:
                    full_window = np.concatenate([prev_half, new_frames], axis=0)
                else:
                    full_window = new_frames
                prev_half = full_window[hop_size:]
                try:
                    self._latent_queue.put(full_window, timeout=0.1)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"[error] Manual navigation loop exception: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("[info] Manual navigation loop stopped")

    def _decode_loop(self):
        print("[info] Decode loop started")
        try:
            while self._running.is_set():
                buf_dur = self.decoder.buffer_duration()
                with self._lock:
                    active_mode = self._active_mode
                target_buffer_high = (
                    self._manual_target_buffer_high()
                    if active_mode == "manual"
                    else self.TARGET_BUFFER_HIGH_POLICY
                )
                if buf_dur > target_buffer_high:
                    time.sleep(0.01 if active_mode == "manual" else 0.05)
                    continue

                try:
                    z_batch = self._latent_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                decode_t0 = time.perf_counter()
                audio = decode_latents(self.vae, z_batch)
                decode_ms = (time.perf_counter() - decode_t0) * 1000.0
                with self._stats_lock:
                    self._runtime_stats["decode_ms_last"] = float(decode_ms)
                    self._runtime_stats["decode_ms_sum"] += float(decode_ms)
                    self._runtime_stats["decode_count"] += 1
                self.decoder.write_frame(audio)
        except Exception as e:
            print(f"[error] Decode loop exception: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("[info] Decode loop stopped")

    def _audio_stats_loop(self):
        stats_interval = max(0.1, float(self.args.audio_stats_interval))
        last_underruns = int(self.decoder.underruns)
        print(f"[info] Audio stats logging enabled (interval={stats_interval:.2f}s)")
        while self._running.is_set():
            time.sleep(stats_interval)
            curr_underruns = int(self.decoder.underruns)
            delta_underruns = curr_underruns - last_underruns
            last_underruns = curr_underruns
            qsize = (
                int(self._latent_queue.qsize()) if self._latent_queue is not None else 0
            )
            with self._stats_lock:
                current_window = int(self._runtime_stats["current_window"])
                current_hop = int(self._runtime_stats["current_hop"])
                window_changes = int(self._runtime_stats["window_changes"])
                decode_ms_last = float(self._runtime_stats["decode_ms_last"])
                decode_count = int(self._runtime_stats["decode_count"])
                decode_ms_avg = float(self._runtime_stats["decode_ms_sum"]) / max(
                    1, decode_count
                )
            print(
                f"[audio] buf={self.decoder.buffer_duration():.3f}s "
                f"queue={qsize} "
                f"decode_ms_last={decode_ms_last:.1f} "
                f"decode_ms_avg={decode_ms_avg:.1f} "
                f"window={current_window}/{current_hop} "
                f"window_changes={window_changes} "
                f"underruns_total={curr_underruns} "
                f"underruns_delta={delta_underruns:+d}"
            )

    def get_extra_state(self) -> dict:
        with self._lock:
            selected_mode = self.selected_mode
            manual_decode_window = int(self._manual_window_size)
            manual_decode_window_active = int(self._manual_window_runtime_size)
            manual_buffer_ratio = float(self._manual_buffer_ratio)
        manual_target_buffer_high = float(
            max(
                self.TARGET_BUFFER_HIGH_MANUAL,
                min(
                    3.0,
                    manual_buffer_ratio
                    * manual_decode_window_active
                    * self.MANUAL_FRAME_SEC,
                ),
            )
        )
        manual_engine_state = self.manual.get_state()
        with self._stats_lock:
            # Ensure all values are JSON-serializable
            faders = manual_engine_state.get(
                "faders", [0.5] * int(self.manual.control_dim)
            )
            if hasattr(faders, "tolist"):
                faders = faders.tolist()
            elif not isinstance(faders, list):
                faders = [float(f) for f in faders]

            # Convert all fader values to float to ensure JSON serialization
            faders_float = [float(f) for f in faders]
            position = manual_engine_state.get("position")
            if hasattr(position, "tolist"):
                position = position.tolist()
            if isinstance(position, list):
                position = [float(v) for v in position]

            manual_info = {
                "nearest_index": int(self._manual_last_index),
                "distance": float(self._manual_last_distance),
                "current_file_id": int(
                    self.frame_file_ids[
                        int(np.clip(self._manual_last_index, 0, self.frame_file_ids.shape[0] - 1))
                    ]
                ),
                "faders": faders_float,
                "control_dim": int(manual_engine_state.get("control_dim", 3)),
                "position": position,
                "decode_window": manual_decode_window,
                "decode_window_active": manual_decode_window_active,
                "decode_window_min": int(self.MANUAL_WINDOW_MIN),
                "decode_window_max": int(self.MANUAL_WINDOW_MAX),
                "buffer_ratio": manual_buffer_ratio,
                "target_buffer_high": manual_target_buffer_high,
                "search_mode": str(manual_engine_state.get("search_mode", "legacy")),
                "reducer": str(manual_engine_state.get("reducer", "pca")),
                "coarse_k": int(manual_engine_state.get("coarse_k", 0)),
                "refine_k": int(manual_engine_state.get("refine_k", 0)),
                "wander_k": int(manual_engine_state.get("wander_k", 1)),
                "wander_speed": float(manual_engine_state.get("wander_speed", 0.0)),
                "is_wandering": bool(manual_engine_state.get("is_wandering", False)),
                "wander_progress": float(
                    manual_engine_state.get("wander_progress", 0.0)
                ),
                "dither": float(self._manual_dither_amount),
            }
        return {
            "transport": {
                "running": bool(self._running.is_set()),
                "selected_mode": selected_mode,
            },
            "navigation_mode": selected_mode,
            "manual": manual_info,
        }

    def handle_ws_message(self, data: dict) -> bool:
        msg_type = data.get("type", "")
        if msg_type == "transport":
            action = data.get("action", "")
            if action == "set_mode":
                mode = data.get("mode", "")
                ok, msg = self.set_mode(mode)
                print(f"[ws] transport set_mode({mode}) -> {msg}")
                return True
            elif action == "start":
                ok, msg = self.start()
                print(f"[ws] transport start -> {msg}")
                return True
            elif action == "stop":
                ok, msg = self.stop()
                print(f"[ws] transport stop -> {msg}")
                return True
            else:
                print(f"[ws] Unknown transport action: {action}")
                return False

        if msg_type in ("random_control", "control"):
            controls = data.get("controls", {})
            if isinstance(controls, dict):
                self.nav.set_random_controls(**controls)
            return True

        if msg_type == "reorganized_control":
            controls = data.get("controls", {})
            if isinstance(controls, dict):
                self.nav.set_reorganized_controls(**controls)
            return True

        if msg_type == "manual_controls":
            faders = data.get("faders", [])
            ok, msg = self.set_manual_faders(faders)
            if not ok:
                print(f"[ws] manual_controls error: {msg}")
            return True

        if msg_type == "manual_wander":
            wander_k = data.get("k")
            wander_speed = data.get("speed")
            ok, msg = self.set_manual_wander_params(k=wander_k, speed=wander_speed)
            if not ok:
                print(f"[ws] manual_wander error: {msg}")
            return True

        if msg_type == "manual_window":
            window_size = data.get("size")
            ok, msg = self.set_manual_window_size(window_size)
            if not ok:
                print(f"[ws] manual_window error: {msg}")
            return True

        if msg_type == "manual_buffer":
            ratio = data.get("ratio")
            ok, msg = self.set_manual_buffer_ratio(ratio)
            if not ok:
                print(f"[ws] manual_buffer error: {msg}")
            return True

        if msg_type == "manual_dither":
            amount = data.get("amount", 0.0)
            ok, msg = self.set_manual_dither(amount)
            print(f"[ws] manual_dither -> {msg}")
            return True

        return False


def main():
    ap = argparse.ArgumentParser(
        description="Real-time decoding with selectable manual/random/reorganized navigation."
    )
    ap.add_argument(
        "--corpus_dir", required=True, help="Directory containing corpus.npz"
    )
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0",
                    help="HuggingFace model ID (legacy, use corpus vae_id instead).")
    ap.add_argument("--vae_id", default="", help="VAE adapter ID override (reads from corpus if empty).")
    ap.add_argument("--vae_weight_path", default="", help="Path to local weight file (for VAEs that require it).")
    ap.add_argument("--osc_ip", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument(
        "--osc_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print incoming OSC messages (including unmapped paths).",
    )
    ap.add_argument(
        "--ws_port",
        type=int,
        default=8765,
        help="WebSocket port for visualization (0 to disable).",
    )
    ap.add_argument(
        "--manual_artifact",
        default=None,
        help="Path to manual navigation artifact (.npz). Defaults to <corpus_dir>/manual_navigation.npz",
    )
    ap.add_argument(
        "--initial_navigation_mode",
        choices=["manual", "random", "reorganized"],
        default="random",
        help="Navigation mode selected at startup.",
    )
    ap.add_argument(
        "--manual_wander_k",
        type=int,
        default=1,
        help="Manual mode wandering neighborhood size (1 disables wandering).",
    )
    ap.add_argument(
        "--manual_wander_speed",
        type=float,
        default=0.0001,
        help="Manual mode wander transition speed (0.0 instant, 1.0 slowest).",
    )
    ap.add_argument(
        "--manual_coarse_k",
        type=int,
        default=96,
        help="Manual two-stage search: coarse candidate count in control space.",
    )
    ap.add_argument(
        "--manual_refine_k",
        type=int,
        default=16,
        help="Manual two-stage search: refined neighbor count in descriptor space.",
    )
    ap.add_argument(
        "--manual_desc_interp_k",
        type=int,
        default=8,
        help="Manual two-stage search: descriptor interpolation neighbors for non-linear reducers.",
    )
    ap.add_argument(
        "--manual_window_size",
        type=int,
        default=6,
        help="Fixed manual decode batch size (larger = smoother, higher control latency).",
    )
    ap.add_argument(
        "--manual_fader_motion_threshold",
        type=float,
        default=0.01,
        help="Minimum max-abs fader delta to count as active motion.",
    )
    ap.add_argument(
        "--manual_buffer_ratio",
        type=float,
        default=0.15,
        help="Manual mode buffer target ratio relative to chunk duration (higher = safer, more latency).",
    )
    ap.add_argument(
        "--manual_dither",
        type=float,
        default=0.0,
        help="Latent micro-dither amount (0.0 = off, 0.01-0.05 typical). "
             "Adds small noise to latent vectors before decoding to break "
             "periodic pitch artifacts from repeated frames.",
    )
    ap.add_argument(
        "--autostart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start transport immediately on launch (default: false).",
    )

    # Decoder controls.
    ap.add_argument(
        "--output_gain", type=float, default=1.0, help="Initial output gain (0-2)."
    )
    ap.add_argument(
        "--audio_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print periodic audio transport stats (buffer duration + underruns delta).",
    )
    ap.add_argument(
        "--audio_stats_interval",
        type=float,
        default=1.0,
        help="Seconds between audio stats logs when --audio_stats is enabled.",
    )

    # Manifold parameters.
    ap.add_argument("--manifold_k", type=int, default=16)
    ap.add_argument("--manifold_n_local", type=int, default=8)
    ap.add_argument("--manifold_n_global", type=int, default=32)
    ap.add_argument("--manifold_sparse_quantile", type=float, default=0.75)

    # Random model/runtime arguments.
    ap.add_argument(
        "--random_model_path",
        default=None,
        help="Random mode checkpoint (.pt). Defaults to latest latent_policy_*.pt in --corpus_dir.",
    )
    ap.add_argument(
        "--random_temperature",
        type=float,
        default=1.0,
        help="Base temperature for random mode policy sampling.",
    )
    ap.add_argument(
        "--random_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stochastically sample in random mode (default: yes).",
    )
    ap.add_argument(
        "--random_timbre_swap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable corpus-locked timbre-aware swapping in random mode.",
    )
    ap.add_argument(
        "--random_recompose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable short-unit recomposition in random mode.",
    )

    # Reorganized model/runtime arguments.
    ap.add_argument(
        "--reorganized_units_path",
        default=None,
        help="Path to policy_v2_units.npz (default: <corpus_dir>/policy_v2_units.npz).",
    )
    ap.add_argument(
        "--reorganized_model_path",
        default=None,
        help="Optional trained reorganized transition model (.pt). Defaults to latest policy_v2_*.pt in --corpus_dir.",
    )
    ap.add_argument(
        "--reorganized_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for reorganized unit selection.",
    )

    # Random control parameters.
    ap.add_argument(
        "--ctrl_phrase_scale",
        type=float,
        default=0.40,
        help="Expected morphology phrase scale (0=short, 1=long, variable duration).",
    )
    ap.add_argument(
        "--ctrl_jump_rate",
        type=float,
        default=0.55,
        help="Recomposition jump rate (0-1).",
    )
    ap.add_argument(
        "--ctrl_timbre_lock",
        type=float,
        default=0.55,
        help="Timbre lock strength (0-1).",
    )
    ap.add_argument(
        "--ctrl_drift",
        type=float,
        default=0.50,
        help="Timbre target drift (0-1).",
    )
    ap.add_argument(
        "--ctrl_repeat_avoid",
        type=float,
        default=0.75,
        help="Anti-repeat/anti-serial pressure (0-1).",
    )
    ap.add_argument(
        "--ctrl_crossfile",
        type=float,
        default=0.70,
        help="Cross-file allowance (0-1).",
    )

    # Reorganized control parameters.
    ap.add_argument(
        "--ctrl_morph_len",
        type=float,
        default=0.50,
        help="Relative morphology unit length target in reorganized mode.",
    )
    ap.add_argument(
        "--ctrl_reorg_jump_rate",
        type=float,
        default=0.60,
        help="Reorganized mode candidate-pool breadth.",
    )
    ap.add_argument(
        "--ctrl_reorg_timbre_lock",
        type=float,
        default=0.45,
        help="Reorganized mode entry/exit timbre continuity strength.",
    )
    ap.add_argument(
        "--ctrl_evolution",
        type=float,
        default=0.60,
        help="Reorganized mode evolution pressure over unit timbre trajectory.",
    )
    ap.add_argument(
        "--ctrl_novelty",
        type=float,
        default=0.50,
        help="Reorganized mode anti-repeat + stochastic novelty pressure.",
    )
    ap.add_argument(
        "--ctrl_reorg_crossfile",
        type=float,
        default=0.70,
        help="Reorganized mode cross-file allowance.",
    )

    # Window size for random/reorganized batched decoding.
    ap.add_argument(
        "--window_size",
        type=int,
        default=2,
        help="Initial window size in latent frames (2-64).",
    )
    ap.add_argument(
        "--fixed_window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable adaptive window sizing and keep --window_size constant.",
    )
    ap.add_argument(
        "--boundary_window_updates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In adaptive mode, apply window-size changes only at window boundaries.",
    )

    args = ap.parse_args()

    corpus_npz = find_corpus_file(args.corpus_dir)
    print(f"[info] Using corpus: {corpus_npz}")

    data = load_corpus(corpus_npz)
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError("Corpus missing latent geometry. Re-run preprocess.py.")

    Z_concat = data["Z_concat"].astype(np.float32)
    Z_mean = data["Z_mean"].astype(np.float32)
    Z_std = data["Z_std"].astype(np.float32)
    corpus_vae_id = str(_read_scalar(data, "vae_id", ""))
    corpus_sr = int(_read_scalar(data, "sr", 44100))
    corpus_latent_hz = float(_read_scalar(data, "latent_hz", 21.5))
    if not np.isfinite(corpus_latent_hz) or corpus_latent_hz <= 0.0:
        raise RuntimeError(f"Corpus latent_hz must be positive, got {corpus_latent_hz}")
    latent_frame_sec = 1.0 / corpus_latent_hz

    manual_artifact_path = _resolve_manual_artifact_path(
        args.corpus_dir, args.manual_artifact
    )
    manual_data = load_manual_artifact(
        manual_artifact_path, expected_frames=Z_concat.shape[0]
    )
    print(f"[info] Using manual artifact: {manual_artifact_path}")

    random_model_path = _resolve_optional_model_path(
        args.corpus_dir, args.random_model_path, "latent_policy_*.pt"
    )
    if random_model_path is None:
        print(
            "[warn] No random model checkpoint found; random mode will use heuristic fallback."
        )
    else:
        print(f"[info] Using random model checkpoint: {random_model_path}")

    reorganized_units_path = _resolve_reorganized_units_path(
        args.corpus_dir, args.reorganized_units_path
    )
    v2_data = None
    if reorganized_units_path is not None:
        v2_data = load_policy_v2_artifact(
            reorganized_units_path,
            expected_frames=Z_concat.shape[0],
        )
        print(f"[info] Using reorganized units artifact: {reorganized_units_path}")
    elif args.initial_navigation_mode == "reorganized":
        raise RuntimeError(
            "Reorganized mode requires policy_v2_units.npz. Re-run preprocess.py or set --reorganized_units_path."
        )
    else:
        print(
            "[warn] No reorganized units artifact found; reorganized mode will be unavailable."
        )

    reorganized_model_path = _resolve_optional_model_path(
        args.corpus_dir, args.reorganized_model_path, "policy_v2_*.pt"
    )
    if reorganized_model_path is None:
        print(
            "[warn] No reorganized model checkpoint found; reorganized mode will run heuristic scoring."
        )
    else:
        print(f"[info] Using reorganized model checkpoint: {reorganized_model_path}")

    nav = load_navigation_engine(
        data=data,
        random_model_path=random_model_path,
        random_temperature=args.random_temperature,
        random_sample=bool(args.random_sample),
        desc_weighted=manual_data["manual_desc_weighted"],
        random_timbre_swap=bool(args.random_timbre_swap),
        random_recompose=bool(args.random_recompose),
        random_phrase_scale=args.ctrl_phrase_scale,
        random_jump_rate=args.ctrl_jump_rate,
        random_timbre_lock=args.ctrl_timbre_lock,
        random_drift=args.ctrl_drift,
        random_repeat_avoid=args.ctrl_repeat_avoid,
        random_crossfile=args.ctrl_crossfile,
        reorganized_enabled=bool(v2_data is not None),
        reorganized_artifact=v2_data,
        reorganized_model_path=reorganized_model_path,
        reorganized_temperature=args.reorganized_temperature,
        reorganized_morph_len=args.ctrl_morph_len,
        reorganized_jump_rate=args.ctrl_reorg_jump_rate,
        reorganized_timbre_lock=args.ctrl_reorg_timbre_lock,
        reorganized_evolution=args.ctrl_evolution,
        reorganized_novelty=args.ctrl_novelty,
        reorganized_crossfile=args.ctrl_reorg_crossfile,
        policy_variant=(
            args.initial_navigation_mode
            if args.initial_navigation_mode in ("random", "reorganized")
            else "random"
        ),
        latent_frame_seconds=latent_frame_sec,
    )

    manifold_cfg = ManifoldConfig(
        k=args.manifold_k,
        n_local=args.manifold_n_local,
        n_global=args.manifold_n_global,
        sparse_quantile=args.manifold_sparse_quantile,
    )
    manifold = ManifoldConstrainedGenerator(nav.GG, geometry, manifold_cfg)

    manual_engine = ManualNavigationEngine(
        manual_points=manual_data["manual_embed_points"],
        fader_p01=manual_data["manual_fader_p01"],
        fader_p99=manual_data["manual_fader_p99"],
        desc_weighted=manual_data["manual_desc_weighted"],
        pca_components=manual_data["manual_pca_components"],
        pca_mean=manual_data["manual_pca_mean"],
        reducer=str(manual_data["manual_embed_reducer"]),
        desc_interp_k=int(args.manual_desc_interp_k),
        coarse_k=int(args.manual_coarse_k),
        refine_k=int(args.manual_refine_k),
        leafsize=int(manual_data["kdtree_leafsize"]),
        wander_k=int(args.manual_wander_k),
        wander_speed=float(args.manual_wander_speed),
    )
    print(
        "[info] Manual wandering config: "
        f"k={int(args.manual_wander_k)}, speed={float(args.manual_wander_speed):.4f}"
    )
    print(
        "[info] Manual decode config: "
        f"window={int(args.manual_window_size)}, "
        f"motion_eps={float(args.manual_fader_motion_threshold):.3f}, "
        f"buffer_ratio={float(args.manual_buffer_ratio):.2f}"
    )
    print(
        "[info] Manual retrieval config: "
        f"artifact_v={int(manual_data['version'])}, reducer={manual_data['manual_embed_reducer']}, "
        f"mode={manual_engine.search_mode}, coarse_k={int(args.manual_coarse_k)}, "
        f"refine_k={int(args.manual_refine_k)}, interp_k={int(args.manual_desc_interp_k)}"
    )

    # Load VAE: prefer vae_id from corpus, fall back to CLI args
    vae_id = args.vae_id or corpus_vae_id
    if vae_id:
        print(f"[info] Loading VAE adapter: {vae_id}")
        vae = load_vae_adapter(vae_id, weight_path=args.vae_weight_path)
    else:
        print("[info] Loading VAE decoder (legacy)...")
        vae = load_vae(args.pretrained)
    decoder = DecoderPlayer(gain=args.output_gain, sr=corpus_sr)

    controller = TransportController(
        args=args,
        nav=nav,
        manual_engine=manual_engine,
        manifold=manifold,
        decoder=decoder,
        vae=vae,
        Z_concat=Z_concat,
        frame_file_ids=manual_data["frame_file_ids"],
        Z_mean=Z_mean,
        Z_std=Z_std,
        initial_mode=args.initial_navigation_mode,
        latent_frame_sec=latent_frame_sec,
    )

    ws_server = None
    ws_thread = None
    shutdown_requested = threading.Event()

    def request_shutdown(reason: str = "web"):
        """Request graceful process shutdown from non-main threads (e.g. WebSocket)."""
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        print(f"[info] Shutdown requested via {reason}")
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as e:
            print(f"[warn] Failed to signal shutdown: {e}")

    if args.ws_port > 0:
        try:
            from stable_audio_wanderer.runtime.ws_server import start_ws_server

            ws_server, ws_thread = start_ws_server(
                nav,
                decoder,
                port=args.ws_port,
                on_exit_request=request_shutdown,
                message_handler=controller.handle_ws_message,
                extra_state_provider=controller.get_extra_state,
                manual_points_3d=manual_data["manual_embed_points"],
                manual_file_ids=manual_data["frame_file_ids"],
                manual_fader_p01=manual_data["manual_fader_p01"],
                manual_fader_p99=manual_data["manual_fader_p99"],
            )
            print(f"[info] WebSocket server running on ws://127.0.0.1:{args.ws_port}")
        except Exception as e:
            print(f"[warn] Failed to start WebSocket server: {e}")

    print(f"[info] Running with {nav.N} segments")
    print(f"[info] Selected mode: {args.initial_navigation_mode}")
    print(
        "[info] Random controls: "
        f"phrase_scale={args.ctrl_phrase_scale}, jump_rate={args.ctrl_jump_rate}, "
        f"timbre_lock={args.ctrl_timbre_lock}, drift={args.ctrl_drift}, "
        f"repeat_avoid={args.ctrl_repeat_avoid}, crossfile={args.ctrl_crossfile}"
    )
    print(
        "[info] Reorganized controls: "
        f"morph_len={args.ctrl_morph_len}, jump_rate={args.ctrl_reorg_jump_rate}, "
        f"timbre_lock={args.ctrl_reorg_timbre_lock}, evolution={args.ctrl_evolution}, "
        f"novelty={args.ctrl_novelty}, crossfile={args.ctrl_reorg_crossfile}"
    )
    print(
        f"[info] Random timbre swap: enabled={bool(args.random_timbre_swap)}, "
        f"descriptors={'yes' if manual_data['manual_desc_weighted'] is not None else 'no'}"
    )
    print(
        f"[info] Random recomposition: enabled={bool(args.random_recompose)}"
    )
    print(
        "[info] Reorganized availability: "
        f"enabled={bool(v2_data is not None)}, "
        f"artifact={'yes' if v2_data is not None else 'no'}, "
        f"model={'yes' if reorganized_model_path is not None else 'no'}, "
        f"temperature={float(args.reorganized_temperature):.2f}"
    )
    print("[info] Transport is idle. Use the web UI Start button to begin decoding.")

    if args.autostart:
        ok, msg = controller.start()
        print(f"[info] Autostart: {msg}")

    try:
        run_server(
            nav,
            decoder,
            ip=args.osc_ip,
            port=args.osc_port,
            manual_controller=controller,
            osc_debug=bool(args.osc_debug),
        )
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        controller.close()
        if ws_server is not None:
            ws_server.shutdown()

    print("[info] Done.")


if __name__ == "__main__":
    main()
