#!/usr/bin/env python3
"""
Real-time decoding with selectable policy/manual navigation modes.
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

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.policy.latent_geometry import load_geometry_from_dict
from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer
from stable_audio_wanderer.runtime.manifold import (
    ManifoldConfig,
    ManifoldConstrainedGenerator,
)
from stable_audio_wanderer.runtime.manual_player import ManualNavigationEngine
from stable_audio_wanderer.runtime.osc_server import run_server
from stable_audio_wanderer.runtime.player import LatentNavigationEngine
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
    policy_path: str = None,
    policy_temperature: float = 1.0,
    policy_sample: bool = True,
    control_width: float = 0.5,
    control_energy: float = 0.5,
    control_gravity: float = 0.5,
    control_memory: float = 0.0,
    control_coherence: float = 0.0,
    control_exploration: float = 0.0,
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
        policy_path=policy_path,
        policy_temperature=policy_temperature,
        policy_sample=policy_sample,
        control_width=control_width,
        control_energy=control_energy,
        control_gravity=control_gravity,
        control_memory=control_memory,
        control_coherence=control_coherence,
        control_exploration=control_exploration,
    )


def _read_scalar(data: dict, key: str, default):
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    return value


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

    if "manual_embed_points" in artifact:
        points = artifact["manual_embed_points"].astype(np.float32)
    elif "manual_pca_points" in artifact:
        points = artifact["manual_pca_points"].astype(np.float32)
    else:
        raise RuntimeError("Manual artifact missing manual_embed_points.")

    p01 = artifact["manual_fader_p01"].astype(np.float32).reshape(-1)
    p99 = artifact["manual_fader_p99"].astype(np.float32).reshape(-1)
    frame_file_ids = artifact["frame_file_ids"].astype(np.int32).reshape(-1)
    frame_t = artifact["frame_t"].astype(np.int32).reshape(-1)
    leaf = int(_read_scalar(artifact, "kdtree_leafsize", 32))
    version = int(_read_scalar(artifact, "version", 0))
    reducer_name = str(_read_scalar(artifact, "manual_embed_reducer", "pca")).lower()
    if reducer_name not in ("pca", "umap"):
        reducer_name = "pca"
    pca_components = None
    pca_mean = None
    desc_weighted = None
    if version == 3:
        if "manual_desc_weighted" not in artifact:
            raise RuntimeError("Manual artifact v3 missing manual_desc_weighted.")
        desc_weighted = artifact["manual_desc_weighted"].astype(np.float32)
        if "manual_pca_components" in artifact and "manual_pca_mean" in artifact:
            pca_components = artifact["manual_pca_components"].astype(np.float32)
            pca_mean = artifact["manual_pca_mean"].astype(np.float32).reshape(-1)
    elif version == 2:
        if "manual_desc_weighted" not in artifact:
            raise RuntimeError("Manual artifact v2 missing manual_desc_weighted.")
        desc_weighted = artifact["manual_desc_weighted"].astype(np.float32)
        if "manual_pca_components" in artifact and "manual_pca_mean" in artifact:
            pca_components = artifact["manual_pca_components"].astype(np.float32)
            pca_mean = artifact["manual_pca_mean"].astype(np.float32).reshape(-1)
        reducer_name = "pca"
        print(
            "[warn] Manual artifact version 2 loaded. "
            "Re-run preprocess.py/train_policy.py for v3 metadata."
        )
    elif version == 1:
        print(
            "[warn] Manual artifact version 1 loaded (legacy MFCC-only retrieval). "
            "Re-run train_policy.py --navigation_mode manual to enable descriptor rerank."
        )
        reducer_name = "pca"
    else:
        raise RuntimeError(f"Unsupported manual artifact version: {version}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"manual_embed_points must be [N, 3], got {points.shape}")
    if points.shape[0] != expected_frames:
        raise RuntimeError(
            f"manual_embed_points frame count mismatch: {points.shape[0]} vs expected {expected_frames}"
        )
    if (
        frame_file_ids.shape[0] != points.shape[0]
        or frame_t.shape[0] != points.shape[0]
    ):
        raise RuntimeError("Manual artifact frame metadata length mismatch.")
    if p01.shape[0] != 3 or p99.shape[0] != 3:
        raise RuntimeError("manual_fader_p01/manual_fader_p99 must both be shape [3].")
    if reducer_name == "pca" and version >= 2 and pca_components is None:
        raise RuntimeError(
            "PCA reducer requires manual_pca_components/manual_pca_mean in artifact."
        )
    if pca_components is not None:
        if pca_components.ndim != 2 or pca_components.shape[0] != 3:
            raise RuntimeError(
                f"manual_pca_components must be [3, D], got {pca_components.shape}"
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


class TransportController:
    """Owns runtime transport state for start/stop and mode selection."""

    POLICY_QUEUE_SIZE = 4
    MANUAL_QUEUE_SIZE = 2
    MANUAL_WINDOW_MIN = 1
    MANUAL_WINDOW_MAX = 64
    TARGET_BUFFER_HIGH_POLICY = 1.0
    TARGET_BUFFER_HIGH_MANUAL = 0.22
    MANUAL_PREBUFFER_SEC = 0.22
    MANUAL_FRAME_SEC = 0.0465

    def __init__(
        self,
        args,
        nav: LatentNavigationEngine,
        manual_engine: ManualNavigationEngine,
        manifold: ManifoldConstrainedGenerator,
        decoder: DecoderPlayer,
        vae,
        Z_concat: np.ndarray,
        Z_mean: np.ndarray,
        Z_std: np.ndarray,
        initial_mode: str,
    ):
        self.args = args
        self.nav = nav
        self.manual = manual_engine
        self.manifold = manifold
        self.decoder = decoder
        self.vae = vae
        self.Z_concat = np.asarray(Z_concat, dtype=np.float32)
        self.Z_mean = np.asarray(Z_mean, dtype=np.float32)
        self.Z_std = np.asarray(Z_std, dtype=np.float32)

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
        self._manual_fader_motion_threshold = float(
            np.clip(float(args.manual_fader_motion_threshold), 0.0, 1.0)
        )
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
        if mode not in ("policy", "manual"):
            return False, f"invalid mode '{mode}'"
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
            running_manual = self._running.is_set() and self._active_mode == "manual"

        self._set_manual_window_stats(self._manual_window_size, changed=changed)
        dropped = 0
        if running_manual and changed:
            dropped = self._drop_pending_latents()
        if dropped > 0:
            return True, f"manual window size set to {clamped} (dropped {dropped} queued batches)"
        return True, f"manual window size set to {clamped}"

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
        if mode == "policy":
            self._start_policy()
        else:
            self._start_manual()

        return True, f"transport started ({mode})"

    def _start_policy(self):
        print("[info] Starting transport (policy mode)")
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
                f"[info] Adaptive window sizing (initial={window_size}, policy-predicted)"
            )

        # Pre-buffer: fill audio buffer with ~1 second of audio before starting stream.
        print("[info] Pre-buffering audio...")
        audio_per_hop = hop_size * 0.0465
        num_hops = max(4, int(1.0 / audio_per_hop) + 1)

        frame_buffer = [self.nav.step() for _ in range(window_size)]
        z_batch_norm = self.manifold.generate_batch(
            frame_buffer, exploration=self.nav.ctrl_exploration
        )
        z_batch_raw = z_batch_norm * self.Z_std + self.Z_mean
        audio = decode_latents(self.vae, z_batch_raw)
        self.decoder.write_frame(audio)
        prev_half = frame_buffer[hop_size:]

        for _ in range(num_hops - 1):
            new_frames = [self.nav.step() for _ in range(hop_size)]
            full_window = prev_half + new_frames
            z_batch_norm = self.manifold.generate_batch(
                full_window, exploration=self.nav.ctrl_exploration
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
                full_window, exploration=self.nav.ctrl_exploration
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
        self._set_manual_window_stats(self._manual_window_size, changed=False)
        print(
            "[info] Manual decode window config: "
            f"window={self._manual_window_size}, queue={self.MANUAL_QUEUE_SIZE}"
        )

        # Keep manual prebuffer short for responsive fader changes.
        print("[info] Pre-buffering manual audio...")
        pre_frames = max(4, int(self.MANUAL_PREBUFFER_SEC / self.MANUAL_FRAME_SEC) + 1)
        pre_remaining = int(pre_frames)
        while pre_remaining > 0:
            batch_size = min(int(self._manual_window_size), pre_remaining)
            z_raw = self._next_manual_latent_batch(batch_size)
            audio = decode_latents(self.vae, z_raw)
            self.decoder.write_frame(audio)
            pre_remaining -= batch_size
        print(
            f"[info] Pre-buffered {pre_frames} manual frames ({self.decoder.buffer_duration():.2f}s)"
        )

        for _ in range(self._latent_queue.maxsize):
            batch_size = self._manual_batch_window_size()
            self._latent_queue.put(self._next_manual_latent_batch(batch_size))
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

        return batch

    def _set_manual_window_stats(self, window_size: int, changed: bool) -> None:
        with self._stats_lock:
            if changed:
                self._runtime_stats["window_changes"] += 1
            self._runtime_stats["current_window"] = int(window_size)
            self._runtime_stats["current_hop"] = int(window_size)

    def _manual_batch_window_size(self) -> int:
        self._set_manual_window_stats(self._manual_window_size, changed=False)
        return int(self._manual_window_size)

    def _policy_nav_loop(self):
        nav_mode = "fixed" if self.args.fixed_window else "adaptive"
        if (not self.args.fixed_window) and self.args.boundary_window_updates:
            nav_mode = "adaptive-boundary"
        print(f"[info] Navigation loop started ({nav_mode} mode)")

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
                        exploration=self.nav.ctrl_exploration,
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
        print("[info] Manual navigation loop started")
        try:
            while self._running.is_set():
                if self._latent_queue.full():
                    time.sleep(0.005)
                    continue
                batch_size = self._manual_batch_window_size()
                z_raw = self._next_manual_latent_batch(batch_size)
                try:
                    self._latent_queue.put(z_raw, timeout=0.1)
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
                    self.TARGET_BUFFER_HIGH_MANUAL
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
                "faders": faders_float,
                "control_dim": int(manual_engine_state.get("control_dim", 3)),
                "position": position,
                "decode_window": manual_decode_window,
                "decode_window_min": int(self.MANUAL_WINDOW_MIN),
                "decode_window_max": int(self.MANUAL_WINDOW_MAX),
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

        return False


def main():
    ap = argparse.ArgumentParser(
        description="Real-time decoding with selectable policy/manual navigation."
    )
    ap.add_argument(
        "--corpus_dir", required=True, help="Directory containing corpus.npz"
    )
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
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
        choices=["policy", "manual"],
        default="policy",
        help="Navigation mode selected at startup.",
    )
    ap.add_argument(
        "--manual_wander_k",
        type=int,
        default=4,
        help="Manual mode wandering neighborhood size (1 disables wandering).",
    )
    ap.add_argument(
        "--manual_wander_speed",
        type=float,
        default=0.5,
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
        "--smoothing",
        type=float,
        default=0.1,
        help="Crossfade smoothing between frames (0-1).",
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

    # Policy controls.
    ap.add_argument(
        "--policy_path", default=None, help="Checkpoint .pt for navigation policy."
    )
    ap.add_argument(
        "--policy_temperature",
        type=float,
        default=1.0,
        help="Base temperature for policy sampling.",
    )
    ap.add_argument(
        "--policy_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stochastically sample from policy (default: yes).",
    )

    # Control parameters (policy mode).
    ap.add_argument(
        "--ctrl_width",
        type=float,
        default=0.5,
        help="Initial width control (0-1): temperature scaling.",
    )
    ap.add_argument(
        "--ctrl_energy",
        type=float,
        default=0.5,
        help="Initial energy control (0-1): displacement magnitude.",
    )
    ap.add_argument(
        "--ctrl_gravity",
        type=float,
        default=0.5,
        help="Initial gravity control (0-1): forward/backward bias.",
    )
    ap.add_argument(
        "--ctrl_memory",
        type=float,
        default=0.0,
        help="Initial memory control (0-1): pull toward recent positions.",
    )
    ap.add_argument(
        "--ctrl_coherence",
        type=float,
        default=0.0,
        help="Initial coherence control (0-1): stay within same file.",
    )
    ap.add_argument(
        "--ctrl_exploration",
        type=float,
        default=0.0,
        help="Initial exploration control (0-1): entropy injection.",
    )

    # Window size for policy batched decoding.
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

    manual_artifact_path = args.manual_artifact
    if manual_artifact_path is None:
        manual_artifact_path = os.path.join(args.corpus_dir, "manual_navigation.npz")
    manual_data = load_manual_artifact(
        manual_artifact_path, expected_frames=Z_concat.shape[0]
    )
    print(f"[info] Using manual artifact: {manual_artifact_path}")

    nav = load_navigation_engine(
        data=data,
        policy_path=args.policy_path,
        policy_temperature=args.policy_temperature,
        policy_sample=bool(args.policy_sample),
        control_width=args.ctrl_width,
        control_energy=args.ctrl_energy,
        control_gravity=args.ctrl_gravity,
        control_memory=args.ctrl_memory,
        control_coherence=args.ctrl_coherence,
        control_exploration=args.ctrl_exploration,
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
        f"k={int(args.manual_wander_k)}, speed={float(args.manual_wander_speed):.2f}"
    )
    print(
        "[info] Manual decode config: "
        f"window={int(args.manual_window_size)}, "
        f"motion_eps={float(args.manual_fader_motion_threshold):.3f}"
    )
    print(
        "[info] Manual retrieval config: "
        f"artifact_v={int(manual_data['version'])}, reducer={manual_data['manual_embed_reducer']}, "
        f"mode={manual_engine.search_mode}, coarse_k={int(args.manual_coarse_k)}, "
        f"refine_k={int(args.manual_refine_k)}, interp_k={int(args.manual_desc_interp_k)}"
    )

    print("[info] Loading VAE decoder...")
    vae = load_vae(args.pretrained)
    decoder = DecoderPlayer(gain=args.output_gain, smoothing=args.smoothing)

    controller = TransportController(
        args=args,
        nav=nav,
        manual_engine=manual_engine,
        manifold=manifold,
        decoder=decoder,
        vae=vae,
        Z_concat=Z_concat,
        Z_mean=Z_mean,
        Z_std=Z_std,
        initial_mode=args.initial_navigation_mode,
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
        f"[info] Controls: width={args.ctrl_width}, energy={args.ctrl_energy}, "
        f"gravity={args.ctrl_gravity}, memory={args.ctrl_memory}"
    )
    print(
        f"[info] Advanced: coherence={args.ctrl_coherence}, exploration={args.ctrl_exploration}"
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
