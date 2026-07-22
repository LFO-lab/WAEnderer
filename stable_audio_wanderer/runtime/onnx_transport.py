"""Unified-Web transport for the CPU SAME-S ONNX presentation backend.

The standalone ``bin/perform.py`` transport intentionally remains Torch based.
This controller owns the narrower Web presentation path: fixed manifest windows,
worker-side ONNX decoding and overlap-add, file-bounded manual windows, and
generation-tagged live window changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .decoder_player import DecoderPlayer, GENERATION_TRANSITION_SAMPLES
from .manual_windows import build_file_bounded_latent_window
from .overlap_add import StreamingFullOverlapAdd
from .wander_windows import WanderWindowPlanner
from ..vae.onnx_decoder import SameSOnnxDecoder


@dataclass(frozen=True)
class _DecodeRequest:
    generation: int
    window: int
    raw_latents: np.ndarray


class OnnxTransportController:
    """Fail-closed transport used only by the unified Web pipeline."""

    LATENT_QUEUE_SIZE = 4
    MIN_PREBUFFER_SECONDS = 0.120

    def __init__(
        self,
        *,
        nav,
        manual_engine,
        manifold,
        player: DecoderPlayer,
        latent_decoder: SameSOnnxDecoder,
        z_concat: np.ndarray,
        file_offsets: np.ndarray,
        frame_file_ids: np.ndarray,
        z_mean: np.ndarray,
        z_std: np.ndarray,
        manual_points: Optional[np.ndarray] = None,
        unit_start_idx: Optional[np.ndarray] = None,
        unit_end_idx: Optional[np.ndarray] = None,
        unit_graph_neighbors: Optional[np.ndarray] = None,
        unit_graph_scores: Optional[np.ndarray] = None,
        initial_mode: str = "random",
        initial_window: int = 2,
    ) -> None:
        self.nav = nav
        self.manual = manual_engine
        self.manifold = manifold
        self.decoder = player
        self.latent_decoder = latent_decoder

        self.Z_concat = np.ascontiguousarray(z_concat, dtype=np.float32)
        self.file_offsets = np.asarray(file_offsets, dtype=np.int64).reshape(-1)
        self.frame_file_ids = np.asarray(frame_file_ids, dtype=np.int32).reshape(-1)
        self.Z_mean = np.asarray(z_mean, dtype=np.float32).reshape(-1)
        self.Z_std = np.asarray(z_std, dtype=np.float32).reshape(-1)
        if self.Z_concat.ndim != 2 or self.Z_concat.shape[1] != 256:
            raise ValueError(f"SAME-S corpus latents must be [N,256], got {self.Z_concat.shape}")
        if self.file_offsets.size < 2 or self.file_offsets[-1] != self.Z_concat.shape[0]:
            raise ValueError("file_offsets must cover every corpus latent frame")
        if self.frame_file_ids.shape[0] != self.Z_concat.shape[0]:
            raise ValueError("frame_file_ids must align with Z_concat")
        if self.Z_mean.shape != (256,) or self.Z_std.shape != (256,):
            raise ValueError("Z_mean and Z_std must both have shape [256]")

        if manual_points is None:
            # Compatibility for direct/test construction. Production passes the
            # corpus manual embedding explicitly.
            coordinate = np.linspace(
                0.0, 1.0, self.Z_concat.shape[0], dtype=np.float32
            )
            manual_points = np.column_stack((coordinate, coordinate))
        self.wander_planner = WanderWindowPlanner(
            self.Z_concat,
            self.file_offsets,
            manual_points,
            self.Z_mean,
            self.Z_std,
            unit_start_idx=unit_start_idx,
            unit_end_idx=unit_end_idx,
            unit_graph_neighbors=unit_graph_neighbors,
            unit_graph_scores=unit_graph_scores,
        )

        requested_window = int(initial_window)
        if requested_window not in latent_decoder.supported_windows:
            raise ValueError(
                f"Decoder window T{requested_window} is unavailable; "
                f"supported: {latent_decoder.supported_windows}"
            )

        self.selected_mode = str(initial_mode)
        self._active_mode = str(initial_mode)
        self._manual_faders = np.full(
            int(self.manual.control_dim), 0.5, dtype=np.float32
        )
        self._manual_last_index = 0
        self._manual_last_distance = 0.0

        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._running = threading.Event()
        self._latent_queue: Optional[queue.Queue[_DecodeRequest]] = None
        self._producer_thread: Optional[threading.Thread] = None
        self._decode_thread: Optional[threading.Thread] = None
        self._decoder_started = False

        self._generation_counter = 0
        self._requested_generation = 0
        self._requested_window = requested_window
        self._generation_windows: dict[int, int] = {}
        self._wander_frame_source = "k_nearest"
        self._wander_frame_order = 0.0
        self._wander_latent_colour = 0.0
        self._wander_reset_serial = 0
        self._transport_error: Optional[str] = None
        self._prebuffering = False
        self._decode_times_ms: deque[float] = deque(maxlen=512)
        self._decode_count = 0
        self._last_decode_ms = 0.0

    def is_running(self) -> bool:
        return self._running.is_set()

    def _drop_pending_requests(self) -> int:
        latent_queue = self._latent_queue
        if latent_queue is None:
            return 0
        dropped = 0
        while True:
            try:
                latent_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped

    def set_mode(self, mode: str) -> Tuple[bool, str]:
        mode = str(mode)
        if mode not in ("random", "reorganized", "manual"):
            return False, f"invalid mode {mode!r}"
        if mode == "reorganized" and not self.nav.has_variant("reorganized"):
            return False, "reorganized mode unavailable (missing units artifact)"
        with self._lock:
            if self._running.is_set():
                return False, "cannot change mode while transport is running"
            self.selected_mode = mode
        return True, f"mode set to {mode}"

    def set_manual_faders(self, faders) -> Tuple[bool, str]:
        try:
            values = np.asarray(faders, dtype=np.float32).reshape(-1)
            if values.shape != (int(self.manual.control_dim),):
                raise ValueError(
                    f"Expected {self.manual.control_dim} controls, got {values.shape[0]}"
                )
            values = np.clip(values, 0.0, 1.0)
        except Exception as exc:
            return False, str(exc)

        with self._lock:
            previous = self._manual_faders.copy()
            self._manual_faders = values.copy()
            running_manual = self._running.is_set() and self._active_mode == "manual"
        if not running_manual:
            self.manual.set_faders(values)
        dropped = self._drop_pending_requests() if running_manual else 0
        delta = float(np.max(np.abs(values - previous)))
        return True, f"manual faders updated (delta={delta:.3f}, dropped={dropped})"

    def set_manual_wander_params(self, k=None, speed=None) -> Tuple[bool, str]:
        try:
            self.manual.set_wander_params(k=k, speed=speed)
        except Exception as exc:
            return False, str(exc)
        if self._running.is_set() and self._active_mode == "manual":
            self._drop_pending_requests()
        return True, "manual wander parameters updated"

    def set_decoder_window(self, size) -> Tuple[bool, str]:
        try:
            window = int(size)
        except (TypeError, ValueError):
            return False, f"invalid decoder window: {size!r}"
        if window not in self.latent_decoder.supported_windows:
            return False, (
                f"decoder window T{window} is unavailable; "
                f"supported: {self.latent_decoder.supported_windows}"
            )

        with self._lock:
            if window == self._requested_window:
                return True, f"decoder window already T{window}"
            self._requested_window = window
            self._wander_reset_serial += 1
            if self._running.is_set():
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = window
                generation = self._requested_generation
            else:
                generation = None
        if generation is not None:
            # Announce before any replacement PCM is available.  This removes a
            # previously staged generation immediately and makes a late result
            # fail at write_hop even if it passed the controller snapshot check.
            self.decoder.request_generation(generation)
            dropped = self._drop_pending_requests()
            return True, (
                f"staging decoder window T{window} as generation {generation} "
                f"(dropped {dropped} stale requests)"
            )
        return True, f"decoder window set to T{window}"

    def set_wander_render_controls(self, controls) -> Tuple[bool, str]:
        """Apply decoder-window rendering controls without renaming Random policy APIs."""
        if not isinstance(controls, dict):
            return False, "wander_render controls must be an object"

        valid_sources = {"k_nearest", "contiguous", "morphology_graph"}
        with self._lock:
            source = self._wander_frame_source
            frame_order = self._wander_frame_order
            latent_colour = self._wander_latent_colour

            candidate_source = controls.get("frame_source")
            if isinstance(candidate_source, str) and candidate_source in valid_sources:
                source = candidate_source

            for key, current in (
                ("frame_order", frame_order),
                ("latent_colour", latent_colour),
            ):
                if key not in controls:
                    continue
                try:
                    parsed = float(controls[key])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(parsed):
                    continue
                if key == "frame_order":
                    frame_order = float(np.clip(parsed, 0.0, 1.0))
                else:
                    latent_colour = float(np.clip(parsed, 0.0, 1.0))

            source_changed = source != self._wander_frame_source
            changed = (
                source_changed
                or frame_order != self._wander_frame_order
                or latent_colour != self._wander_latent_colour
            )
            if not changed:
                return True, "wander render controls unchanged"

            self._wander_frame_source = source
            self._wander_frame_order = frame_order
            self._wander_latent_colour = latent_colour
            if source_changed:
                self._wander_reset_serial += 1

            if self._running.is_set() and self._active_mode == "random":
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = self._requested_window
                generation = self._requested_generation
            else:
                generation = None

        if generation is not None:
            self.decoder.request_generation(generation)
            dropped = self._drop_pending_requests()
            return True, (
                f"staging Wander render generation {generation} "
                f"(dropped {dropped} stale requests)"
            )
        return True, "wander render controls updated"

    def reset_wander(self, idx=None) -> Tuple[bool, str]:
        self.nav.reset_policy(idx=idx)
        with self._lock:
            self._wander_reset_serial += 1
            if self._running.is_set() and self._active_mode == "random":
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = self._requested_window
                generation = self._requested_generation
            else:
                generation = None
        if generation is not None:
            self.decoder.request_generation(generation)
            self._drop_pending_requests()
        return True, "Wander reset"

    def _cleanup_previous_run(self) -> None:
        self._running.clear()
        if self._producer_thread is not None:
            self._producer_thread.join()
            self._producer_thread = None
        if self._decode_thread is not None:
            self._decode_thread.join()
            self._decode_thread = None
        if self._decoder_started:
            self.decoder.stop()
            self._decoder_started = False
        self.decoder.reset_buffers()
        self._latent_queue = None

    def start(self) -> Tuple[bool, str]:
        with self._lock:
            if self._running.is_set():
                return False, "transport already running"
        # A failed run may have stopped its worker while the audio stream finishes
        # the error fade.  A new Start is the only operation that clears the latch.
        self._cleanup_previous_run()

        with self._lock:
            mode = self.selected_mode
            if mode != "manual" and not self.nav.has_variant(mode):
                return False, f"{mode} mode unavailable"
            if mode != "manual" and not self.nav.set_policy_variant(mode):
                return False, f"could not select navigation mode {mode}"
            self._active_mode = mode
            self._generation_counter += 1
            generation = self._generation_counter
            window = self._requested_window
            self._requested_generation = generation
            self._generation_windows = {generation: window}
            self._transport_error = None
            self._prebuffering = True
            self._wander_reset_serial += 1

        self.decoder.request_generation(generation)

        with self._stats_lock:
            self._decode_times_ms.clear()
            self._decode_count = 0
            self._last_decode_ms = 0.0

        self._latent_queue = queue.Queue(maxsize=self.LATENT_QUEUE_SIZE)
        self._running.set()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="saw-onnx-latent-producer",
        )
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            daemon=True,
            name="saw-onnx-decode",
        )
        self._producer_thread.start()
        self._decode_thread.start()
        return True, f"transport prebuffering ({mode}, T{window})"

    def stop(self) -> Tuple[bool, str]:
        had_resources = any(
            (
                self._running.is_set(),
                self._producer_thread is not None,
                self._decode_thread is not None,
                self._decoder_started,
            )
        )
        self._cleanup_previous_run()
        with self._lock:
            self._prebuffering = False
        return True, "transport stopped" if had_resources else "already stopped"

    def close(self) -> None:
        self.stop()
        self.decoder.close()

    def _snapshot_request(self) -> tuple[int, int, str]:
        with self._lock:
            return (
                int(self._requested_generation),
                int(self._requested_window),
                str(self._active_mode),
            )

    def _manual_request(self, generation: int, window: int) -> _DecodeRequest:
        with self._lock:
            faders = self._manual_faders.copy()
        # Exactly one spatial query chooses the source-file anchor for a decode.
        frame = self.manual.step_with_faders(faders)
        anchor = int(np.clip(frame.nearest_index, 0, self.Z_concat.shape[0] - 1))
        raw, _plan = build_file_bounded_latent_window(
            self.Z_concat,
            self.file_offsets,
            anchor,
            window,
            mean=self.Z_mean,
            std=self.Z_std,
        )
        with self._stats_lock:
            self._manual_last_index = anchor
            self._manual_last_distance = float(frame.distance)
        return _DecodeRequest(generation, window, raw)

    def _wander_request(
        self,
        generation: int,
        window: int,
        previous_generation: Optional[int],
        planner_reset_serial: int,
    ) -> tuple[_DecodeRequest, int]:
        """Advance the policy at the existing cadence, then plan one full T window."""
        hop = (window + 1) // 2
        step_count = window if generation != previous_generation else hop
        anchor = None
        for _ in range(step_count):
            anchor = self.nav.step(fixed_retrieval_window=window)
        if anchor is None:
            raise RuntimeError("Wander navigation produced no anchor")
        anchor_frame = int(getattr(anchor, "nearest_idx", anchor))
        with self._lock:
            frame_source = self._wander_frame_source
            frame_order = self._wander_frame_order
            latent_colour = self._wander_latent_colour
            reset_serial = self._wander_reset_serial
        if reset_serial != planner_reset_serial:
            self.wander_planner.reset()
            planner_reset_serial = reset_serial
        controls = self.nav.get_random_controls()
        planned = self.wander_planner.plan(
            anchor_frame,
            window,
            frame_source=frame_source,
            frame_order=frame_order,
            latent_colour=latent_colour,
            **controls,
        )
        return _DecodeRequest(generation, window, planned.raw_latents), planner_reset_serial

    def _producer_loop(self) -> None:
        previous_frames = []
        previous_generation = None
        planner_reset_serial = -1
        try:
            while self._running.is_set():
                latent_queue = self._latent_queue
                if latent_queue is None:
                    return
                if latent_queue.full():
                    time.sleep(0.002)
                    continue

                generation, window, mode = self._snapshot_request()
                if mode == "manual":
                    request = self._manual_request(generation, window)
                elif mode == "random":
                    request, planner_reset_serial = self._wander_request(
                        generation,
                        window,
                        previous_generation,
                        planner_reset_serial,
                    )
                    previous_generation = generation
                else:
                    def next_frame():
                        return self.nav.step(fixed_retrieval_window=window)

                    hop = (window + 1) // 2
                    if generation != previous_generation:
                        frames = [next_frame() for _ in range(window)]
                    else:
                        tail_count = window - hop
                        tail = list(previous_frames[-tail_count:]) if tail_count else []
                        if len(tail) < tail_count:
                            tail = [next_frame() for _ in range(tail_count - len(tail))] + tail
                        frames = tail + [next_frame() for _ in range(hop)]
                    z_norm = self.manifold.generate_batch(
                        frames,
                        exploration=self.nav.get_active_jump_rate(variant=mode),
                    )
                    raw = np.ascontiguousarray(
                        z_norm * self.Z_std[None, :] + self.Z_mean[None, :],
                        dtype=np.float32,
                    )
                    request = _DecodeRequest(generation, window, raw)
                    previous_frames = frames
                    previous_generation = generation

                try:
                    latent_queue.put(request, timeout=0.1)
                except queue.Full:
                    continue
        except Exception as exc:
            self._latch_runtime_error(f"latent producer failed: {exc}")

    def _target_prebuffer_seconds(self, window: int) -> float:
        metadata = self.latent_decoder.metadata_for(window)
        return max(
            self.MIN_PREBUFFER_SECONDS,
            metadata.audio_hop_samples / float(metadata.sample_rate),
        )

    def _decode_loop(self) -> None:
        assemblers: dict[int, StreamingFullOverlapAdd] = {}
        try:
            while self._running.is_set():
                generation, window, _mode = self._snapshot_request()
                target_buffer = self._target_prebuffer_seconds(window)
                if self.decoder.generation_buffer_duration(generation) >= target_buffer:
                    time.sleep(0.002)
                    continue

                latent_queue = self._latent_queue
                if latent_queue is None:
                    return
                try:
                    request = latent_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if (
                    request.generation != generation
                    or request.window != window
                    or not self._running.is_set()
                ):
                    continue

                decoded = self.latent_decoder.decode(request.raw_latents)
                latest_generation, latest_window, _ = self._snapshot_request()
                if (
                    request.generation != latest_generation
                    or request.window != latest_window
                    or not self._running.is_set()
                ):
                    continue

                assembler = assemblers.get(request.generation)
                if assembler is None:
                    assembler = StreamingFullOverlapAdd(
                        decoded.metadata.audio_hop_samples,
                        channels=decoded.metadata.channels,
                    )
                    assemblers = {request.generation: assembler}
                hop = assembler.push(decoded.audio.T)

                latest_generation, latest_window, _ = self._snapshot_request()
                if (
                    request.generation != latest_generation
                    or request.window != latest_window
                    or not self._running.is_set()
                ):
                    continue
                if not self.decoder.write_hop(hop, generation=request.generation):
                    continue

                with self._stats_lock:
                    self._last_decode_ms = float(decoded.decode_time_ms)
                    self._decode_times_ms.append(float(decoded.decode_time_ms))
                    self._decode_count += 1

                if not self._decoder_started:
                    buffered = self.decoder.generation_buffer_duration(request.generation)
                    if buffered >= target_buffer:
                        self.decoder.start()
                        self._decoder_started = True
                        with self._lock:
                            self._prebuffering = False
        except Exception as exc:
            self._latch_runtime_error(f"ONNX decoder runtime failure: {exc}")

    def _latch_runtime_error(self, message: str) -> None:
        with self._lock:
            if self._transport_error is None:
                self._transport_error = str(message)
            self._prebuffering = False
        self._running.clear()
        if self._decoder_started:
            self.decoder.fade_to_silence(GENERATION_TRANSITION_SAMPLES)

    def _decoder_state(self) -> dict:
        with self._lock:
            requested_generation = int(self._requested_generation)
            requested_window = int(self._requested_window)
            error = self._transport_error
            prebuffering = bool(self._prebuffering)
            running = bool(self._running.is_set())
            generation_windows = dict(self._generation_windows)
        current_generation = self.decoder.current_generation
        selected_window = generation_windows.get(
            current_generation, requested_window
        )
        player_state = self.decoder.get_state()
        transition_status = str(player_state.get("transition_status", "idle"))
        if error:
            transition_status = "error"
        elif prebuffering:
            transition_status = "prebuffering"
        elif (
            running
            and current_generation != requested_generation
            and transition_status == "idle"
        ):
            transition_status = "staging"

        metadata = self.latent_decoder.metadata_for(requested_window)
        with self._stats_lock:
            timings = np.asarray(self._decode_times_ms, dtype=np.float64)
            timing = {
                "last_ms": float(self._last_decode_ms),
                "p99_ms": float(np.percentile(timings, 99.0)) if timings.size else 0.0,
                "count": int(self._decode_count),
            }
        info = self.latent_decoder.info
        resource_path = getattr(info, "resource_path", None)
        state = {
            "backend": self.latent_decoder.info.backend,
            "provider": self.latent_decoder.info.provider,
            "vae_id": self.latent_decoder.info.vae_id,
            "supported_windows": list(self.latent_decoder.supported_windows),
            "selected_window": int(selected_window),
            "requested_window": requested_window,
            "latent_hop": int(metadata.latent_hop),
            "audio_hop_samples": int(metadata.audio_hop_samples),
            "audio_window_samples": int(metadata.audio_window_samples),
            "decode_timing": timing,
            "transition_status": transition_status,
            "generation": current_generation,
            "requested_generation": requested_generation,
            "underruns": int(player_state.get("underruns", 0)),
            "buffer_duration": float(player_state.get("buffer_duration", 0.0)),
            "error": error,
        }
        if resource_path is not None:
            state["resource_path"] = str(resource_path)
            state["model_path"] = str(info.model_path)
        else:
            state["bundle_path"] = str(info.bundle_path)
        return state

    def get_extra_state(self) -> dict:
        with self._lock:
            selected_mode = self.selected_mode
            error = self._transport_error
            faders = self._manual_faders.tolist()
            requested_window = int(self._requested_window)
            frame_source = self._wander_frame_source
            frame_order = float(self._wander_frame_order)
            latent_colour = float(self._wander_latent_colour)
        manual_state = self.manual.get_state()
        wander_diagnostics = self.wander_planner.last_diagnostics
        effective_source = (
            "contiguous"
            if frame_source == "morphology_graph"
            and not self.wander_planner.graph_available
            else frame_source
        )
        if (
            wander_diagnostics is not None
            and wander_diagnostics.requested_frame_source == frame_source
        ):
            effective_source = wander_diagnostics.effective_frame_source
        with self._stats_lock:
            manual_index = int(self._manual_last_index)
            manual_distance = float(self._manual_last_distance)
        return {
            "transport": {
                "running": bool(self._running.is_set()),
                "selected_mode": selected_mode,
                "error": error,
            },
            "navigation_mode": selected_mode,
            "wander_render": {
                "frame_source": frame_source,
                "requested_frame_source": frame_source,
                "effective_frame_source": effective_source,
                "graph_available": bool(self.wander_planner.graph_available),
                "frame_order": frame_order,
                "latent_colour": latent_colour,
                "seed": int(self.wander_planner.seed),
            },
            "manual": {
                **manual_state,
                "nearest_index": manual_index,
                "distance": manual_distance,
                "current_file_id": int(
                    self.frame_file_ids[
                        int(np.clip(manual_index, 0, self.frame_file_ids.shape[0] - 1))
                    ]
                ),
                "faders": [float(value) for value in faders],
                "decode_window": requested_window,
                "decode_window_active": requested_window,
                "decode_window_min": int(min(self.latent_decoder.supported_windows)),
                "decode_window_max": int(max(self.latent_decoder.supported_windows)),
                "dither": 0.0,
            },
            "decoder": self._decoder_state(),
        }

    def handle_ws_message(self, data: dict) -> bool:
        msg_type = data.get("type", "")
        if msg_type == "transport":
            action = data.get("action", "")
            if action == "set_mode":
                ok, message = self.set_mode(data.get("mode", ""))
            elif action == "start":
                ok, message = self.start()
            elif action == "stop":
                ok, message = self.stop()
            else:
                return False
            print(f"[ws] ONNX transport {action}: {message}")
            return True
        if msg_type == "decoder_window":
            ok, message = self.set_decoder_window(data.get("size"))
            print(f"[ws] decoder window: {message}")
            return True
        if msg_type == "wander_render":
            ok, message = self.set_wander_render_controls(data.get("controls", {}))
            print(f"[ws] wander render: {message}")
            return True
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
            self.set_manual_faders(data.get("faders", []))
            return True
        if msg_type == "manual_wander":
            self.set_manual_wander_params(k=data.get("k"), speed=data.get("speed"))
            return True
        if msg_type == "reset":
            ok, message = self.reset_wander(idx=data.get("index"))
            print(f"[ws] reset: {message}")
            return True
        if msg_type in ("manual_window", "manual_dither"):
            # These legacy controls are intentionally unavailable in Web-ONNX mode.
            return True
        return False


__all__ = ["OnnxTransportController"]
