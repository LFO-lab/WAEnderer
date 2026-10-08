"""Unified-Web VAE transport over a backend-neutral decoder contract.

The standalone ``bin/perform.py`` transport intentionally remains Torch based.
This controller owns the Web presentation path: validated decoder windows,
worker-side decoding and overlap-add, manual sequences/textures, and
generation-tagged live window changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .decoder_player import DecoderPlayer, GENERATION_TRANSITION_SAMPLES
from .manual_windows import build_file_bounded_latent_window
from .overlap_add import StreamingFullOverlapAdd
from .wander_windows import WanderWindowPlanner
from .window_controls import AdaptiveWindow, ManualTexture
from ..vae.decoder_contract import LatentDecoder


@dataclass(frozen=True)
class _DecodeRequest:
    generation: int
    window: int
    raw_latents: np.ndarray
    frame_indices: Tuple[int, ...]


@dataclass
class _DecodeLane:
    generation: int
    window: int
    planner: object
    texture: object
    texture_tail: object = None
    texture_content: object = None
    previous_frames: list = field(default_factory=list)
    previous_generation: object = None
    reset_serial: int = -1
    future: object = None
    request: object = None
    next_request: object = None
    assembler: object = None
    prepared: list = field(default_factory=list)
    prepared_samples: int = 0
    published: bool = False


class DecoderTransportController:
    """Fail-closed transport used only by the unified Web pipeline."""

    MIN_PREBUFFER_SECONDS = 0.120
    PREBUFFER_HOPS = 2

    def __init__(
        self,
        *,
        nav,
        manual_engine,
        manifold,
        player: DecoderPlayer,
        latent_decoder: LatentDecoder,
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
        initial_mode: str = "wander",
        initial_window: int = 2,
    ) -> None:
        self.nav = nav
        self.manual = manual_engine
        self.manifold = manifold
        self.decoder = player
        self.latent_decoder = latent_decoder
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._resources_closed = False

        self.Z_concat = np.ascontiguousarray(z_concat, dtype=np.float32)
        self.file_offsets = np.asarray(file_offsets, dtype=np.int64).reshape(-1)
        self.frame_file_ids = np.asarray(frame_file_ids, dtype=np.int32).reshape(-1)
        self.Z_mean = np.asarray(z_mean, dtype=np.float32).reshape(-1)
        self.Z_std = np.asarray(z_std, dtype=np.float32).reshape(-1)
        latent_dim = latent_decoder.metadata_for(initial_window).latent_dim
        if self.Z_concat.ndim != 2 or self.Z_concat.shape[1] != latent_dim:
            raise ValueError(f"Corpus latents must be [N,{latent_dim}], got {self.Z_concat.shape}")
        if self.file_offsets.size < 2 or self.file_offsets[-1] != self.Z_concat.shape[0]:
            raise ValueError("file_offsets must cover every corpus latent frame")
        if self.frame_file_ids.shape[0] != self.Z_concat.shape[0]:
            raise ValueError("frame_file_ids must align with Z_concat")
        if self.Z_mean.shape != (latent_dim,) or self.Z_std.shape != (latent_dim,):
            raise ValueError(f"Z_mean and Z_std must both have shape [{latent_dim}]")

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
        self._decode_thread: Optional[threading.Thread] = None
        self._decoder_started = False

        self._generation_counter = 0
        self._hard_generation = 0
        self._live_generations = set()
        self._requested_generation = 0
        self._requested_window = requested_window
        self._adaptive_generation = None
        self._navigation_seconds = 0.0
        self._window_mode = "fixed"
        self._window_min = min(latent_decoder.supported_windows)
        self._window_max = max(latent_decoder.supported_windows)
        self._manual_content = "source"
        self._variation_amount = 0.25
        self._adaptive = AdaptiveWindow()
        self._manual_input_motion = AdaptiveWindow()
        self._manual_texture = ManualTexture(self.Z_concat, self.Z_mean, self.Z_std)
        self._texture_tail = None
        self._texture_content = None
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

    def set_mode(self, mode: str) -> Tuple[bool, str]:
        from stable_audio_wanderer.navigation_modes import normalize_mode
        mode = normalize_mode(mode)
        if mode not in ("wander", "reorganized", "manual"):
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
            if not np.all(np.isfinite(values)):
                raise ValueError("manual controls must be finite")
            values = np.clip(values, 0.0, 1.0)
        except Exception as exc:
            return False, str(exc)

        with self._lock:
            previous = self._manual_faders.copy()
            self._manual_faders = values.copy()
            self._manual_input_motion.observe(values, time.monotonic())
            running_manual = self._running.is_set() and self._active_mode == "manual"
        if not running_manual:
            self.manual.set_faders(values)
        delta = float(np.max(np.abs(values - previous)))
        return True, f"manual faders updated (delta={delta:.3f})"

    def set_manual_wander_params(self, k=None, speed=None) -> Tuple[bool, str]:
        try:
            self.manual.set_wander_params(k=k, speed=speed)
        except Exception as exc:
            return False, str(exc)
        return True, "manual wander parameters updated"

    def set_decoder_window(self, size, *, adaptive=False) -> Tuple[bool, str]:
        try:
            window = int(size)
        except (TypeError, ValueError):
            return False, f"invalid decoder window: {size!r}"
        if window not in self.latent_decoder.supported_windows:
            return False, f"decoder window T{window} is unavailable; supported: {self.latent_decoder.supported_windows}"
        with self._lock:
            if window == self._requested_window:
                return True, f"decoder window already T{window}"
            self._requested_window = window
            if self._running.is_set():
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = window
                self._adaptive_generation = self._requested_generation if adaptive else None
                keep = self._live_generations | {self._requested_generation, self._hard_generation}
                self._generation_windows = {g: t for g, t in self._generation_windows.items() if g in keep}
        # A T request changes intent only. The scheduler finishes one candidate
        # while sustaining the audible stream, then pursues the newest intent.
        return True, f"decoder window requested T{window}"

    def _announce_generation(self, generation):
        """Hard changes retain strict stale-result rejection (unlike T changes)."""
        with self._lock:
            if generation < self._hard_generation:
                return
            self._hard_generation = generation
            self.decoder.request_generation(generation)

    def set_window_controls(self, controls):
        if not isinstance(controls, dict):
            return False, "window controls must be an object"
        try:
            mode = controls.get("mode", self._window_mode)
            content = controls.get("content", self._manual_content)
            low = int(controls.get("minimum", self._window_min))
            high = int(controls.get("maximum", self._window_max))
            amount = float(controls.get("variation", self._variation_amount))
            if mode not in ("fixed", "adaptive") or content not in ("source", "held", "variation"):
                raise ValueError("invalid window mode or manual content")
            if low not in self.latent_decoder.supported_windows or high not in self.latent_decoder.supported_windows or low > high:
                raise ValueError("invalid adaptive window range")
            if not np.isfinite(amount) or not 0 <= amount <= 1:
                raise ValueError("variation must be between zero and one")
        except (ValueError, TypeError, OverflowError) as exc:
            return False, str(exc)
        with self._lock:
            changed_content = (content, amount) != (self._manual_content, self._variation_amount)
            self._window_mode, self._window_min, self._window_max = mode, low, high
            self._manual_content, self._variation_amount = content, amount
            generation = None
            if changed_content and self._running.is_set() and self._active_mode == "manual":
                self._generation_counter += 1
                generation = self._requested_generation = self._generation_counter
                self._generation_windows[generation] = self._requested_window
        if generation is not None:
            self._announce_generation(generation)
        return True, "window controls updated"

    def _observe_centre(self, index, frames):
        index = int(np.clip(index, 0, len(self.Z_concat) - 1))
        metadata = self.latent_decoder.metadata_for(self._requested_window)
        with self._lock:
            self._navigation_seconds += frames * metadata.samples_per_latent / metadata.sample_rate
            self._adaptive.observe((self.Z_concat[index] - self.Z_mean) / np.maximum(self.Z_std, 1e-6), self._navigation_seconds)

    def _adapt_window(self):
        with self._lock:
            if self._window_mode != "adaptive":
                return
            current = self._requested_window
            windows = [t for t in self.latent_decoder.supported_windows
                       if self._window_min <= t <= self._window_max]
        # Finish the previous staged transition before proposing another one.
        state = self.decoder.get_state()
        if (self.decoder.current_generation != self._requested_generation
                or state.get("transition_status", "idle") != "idle"):
            return
        now = time.monotonic()
        with self._lock:
            input_motion = self._manual_input_motion
            if self._active_mode == "manual" and input_motion.timestamp is not None:
                # Input events can shorten T even while a long decode is in flight.
                elapsed = max(0.0, now - input_motion.timestamp)
                self._adaptive.motion = max(self._adaptive.motion, input_motion.motion * np.exp(-elapsed / 0.25))
            target = self._adaptive.choose(current, windows, now)
        if target != current:
            self.set_decoder_window(target, adaptive=True)

    def set_wander_render_controls(self, controls) -> Tuple[bool, str]:
        """Apply decoder-window rendering controls without renaming Wander policy APIs."""
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

            if self._running.is_set() and self._active_mode == "wander":
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = self._requested_window
                generation = self._requested_generation
            else:
                generation = None

        if generation is not None:
            self._announce_generation(generation)
            return True, (
                f"staging Wander render generation {generation} "
            )
        return True, "wander render controls updated"

    def reset_wander(self, idx=None) -> Tuple[bool, str]:
        self.nav.reset_policy(idx=idx)
        with self._lock:
            self._wander_reset_serial += 1
            if self._running.is_set() and self._active_mode == "wander":
                self._generation_counter += 1
                self._requested_generation = self._generation_counter
                self._generation_windows[self._requested_generation] = self._requested_window
                generation = self._requested_generation
            else:
                generation = None
        if generation is not None:
            self._announce_generation(generation)
        return True, "Wander reset"

    def _cleanup_previous_run(self) -> None:
        # Serialize against the producer's publication/start critical section.
        # Stop playback immediately, then drain computations before resetting PCM.
        with self._lock:
            self._running.clear()
            if self._decoder_started:
                self.decoder.stop()
                self._decoder_started = False
        if self._decode_thread is not None:
            self._decode_thread.join()
            self._decode_thread = None
        self.decoder.reset_buffers()

    def start(self) -> Tuple[bool, str]:
        with self._lifecycle_lock:
            if self._closed:
                return False, "transport is closed"
            return self._start()

    def _start(self) -> Tuple[bool, str]:
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
            self._live_generations = {generation}
            self._transport_error = None
            self._prebuffering = True
            self._wander_reset_serial += 1

        self._adaptive = AdaptiveWindow()
        self._manual_input_motion = AdaptiveWindow()
        self._manual_input_motion.observe(self._manual_faders, time.monotonic())
        self._manual_texture = ManualTexture(self.Z_concat, self.Z_mean, self.Z_std)
        self._texture_tail = None
        self._texture_content = None
        self._announce_generation(generation)
        hold_index = self._stationary_hold_index(mode)
        hold_metadata = self.latent_decoder.metadata_for(window)
        self.decoder.capture_presentation_hold(
            hold_index,
            generation=generation,
            samples_per_frame=int(hold_metadata.samples_per_latent),
        )

        with self._stats_lock:
            self._decode_times_ms.clear()
            self._decode_count = 0
            self._last_decode_ms = 0.0

        self._running.set()
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            daemon=True,
            name="saw-decoder-transport",
        )
        self._decode_thread.start()
        return True, f"transport prebuffering ({mode}, T{window})"

    def stop(self) -> Tuple[bool, str]:
        with self._lifecycle_lock:
            return self._stop()

    def _stop(self) -> Tuple[bool, str]:
        had_resources = any(
            (
                self._running.is_set(),
                self._decode_thread is not None,
                self._decoder_started,
            )
        )
        self._cleanup_previous_run()
        with self._lock:
            self._prebuffering = False
        return True, "transport stopped" if had_resources else "already stopped"

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._resources_closed:
                return
            self._closed = True
            self._stop()
            self.decoder.close()
            self._resources_closed = True

    def _snapshot_request(self) -> tuple[int, int, str]:
        with self._lock:
            return (
                int(self._requested_generation),
                int(self._requested_window),
                str(self._active_mode),
            )

    def _stationary_hold_index(self, mode: str) -> int:
        """Capture the visible cursor before the producer advances navigation."""
        if mode == "manual":
            state = self.manual.get_state()
            candidate = state.get("nearest_index", self._manual_last_index)
        else:
            state = self.nav.get_state()
            candidate = state.get("policy_index", 0)
        return int(np.clip(round(float(candidate)), 0, self.Z_concat.shape[0] - 1))

    def _manual_request(self, generation: int, window: int) -> _DecodeRequest:
        with self._lock:
            faders = self._manual_faders.copy()
        # Exactly one spatial query chooses the source-file anchor for a decode.
        frame = self.manual.step_with_faders(faders)
        anchor = int(np.clip(frame.nearest_index, 0, self.Z_concat.shape[0] - 1))
        self._observe_centre(anchor, (window + 1) // 2)
        raw, plan = build_file_bounded_latent_window(
            self.Z_concat,
            self.file_offsets,
            anchor,
            window,
            mean=self.Z_mean,
            std=self.Z_std,
        )
        with self._lock:
            content, amount = self._manual_content, self._variation_amount
        indices = tuple(int(index) for index in plan.frame_indices)
        if self._texture_content != content:
            self._texture_tail = None
            self._texture_content = content
            self._manual_texture = ManualTexture(self.Z_concat, self.Z_mean, self.Z_std)
        if content != "source":
            hop = (window + 1) // 2
            tail_count = window - hop
            tail = self._texture_tail
            if tail is None or len(tail) < tail_count:
                raw = self._manual_texture.render(anchor, window, content, amount)
            else:
                raw = np.concatenate([tail[-tail_count:], self._manual_texture.render(anchor, hop, content, amount)])
            self._texture_tail = raw[hop:].copy()
            # These indices identify the texture centre, not exact source samples.
            indices = (anchor,) * window
        else:
            self._texture_tail = None
        with self._stats_lock:
            self._manual_last_index = anchor
            self._manual_last_distance = float(frame.distance)
        return _DecodeRequest(
            generation,
            window,
            raw,
            indices,
        )

    def _wander_request(
        self,
        generation: int,
        window: int,
        previous_generation: Optional[int],
        planner_reset_serial: int,
    ) -> tuple[_DecodeRequest, int]:
        """Advance the policy at the existing cadence, then plan one full T window."""
        hop = (window + 1) // 2
        step_count = window if (generation != previous_generation and generation != self._adaptive_generation) else hop
        anchor = None
        for _ in range(step_count):
            anchor = self.nav.step(fixed_retrieval_window=window)
        if anchor is None:
            raise RuntimeError("Wander navigation produced no anchor")
        anchor_frame = int(getattr(anchor, "nearest_idx", anchor))
        centre_index = int(np.clip(anchor_frame, 0, len(self.Z_concat) - 1))
        self._observe_centre(centre_index, step_count)
        with self._lock:
            frame_source = self._wander_frame_source
            frame_order = self._wander_frame_order
            latent_colour = self._wander_latent_colour
            reset_serial = self._wander_reset_serial
        if reset_serial != planner_reset_serial:
            self.wander_planner.reset()
            planner_reset_serial = reset_serial
        controls = self.nav.get_wander_controls()
        planned = self.wander_planner.plan(
            anchor_frame,
            window,
            frame_source=frame_source,
            frame_order=frame_order,
            latent_colour=latent_colour,
            **controls,
        )
        return (
            _DecodeRequest(
                generation,
                window,
                planned.raw_latents,
                tuple(int(index) for index in planned.input_frames),
            ),
            planner_reset_serial,
        )

    def _target_prebuffer_seconds(self, window: int) -> float:
        # Keep a spare hop while navigation and inference prepare the next one.
        # One-hop refill margins proved insufficient with real policy transitions.
        metadata = self.latent_decoder.metadata_for(window)
        return max(
            self.MIN_PREBUFFER_SECONDS,
            self.PREBUFFER_HOPS * metadata.audio_hop_samples / float(metadata.sample_rate),
        )

    def _new_lane(self, generation, window, source=None):
        planner = source.planner if source else self.wander_planner
        if source:
            planner = planner.fork() if hasattr(planner, "fork") else copy.deepcopy(planner)
        texture = copy.copy(source.texture if source else self._manual_texture)
        texture.rng = copy.deepcopy(texture.rng)
        return _DecodeLane(generation, window, planner, texture,
                           texture_tail=source.texture_tail if source else self._texture_tail,
                           texture_content=source.texture_content if source else self._texture_content,
                           reset_serial=source.reset_serial if source else -1)

    def _lane_request(self, lane, mode):
        # One coordinator owns all navigation. Each stream has private planner,
        # texture, overlap-add and frame-tail state; corpus data is shared.
        saved = (self.wander_planner, self._manual_texture, self._texture_tail, self._texture_content)
        self.wander_planner, self._manual_texture = lane.planner, lane.texture
        self._texture_tail, self._texture_content = lane.texture_tail, lane.texture_content
        try:
            if mode == "manual":
                request = self._manual_request(lane.generation, lane.window)
            elif mode == "wander":
                request, lane.reset_serial = self._wander_request(
                    lane.generation, lane.window, lane.previous_generation, lane.reset_serial)
            else:
                hop = (lane.window + 1) // 2
                tail_count = lane.window - hop
                frames = list(lane.previous_frames[-tail_count:]) if tail_count else []
                while len(frames) < lane.window:
                    frames.append(self.nav.step(fixed_retrieval_window=lane.window))
                z = self.manifold.generate_batch(frames, exploration=self.nav.get_active_jump_rate(variant=mode))
                request = _DecodeRequest(lane.generation, lane.window,
                    np.ascontiguousarray(z*self.Z_std[None, :] + self.Z_mean[None, :], dtype=np.float32),
                    tuple(int(getattr(frame, "nearest_idx", frame)) for frame in frames))
                lane.previous_frames = frames
            lane.previous_generation = lane.generation
            lane.texture = self._manual_texture
            lane.texture_tail, lane.texture_content = self._texture_tail, self._texture_content
            return request
        finally:
            self.wander_planner, self._manual_texture, self._texture_tail, self._texture_content = saved

    def _validate_request(self, request):
        if len(request.frame_indices) != request.window:
            raise RuntimeError("decode request provenance does not match its latent window")
        if any(index < 0 or index >= len(self.Z_concat) for index in request.frame_indices):
            raise RuntimeError("decode request provenance contains an out-of-range corpus index")

    def _finish_decode(self, lane):
        decoded = lane.future.result()
        lane.future = None
        request = lane.request
        if lane.assembler is None:
            lane.assembler = StreamingFullOverlapAdd(decoded.metadata.audio_hop_samples,
                                                     channels=decoded.metadata.channels)
        hop = lane.assembler.push(decoded.audio.T)
        indices = request.frame_indices[:int(decoded.metadata.latent_hop)]
        samples = int(decoded.metadata.samples_per_latent)
        if len(indices)*samples != hop.shape[1]:
            raise RuntimeError("decode hop provenance does not match emitted PCM: "
                               f"{len(indices)} frames * {samples} samples != {hop.shape[1]} hop samples")
        lane.prepared.append((hop, indices, samples))
        lane.prepared_samples += hop.shape[1]
        with self._stats_lock:
            self._last_decode_ms = float(decoded.decode_time_ms)
            self._decode_times_ms.append(float(decoded.decode_time_ms))
            self._decode_count += 1

    def _decode_loop(self) -> None:
        active = candidate = None
        retired = []
        # LatentDecoder must safely handle overlapping independent requests.
        # At most two calls exist, with no job backlog; scheduling is unchanged.
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="saw-decoder-inference")
        try:
            while self._running.is_set():
                self._adapt_window()
                requested, window, mode = self._snapshot_request()
                with self._lock:
                    hard = self._hard_generation
                if active is None or hard > active.generation:
                    for lane in (active, candidate):
                        if lane and lane.future:
                            retired.append(lane.future)
                    # Start the hard generation even if a newer soft request
                    # arrived while it was being installed.
                    generation = hard or requested
                    active = self._new_lane(generation, self._generation_windows.get(generation, window))
                    active.published = True
                    candidate = None
                retired = [future for future in retired if not future.done()]
                if candidate and candidate.published and self.decoder.current_generation == candidate.generation:
                    if active.future:
                        retired.append(active.future)
                    active, candidate = candidate, None
                    self.wander_planner = active.planner
                    # Keep only metadata that may still be presented/requested.
                    with self._lock:
                        keep = {active.generation, self._requested_generation, self._hard_generation}
                        self._generation_windows = {g: t for g, t in self._generation_windows.items() if g in keep}
                if (candidate is None and requested != active.generation and self._decoder_started
                        and self.decoder.get_state().get("transition_status", "idle") == "idle"):
                    candidate = self._new_lane(requested, window, source=active)
                with self._lock:
                    self._live_generations = {lane.generation for lane in (active, candidate) if lane}
                    for lane in (active, candidate):
                        if lane:
                            self._generation_windows[lane.generation] = lane.window
                for lane in (active, candidate):
                    if lane is None:
                        continue
                    if lane.future and lane.future.done():
                        self._finish_decode(lane)
                    target = self._target_prebuffer_seconds(lane.window)
                    # Have enough audio for the crossfade before exposing a candidate.
                    ready_samples = max(int(target*self.decoder.sr), GENERATION_TRANSITION_SAMPLES)
                    with self._lock:
                        valid = self._running.is_set() and self._hard_generation <= lane.generation
                        if valid and not lane.published and lane.prepared_samples >= ready_samples:
                            lane.published = self.decoder.request_generation(lane.generation, continue_current=True)
                        if valid and lane.published:
                            for hop, indices, samples in lane.prepared:
                                self.decoder.write_hop(hop, generation=lane.generation,
                                    frame_indices=indices, samples_per_frame=samples)
                            lane.prepared.clear()
                            lane.prepared_samples = 0
                            if not self._decoder_started and self.decoder.generation_buffer_duration(lane.generation) >= target:
                                self.decoder.start()
                                self._decoder_started = True
                                self._prebuffering = False
                    buffered = (self.decoder.generation_buffer_duration(lane.generation) if lane.published
                                else lane.prepared_samples / float(self.decoder.sr))
                    threshold = target if lane.published else ready_samples / float(self.decoder.sr)
                    busy = len(retired) + sum(bool(x and x.future) for x in (active, candidate))
                    if valid and lane.future is None and buffered < threshold and busy < 2:
                        lane.request = lane.next_request or self._lane_request(lane, mode)
                        lane.next_request = None
                        self._validate_request(lane.request)
                        with self._lock:
                            if self._running.is_set() and self._hard_generation <= lane.generation:
                                lane.future = pool.submit(self.latent_decoder.decode, lane.request.raw_latents)
                    # Prepare one next latent window while inference is busy.
                    # Planning after completion leaves navigation time outside
                    # the decode hop budget and can starve otherwise fast ORT.
                    # A lane owns its planner; no additional inference is queued.
                    if valid and lane.future is not None and lane.next_request is None and buffered < threshold:
                        lane.next_request = self._lane_request(lane, mode)
                        self._validate_request(lane.next_request)
                time.sleep(0.002)
        except Exception as exc:
            self._latch_runtime_error(f"Decoder runtime failure ({self.latent_decoder.info.backend}): {exc}")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

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
            "backend": info.backend,
            "provider": info.provider,
            "vae_id": info.vae_id,
            "supported_windows": list(self.latent_decoder.supported_windows),
            "window_controls": {"mode": self._window_mode, "minimum": self._window_min,
                                "maximum": self._window_max, "content": self._manual_content,
                                "variation": self._variation_amount},
            "manual_provenance": "source_frames" if self._manual_content == "source" else "texture_centre",
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
        elif getattr(info, "bundle_path", None) is not None:
            state["bundle_path"] = str(info.bundle_path)
        elif getattr(info, "model_path", None) is not None:
            state["model_path"] = str(info.model_path)
        return state

    def get_erae_state(self) -> dict:
        """Lightweight OSC snapshot; no audio callback work or decode statistics."""
        with self._lock:
            running = self._running.is_set()
            mode = self._active_mode if running else self.selected_mode
            requested = int(self._requested_window)
            requested_generation = self._requested_generation
            windows = dict(self._generation_windows)
            error = self._transport_error or ""
            prebuffering = self._prebuffering
            window_mode = self._window_mode
        presentation = self.decoder.get_presentation_state() or {}
        index = presentation.get("index")
        valid = isinstance(index, (int, np.integer)) and 0 <= index < len(self.Z_concat)
        generation = presentation.get("generation")
        if generation is None:
            generation = self.decoder.current_generation
        active = windows.get(generation, -1)
        transition = str(self.decoder.get_state().get("transition_status", "idle"))
        if error:
            transition = "error"
        elif prebuffering:
            transition = "prebuffering"
        elif running and generation != requested_generation and transition == "idle":
            transition = "staging"
        return dict(running=running, mode=mode, valid=bool(valid),
                    index=int(index) if valid else -1,
                    generation=int(generation) if generation is not None else -1,
                    requested_window=requested, active_window=int(active),
                    transition=transition, window_mode=window_mode, error=error)

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
        state = {
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
        presentation = self.decoder.get_presentation_state()
        if isinstance(presentation, dict):
            state["presentation"] = dict(presentation)
        return state

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
            print(f"[ws] Decoder transport {action}: {message}")
            return True
        if msg_type == "window_controls":
            ok, message = self.set_window_controls(data.get("controls", {}))
            print(f"[ws] window controls: {message}")
            return True
        if msg_type == "decoder_window":
            ok, message = self.set_decoder_window(data.get("size"))
            print(f"[ws] decoder window: {message}")
            return True
        if msg_type == "wander_render":
            ok, message = self.set_wander_render_controls(data.get("controls", {}))
            print(f"[ws] wander render: {message}")
            return True
        if msg_type in ("wander_control", "random_control", "control"):
            controls = data.get("controls", {})
            if isinstance(controls, dict):
                self.nav.set_wander_controls(**controls)
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
            # These legacy controls are intentionally unavailable in Web mode.
            return True
        return False


__all__ = ["DecoderTransportController"]
