"""Observed-frame decoder-window planning for unified-Web Wander mode.

The planner mirrors the JUCE rendering order:

1. select observed corpus frames,
2. optionally disorder only the interior frame order,
3. add deterministic, temporally smoothed latent colour in normalized space,
4. denormalize the complete ``[T, D]`` window for the decoder.

Morphology-graph traversal is stateful across windows.  All other helpers are
pure so the exact JUCE PRNG and hash behaviour can be tested independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .manual_windows import plan_file_bounded_latent_window


FRAME_SOURCE_K_NEAREST = "k_nearest"
FRAME_SOURCE_CONTIGUOUS = "contiguous"
FRAME_SOURCE_MORPHOLOGY_GRAPH = "morphology_graph"
FRAME_SOURCES = frozenset(
    (
        FRAME_SOURCE_K_NEAREST,
        FRAME_SOURCE_CONTIGUOUS,
        FRAME_SOURCE_MORPHOLOGY_GRAPH,
    )
)

DEFAULT_WANDER_SEED = 0x476F6F64
_UINT32_MASK = 0xFFFFFFFF
_MAX_RECENT_GRAPH_UNITS = 24
_TARGET_PLANNED_GRAPH_UNITS = 12


def _u32(value: int) -> int:
    return int(value) & _UINT32_MASK


def _round_to_int(value: float) -> int:
    """Match JUCE ``roundToInt`` for the non-negative values used here."""

    return int(math.floor(float(value) + 0.5))


def _clamp01(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return min(1.0, max(0.0, result))


def mix_frame_order_seed(
    seed: int,
    chunk_start_frame: int,
    latent_window: int,
    decode_window_index: int,
) -> int:
    """Return the exact 32-bit seed mixed by the JUCE frame-order pass."""

    value = _u32(seed) if int(seed) != 0 else 1
    value ^= _u32(_u32(chunk_start_frame) + 0x9E3779B9)
    value = _u32(value * 0x85EBCA6B)
    value ^= _u32(_u32(latent_window) * 0xC2B2AE35)
    value = _u32(value * 0x27D4EB2D)
    value ^= _u32(_u32(decode_window_index) + 0x165667B1)
    value ^= value >> 16
    return value if value != 0 else 1


def _next_frame_order_random(state: int) -> Tuple[int, int]:
    state = _u32(_u32(state) * 1664525 + 1013904223)
    return state, state


def _is_strictly_stepwise_frame_order(frames: np.ndarray) -> bool:
    if frames.size < 2:
        return True
    for previous, current in zip(frames[:-1], frames[1:]):
        if int(current) != int(previous) + 1 and int(current) != int(previous):
            return False
    return True


def apply_frame_order_disorder(
    frame_indices: Sequence[int],
    frame_order: float,
    seed: int,
    chunk_start_frame: int,
    latent_window: int,
    decode_window_index: int,
) -> np.ndarray:
    """Apply the JUCE seeded interior shuffle while preserving both endpoints."""

    amount = _clamp01(frame_order, "frame_order")
    ordered = np.asarray(frame_indices, dtype=np.int64).reshape(-1).copy()
    if amount <= 0.0 or ordered.size <= 3:
        return ordered

    interior_count = int(ordered.size) - 2
    maximum_swaps = max(0, interior_count - 1)
    swap_count = _round_to_int(amount * maximum_swaps)
    if amount > 0.0 and maximum_swaps > 0:
        swap_count = max(1, swap_count)

    random_state = mix_frame_order_seed(
        seed, chunk_start_frame, latent_window, decode_window_index
    )
    for step in range(swap_count):
        target_interior = 1 + (interior_count - 1 - step)
        candidate_count = interior_count - step
        random_state, random_value = _next_frame_order_random(random_state)
        random_interior = 1 + int(random_value % candidate_count)
        ordered[target_interior], ordered[random_interior] = (
            ordered[random_interior],
            ordered[target_interior],
        )

    if amount > 0.0 and _is_strictly_stepwise_frame_order(ordered) and interior_count > 1:
        ordered[1], ordered[2] = ordered[2], ordered[1]
    return ordered


def _avalanche_latent_colour_hash(value: int) -> int:
    value = _u32(value)
    value ^= value >> 16
    value = _u32(value * 0x7FEB352D)
    value ^= value >> 15
    value = _u32(value * 0x846CA68B)
    value ^= value >> 16
    return _u32(value)


def _hash_to_signed_unit_float(value: int) -> np.float32:
    scale = 1.0 / 2147483647.5
    return np.float32(float(_u32(value)) * scale - 1.0)


def latent_colour_raw_noise(seed: int, frame: int, dimension: int) -> np.float32:
    """Return JUCE's deterministic raw latent-colour sample."""

    value = _u32(seed) if int(seed) != 0 else 1
    value ^= _u32(_u32(frame) * 0x9E3779B9)
    value ^= _u32(_u32(dimension) * 0x85EBCA6B)
    value ^= 0xC2B2AE35
    return _hash_to_signed_unit_float(_avalanche_latent_colour_hash(value))


def latent_colour_smoothed_noise(
    seed: int,
    frame: int,
    dimension: int,
    sequence_start: int,
    sequence_end: int,
) -> np.float32:
    """Return the JUCE five-tap ``[1,4,6,4,1]/16`` colour sample."""

    if int(sequence_end) <= int(sequence_start):
        sequence_start = int(frame)
        sequence_end = int(frame) + 1

    def clamped(candidate: int) -> int:
        return min(int(sequence_end) - 1, max(int(sequence_start), int(candidate)))

    n0 = latent_colour_raw_noise(seed, clamped(int(frame) - 2), dimension)
    n1 = latent_colour_raw_noise(seed, clamped(int(frame) - 1), dimension)
    n2 = latent_colour_raw_noise(seed, clamped(int(frame)), dimension)
    n3 = latent_colour_raw_noise(seed, clamped(int(frame) + 1), dimension)
    n4 = latent_colour_raw_noise(seed, clamped(int(frame) + 2), dimension)
    value = np.float32(n0 + np.float32(4.0) * n1)
    value = np.float32(value + np.float32(6.0) * n2)
    value = np.float32(value + np.float32(4.0) * n3)
    value = np.float32(value + n4)
    return np.float32(value * np.float32(1.0 / 16.0))


def _raw_noise_vector(seed: int, frame: int, latent_dim: int) -> np.ndarray:
    """Vectorized equivalent of :func:`latent_colour_raw_noise`."""

    dimensions = np.arange(int(latent_dim), dtype=np.uint64)
    base = _u32(seed) if int(seed) != 0 else 1
    values = np.full(int(latent_dim), base, dtype=np.uint64)
    values ^= np.uint64(_u32(_u32(frame) * 0x9E3779B9))
    values ^= (dimensions * np.uint64(0x85EBCA6B)) & np.uint64(_UINT32_MASK)
    values ^= np.uint64(0xC2B2AE35)
    values &= np.uint64(_UINT32_MASK)
    values ^= values >> np.uint64(16)
    values = (values * np.uint64(0x7FEB352D)) & np.uint64(_UINT32_MASK)
    values ^= values >> np.uint64(15)
    values = (values * np.uint64(0x846CA68B)) & np.uint64(_UINT32_MASK)
    values ^= values >> np.uint64(16)
    values &= np.uint64(_UINT32_MASK)
    scale = 1.0 / 2147483647.5
    return (values.astype(np.float64) * scale - 1.0).astype(np.float32)


@dataclass(frozen=True)
class WanderGraphControls:
    """Accepted Wander controls mapped onto JUCE graph controls."""

    attractor_x: float
    attractor_y: float
    segment_length: float
    continuity: float
    radius: float
    wander: float
    motion_rate: float
    novelty: float
    crossfile_penalty: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WanderWindowDiagnostics:
    requested_frame_source: str
    effective_frame_source: str
    graph_available: bool
    graph_error: Optional[str]
    anchor_frame: int
    chunk_start_frame: int
    decode_window_index: int
    frame_order: float
    latent_colour: float
    seed: int
    k_nearest_distances_nondecreasing: bool
    planned_units: Tuple[int, ...]
    recent_units: Tuple[int, ...]
    graph_controls: WanderGraphControls

    @property
    def planned_unit_count(self) -> int:
        return len(self.planned_units)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["planned_unit_count"] = self.planned_unit_count
        return result


@dataclass(frozen=True)
class WanderLatentWindow:
    """One complete raw decoder input plus observed-frame diagnostics."""

    raw_latents: np.ndarray
    source_frames: Tuple[int, ...]
    input_frames: Tuple[int, ...]
    diagnostics: WanderWindowDiagnostics


@dataclass(frozen=True)
class _GraphPlanSignature:
    anchor_unit: int
    controls: WanderGraphControls
    seed: int


class WanderWindowPlanner:
    """Build complete observed-frame windows for unified-Web Wander mode."""

    def __init__(
        self,
        z_concat: np.ndarray,
        file_offsets: Sequence[int],
        manual_points: np.ndarray,
        z_mean: np.ndarray,
        z_std: np.ndarray,
        *,
        unit_start_idx: Optional[Sequence[int]] = None,
        unit_end_idx: Optional[Sequence[int]] = None,
        unit_graph_neighbors: Optional[np.ndarray] = None,
        unit_graph_scores: Optional[np.ndarray] = None,
        seed: int = DEFAULT_WANDER_SEED,
    ) -> None:
        latents = np.asarray(z_concat, dtype=np.float32)
        if latents.ndim != 2 or latents.shape[0] <= 0 or latents.shape[1] <= 0:
            raise ValueError(f"z_concat must be a non-empty [N,D] array, got {latents.shape}")
        if not np.all(np.isfinite(latents)):
            raise ValueError("z_concat contains non-finite values")
        self._latents = np.ascontiguousarray(latents, dtype=np.float32)
        self._frame_count, self._latent_dim = self._latents.shape

        offsets_input = np.asarray(file_offsets)
        if offsets_input.ndim != 1 or offsets_input.size < 2:
            raise ValueError("file_offsets must be [num_files + 1]")
        if not np.all(np.isfinite(offsets_input)):
            raise ValueError("file_offsets contains non-finite values")
        offsets = offsets_input.astype(np.int64)
        if not np.array_equal(offsets_input, offsets):
            raise ValueError("file_offsets must contain integer positions")
        if offsets[0] != 0 or offsets[-1] != self._frame_count:
            raise ValueError("file_offsets must start at 0 and cover all corpus frames")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("file_offsets must be nondecreasing")
        if np.any(offsets[1:] == offsets[:-1]):
            raise ValueError("file_offsets cannot contain empty source files")
        self._file_offsets = offsets

        points = np.asarray(manual_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] != self._frame_count or points.shape[1] < 2:
            raise ValueError(
                "manual_points must have shape [N, at least 2] aligned with z_concat"
            )
        if not np.all(np.isfinite(points[:, :2])):
            raise ValueError("manual_points contains non-finite map coordinates")
        self._manual_points = np.ascontiguousarray(points[:, :2], dtype=np.float32)

        mean = np.asarray(z_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(z_std, dtype=np.float32).reshape(-1)
        if mean.shape != (self._latent_dim,) or std.shape != (self._latent_dim,):
            raise ValueError(
                f"z_mean and z_std must both have shape [{self._latent_dim}]"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("z_mean and z_std must be finite")
        self._mean = mean.copy()
        self._std = std.copy()
        self.seed = _u32(seed) if int(seed) != 0 else 1

        self._unit_start = np.empty(0, dtype=np.int64)
        self._unit_end = np.empty(0, dtype=np.int64)
        self._unit_file = np.empty(0, dtype=np.int64)
        self._neighbors = np.empty((0, 0), dtype=np.int64)
        self._scores: Optional[np.ndarray] = None
        self.graph_available, self._graph_error = self._prepare_graph(
            unit_start_idx,
            unit_end_idx,
            unit_graph_neighbors,
            unit_graph_scores,
        )

        self._decode_window_index = 0
        self._graph_state_valid = False
        self._graph_unit = -1
        self._graph_frame = -1
        self._graph_anchor_unit = -1
        self._recent_units: list[int] = []
        self._planned_units: list[int] = []
        self._plan_signature: Optional[_GraphPlanSignature] = None

        initial_controls = self._map_graph_controls(0, 0.4, 0.55, 0.55, 0.5, 0.75, 0.7)
        self.last_diagnostics = WanderWindowDiagnostics(
            requested_frame_source=FRAME_SOURCE_K_NEAREST,
            effective_frame_source=FRAME_SOURCE_K_NEAREST,
            graph_available=self.graph_available,
            graph_error=self._graph_error,
            anchor_frame=-1,
            chunk_start_frame=-1,
            decode_window_index=0,
            frame_order=0.0,
            latent_colour=0.0,
            seed=self.seed,
            k_nearest_distances_nondecreasing=True,
            planned_units=(),
            recent_units=(),
            graph_controls=initial_controls,
        )

    def _prepare_graph(
        self,
        starts_input: Optional[Sequence[int]],
        ends_input: Optional[Sequence[int]],
        neighbors_input: Optional[np.ndarray],
        scores_input: Optional[np.ndarray],
    ) -> Tuple[bool, Optional[str]]:
        if starts_input is None or ends_input is None or neighbors_input is None:
            return False, "missing morphology graph arrays"
        try:
            starts_raw = np.asarray(starts_input)
            ends_raw = np.asarray(ends_input)
            neighbors_raw = np.asarray(neighbors_input)
            if not (
                np.all(np.isfinite(starts_raw))
                and np.all(np.isfinite(ends_raw))
                and np.all(np.isfinite(neighbors_raw))
            ):
                raise ValueError("graph integer arrays contain non-finite values")
            starts = starts_raw.astype(np.int64).reshape(-1)
            ends = ends_raw.astype(np.int64).reshape(-1)
            neighbors = neighbors_raw.astype(np.int64)
            if not np.array_equal(starts_raw.reshape(-1), starts):
                raise ValueError("unit_start_idx must contain integers")
            if not np.array_equal(ends_raw.reshape(-1), ends):
                raise ValueError("unit_end_idx must contain integers")
            if not np.array_equal(neighbors_raw, neighbors):
                raise ValueError("unit_graph_neighbors must contain integers")
            unit_count = int(starts.size)
            if unit_count <= 0 or ends.shape != starts.shape:
                raise ValueError("unit boundary arrays are empty or misaligned")
            if neighbors.ndim != 2 or neighbors.shape[0] != unit_count or neighbors.shape[1] <= 0:
                raise ValueError("unit_graph_neighbors must be [units, K]")
            if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends > self._frame_count):
                raise ValueError("unit boundaries contain invalid frame ranges")
            if np.any(neighbors < -1) or np.any(neighbors >= unit_count):
                raise ValueError("unit_graph_neighbors contains invalid unit ids")

            unit_file = np.searchsorted(self._file_offsets, starts, side="right") - 1
            end_file = np.searchsorted(self._file_offsets, ends - 1, side="right") - 1
            if np.any(unit_file != end_file):
                raise ValueError("morphology units may not cross source-file boundaries")

            scores = None
            if scores_input is not None:
                candidate_scores = np.asarray(scores_input, dtype=np.float32)
                if candidate_scores.shape == neighbors.shape and np.all(np.isfinite(candidate_scores)):
                    scores = np.ascontiguousarray(candidate_scores, dtype=np.float32)

            self._unit_start = starts
            self._unit_end = ends
            self._unit_file = unit_file.astype(np.int64)
            self._neighbors = np.ascontiguousarray(neighbors, dtype=np.int64)
            self._scores = scores
            return True, None
        except (TypeError, ValueError, OverflowError) as exc:
            return False, str(exc)

    def fork(self):
        """Copy traversal state for a staged stream while sharing corpus arrays."""
        result = copy.copy(self)
        result._recent_units = list(self._recent_units)
        result._planned_units = list(self._planned_units)
        return result

    def reset(self) -> None:
        """Clear decode counters and all retained morphology traversal state."""

        self._decode_window_index = 0
        self._graph_state_valid = False
        self._graph_unit = -1
        self._graph_frame = -1
        self._graph_anchor_unit = -1
        self._recent_units.clear()
        self._clear_graph_plan()
        self.last_diagnostics = replace(
            self.last_diagnostics,
            anchor_frame=-1,
            chunk_start_frame=-1,
            decode_window_index=0,
            planned_units=(),
            recent_units=(),
        )

    def _sequence_bounds(self, frame: int) -> Tuple[int, int]:
        file_index = int(np.searchsorted(self._file_offsets, int(frame), side="right") - 1)
        file_index = min(self._file_offsets.size - 2, max(0, file_index))
        return int(self._file_offsets[file_index]), int(self._file_offsets[file_index + 1])

    def _map_graph_controls(
        self,
        anchor: int,
        phrase_scale: float,
        jump_rate: float,
        timbre_lock: float,
        drift: float,
        repeat_avoid: float,
        crossfile: float,
    ) -> WanderGraphControls:
        point = self._manual_points[int(anchor), :2]
        drift_value = _clamp01(drift, "drift")
        return WanderGraphControls(
            attractor_x=float(point[0]),
            attractor_y=float(point[1]),
            segment_length=_clamp01(phrase_scale, "phrase_scale"),
            continuity=1.0 - _clamp01(jump_rate, "jump_rate"),
            radius=1.0 - _clamp01(timbre_lock, "timbre_lock"),
            wander=drift_value,
            motion_rate=drift_value,
            novelty=_clamp01(repeat_avoid, "repeat_avoid"),
            crossfile_penalty=1.0 - _clamp01(crossfile, "crossfile"),
        )

    def _build_k_nearest(self, anchor: int, window_size: int) -> Tuple[np.ndarray, bool]:
        target = self._manual_points[int(anchor)]
        delta = self._manual_points - target[None, :]
        distances = np.einsum("nd,nd->n", delta, delta, dtype=np.float32)
        frame_ids = np.arange(self._frame_count, dtype=np.int64)
        order = np.lexsort((frame_ids, distances.astype(np.float64)))
        frames = order[: min(window_size, self._frame_count)].astype(np.int64)
        if frames.size == 0:
            frames = np.asarray([anchor], dtype=np.int64)
        if frames.size < window_size:
            frames = np.pad(frames, (0, window_size - frames.size), constant_values=int(frames[-1]))
        selected_distances = distances[frames]
        nondecreasing = bool(
            selected_distances.size < 2
            or np.all(selected_distances[1:] >= selected_distances[:-1])
        )
        return frames, nondecreasing

    def _build_contiguous(self, anchor: int, window_size: int) -> Tuple[np.ndarray, int]:
        plan = plan_file_bounded_latent_window(
            anchor,
            window_size,
            self._file_offsets,
            frame_count=self._frame_count,
        )
        return np.asarray(plan.frame_indices, dtype=np.int64), int(plan.chunk_start)

    def _find_unit_for_frame(self, frame: int) -> int:
        if not self.graph_available:
            return -1
        matches = np.flatnonzero((self._unit_start <= frame) & (frame < self._unit_end))
        return int(matches[0]) if matches.size else -1

    def _remember_unit_in(self, recent: list[int], unit: int) -> None:
        if unit < 0 or (recent and recent[-1] == unit):
            return
        recent.append(int(unit))
        if len(recent) > _MAX_RECENT_GRAPH_UNITS:
            del recent[: len(recent) - _MAX_RECENT_GRAPH_UNITS]

    def _repeat_cost(self, recent: Sequence[int], unit: int) -> float:
        best = 0.0
        for rank, previous in enumerate(reversed(recent)):
            if int(previous) == int(unit):
                best = max(best, max(0.0, 1.0 - 0.08 * float(rank)))
        return best

    def _unit_map_distance_squared(self, unit: int, controls: WanderGraphControls) -> float:
        start = int(self._unit_start[unit])
        length = int(self._unit_end[unit] - start)
        samples = min(8, max(1, length))
        best = math.inf
        for sample in range(samples):
            offset = 0 if samples <= 1 else _round_to_int(sample * (length - 1) / (samples - 1))
            point = self._manual_points[start + offset]
            dx = float(point[0]) - controls.attractor_x
            dy = float(point[1]) - controls.attractor_y
            best = min(best, dx * dx + dy * dy)
        return best

    @staticmethod
    def _region_cost(distance_squared: float, radius: float) -> float:
        region_radius = 0.05 + min(1.0, max(0.0, radius)) * 0.65
        limit_squared = region_radius * region_radius
        if distance_squared <= limit_squared:
            return 0.0
        return (distance_squared - limit_squared) / max(0.001, 2.0 - limit_squared)

    def _choose_next_unit(
        self,
        current_unit: int,
        anchor_unit: int,
        controls: WanderGraphControls,
        recent: Sequence[int],
        decision_index: int,
    ) -> int:
        if not self.graph_available:
            return int(current_unit)
        unit_count = int(self._unit_start.size)
        current_unit = min(unit_count - 1, max(0, int(current_unit)))
        lengths = self._unit_end - self._unit_start
        min_length = float(np.min(lengths))
        max_length = float(np.max(lengths))
        target_length = min_length + controls.segment_length * (max_length - min_length)
        length_denominator = max(1.0, max_length - min_length)
        candidates: dict[int, tuple] = {}

        for slot, candidate_value in enumerate(self._neighbors[current_unit]):
            unit = int(candidate_value)
            if unit < 0 or unit >= unit_count:
                continue
            graph_score = float(self._scores[current_unit, slot]) if self._scores is not None else 0.0
            map_distance = self._unit_map_distance_squared(unit, controls)
            region_cost = self._region_cost(map_distance, controls.radius)
            unit_length = float(lengths[unit])
            length_cost = abs(unit_length - target_length) / length_denominator
            repeat_cost = self._repeat_cost(recent, unit)
            crossfile_cost = float(self._unit_file[unit] != self._unit_file[current_unit])
            graph_weight = 0.20 + 0.85 * controls.continuity if self._scores is not None else 0.0
            map_weight = 0.10 + 0.55 * controls.continuity
            region_weight = 0.35 + 1.15 * controls.continuity
            length_weight = 0.10 + 0.35 * controls.continuity
            repeat_weight = 0.20 + 1.10 * controls.novelty
            crossfile_weight = 1.30 * controls.crossfile_penalty
            distance = (
                graph_weight * graph_score
                + map_weight * map_distance
                + region_weight * region_cost
                + length_weight * length_cost
                + repeat_weight * repeat_cost
                + crossfile_weight * crossfile_cost
            )
            if unit == anchor_unit:
                distance *= 0.70 + 0.25 * controls.continuity

            scored = (
                distance,
                graph_score,
                map_distance,
                region_cost,
                repeat_cost,
                length_cost,
                unit,
            )
            previous = candidates.get(unit)
            if previous is None or scored[0] < previous[0]:
                candidates[unit] = scored

        if not candidates:
            return current_unit
        ranked = sorted(candidates.values())
        exploration = min(
            1.0,
            max(
                0.0,
                0.45 * (1.0 - controls.continuity)
                + 0.25 * controls.novelty
                + 0.20 * controls.wander
                + 0.10 * controls.motion_rate,
            ),
        )
        pool_size = min(
            len(ranked),
            max(1, 1 + _round_to_int(exploration * float(len(ranked) - 1))),
        )
        state = mix_frame_order_seed(
            self.seed,
            int(self._unit_start[current_unit]),
            int(lengths[current_unit]),
            decision_index,
        )
        _, random_value = _next_frame_order_random(state)
        return int(ranked[int(random_value % pool_size)][-1])

    def _clear_graph_plan(self) -> None:
        self._planned_units.clear()
        self._plan_signature = None

    @staticmethod
    def _plan_matches(
        signature: Optional[_GraphPlanSignature],
        anchor_unit: int,
        controls: WanderGraphControls,
        seed: int,
    ) -> bool:
        if signature is None or signature.anchor_unit != anchor_unit or signature.seed != seed:
            return False
        old = signature.controls
        attractor_distance = (
            (controls.attractor_x - old.attractor_x) ** 2
            + (controls.attractor_y - old.attractor_y) ** 2
        )
        return (
            attractor_distance <= 0.01
            and abs(controls.wander - old.wander) <= 0.05
            and abs(controls.motion_rate - old.motion_rate) <= 0.05
            and abs(controls.radius - old.radius) <= 0.02
            and abs(controls.continuity - old.continuity) <= 0.02
            and abs(controls.novelty - old.novelty) <= 0.02
            and abs(controls.segment_length - old.segment_length) <= 0.02
            and abs(controls.crossfile_penalty - old.crossfile_penalty) <= 0.02
        )

    def _ensure_planned_units(
        self, current_unit: int, anchor_unit: int, controls: WanderGraphControls
    ) -> None:
        if not self.graph_available or controls.wander <= 0.0:
            self._clear_graph_plan()
            return
        if not self._plan_matches(self._plan_signature, anchor_unit, controls, self.seed):
            self._planned_units.clear()
            self._plan_signature = _GraphPlanSignature(anchor_unit, controls, self.seed)

        planning_recent = list(self._recent_units)
        planning_unit = int(current_unit)
        for unit in self._planned_units:
            planning_unit = int(unit)
            self._remember_unit_in(planning_recent, planning_unit)

        while len(self._planned_units) < _TARGET_PLANNED_GRAPH_UNITS:
            decision_index = self._decode_window_index + len(self._planned_units) + 1
            next_unit = self._choose_next_unit(
                planning_unit,
                anchor_unit,
                controls,
                planning_recent,
                decision_index,
            )
            if next_unit < 0:
                break
            self._planned_units.append(next_unit)
            planning_unit = next_unit
            self._remember_unit_in(planning_recent, planning_unit)

    def _consume_planned_unit(
        self, current_unit: int, anchor_unit: int, controls: WanderGraphControls
    ) -> int:
        self._ensure_planned_units(current_unit, anchor_unit, controls)
        if self._planned_units:
            return int(self._planned_units.pop(0))
        return self._choose_next_unit(
            current_unit,
            anchor_unit,
            controls,
            self._recent_units,
            self._decode_window_index,
        )

    def _build_morphology_graph(
        self, anchor: int, window_size: int, controls: WanderGraphControls
    ) -> np.ndarray:
        if not self.graph_available:
            return np.empty(0, dtype=np.int64)
        anchor_unit = self._find_unit_for_frame(anchor)
        if anchor_unit < 0:
            return np.empty(0, dtype=np.int64)

        should_anchor = not self._graph_state_valid or controls.wander <= 0.0
        if should_anchor:
            self._graph_unit = anchor_unit
            self._graph_anchor_unit = anchor_unit
            self._graph_frame = min(
                int(self._unit_end[anchor_unit]) - 1,
                max(int(self._unit_start[anchor_unit]), anchor),
            )
            self._graph_state_valid = True
            self._recent_units.clear()
            self._remember_unit_in(self._recent_units, anchor_unit)
        else:
            self._graph_anchor_unit = anchor_unit

        unit_count = int(self._unit_start.size)
        unit = min(unit_count - 1, max(0, int(self._graph_unit)))
        frame = min(
            int(self._unit_end[unit]) - 1,
            max(int(self._unit_start[unit]), int(self._graph_frame)),
        )
        self._ensure_planned_units(unit, anchor_unit, controls)
        frames: list[int] = []
        for _ in range(window_size):
            frames.append(frame)
            frame += 1
            while frame >= int(self._unit_end[unit]):
                previous_unit = unit
                unit = self._consume_planned_unit(unit, anchor_unit, controls)
                frame = int(self._unit_start[unit])
                if unit != previous_unit or not self._recent_units:
                    self._remember_unit_in(self._recent_units, unit)
                self._ensure_planned_units(unit, anchor_unit, controls)

        self._graph_unit = unit
        self._graph_frame = frame
        self._graph_state_valid = True
        return np.asarray(frames, dtype=np.int64)

    def _apply_latent_colour(self, frames: np.ndarray, amount: float) -> np.ndarray:
        selected = np.ascontiguousarray(self._latents[frames], dtype=np.float32)
        if amount <= 0.0:
            return selected
        amount_f32 = np.float32(amount)
        sigma = np.float32(np.float32(2.0) * amount_f32 * amount_f32)
        noise = np.empty_like(selected, dtype=np.float32)
        raw_cache: dict[int, np.ndarray] = {}

        for row, frame_value in enumerate(frames):
            frame = int(frame_value)
            sequence_start, sequence_end = self._sequence_bounds(frame)
            neighbors = (
                max(sequence_start, min(sequence_end - 1, frame - 2)),
                max(sequence_start, min(sequence_end - 1, frame - 1)),
                frame,
                max(sequence_start, min(sequence_end - 1, frame + 1)),
                max(sequence_start, min(sequence_end - 1, frame + 2)),
            )
            raw = []
            for neighbor in neighbors:
                if neighbor not in raw_cache:
                    raw_cache[neighbor] = _raw_noise_vector(self.seed, neighbor, self._latent_dim)
                raw.append(raw_cache[neighbor])
            smoothed = np.add(raw[0], np.float32(4.0) * raw[1], dtype=np.float32)
            smoothed = np.add(smoothed, np.float32(6.0) * raw[2], dtype=np.float32)
            smoothed = np.add(smoothed, np.float32(4.0) * raw[3], dtype=np.float32)
            smoothed = np.add(smoothed, raw[4], dtype=np.float32)
            noise[row] = np.float32(1.0 / 16.0) * smoothed

        return np.add(selected, sigma * noise, dtype=np.float32)

    def plan(
        self,
        anchor_frame: int,
        window_size: int,
        *,
        frame_source: str = FRAME_SOURCE_K_NEAREST,
        frame_order: float = 0.0,
        latent_colour: float = 0.0,
        phrase_scale: float = 0.4,
        jump_rate: float = 0.55,
        timbre_lock: float = 0.55,
        drift: float = 0.5,
        repeat_avoid: float = 0.75,
        crossfile: float = 0.7,
    ) -> WanderLatentWindow:
        """Plan and assemble one complete raw ``[T,D]`` decoder window."""

        requested_source = str(frame_source).strip().lower()
        if requested_source not in FRAME_SOURCES:
            raise ValueError(f"unsupported Wander frame source {frame_source!r}")
        window = int(window_size)
        if window <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        anchor = int(np.clip(int(anchor_frame), 0, self._frame_count - 1))
        order_amount = _clamp01(frame_order, "frame_order")
        colour_amount = _clamp01(latent_colour, "latent_colour")
        graph_controls = self._map_graph_controls(
            anchor,
            phrase_scale,
            jump_rate,
            timbre_lock,
            drift,
            repeat_avoid,
            crossfile,
        )

        effective_source = requested_source
        knn_nondecreasing = False
        if requested_source == FRAME_SOURCE_K_NEAREST:
            source_frames, knn_nondecreasing = self._build_k_nearest(anchor, window)
            chunk_start = anchor
            self._clear_graph_plan()
        elif requested_source == FRAME_SOURCE_MORPHOLOGY_GRAPH and self.graph_available:
            source_frames = self._build_morphology_graph(anchor, window, graph_controls)
            if source_frames.size == 0:
                source_frames, chunk_start = self._build_contiguous(anchor, window)
                effective_source = FRAME_SOURCE_CONTIGUOUS
            else:
                chunk_start = int(source_frames[0])
        else:
            if requested_source == FRAME_SOURCE_MORPHOLOGY_GRAPH:
                effective_source = FRAME_SOURCE_CONTIGUOUS
            source_frames, chunk_start = self._build_contiguous(anchor, window)
            self._clear_graph_plan()

        input_frames_array = apply_frame_order_disorder(
            source_frames,
            order_amount,
            self.seed,
            chunk_start,
            window,
            self._decode_window_index,
        )
        normalized = self._apply_latent_colour(input_frames_array, colour_amount)
        raw_latents = np.add(
            np.multiply(normalized, self._std[None, :], dtype=np.float32),
            self._mean[None, :],
            dtype=np.float32,
        )
        raw_latents = np.ascontiguousarray(raw_latents, dtype=np.float32)

        diagnostics = WanderWindowDiagnostics(
            requested_frame_source=requested_source,
            effective_frame_source=effective_source,
            graph_available=self.graph_available,
            graph_error=self._graph_error,
            anchor_frame=anchor,
            chunk_start_frame=int(chunk_start),
            decode_window_index=self._decode_window_index,
            frame_order=order_amount,
            latent_colour=colour_amount,
            seed=self.seed,
            k_nearest_distances_nondecreasing=knn_nondecreasing,
            planned_units=tuple(int(unit) for unit in self._planned_units),
            recent_units=tuple(int(unit) for unit in self._recent_units),
            graph_controls=graph_controls,
        )
        self.last_diagnostics = diagnostics
        self._decode_window_index += 1
        return WanderLatentWindow(
            raw_latents=raw_latents,
            source_frames=tuple(int(frame) for frame in source_frames),
            input_frames=tuple(int(frame) for frame in input_frames_array),
            diagnostics=diagnostics,
        )


__all__ = [
    "DEFAULT_WANDER_SEED",
    "FRAME_SOURCE_CONTIGUOUS",
    "FRAME_SOURCE_K_NEAREST",
    "FRAME_SOURCE_MORPHOLOGY_GRAPH",
    "FRAME_SOURCES",
    "WanderGraphControls",
    "WanderLatentWindow",
    "WanderWindowDiagnostics",
    "WanderWindowPlanner",
    "apply_frame_order_disorder",
    "latent_colour_raw_noise",
    "latent_colour_smoothed_noise",
    "mix_frame_order_seed",
]
