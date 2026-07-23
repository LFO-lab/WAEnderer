"""
Realtime decoder audio output using sounddevice with dual-buffer architecture.

"""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("sounddevice is required for realtime decoding. Install with: pip install sounddevice") from exc

from ..config import SR


# Adaptive crossfade constants
CROSSFADE_RATIO = 0.25  # 25% of incoming window as crossfade region
MIN_CROSSFADE_SAMPLES = 64  # Minimum crossfade length
GENERATION_TRANSITION_SAMPLES = 256
PRESENTATION_HISTORY_SIZE = 128


def log_fade_in(length: int) -> np.ndarray:
    """
    Logarithmic fade-in curve (perceptually linear loudness increase).

    Maps [0, 1] -> [0, 1] using log1p for smooth perceptual transition.
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, 1, length, dtype=np.float32)
    return (np.log1p(t * (np.e - 1)) / np.log(np.e)).astype(np.float32)


def log_fade_out(length: int) -> np.ndarray:
    """Logarithmic fade-out (complement of fade-in)."""
    return 1.0 - log_fade_in(length)


def compute_crossfade_length(incoming_window_samples: int) -> int:
    """
    Compute crossfade length based on incoming window size.

    """
    return max(MIN_CROSSFADE_SAMPLES, int(incoming_window_samples * CROSSFADE_RATIO))


def _stereo_channel_last(audio: np.ndarray) -> np.ndarray:
    """Return contiguous float32 PCM in ``[samples, 2]`` layout."""
    pcm = np.asarray(audio, dtype=np.float32)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.ndim != 2:
        raise ValueError(f"PCM must be one- or two-dimensional, got {pcm.shape}")

    # Decoder/OLA code uses [channels, samples]; sounddevice uses [samples, channels].
    if pcm.shape[0] in (1, 2) and pcm.shape[1] > 2:
        pcm = pcm.T
    if pcm.shape[1] == 1:
        pcm = np.repeat(pcm, 2, axis=1)
    if pcm.shape[1] != 2:
        raise ValueError(f"PCM must have one or two channels, got {pcm.shape}")
    if not np.all(np.isfinite(pcm)):
        raise ValueError("PCM contains non-finite samples")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def _validated_provenance(
    pcm_samples: int,
    frame_indices,
    samples_per_frame,
) -> Tuple[Optional[Tuple[int, ...]], Optional[int]]:
    """Validate and normalize optional frame provenance for one PCM chunk."""
    if frame_indices is None and samples_per_frame is None:
        return None, None
    if frame_indices is None or samples_per_frame is None:
        raise ValueError(
            "frame_indices and samples_per_frame must be provided together"
        )

    try:
        raw_indices = tuple(frame_indices)
        indices = tuple(int(index) for index in raw_indices)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frame_indices must be an iterable of integers") from exc
    if not indices:
        raise ValueError("frame_indices must not be empty")
    if any(index < 0 or index != raw for index, raw in zip(indices, raw_indices)):
        raise ValueError("frame_indices must contain non-negative integers")

    try:
        frame_samples = int(samples_per_frame)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("samples_per_frame must be a positive integer") from exc
    if frame_samples <= 0 or frame_samples != samples_per_frame:
        raise ValueError("samples_per_frame must be a positive integer")
    expected_samples = len(indices) * frame_samples
    if expected_samples != int(pcm_samples):
        raise ValueError(
            "PCM/provenance length mismatch: "
            f"{len(indices)} frame indices * {frame_samples} samples != "
            f"{pcm_samples} PCM samples"
        )
    return indices, frame_samples


@dataclass(frozen=True)
class _PcmChunk:
    pcm: np.ndarray
    frame_indices: Optional[Tuple[int, ...]] = None
    samples_per_frame: Optional[int] = None

    def __len__(self) -> int:
        return int(len(self.pcm))


class _IndexRing:
    """Fixed-capacity index history with no storage growth on append."""

    __slots__ = ("_values", "_start", "_size", "_overflowed")

    def __init__(self, capacity: int) -> None:
        self._values = [0] * int(capacity)
        self._start = 0
        self._size = 0
        self._overflowed = False

    @property
    def capacity(self) -> int:
        return len(self._values)

    @property
    def size(self) -> int:
        return self._size

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def append(self, value: int) -> None:
        if self._size < self.capacity:
            slot = (self._start + self._size) % self.capacity
            self._size += 1
        else:
            slot = self._start
            self._start = (self._start + 1) % self.capacity
            self._overflowed = True
        self._values[slot] = int(value)

    def get(self, offset: int) -> int:
        if not 0 <= int(offset) < self._size:
            raise IndexError(offset)
        return int(self._values[(self._start + int(offset)) % self.capacity])

    def clear(self) -> None:
        self._start = 0
        self._size = 0
        self._overflowed = False

    def to_list(self) -> list[int]:
        return [self.get(offset) for offset in range(self._size)]


@dataclass
class _PresentationCursor:
    """Cursor for one generation; events stay private until it is presented."""

    generation: int
    index: Optional[int] = None
    samples_into_frame: int = 0
    samples_per_frame: Optional[int] = None
    events: _IndexRing = field(
        default_factory=lambda: _IndexRing(PRESENTATION_HISTORY_SIZE)
    )
    chunk: Optional[_PcmChunk] = None
    chunk_pos: int = 0


class GenerationPcmBuffer:
    """Prepared-PCM queue with a latest-wins generation handoff.

    This class performs no decoding and allocates no model data.  A producer
    enqueues complete PCM hops.  The consumer renders the current generation
    until at least one callback-sized block from a newer generation is ready,
    then crossfades over exactly ``transition_samples`` samples and discards
    the remaining old queue.
    """

    def __init__(self, channels: int = 2, transition_samples: int = GENERATION_TRANSITION_SAMPLES):
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if int(transition_samples) <= 0:
            raise ValueError("transition_samples must be positive")
        self.channels = int(channels)
        self.transition_samples = int(transition_samples)
        self._empty = np.zeros((0, self.channels), dtype=np.float32)
        self._empty_chunk = _PcmChunk(self._empty)
        self._transition_fade_in = np.linspace(
            0.0, 1.0, self.transition_samples, dtype=np.float32
        )
        self._transition_fade_out = 1.0 - self._transition_fade_in
        self._transition_scratch = np.empty(
            (self.transition_samples, self.channels), dtype=np.float32
        )
        self._fade_curve = np.zeros((0,), dtype=np.float32)

        # Presentation survives queue resets so Stop and the next prebuffer hold
        # the last position that was actually heard.
        self._presentation_index = None
        self._presentation_recent = _IndexRing(PRESENTATION_HISTORY_SIZE)
        self._presentation_generation = None
        self._presentation_samples_into_frame = 0
        self._presentation_samples_per_frame = None
        self.reset()

    def reset(self):
        self._active = self._empty_chunk
        self._active_pos = 0
        self._queue = deque()
        self._current_generation = None
        self._current_cursor = None

        self._pending_active = self._empty_chunk
        self._pending_pos = 0
        self._pending_queue = deque()
        self._pending_generation = None
        self._pending_cursor = None
        self._transition_progress = 0

        # A successor requested during an in-progress crossfade waits here.
        # Finishing the audible ramp avoids jumping back to the prior stream;
        # only the newest successor is retained.
        self._next_pending_active = self._empty_chunk
        self._next_pending_pos = 0
        self._next_pending_queue = deque()
        self._next_pending_generation = None
        self._next_pending_cursor = None

        # A producer announces a requested generation before its PCM is ready.
        # This watermark lets the callback keep consuming the committed stream
        # while immediately invalidating a superseded staged stream.
        self._requested_generation = None

        self._fade_total = 0
        self._fade_progress = 0
        self._last_active_sample = np.zeros((self.channels,), dtype=np.float32)
        self._last_active_sample_valid = False

    def capture_presentation_hold(
        self,
        index: int,
        *,
        generation=None,
        samples_per_frame=None,
    ) -> bool:
        """Capture the first stationary UI position without moving an old cursor."""
        if self._presentation_index is not None:
            return False
        try:
            hold_index = int(index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "presentation index must be a non-negative integer"
            ) from exc
        if hold_index < 0 or hold_index != index:
            raise ValueError("presentation index must be a non-negative integer")
        if samples_per_frame is not None:
            try:
                frame_samples = int(samples_per_frame)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "samples_per_frame must be a positive integer"
                ) from exc
            if frame_samples <= 0 or frame_samples != samples_per_frame:
                raise ValueError("samples_per_frame must be a positive integer")
        else:
            frame_samples = None

        self._presentation_index = hold_index
        self._presentation_recent.append(hold_index)
        self._presentation_generation = (
            None if generation is None else int(generation)
        )
        self._presentation_samples_into_frame = 0
        self._presentation_samples_per_frame = frame_samples
        return True

    def get_presentation_state(self):
        """Return the current audible cursor, or ``None`` before it is captured."""
        if self._presentation_index is None:
            return None
        return {
            "index": int(self._presentation_index),
            "recent_indices": self._presentation_recent.to_list(),
            "generation": self._presentation_generation,
            "samples_into_frame": int(self._presentation_samples_into_frame),
            "samples_per_frame": self._presentation_samples_per_frame,
        }

    @property
    def current_generation(self):
        return self._current_generation

    @property
    def pending_generation(self):
        if self._next_pending_generation is not None:
            return self._next_pending_generation
        return self._pending_generation

    @property
    def transition_status(self) -> str:
        if self._pending_generation is None:
            return "idle"
        if self._transition_progress > 0:
            return "crossfading"
        return "staging"

    def _remaining(self, active, pos, chunks) -> int:
        return max(0, len(active) - int(pos)) + sum(len(chunk) for chunk in chunks)

    def buffered_samples(self, generation=None) -> int:
        if generation is None or generation == self._current_generation:
            return self._remaining(self._active, self._active_pos, self._queue)
        if generation == self._pending_generation:
            return self._remaining(
                self._pending_active, self._pending_pos, self._pending_queue
            )
        if generation == self._next_pending_generation:
            return self._remaining(
                self._next_pending_active,
                self._next_pending_pos,
                self._next_pending_queue,
            )
        return 0

    def _clear_next_pending(self) -> None:
        self._next_pending_active = self._empty_chunk
        self._next_pending_pos = 0
        self._next_pending_queue.clear()
        self._next_pending_generation = None
        self._next_pending_cursor = None

    def request_generation(self, generation: int) -> bool:
        """Declare the newest requested generation without interrupting playback.

        Older staged PCM is discarded immediately unless it is already audible
        in a crossfade; that ramp alone is allowed to finish before the newest
        successor takes over.  The committed generation remains available
        until replacement PCM can complete a transition.  Enqueue rejects
        results below this watermark, closing the decode-result/write race.
        """
        gen = int(generation)
        if self._requested_generation is not None and gen < self._requested_generation:
            return False
        self._requested_generation = gen
        if self._pending_generation is not None and self._pending_generation < gen:
            if self._transition_progress > 0:
                # This generation is already audible.  Finish only its current
                # ramp, then transition to the newest request.
                self._clear_next_pending()
                self._next_pending_generation = gen
                self._next_pending_cursor = _PresentationCursor(gen)
            else:
                self._pending_active = self._empty_chunk
                self._pending_pos = 0
                self._pending_queue.clear()
                self._pending_generation = None
                self._pending_cursor = None
                self._transition_progress = 0
        elif (
            self._next_pending_generation is not None
            and self._next_pending_generation < gen
        ):
            self._clear_next_pending()
            self._next_pending_generation = gen
            self._next_pending_cursor = _PresentationCursor(gen)
        return True

    def enqueue(
        self,
        audio: np.ndarray,
        generation: int = 0,
        frame_indices=None,
        samples_per_frame=None,
    ) -> bool:
        pcm = _stereo_channel_last(audio)
        if len(pcm) == 0:
            return False
        indices, frame_samples = _validated_provenance(
            len(pcm), frame_indices, samples_per_frame
        )
        chunk = _PcmChunk(pcm, indices, frame_samples)
        gen = int(generation)

        if self._requested_generation is not None and gen < self._requested_generation:
            return False

        if self._current_generation is None:
            self._current_generation = gen
            self._current_cursor = _PresentationCursor(gen)
            self._queue.append(chunk)
            return True

        if self._next_pending_generation is not None:
            if gen < self._next_pending_generation:
                return False
            if gen != self._next_pending_generation:
                self._clear_next_pending()
                self._next_pending_generation = gen
                self._next_pending_cursor = _PresentationCursor(gen)
            self._next_pending_queue.append(chunk)
            return True

        if (
            self._pending_generation is not None
            and self._transition_progress > 0
            and gen > self._pending_generation
        ):
            self._next_pending_generation = gen
            self._next_pending_cursor = _PresentationCursor(gen)
            self._next_pending_queue.append(chunk)
            return True

        newest = (
            self._pending_generation
            if self._pending_generation is not None
            else self._current_generation
        )
        if gen < newest:
            return False

        if gen == self._current_generation and self._pending_generation is None:
            if self._current_cursor is None:
                self._current_cursor = _PresentationCursor(gen)
            self._queue.append(chunk)
            return True

        if gen != self._pending_generation:
            # A rapid change supersedes every staged sample from the prior request.
            self._pending_active = self._empty_chunk
            self._pending_pos = 0
            self._pending_queue.clear()
            self._pending_generation = gen
            self._pending_cursor = _PresentationCursor(gen)
            self._transition_progress = 0
        self._pending_queue.append(chunk)
        return True

    @classmethod
    def _pull_into(
        cls,
        output,
        active,
        pos,
        chunks,
        frames: int,
        *,
        cursor: Optional[_PresentationCursor] = None,
        audible_samples: int = 0,
        activate_final_boundary: bool = True,
    ):
        output[:frames].fill(0.0)
        written = 0
        audible_remaining = max(0, int(audible_samples))
        while written < frames:
            remaining = len(active) - pos
            if remaining <= 0:
                if not chunks:
                    break
                active = chunks.popleft()
                pos = 0
                remaining = len(active)
            count = min(remaining, frames - written)
            output[written : written + count] = active.pcm[pos : pos + count]
            if audible_remaining > 0:
                audible = min(count, audible_remaining)
                cls._advance_cursor(
                    cursor,
                    active,
                    pos,
                    audible,
                    activate_final_boundary=(
                        activate_final_boundary or audible < audible_remaining
                    ),
                )
                audible_remaining -= audible
            written += count
            pos += count
        return written, active, pos

    def _promote_pending(self):
        old_queue = self._queue
        self._active = self._pending_active
        self._active_pos = self._pending_pos
        self._queue = self._pending_queue
        self._current_generation = self._pending_generation
        self._current_cursor = self._pending_cursor

        old_queue.clear()
        if self._next_pending_generation is not None:
            self._pending_active = self._next_pending_active
            self._pending_pos = self._next_pending_pos
            self._pending_queue = self._next_pending_queue
            self._pending_generation = self._next_pending_generation
            self._pending_cursor = self._next_pending_cursor
            self._next_pending_active = self._empty_chunk
            self._next_pending_pos = 0
            self._next_pending_queue = old_queue
            self._next_pending_generation = None
            self._next_pending_cursor = None
        else:
            self._pending_active = self._empty_chunk
            self._pending_pos = 0
            self._pending_queue = old_queue
            self._pending_generation = None
            self._pending_cursor = None
        self._transition_progress = 0

    def _clear_after_fade(self):
        self._active = self._empty_chunk
        self._active_pos = 0
        self._queue.clear()
        # The producer recreates this cursor if same-generation PCM resumes.
        # Avoid constructing presentation storage on the audio callback.
        self._current_cursor = None
        self._pending_active = self._empty_chunk
        self._pending_pos = 0
        self._pending_queue.clear()
        self._pending_generation = None
        self._pending_cursor = None
        self._transition_progress = 0
        self._clear_next_pending()
        self._fade_total = 0
        self._fade_progress = 0

    def fade_to_silence(self, samples: int = GENERATION_TRANSITION_SAMPLES):
        self._fade_total = max(1, int(samples))
        self._fade_progress = 0
        self._fade_curve = np.linspace(
            1.0, 0.0, self._fade_total, dtype=np.float32
        )

    def _mix_pending_into(
        self,
        output,
        frames: int,
        *,
        audible_samples: int = 0,
        activate_final_boundary: bool = True,
    ) -> int:
        written = 0
        audible_remaining = max(0, int(audible_samples))
        while written < frames:
            remaining = len(self._pending_active) - self._pending_pos
            if remaining <= 0:
                if not self._pending_queue:
                    break
                self._pending_active = self._pending_queue.popleft()
                self._pending_pos = 0
                remaining = len(self._pending_active)
            count = min(remaining, frames - written)
            source = self._pending_active.pcm[
                self._pending_pos : self._pending_pos + count
            ]
            if audible_remaining > 0:
                audible = min(count, audible_remaining)
                self._advance_cursor(
                    self._pending_cursor,
                    self._pending_active,
                    self._pending_pos,
                    audible,
                    activate_final_boundary=(
                        activate_final_boundary or audible < audible_remaining
                    ),
                )
                audible_remaining -= audible
            fade_remaining = self.transition_samples - self._transition_progress
            fade_count = min(count, max(0, fade_remaining))
            if fade_count > 0:
                fade_start = self._transition_progress
                fade_end = fade_start + fade_count
                fade_in = self._transition_fade_in[fade_start:fade_end]
                fade_out = self._transition_fade_out[fade_start:fade_end]
                destination = output[written : written + fade_count]
                np.multiply(destination, fade_out[:, None], out=destination)
                scratch = self._transition_scratch[:fade_count]
                np.multiply(source[:fade_count], fade_in[:, None], out=scratch)
                np.add(destination, scratch, out=destination)
                self._transition_progress += fade_count
            if count > fade_count:
                output[written + fade_count : written + count] = source[fade_count:count]
            written += count
            self._pending_pos += count
        return written

    @staticmethod
    def _advance_cursor(
        cursor: Optional[_PresentationCursor],
        chunk: _PcmChunk,
        start: int,
        count: int,
        *,
        activate_final_boundary: bool = True,
    ) -> None:
        if cursor is None or count <= 0:
            return
        if chunk.frame_indices is None or chunk.samples_per_frame is None:
            # Untagged legacy PCM holds the last authoritative position.  Break
            # chunk continuity so the next tagged hop starts a fresh event.
            cursor.chunk = None
            cursor.chunk_pos = 0
            return

        indices = chunk.frame_indices
        frame_samples = int(chunk.samples_per_frame)
        position = int(start)
        end = min(len(chunk), position + int(count))
        if position >= end:
            return

        if cursor.chunk is not chunk or cursor.chunk_pos != position:
            frame_offset = min(position // frame_samples, len(indices) - 1)
            cursor.index = int(indices[frame_offset])
            cursor.samples_per_frame = frame_samples
            cursor.samples_into_frame = position - frame_offset * frame_samples
            cursor.events.append(cursor.index)
            cursor.chunk = chunk
            cursor.chunk_pos = position

        while position < end:
            frame_offset = min(position // frame_samples, len(indices) - 1)
            within_frame = position - frame_offset * frame_samples
            step = min(frame_samples - within_frame, end - position)
            cursor.index = int(indices[frame_offset])
            cursor.samples_per_frame = frame_samples
            cursor.samples_into_frame = within_frame + step
            position += step

            # At an exact boundary, the playback clock now points at the next
            # represented frame even if the callback ends on that boundary.
            if (
                position < len(chunk)
                and position % frame_samples == 0
                and (position < end or activate_final_boundary)
            ):
                next_offset = position // frame_samples
                cursor.index = int(indices[next_offset])
                cursor.samples_into_frame = 0
                cursor.events.append(cursor.index)

        cursor.chunk = chunk
        cursor.chunk_pos = position

    @staticmethod
    def _advance_to_queued_chunk(
        cursor: Optional[_PresentationCursor],
        active: _PcmChunk,
        active_pos: int,
        chunks,
    ) -> None:
        """Resolve an exact hop boundary to an already-buffered next frame."""
        if (
            cursor is None
            or cursor.chunk is not active
            or cursor.chunk_pos != int(active_pos)
            or int(active_pos) != len(active)
            or not chunks
        ):
            return
        next_chunk = chunks[0]
        if (
            next_chunk.frame_indices is None
            or next_chunk.samples_per_frame is None
        ):
            return
        cursor.index = int(next_chunk.frame_indices[0])
        cursor.samples_into_frame = 0
        cursor.samples_per_frame = int(next_chunk.samples_per_frame)
        cursor.events.append(cursor.index)
        cursor.chunk = next_chunk
        cursor.chunk_pos = 0

    def _commit_cursor(self, cursor: Optional[_PresentationCursor]) -> None:
        if cursor is None or cursor.index is None:
            return

        events = cursor.events
        # A first-run hold already represents the first frame at time zero.
        # Do not add that same point twice when its PCM begins, while preserving
        # genuine repeated indices inside the provenance stream.
        first_event = 0
        if (
            events.size > 0
            and not events.overflowed
            and self._presentation_index == events.get(0)
            and self._presentation_generation == cursor.generation
            and self._presentation_samples_into_frame == 0
        ):
            first_event = 1
        event_offset = first_event
        while event_offset < events.size:
            self._presentation_recent.append(events.get(event_offset))
            event_offset += 1
        events.clear()

        self._presentation_index = int(cursor.index)
        self._presentation_generation = int(cursor.generation)
        self._presentation_samples_into_frame = int(cursor.samples_into_frame)
        self._presentation_samples_per_frame = cursor.samples_per_frame

    def render_into(self, output: np.ndarray) -> bool:
        """Render prepared PCM into a caller-owned buffer; return underrun state."""
        if output.ndim != 2 or output.shape[1] != self.channels:
            raise ValueError(
                f"output must be [frames,{self.channels}], got {output.shape}"
            )
        frames = int(output.shape[0])
        if frames == 0:
            return False

        if self._fade_total > 0:
            fade_remaining = max(0, self._fade_total - self._fade_progress)
            presentation_limit = min(frames, fade_remaining)
            fade_finishes = fade_remaining <= frames
        else:
            presentation_limit = frames
            fade_finishes = False

        transition_remaining = 0
        transition_will_run = False
        if self._pending_generation is not None:
            pending_available = self.buffered_samples(self._pending_generation)
            transition_remaining = max(
                0, self.transition_samples - self._transition_progress
            )
            needed = max(frames, transition_remaining)
            transition_will_run = pending_available >= needed

        transition_finishes = (
            transition_will_run and presentation_limit >= transition_remaining
        )
        current_continues = not fade_finishes and not transition_finishes
        pending_continues = not fade_finishes
        outgoing_audible = min(
            frames,
            transition_remaining if transition_will_run else frames,
            presentation_limit,
        )
        replacement_audible = (
            min(frames, presentation_limit) if transition_will_run else 0
        )

        old_written, self._active, self._active_pos = self._pull_into(
            output,
            self._active,
            self._active_pos,
            self._queue,
            frames,
            cursor=self._current_cursor,
            audible_samples=outgoing_audible,
            activate_final_boundary=current_continues,
        )
        if old_written > 0:
            self._last_active_sample[:] = output[old_written - 1]
            self._last_active_sample_valid = True
        replacement_written = 0
        outgoing_shortfall = False

        if transition_will_run:
            if old_written < frames:
                # A late replacement can arrive after the outgoing queue has
                # fewer than 256 samples left.  Extend its terminal sample so
                # the fixed-length transition stays continuous, and still
                # report the source shortfall as an underrun.
                outgoing_shortfall = True
                if self._last_active_sample_valid:
                    output[old_written:frames] = self._last_active_sample
            replacement_written = self._mix_pending_into(
                output,
                frames,
                audible_samples=replacement_audible,
                activate_final_boundary=pending_continues,
            )

        if self._fade_total > 0:
            remaining = self._fade_total - self._fade_progress
            fade_count = min(frames, max(0, remaining))
            presentation_limit = fade_count
            if fade_count > 0:
                fade_end = self._fade_progress + fade_count
                output[:fade_count] *= self._fade_curve[
                    self._fade_progress : fade_end, None
                ]
                self._fade_progress += fade_count
            if fade_count < frames:
                output[fade_count:] = 0.0

        if current_continues:
            self._advance_to_queued_chunk(
                self._current_cursor,
                self._active,
                self._active_pos,
                self._queue,
            )
        if pending_continues:
            self._advance_to_queued_chunk(
                self._pending_cursor,
                self._pending_active,
                self._pending_pos,
                self._pending_queue,
            )

        if transition_finishes:
            self._commit_cursor(self._current_cursor)
            self._promote_pending()
            self._commit_cursor(self._current_cursor)
        else:
            self._commit_cursor(self._current_cursor)

        if fade_finishes:
            self._clear_after_fade()

        fully_covered = old_written >= frames or replacement_written >= frames
        return outgoing_shortfall or not fully_covered

    def render(self, frames: int):
        """Allocate and render a block (convenience API for tests/non-RT writers)."""
        frames = max(0, int(frames))
        output = np.empty((frames, self.channels), dtype=np.float32)
        underrun = self.render_into(output)
        return output, underrun


class DecoderPlayer:
    """
    Streaming output for prepared PCM, with a legacy Torch frame adapter.

    Architecture:
    - Audio callback (sounddevice thread): consumes prepared PCM, applies the
      prebuilt generation transition and output gain, and zero-fills underruns.
    - Decode worker: supplies exact ONNX/OLA hops through :meth:`write_hop`.
    - Standalone Torch transport: retains its historical adaptive frame
      crossfade through :meth:`write_frame`.

    If callback streams are unavailable (e.g. CFFI callback allocation blocked),
    automatically falls back to blocking stream writes from a Python worker thread.

    No model inference or overlap-add runs in the callback.
    """

    def __init__(
        self,
        sr: int = SR,
        gain: float = 1.0,
        blocksize: int = 0,
    ):
        self.sr = int(sr)
        self._gain = float(gain)
        self.frame_samples = 0
        self.frame_duration = None
        self.underruns = 0

        # Prepared PCM only.  ONNX decoding and OLA happen on the decode worker.
        self._pcm_buffer = GenerationPcmBuffer()
        self._queue_lock = threading.Lock()

        # Adaptive crossfade state
        self._crossfade_buffer = None  # Tail from previous frame for blending

        self._blocksize = int(blocksize)
        self._running = threading.Event()
        self._writer_thread = None
        self._use_callback_stream = True

        force_blocking = os.environ.get("STABLE_AUDIO_BLOCKING_STREAM", "").lower() in ("1", "true", "yes")
        if force_blocking:
            self._use_callback_stream = False
            self._stream = sd.OutputStream(
                samplerate=self.sr,
                channels=2,
                dtype="float32",
                blocksize=self._blocksize,
            )
            print("[warn] STABLE_AUDIO_BLOCKING_STREAM enabled: using blocking audio stream mode")
        else:
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.sr,
                    channels=2,
                    dtype="float32",
                    blocksize=self._blocksize,
                    callback=self._callback,
                )
            except MemoryError as exc:
                # Some macOS environments deny writable+executable memory for ffi callbacks.
                self._use_callback_stream = False
                self._stream = sd.OutputStream(
                    samplerate=self.sr,
                    channels=2,
                    dtype="float32",
                    blocksize=self._blocksize,
                )
                print("[warn] Callback stream unavailable; falling back to blocking audio stream mode")
                print(f"[warn] Original callback error: {exc}")

    def start(self):
        self._stream.start()
        if not self._use_callback_stream:
            self._running.set()
            self._writer_thread = threading.Thread(target=self._blocking_writer_loop, daemon=True)
            self._writer_thread.start()

    def stop(self):
        self._running.clear()
        if self._writer_thread is not None:
            self._writer_thread.join()
            self._writer_thread = None
        self._stream.stop()

    def reset_buffers(self):
        """
        Drop queued/active audio and reset crossfade state.
        Useful when restarting transport without recreating the stream.
        """
        with self._queue_lock:
            self._pcm_buffer.reset()
        self._crossfade_buffer = None

    def close(self):
        self._stream.close()

    def set_gain(self, value: float):
        self._gain = float(max(0.0, value))

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback - runs in sounddevice's audio thread."""
        if getattr(status, "output_underflow", False):
            self.underruns += 1
        with self._queue_lock:
            underrun = self._pcm_buffer.render_into(outdata)
        outdata *= float(self._gain)
        if underrun:
            self.underruns += 1

    def _blocking_writer_loop(self):
        """
        Stream writer loop for environments that cannot allocate CFFI callbacks.
        """
        frames = self._blocksize if self._blocksize > 0 else 1024
        chunk = np.empty((frames, 2), dtype=np.float32)
        while self._running.is_set():
            with self._queue_lock:
                underrun = self._pcm_buffer.render_into(chunk)
            if underrun:
                self.underruns += 1

            try:
                chunk *= float(self._gain)
                self._stream.write(chunk)
            except Exception:
                self.underruns += 1
                time.sleep(0.01)

    def write_hop(
        self,
        audio: np.ndarray,
        generation: int = 0,
        frame_indices=None,
        samples_per_frame=None,
    ) -> bool:
        """Atomically queue one PCM hop and its optional frame provenance."""
        pcm = _stereo_channel_last(audio)
        with self._queue_lock:
            accepted = self._pcm_buffer.enqueue(
                pcm,
                generation=int(generation),
                frame_indices=frame_indices,
                samples_per_frame=samples_per_frame,
            )
            if accepted:
                self.frame_samples = int(len(pcm))
                self.frame_duration = self.frame_samples / float(self.sr)
            return accepted

    def capture_presentation_hold(
        self,
        index: int,
        *,
        generation=None,
        samples_per_frame=None,
    ) -> bool:
        """Capture a first-run stationary cursor before producer prebuffering."""
        with self._queue_lock:
            return self._pcm_buffer.capture_presentation_hold(
                index,
                generation=generation,
                samples_per_frame=samples_per_frame,
            )

    def get_presentation_state(self):
        """Return an atomic playback-clocked presentation snapshot."""
        with self._queue_lock:
            return self._pcm_buffer.get_presentation_state()

    def request_generation(self, generation: int) -> bool:
        """Invalidate staged PCM older than ``generation`` while current PCM plays."""
        with self._queue_lock:
            return self._pcm_buffer.request_generation(int(generation))

    def fade_to_silence(self, samples: int = GENERATION_TRANSITION_SAMPLES):
        """Request a callback-side fade, used after a latched runtime failure."""
        with self._queue_lock:
            self._pcm_buffer.fade_to_silence(samples)

    @property
    def current_generation(self):
        with self._queue_lock:
            return self._pcm_buffer.current_generation

    def generation_buffer_duration(self, generation: int) -> float:
        with self._queue_lock:
            samples = self._pcm_buffer.buffered_samples(int(generation))
        return samples / float(self.sr)

    def write_frame(self, audio: np.ndarray):
        """
        Apply adaptive logarithmic crossfade, then queue for playback.

        Crossfade length is proportional to the incoming window size:
        - Shorter windows (transients) -> shorter crossfades (preserve attack)
        - Longer windows (sustained) -> longer crossfades (smooth blend)
        """
        audio = _stereo_channel_last(audio)

        incoming_samples = len(audio)
        crossfade_len = compute_crossfade_length(incoming_samples)

        # Update frame tracking
        self.frame_samples = incoming_samples
        self.frame_duration = self.frame_samples / float(self.sr)

        if self._crossfade_buffer is None:
            # First frame: no crossfade needed, just save tail
            if crossfade_len < incoming_samples:
                self._crossfade_buffer = audio[-crossfade_len:].copy()
                output = audio[:-crossfade_len]
            else:
                # Very short frame - output all but save for blending
                self._crossfade_buffer = audio.copy()
                output = np.zeros((0, 2), dtype=np.float32)
        else:
            # Crossfade region: blend previous tail with new head
            prev_tail = self._crossfade_buffer
            prev_len = len(prev_tail)

            # Use minimum of prev tail and new crossfade length
            blend_len = min(prev_len, crossfade_len, incoming_samples)

            if blend_len > 0:
                # Extract regions
                new_head = audio[:blend_len]

                # Apply logarithmic crossfade
                fade_out = log_fade_out(blend_len)[:, None]  # [blend_len, 1] for stereo
                fade_in = log_fade_in(blend_len)[:, None]

                # Trim prev_tail to match blend length
                prev_tail_trimmed = prev_tail[-blend_len:]

                blended = prev_tail_trimmed * fade_out + new_head * fade_in

                # Determine body and tail regions
                if crossfade_len < incoming_samples:
                    new_body = audio[blend_len:-crossfade_len]
                    new_tail = audio[-crossfade_len:]
                else:
                    new_body = np.zeros((0, 2), dtype=np.float32)
                    new_tail = audio[blend_len:] if blend_len < incoming_samples else audio

                # Concatenate: blended region + body
                if len(new_body) > 0:
                    output = np.concatenate([blended, new_body], axis=0)
                else:
                    output = blended

                # Save new tail for next crossfade
                self._crossfade_buffer = new_tail.copy() if len(new_tail) > 0 else blended[-crossfade_len:].copy()
            else:
                # Edge case: no blending possible, just output
                output = audio[:-crossfade_len] if crossfade_len < incoming_samples else np.zeros((0, 2), dtype=np.float32)
                self._crossfade_buffer = audio[-crossfade_len:].copy() if crossfade_len < incoming_samples else audio.copy()

        # Queue output for playback (with gain)
        if len(output) > 0:
            with self._queue_lock:
                self._pcm_buffer.enqueue(output, generation=0)

    def buffer_duration(self) -> float:
        """Return approximate buffered audio duration in seconds."""
        with self._queue_lock:
            samples = self._pcm_buffer.buffered_samples()
        return samples / float(self.sr)

    def get_state(self) -> dict:
        with self._queue_lock:
            generation = self._pcm_buffer.current_generation
            pending_generation = self._pcm_buffer.pending_generation
            transition_status = self._pcm_buffer.transition_status
            buffer_duration = (
                self._pcm_buffer.buffered_samples() / float(self.sr)
            )
        return {
            "gain": self._gain,
            "frame_samples": int(self.frame_samples),
            "underruns": int(self.underruns),
            "buffer_duration": buffer_duration,
            "generation": generation,
            "pending_generation": pending_generation,
            "transition_status": transition_status,
        }
