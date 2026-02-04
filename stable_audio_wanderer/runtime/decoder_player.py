"""
Realtime decoder audio output using sounddevice with dual-buffer architecture.

Supports two crossfade modes:
1. Fixed Hann overlap-add (COLA compliant, 50% overlap) - default
2. Adaptive logarithmic crossfade (variable window sizes) - for dynamic decoding

Logarithmic crossfades maintain perceived loudness better than linear crossfades
because human hearing perceives loudness logarithmically.
"""
import threading
from collections import deque
import numpy as np
from scipy.signal.windows import hann

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("sounddevice is required for realtime decoding. Install with: pip install sounddevice") from exc

from ..config import SR


# Adaptive crossfade constants
CROSSFADE_RATIO = 0.25  # 25% of incoming window as crossfade region
MIN_CROSSFADE_SAMPLES = 64  # Minimum crossfade length


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

    Shorter windows (transients) -> shorter crossfades (preserve attack)
    Longer windows (sustained) -> longer crossfades (smooth blend)
    """
    return max(MIN_CROSSFADE_SAMPLES, int(incoming_window_samples * CROSSFADE_RATIO))


class DecoderPlayer:
    """
    Streaming audio output with dual-buffer architecture and overlap-add.

    Architecture:
    - Audio callback (sounddevice thread): Pulls from chunk queue
    - Decode thread: Applies crossfade + overlap-add, queues result
    - Two modes: fixed Hann (COLA) or adaptive logarithmic (variable windows)

    This eliminates lock contention and ensures continuous playback.
    """

    def __init__(
        self,
        sr: int = SR,
        gain: float = 1.0,
        smoothing: float = 0.1,
        blocksize: int = 0,
        adaptive_crossfade: bool = False,
    ):
        self.sr = int(sr)
        self._gain = float(gain)
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0))  # Kept for API compat
        self.frame_samples = 0
        self.frame_duration = None
        self.underruns = 0
        self._adaptive_crossfade = bool(adaptive_crossfade)

        # Dual-buffer: active chunk being played + queue of ready chunks
        self._active_chunk = np.zeros((0, 2), dtype=np.float32)
        self._active_pos = 0  # Read position in active chunk
        self._chunk_queue = deque()  # Ready chunks waiting to play
        self._queue_lock = threading.Lock()  # Only held briefly during swap

        # Overlap-add state (Hann windowing for COLA compliance)
        self._overlap_samples = 0       # Will be set on first chunk
        self._window = None             # Hann window (periodic)
        self._overlap_buffer = None     # Previous chunk's second half (windowed)

        # Adaptive crossfade state (for variable window sizes)
        self._adaptive_overlap_buffer = None  # Tail from previous frame

        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()

    def close(self):
        self._stream.close()

    def set_gain(self, value: float):
        self._gain = float(max(0.0, value))

    def set_smoothing(self, value: float):
        self._smoothing = float(np.clip(value, 0.0, 1.0))

    def set_adaptive_crossfade(self, enabled: bool):
        """Enable or disable adaptive logarithmic crossfade mode."""
        self._adaptive_crossfade = bool(enabled)
        if enabled:
            # Reset adaptive buffer when switching modes
            self._adaptive_overlap_buffer = None

    def _setup_overlap(self, chunk_len: int):
        """Initialize Hann window for 50% overlap-add (COLA compliant)."""
        self._overlap_samples = chunk_len // 2  # 50% overlap

        # Periodic Hann window (sym=False) satisfies COLA at 50% overlap
        self._window = hann(chunk_len, sym=False).astype(np.float32)

        # Initialize overlap buffer with zeros (for first chunk)
        self._overlap_buffer = np.zeros((self._overlap_samples, 2), dtype=np.float32)

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback - runs in sounddevice's audio thread."""
        if status.output_underflow:
            self.underruns += 1

        out_pos = 0

        while out_pos < frames:
            # Check if we need more data from active chunk
            active_remaining = len(self._active_chunk) - self._active_pos

            if active_remaining <= 0:
                # Try to get next chunk from queue
                with self._queue_lock:
                    if self._chunk_queue:
                        self._active_chunk = self._chunk_queue.popleft()
                        self._active_pos = 0
                        active_remaining = len(self._active_chunk)
                    else:
                        # No chunks available - output silence for remaining frames
                        outdata[out_pos:] = 0
                        self.underruns += 1
                        return

            # Copy what we can from active chunk
            to_copy = min(active_remaining, frames - out_pos)
            outdata[out_pos:out_pos + to_copy] = self._active_chunk[self._active_pos:self._active_pos + to_copy]
            self._active_pos += to_copy
            out_pos += to_copy

    def write_frame(self, audio: np.ndarray):
        """Apply crossfade and overlap-add, then queue for playback.

        Routes to either fixed Hann overlap-add or adaptive logarithmic crossfade
        based on the adaptive_crossfade setting.
        """
        if self._adaptive_crossfade:
            return self._write_frame_adaptive(audio)
        return self._write_frame_hann(audio)

    def _write_frame_hann(self, audio: np.ndarray):
        """Apply Hann window and overlap-add, then queue for playback (COLA compliant)."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)

        # Initialize overlap-add on first chunk
        if self._window is None:
            self._setup_overlap(len(audio))
            self.frame_samples = len(audio)
            self.frame_duration = self.frame_samples / float(self.sr)

        # Apply Hann window to entire chunk
        windowed = audio * self._window[:, None]

        # Split into first half and second half
        half = self._overlap_samples
        first_half = windowed[:half]
        second_half = windowed[half:]

        # Overlap-add: previous second_half + current first_half
        blended = self._overlap_buffer + first_half

        # Apply gain to the blended output
        output = blended * self._gain

        with self._queue_lock:
            self._chunk_queue.append(output)

        # Save current second half for next overlap
        self._overlap_buffer = second_half.copy()

    def _write_frame_adaptive(self, audio: np.ndarray):
        """
        Apply logarithmic crossfade with length based on incoming window.

        Unlike Hann overlap-add (which requires fixed window sizes), this
        approach handles arbitrary window size transitions smoothly.

        Crossfade length is proportional to the incoming window size:
        - Shorter windows (transients) -> shorter crossfades (preserve attack)
        - Longer windows (sustained) -> longer crossfades (smooth blend)
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)

        incoming_samples = len(audio)
        crossfade_len = compute_crossfade_length(incoming_samples)

        # Update frame tracking
        self.frame_samples = incoming_samples
        self.frame_duration = self.frame_samples / float(self.sr)

        if self._adaptive_overlap_buffer is None:
            # First frame: no crossfade needed, just save tail
            if crossfade_len < incoming_samples:
                self._adaptive_overlap_buffer = audio[-crossfade_len:].copy()
                output = audio[:-crossfade_len]
            else:
                # Very short frame - output all but save for blending
                self._adaptive_overlap_buffer = audio.copy()
                output = np.zeros((0, 2), dtype=np.float32)
        else:
            # Crossfade region: blend previous tail with new head
            prev_tail = self._adaptive_overlap_buffer
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
                self._adaptive_overlap_buffer = new_tail.copy() if len(new_tail) > 0 else blended[-crossfade_len:].copy()
            else:
                # Edge case: no blending possible, just output
                output = audio[:-crossfade_len] if crossfade_len < incoming_samples else np.zeros((0, 2), dtype=np.float32)
                self._adaptive_overlap_buffer = audio[-crossfade_len:].copy() if crossfade_len < incoming_samples else audio.copy()

        # Queue output for playback (with gain)
        if len(output) > 0:
            with self._queue_lock:
                self._chunk_queue.append(output * self._gain)

    def buffer_duration(self) -> float:
        """Return approximate buffered audio duration in seconds."""
        with self._queue_lock:
            queued_samples = sum(len(c) for c in self._chunk_queue)
        active_remaining = max(0, len(self._active_chunk) - self._active_pos)
        return (queued_samples + active_remaining) / float(self.sr)

    def get_state(self) -> dict:
        return {
            "gain": self._gain,
            "smoothing": self._smoothing,
            "frame_samples": int(self.frame_samples),
            "underruns": int(self.underruns),
            "buffer_duration": self.buffer_duration(),
            "adaptive_crossfade": self._adaptive_crossfade,
        }
