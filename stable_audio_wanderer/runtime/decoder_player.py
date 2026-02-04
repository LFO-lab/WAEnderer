"""
Realtime decoder audio output using sounddevice with dual-buffer architecture.

Uses 50% overlap-add with Hann windowing for seamless transitions.
Hann window satisfies COLA (Constant Overlap-Add) at 50% overlap.
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


class DecoderPlayer:
    """
    Streaming audio output with dual-buffer architecture and overlap-add.

    Architecture:
    - Audio callback (sounddevice thread): Pulls from chunk queue
    - Decode thread: Applies Hann window + overlap-add, queues result
    - 50% overlap with Hann window satisfies COLA for seamless reconstruction

    This eliminates lock contention and ensures continuous playback.
    """

    def __init__(
        self,
        sr: int = SR,
        gain: float = 1.0,
        smoothing: float = 0.1,
        blocksize: int = 0,
    ):
        self.sr = int(sr)
        self._gain = float(gain)
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0))  # Kept for API compat
        self.frame_samples = 0
        self.frame_duration = None
        self.underruns = 0

        # Dual-buffer: active chunk being played + queue of ready chunks
        self._active_chunk = np.zeros((0, 2), dtype=np.float32)
        self._active_pos = 0  # Read position in active chunk
        self._chunk_queue = deque()  # Ready chunks waiting to play
        self._queue_lock = threading.Lock()  # Only held briefly during swap

        # Overlap-add state (Hann windowing for COLA compliance)
        self._overlap_samples = 0       # Will be set on first chunk
        self._window = None             # Hann window (periodic)
        self._overlap_buffer = None     # Previous chunk's second half (windowed)

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
        """Apply Hann window and overlap-add, then queue for playback."""
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
        }
