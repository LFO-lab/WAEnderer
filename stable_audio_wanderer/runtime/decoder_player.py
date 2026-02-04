"""
Realtime decoder audio output using sounddevice with dual-buffer architecture.
"""
import threading
from collections import deque
import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("sounddevice is required for realtime decoding. Install with: pip install sounddevice") from exc

from ..config import SR


class DecoderPlayer:
    """
    Streaming audio output with dual-buffer architecture for gap-free playback.

    Architecture:
    - Audio callback (sounddevice thread): Pulls from active buffer
    - Decode thread: Fills chunks into ready queue
    - When active buffer exhausted, swap in next chunk from ready queue

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
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.frame_samples = 0
        self.frame_duration = None
        self.underruns = 0

        # Dual-buffer: active chunk being played + queue of ready chunks
        self._active_chunk = np.zeros((0, 2), dtype=np.float32)
        self._active_pos = 0  # Read position in active chunk
        self._chunk_queue = deque()  # Ready chunks waiting to play
        self._queue_lock = threading.Lock()  # Only held briefly during swap

        # For crossfade smoothing between chunks
        self._prev_tail = None

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
        """Queue a decoded audio chunk for playback."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)

        if self.frame_samples == 0:
            self.frame_samples = int(audio.shape[0])
            self.frame_duration = self.frame_samples / float(self.sr)

        # Apply smoothing crossfade with previous tail
        fade_len = int(self._smoothing * audio.shape[0])
        if fade_len > 0 and self._prev_tail is not None:
            fade_len = min(fade_len, self._prev_tail.shape[0], audio.shape[0])
            if fade_len > 0:
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = 1.0 - fade_in
                audio[:fade_len] = (
                    audio[:fade_len] * fade_in[:, None] +
                    self._prev_tail[-fade_len:] * fade_out[:, None]
                )
        if fade_len > 0:
            self._prev_tail = audio[-fade_len:].copy()
        else:
            self._prev_tail = None

        # Apply gain and queue the chunk
        chunk = audio * self._gain

        with self._queue_lock:
            self._chunk_queue.append(chunk)

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
