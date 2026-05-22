"""
Realtime decoder audio output using sounddevice with dual-buffer architecture.

"""
import os
import threading
import time
from collections import deque
import numpy as np

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

    """
    return max(MIN_CROSSFADE_SAMPLES, int(incoming_window_samples * CROSSFADE_RATIO))


class DecoderPlayer:
    """
    Streaming audio output with dual-buffer architecture and adaptive crossfade.

    Architecture:
    - Audio callback (sounddevice thread): Pulls from chunk queue
    - Decode thread: Applies adaptive logarithmic crossfade, queues result

    If callback streams are unavailable (e.g. CFFI callback allocation blocked),
    automatically falls back to blocking stream writes from a Python worker thread.

    Crossfade length scales with incoming window size (shorter windows ->
    shorter crossfades to preserve transients; longer windows -> longer
    crossfades for smooth blending).

    This eliminates lock contention and ensures continuous playback.
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

        # Dual-buffer: active chunk being played + queue of ready chunks
        self._active_chunk = np.zeros((0, 2), dtype=np.float32)
        self._active_pos = 0  # Read position in active chunk
        self._chunk_queue = deque()  # Ready chunks waiting to play
        self._queue_lock = threading.Lock()  # Only held briefly during swap

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
            self._writer_thread.join(timeout=1.0)
            self._writer_thread = None
        self._stream.stop()

    def reset_buffers(self):
        """
        Drop queued/active audio and reset crossfade state.
        Useful when restarting transport without recreating the stream.
        """
        with self._queue_lock:
            self._chunk_queue.clear()
            self._active_chunk = np.zeros((0, 2), dtype=np.float32)
            self._active_pos = 0
        self._crossfade_buffer = None

    def close(self):
        self._stream.close()

    def set_gain(self, value: float):
        self._gain = float(max(0.0, value))

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

    def _blocking_writer_loop(self):
        """
        Stream writer loop for environments that cannot allocate CFFI callbacks.
        """
        while self._running.is_set():
            chunk = None
            with self._queue_lock:
                if self._chunk_queue:
                    chunk = self._chunk_queue.popleft()

            if chunk is None:
                time.sleep(0.002)
                continue

            try:
                self._stream.write(chunk)
            except Exception:
                self.underruns += 1
                time.sleep(0.01)

    def write_frame(self, audio: np.ndarray):
        """
        Apply adaptive logarithmic crossfade, then queue for playback.

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
            "frame_samples": int(self.frame_samples),
            "underruns": int(self.underruns),
            "buffer_duration": self.buffer_duration(),
        }
