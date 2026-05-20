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


class BufferedAudioQueue:
    """
    Crossfaded audio queue used by DecoderPlayer and offline benchmarks.

    This class owns the non-device portion of the runtime audio path:
    decoded frames are overlap-added, converted to stereo if needed, and
    appended to an output queue. A realtime player can drain the queue to an
    audio device, while evaluation code can stop after the queue-write step and
    treat that as "audio buffer ready".
    """

    def __init__(self, sr: int, gain: float = 1.0):
        self.sr = int(sr)
        self._gain = float(gain)
        self.frame_samples = 0
        self.frame_duration = None

        self._active_chunk = np.zeros((0, 2), dtype=np.float32)
        self._active_pos = 0
        self._chunk_queue = deque()
        self._queue_lock = threading.Lock()
        self._crossfade_buffer = None

    @property
    def gain(self) -> float:
        return float(self._gain)

    def set_gain(self, value: float):
        self._gain = float(max(0.0, value))

    def clear_output(self, preserve_crossfade: bool = False):
        """
        Drop queued audio.

        Args:
            preserve_crossfade: When true, keep the overlap-add tail so the next
                write behaves like steady-state streaming. Benchmarks use this to
                avoid unbounded queue growth without resetting the blend state.
        """
        with self._queue_lock:
            self._chunk_queue.clear()
            self._active_chunk = np.zeros((0, 2), dtype=np.float32)
            self._active_pos = 0
        if not preserve_crossfade:
            self._crossfade_buffer = None

    def write_frame(self, audio: np.ndarray):
        """
        Apply adaptive logarithmic crossfade, then queue the result.

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

        self.frame_samples = incoming_samples
        self.frame_duration = self.frame_samples / float(self.sr)

        if self._crossfade_buffer is None:
            if crossfade_len < incoming_samples:
                self._crossfade_buffer = audio[-crossfade_len:].copy()
                output = audio[:-crossfade_len]
            else:
                self._crossfade_buffer = audio.copy()
                output = np.zeros((0, 2), dtype=np.float32)
        else:
            prev_tail = self._crossfade_buffer
            prev_len = len(prev_tail)
            blend_len = min(prev_len, crossfade_len, incoming_samples)

            if blend_len > 0:
                new_head = audio[:blend_len]
                fade_out = log_fade_out(blend_len)[:, None]
                fade_in = log_fade_in(blend_len)[:, None]
                prev_tail_trimmed = prev_tail[-blend_len:]
                blended = prev_tail_trimmed * fade_out + new_head * fade_in

                if crossfade_len < incoming_samples:
                    new_body = audio[blend_len:-crossfade_len]
                    new_tail = audio[-crossfade_len:]
                else:
                    new_body = np.zeros((0, 2), dtype=np.float32)
                    new_tail = (
                        audio[blend_len:]
                        if blend_len < incoming_samples
                        else audio
                    )

                if len(new_body) > 0:
                    output = np.concatenate([blended, new_body], axis=0)
                else:
                    output = blended

                if len(new_tail) > 0:
                    self._crossfade_buffer = new_tail.copy()
                else:
                    self._crossfade_buffer = blended[-crossfade_len:].copy()
            else:
                if crossfade_len < incoming_samples:
                    output = audio[:-crossfade_len]
                    self._crossfade_buffer = audio[-crossfade_len:].copy()
                else:
                    output = np.zeros((0, 2), dtype=np.float32)
                    self._crossfade_buffer = audio.copy()

        if len(output) > 0:
            with self._queue_lock:
                self._chunk_queue.append(output * self._gain)

    def read(self, frames: int):
        """
        Pull up to *frames* samples from the queue.

        Returns:
            Tuple[np.ndarray, bool]:
                - output chunk with shape [frames, 2]
                - underflow flag indicating silence padding was needed
        """
        frames = int(max(0, frames))
        out = np.zeros((frames, 2), dtype=np.float32)
        out_pos = 0
        underflow = False

        while out_pos < frames:
            active_remaining = len(self._active_chunk) - self._active_pos
            if active_remaining <= 0:
                with self._queue_lock:
                    if self._chunk_queue:
                        self._active_chunk = self._chunk_queue.popleft()
                        self._active_pos = 0
                        active_remaining = len(self._active_chunk)
                    else:
                        underflow = True
                        break

            to_copy = min(active_remaining, frames - out_pos)
            out[out_pos:out_pos + to_copy] = self._active_chunk[
                self._active_pos:self._active_pos + to_copy
            ]
            self._active_pos += to_copy
            out_pos += to_copy

        return out, underflow

    def pop_chunk(self):
        """Pop the next fully prepared chunk, or None if nothing is queued."""
        with self._queue_lock:
            if self._chunk_queue:
                return self._chunk_queue.popleft()
        return None

    def flush_tail(self):
        """
        Queue and return the held overlap-add tail.

        Runtime playback normally keeps this tail until the next decoded frame
        arrives. Offline evaluation can call this once after the final write to
        reconstruct the full waveform using the exact same crossfade state.
        """
        if self._crossfade_buffer is None or len(self._crossfade_buffer) == 0:
            return None
        tail = self._crossfade_buffer.copy() * self._gain
        self._crossfade_buffer = None
        with self._queue_lock:
            self._chunk_queue.append(tail)
        return tail

    def buffer_duration(self) -> float:
        """Return approximate buffered audio duration in seconds."""
        with self._queue_lock:
            queued_samples = sum(len(c) for c in self._chunk_queue)
        active_remaining = max(0, len(self._active_chunk) - self._active_pos)
        return (queued_samples + active_remaining) / float(self.sr)

    def get_state(self) -> dict:
        return {
            "gain": float(self._gain),
            "frame_samples": int(self.frame_samples),
            "buffer_duration": self.buffer_duration(),
        }


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
        self._buffer = BufferedAudioQueue(sr=self.sr, gain=gain)
        self.underruns = 0

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
        self._buffer.clear_output(preserve_crossfade=False)

    def close(self):
        self._stream.close()

    def set_gain(self, value: float):
        self._buffer.set_gain(value)

    @property
    def frame_samples(self) -> int:
        return int(self._buffer.frame_samples)

    @property
    def frame_duration(self):
        return self._buffer.frame_duration

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback - runs in sounddevice's audio thread."""
        if status.output_underflow:
            self.underruns += 1
        chunk, underflow = self._buffer.read(frames)
        outdata[:] = chunk
        if underflow:
            self.underruns += 1

    def _blocking_writer_loop(self):
        """
        Stream writer loop for environments that cannot allocate CFFI callbacks.
        """
        while self._running.is_set():
            chunk = self._buffer.pop_chunk()
            if chunk is None:
                time.sleep(0.002)
                continue

            try:
                self._stream.write(chunk)
            except Exception:
                self.underruns += 1
                time.sleep(0.01)

    def write_frame(self, audio: np.ndarray):
        self._buffer.write_frame(audio)

    def buffer_duration(self) -> float:
        return self._buffer.buffer_duration()

    def get_state(self) -> dict:
        state = self._buffer.get_state()
        state["underruns"] = int(self.underruns)
        return state
