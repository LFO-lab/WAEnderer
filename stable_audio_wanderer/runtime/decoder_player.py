"""
Realtime decoder audio output using sounddevice.
"""
import threading
import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("sounddevice is required for realtime decoding. Install with: pip install sounddevice") from exc

from ..config import SR


class DecoderPlayer:
    """Streaming audio output with optional crossfade smoothing."""

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

        self._buffer = np.zeros((0, 2), dtype=np.float32)
        self._buffer_lock = threading.Lock()
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
        if status.output_underflow:
            self.underruns += 1
        with self._buffer_lock:
            if self._buffer.shape[0] >= frames:
                out = self._buffer[:frames]
                self._buffer = self._buffer[frames:]
            else:
                out = np.zeros((frames, 2), dtype=np.float32)
                if self._buffer.shape[0] > 0:
                    out[: self._buffer.shape[0]] = self._buffer
                    self._buffer = np.zeros((0, 2), dtype=np.float32)
                self.underruns += 1
        outdata[:] = out

    def write_frame(self, audio: np.ndarray):
        """Queue a decoded audio frame for playback."""
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

        audio = audio * self._gain

        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, audio], axis=0)

    def get_state(self) -> dict:
        return {
            "gain": self._gain,
            "smoothing": self._smoothing,
            "frame_samples": int(self.frame_samples),
            "underruns": int(self.underruns),
        }
