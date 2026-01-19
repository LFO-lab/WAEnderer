"""
pyo-based granular synthesis player for corpus playback.
Uses Looper for precise position-based triggered grain playback.

Multi-stream granular synthesis for resynthesis-quality playback:
- Multiple concurrent grain streams for high grain density
- Short grains with high overlap for smooth time-stretching
- Small deviations on position, duration, and rate for natural sound
"""
import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

try:
    import pyo
except ImportError:
    raise ImportError("pyo is required for grain playback. Install with: pip install pyo")

from ..config import SR, LATENT_HZ
from ..io.corpus_io import load_grain_manifest
from ..io.audio_io import load_wav

# Minimum trigger rate matches the latent encoder rate
MIN_TRIGGER_RATE = LATENT_HZ  # 21.5 Hz
# Default grain duration matches one latent frame
DEFAULT_GRAIN_DUR = 1.0 / LATENT_HZ  # ~0.0465s


def _make_envelope_table(env_type: str, size: int = 32768) -> pyo.PyoTableObject:
    """Create envelope table for grain shaping."""
    if env_type == "hann":
        return pyo.HannTable(size=size)
    elif env_type == "blackman":
        # Blackman window - better frequency resolution
        return pyo.WinTable(winfunc=pyo.winfunc_blackman, size=size)
    elif env_type == "blackman_harris":
        # Blackman-Harris window - even better sidelobe suppression
        return pyo.WinTable(winfunc=pyo.winfunc_blackman_harris, size=size)
    elif env_type == "kaiser":
        # Kaiser window - adjustable beta for trade-off
        return pyo.WinTable(winfunc=lambda x: np.kaiser(x, beta=5), size=size)
    elif env_type == "gaussian":
        # Gaussian window - smooth, no sidelobes
        return pyo.WinTable(winfunc=lambda x: np.exp(-0.5 * ((x - size/2) / (size/6))**2), size=size)
    elif env_type == "hamming":
        return pyo.HammingTable(size=size)
    elif env_type == "triangle":
        return pyo.LinTable([(0, 0), (size // 2, 1), (size, 0)], size=size)
    elif env_type == "trapezoid":
        a = size // 10
        return pyo.LinTable([(0, 0), (a, 1), (size - a, 1), (size, 0)], size=size)
    elif env_type == "expodec":
        return pyo.ExpTable([(0, 1), (size, 0.001)], exp=5, size=size)
    elif env_type == "rexpodec":
        return pyo.ExpTable([(0, 0.001), (size, 1)], exp=5, size=size)
    else:
        return pyo.HannTable(size=size)


def _find_nearest_zero_crossing(
    samples: np.ndarray,
    target_sample: int,
    search_radius: int = 256,
    prefer: str = "nearest"  # "nearest", "before", "after"
) -> int:
    """Find nearest zero crossing to target sample.

    Args:
        samples: Audio samples as numpy array
        target_sample: Target sample index to search around
        search_radius: Number of samples to search in each direction
        prefer: Which crossing to prefer - "nearest", "before", or "after"

    Returns:
        Sample index of nearest zero crossing, or target_sample if none found
    """
    start = max(0, target_sample - search_radius)
    end = min(len(samples), target_sample + search_radius)

    window = samples[start:end]
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs) != 0)[0] + start

    if len(crossings) == 0:
        return target_sample

    distances = crossings - target_sample
    if prefer == "before":
        before = crossings[distances <= 0]
        return int(before[-1]) if len(before) > 0 else target_sample
    elif prefer == "after":
        after = crossings[distances >= 0]
        return int(after[0]) if len(after) > 0 else target_sample

    return int(crossings[np.argmin(np.abs(distances))])


class LooperVoice:
    """
    A grain voice using Looper for position-accurate playback.
    Looper supports start position and duration control.
    """
    
    def __init__(
        self,
        server: pyo.Server,
        table: pyo.SndTable,
        env_table: pyo.PyoTableObject,
        sr: int,
        mul: float = 1.0,
    ):
        self.server = server
        self.table = table
        self.env_table = env_table
        self.sr = sr
        self._mul = mul
        self._table_dur = table.getDur()
        self._table_size = table.getSize()

        # Cache table samples for zero-crossing detection
        self._table_samples = np.array(table.getTable())

        # Looper for playback with start position control
        # mode=1 means one-shot (no loop)
        self._looper = pyo.Looper(
            table=self.table,
            dur=0.1,
            xfade=0.01, # 10ms crossfade
            mode=1,  # One-shot
            xfadeshape=1, # smooth fade shape
            startfromloop=False,
            interp=1,
            autosmooth=True,
            mul=mul,
        ).stop()
        
        # Envelope applied to looper output
        self._trig = pyo.Trig()
        self._env = pyo.TrigEnv(
            self._trig,
            table=self.env_table,
            dur=0.1,
            mul=1.0,
        )
        
        # Multiply looper by envelope
        self._output = pyo.Sig(self._looper, mul=self._env)

        # DC blocker
        self._dc_blocker = pyo.DCBlock(self._output)
        
        # Filter
        self._filter = pyo.Biquad(
            self._dc_blocker,
            freq=20000,
            q=1.0,
            type=0,
        )
        
        # Pan
        self._panner = pyo.Pan(self._filter, outs=2, pan=0.5)
        
    def trigger(
        self,
        start_sec: float,
        dur_sec: float,
        pitch: float,
        amp: float,
        pan: float,
        filter_freq: float,
        filter_q: float,
        align_zero_crossing: bool = False,
    ):
        """Trigger grain playback at specified position.

        Args:
            start_sec: Start position in seconds
            dur_sec: Grain duration in seconds
            pitch: Playback pitch ratio
            amp: Amplitude
            pan: Pan position (0-1)
            filter_freq: Filter cutoff frequency
            filter_q: Filter resonance
            align_zero_crossing: If True, align start to nearest zero crossing
        """
        # === Parameter validation for sound quality ===

        # Validate duration - must be positive and reasonable
        min_dur_sec = 0.001
        # Prevent clicks at audio file boundaries by ensuring grains do not extend beyond file
        max_start_sec = self._table_dur - min_dur_sec
        max_dur_for_start = self._table_dur - float(np.clip(start_sec, 0.0, max_start_sec))
        max_dur_sec = max(min_dur_sec, max_dur_for_start)
        dur_sec = float(np.clip(dur_sec, min_dur_sec, max_dur_sec))

        # Validate start position - clamp to valid table range
        start_sec = float(np.clip(start_sec, 0.0, max_start_sec))

        # Zero-crossing alignment for click reduction
        if align_zero_crossing and len(self._table_samples) > 0:
            target_sample = int(start_sec * self.sr)
            # Max shift of 2ms to preserve timing accuracy
            max_shift_samples = int(0.002 * self.sr)
            aligned_sample = _find_nearest_zero_crossing(
                self._table_samples, target_sample, search_radius=max_shift_samples
            )
            start_sec = float(aligned_sample / self.sr)
            start_sec = float(np.clip(start_sec, 0.0, max_start_sec))

        # Validate pitch - must be non-zero, clamp to reasonable range
        pitch = float(np.clip(pitch, -4.0, 4.0))

        # Clamp other parameters
        amp = float(np.clip(amp, 0.0, 2.0))
        pan = float(np.clip(pan, 0.0, 1.0))
        filter_freq = float(np.clip(filter_freq, 20, 20000))
        filter_q = float(np.clip(filter_q, 0.5, 10.0))
        nyquist = self.sr / 2.0
        if abs(pitch) > 1.0:
            max_freq = nyquist / abs(pitch)
            filter_freq = float(np.clip(filter_freq, 20, max_freq))

        # Anti-aliasing filter
        nyquist = self.sr / 2.0
        if abs(pitch) > 1.0:
            # Prevent aliasing by limiting to nyquist / pitch
            max_freq = nyquist / abs(pitch)
            filter_freq = min(filter_freq, max_freq)

        # Set looper parameters
        self._looper.setStart(start_sec)
        self._looper.setDur(dur_sec)
        self._looper.setPitch(pitch)
        self._looper.setMul(amp * self._mul)

        # Set envelope duration to match grain duration
        env_dur = dur_sec 
        self._env.setDur(env_dur)

        # Set filter
        self._filter.setFreq(filter_freq)
        self._filter.setQ(filter_q)

        # Set pan
        self._panner.setPan(pan)

        # Check if envelope is still active before retriggering.
        # If envelope > 10% of peak, fade out quickly before restarting.
        env_val = self._env.get(True)
        # get(True) returns the current value of the envelope generator
        should_fade = False
        try:
            env_curr = float(env_val)
            should_fade = env_curr > 0.1
        except Exception:
            should_fade = False
        if should_fade:
            # Fade out quickly
            self._env.setDur(0.01)
            self._env.play()
            time.sleep(0.012)
        
        self._looper.play()
        self._trig.stop()
        self._trig.play()
        
    def out(self, chnl: int = 0):
        """Send to output."""
        self._panner.out(chnl)
        return self


class PhaseCoherenceManager:
    """Track phase across grains for coherent resynthesis.

    Maintains phase continuity per audio region to reduce phasing
    artifacts during time-stretching and pitch-shifting operations.
    """

    def __init__(self, sr: int):
        self.sr = sr
        self._region_phases: Dict[int, float] = {}
        self._region_size_sec = 0.05  # 50ms regions
        self._global_phase = 0.0
        self._last_position = 0.0
        self._last_time = time.perf_counter()

    def get_region_key(self, position_sec: float) -> int:
        """Get region key for a position."""
        return int(position_sec / self._region_size_sec)

    def get_phase_adjustment(self, position_sec: float, pitch: float,
                             grain_dur_sec: float) -> float:
        """
        Return micro-timing adjustment for phase alignment.

        Tracks phase accumulation per position region.
        Returns adjustment in seconds (0 to 10% of grain duration).

        Args:
            position_sec: Current playback position in seconds
            pitch: Playback pitch ratio
            grain_dur_sec: Grain duration in seconds

        Returns:
            Time adjustment in seconds for phase alignment
        """
        region_key = self.get_region_key(position_sec)
        current_time = time.perf_counter()
        dt = current_time - self._last_time

        # Update global phase based on position delta
        if dt < 1.0:  # Ignore large time gaps
            position_delta = position_sec - self._last_position
            self._global_phase += position_delta * self.sr * abs(pitch)
            self._global_phase = self._global_phase % (self.sr * 10)

        self._last_position = position_sec
        self._last_time = current_time

        # Get or initialize region phase
        if region_key not in self._region_phases:
            self._region_phases[region_key] = self._global_phase

        region_phase = self._region_phases[region_key]

        # Update region for next grain
        self._region_phases[region_key] = (
            region_phase + grain_dur_sec * self.sr * abs(pitch)
        )

        # Cleanup old regions (keep last 100)
        if len(self._region_phases) > 100:
            sorted_keys = sorted(self._region_phases.keys())
            for key in sorted_keys[:-100]:
                del self._region_phases[key]

        # Return micro-adjustment (phase offset mapped to time)
        phase_offset_samples = region_phase % self.sr
        phase_offset_sec = phase_offset_samples / self.sr
        return (phase_offset_sec % grain_dur_sec) * 0.1  # 10% influence

    def reset(self):
        """Clear phase state."""
        self._region_phases.clear()
        self._global_phase = 0.0
        self._last_position = 0.0
        self._last_time = time.perf_counter()


class GrainPlayer:
    """
    pyo-based granular synthesis player with full parameter control.
    
    Uses Looper-based voices for accurate position-based playback.
    Each trigger_grain() call plays a grain at the specified corpus segment.
    """
    
    def __init__(
        self,
        manifest_path: str,
        num_voices: int = 64, # used to be 8
        buffersize: int = 512,
        audio: str = "portaudio",
    ):
        self.manifest = load_grain_manifest(manifest_path)
        self.num_voices = num_voices
        self.buffersize = buffersize
        self.sr = self.manifest["sr"]
        
        # Grain metadata
        self.offsets = self.manifest["offsets"]
        self.lengths = self.manifest["lengths"]
        self.file_ids = self.manifest["file_ids"]
        self.segment_ids = self.manifest["segment_ids"]
        self.grain_paths = self.manifest["grain_paths"]
        self.grain_sec = self.manifest["grain_sec"]
        
        # Build segment_id -> grain index mapping
        self._segment_to_grain: Dict[int, int] = {}
        for grain_idx, seg_id in enumerate(self.segment_ids):
            self._segment_to_grain[int(seg_id)] = grain_idx
        
        # pyo server
        self.server: Optional[pyo.Server] = None
        self._tables: Dict[int, pyo.SndTable] = {}
        self._table_durs: Dict[int, float] = {}
        self._voices: List[Tuple[int, LooperVoice]] = []
        self._voice_idx = 0
        
        # Envelope
        self._env_table: Optional[pyo.PyoTableObject] = None
        self._env_type = "hann"
        
        # State
        self._running = False
        self._trigger_rate = LATENT_HZ  # Default to latent rate (21.5 Hz)
        self._trigger_jitter = 0.0
        self._master_amp = 1.0
        
        # Granular parameters
        self._pitch = 1.0
        self._pitch_spread = 0.0
        self._grain_dur = DEFAULT_GRAIN_DUR  # ~0.0465s (1/21.5 Hz)
        self._grain_dur_spread = 0.0
        self._position_spread = 0.0
        self._pan = 0.5
        self._pan_spread = 0.0
        self._filter_freq = 20000
        self._filter_q = 1.0
        self._reverse_prob = 0.0

        # Audio quality improvements
        self._zero_crossing_align = False  # Zero-crossing alignment for click reduction
        self._phase_coherence_enabled = False  # Phase coherence (experimental)
        self._phase_managers: Dict[int, PhaseCoherenceManager] = {}

        self._lock = threading.Lock()

        # Validate corpus is not empty
        if len(self.segment_ids) == 0:
            raise ValueError("Manifest contains no segments - cannot initialize player")
        if len(self.grain_paths) == 0:
            raise ValueError("Manifest contains no grain paths - cannot load audio")

    def _load_tables(self):
        """Load grain buffers into pyo SndTables."""
        unique_files = set(int(fid) for fid in self.file_ids)
        failed_files = []

        for fid in unique_files:
            if fid < len(self.grain_paths) and self.grain_paths[fid]:
                path = self.grain_paths[fid]
                if os.path.isfile(path):
                    try:
                        table = pyo.SndTable(path)
                        self._tables[fid] = table
                        self._table_durs[fid] = table.getDur()
                        # Create phase coherence manager for this table
                        self._phase_managers[fid] = PhaseCoherenceManager(self.sr)
                        print(f"[grain] Loaded table {fid}: {os.path.basename(path)} "
                              f"({table.getDur():.2f}s, {table.getSize()} samples)")
                    except Exception as e:
                        failed_files.append((fid, path, str(e)))
                        print(f"[error] Failed to load table {fid} from {path}: {e}")

        if failed_files:
            print(f"[warn] {len(failed_files)} of {len(unique_files)} tables failed to load")

        if not self._tables:
            raise RuntimeError("No audio tables could be loaded - cannot start playback")
                    
    def _create_envelope(self):
        """Create the grain envelope table."""
        self._env_table = _make_envelope_table(self._env_type)
                    
    def _create_voices(self):
        """Create voice pool for polyphonic playback."""
        self._voices = []
        
        if not self._tables:
            print("[warn] No tables loaded")
            return
            
        voices_per_table = max(1, self.num_voices // len(self._tables))
        
        for fid, table in self._tables.items():
            for _ in range(voices_per_table):
                voice = LooperVoice(
                    self.server,
                    table,
                    self._env_table,
                    self.sr,
                    mul=self._master_amp,
                )
                voice.out()
                self._voices.append((fid, voice))
        
        while len(self._voices) < self.num_voices and self._tables:
            fid = list(self._tables.keys())[0]
            voice = LooperVoice(
                self.server,
                self._tables[fid],
                self._env_table,
                self.sr,
                mul=self._master_amp,
            )
            voice.out()
            self._voices.append((fid, voice))
            
        print(f"[grain] Created {len(self._voices)} voices")
    
    def boot(self):
        """Initialize and boot the pyo server."""
        if self.server is not None:
            return
            
        self.server = pyo.Server(
            sr=self.sr,
            nchnls=2,
            buffersize=self.buffersize,
            duplex=0,
        )
        self.server.boot()
        
        self._create_envelope()
        self._load_tables()
        self._create_voices()
        
    def start(self):
        """Start audio output."""
        if self.server is None:
            self.boot()
        self.server.start()
        self._running = True
        
    def stop(self):
        """Stop audio output."""
        self._running = False
        if self.server is not None:
            self.server.stop()
            
    def shutdown(self):
        """Shutdown the pyo server completely."""
        self.stop()

        # Stop all voices explicitly to prevent clicks
        for fid, voice in self._voices:
            try:
                voice._looper.stop()
                voice._output.stop()
            except Exception:
                pass  # Voice may already be stopped

        if self.server is not None:
            try:
                self.server.stop()
                self.server.shutdown()
            except Exception:
                pass
            self.server = None

        self._tables.clear()
        self._table_durs.clear()
        self._voices.clear()
        
    def _compute_grain_params(self, grain_dur_override: float = None) -> dict:
        """Compute randomized grain parameters.

        Args:
            grain_dur_override: Optional grain duration override (from scheduler)
        """
        pitch = self._pitch
        if self._pitch_spread > 0:
            semitone_offset = (np.random.random() - 0.5) * 2 * self._pitch_spread
            pitch *= 2 ** (semitone_offset / 12.0)

        if self._reverse_prob > 0 and np.random.random() < self._reverse_prob:
            pitch = -abs(pitch)

        # Use override if provided, otherwise use internal setting
        dur = grain_dur_override if grain_dur_override is not None else self._grain_dur
        if self._grain_dur_spread > 0:
            dur_factor = 1.0 + (np.random.random() - 0.5) * 2 * self._grain_dur_spread
            dur *= dur_factor
            dur = max(0.01, min(1.0, dur))

        pan = self._pan
        if self._pan_spread > 0:
            pan_offset = (np.random.random() - 0.5) * 2 * self._pan_spread
            pan = float(np.clip(pan + pan_offset, 0.0, 1.0))

        # Ensure all values are Python native floats for pyo compatibility
        return {"pitch": float(pitch), "dur": float(dur), "pan": float(pan)}
        
    def trigger_grain(
        self,
        segment_idx: int,
        amp: float = 1.0,
        pan: float = None,
        grain_dur: float = None,
        position_spread: float = None,
    ) -> bool:
        """
        Trigger grain for the given segment index.

        Args:
            segment_idx: Corpus segment index
            amp: Amplitude (0-1)
            pan: Pan position, None uses internal setting
            grain_dur: Grain duration override, None uses internal setting
            position_spread: Position spread override, None uses internal setting

        Returns:
            True if grain was triggered
        """
        if segment_idx not in self._segment_to_grain:
            return False

        grain_idx = self._segment_to_grain[segment_idx]
        file_id = int(self.file_ids[grain_idx])
        offset_samples = int(self.offsets[grain_idx])
        length_samples = int(self.lengths[grain_idx])

        if file_id not in self._tables:
            return False

        with self._lock:
            # Convert samples to seconds
            start_sec = float(offset_samples / self.sr)
            grain_dur_sec = float(length_samples / self.sr)

            # Use override or internal setting for position spread
            effective_pos_spread = position_spread if position_spread is not None else self._position_spread

            # Apply position spread
            if effective_pos_spread > 0:
                max_jitter_sec = grain_dur_sec * effective_pos_spread * 0.5
                jitter = (np.random.random() - 0.5) * 2 * max_jitter_sec
                start_sec = float(max(0, start_sec + jitter))

            # Find voice for this file
            matching = [(i, fid, v) for i, (fid, v) in enumerate(self._voices) if fid == file_id]

            if not matching:
                if not self._voices:
                    return False
                idx = self._voice_idx % len(self._voices)
                _, voice = self._voices[idx]
            else:
                voice_idx = self._voice_idx % len(matching)
                _, _, voice = matching[voice_idx]

            params = self._compute_grain_params(grain_dur_override=grain_dur)

            # Apply phase coherence adjustment if enabled
            if self._phase_coherence_enabled and file_id in self._phase_managers:
                phase_manager = self._phase_managers[file_id]
                phase_adj = phase_manager.get_phase_adjustment(
                    start_sec, params["pitch"], params["dur"]
                )
                start_sec = float(start_sec + phase_adj)

            # Ensure all values are native Python floats for pyo compatibility
            voice.trigger(
                start_sec=start_sec,
                dur_sec=params["dur"],
                pitch=params["pitch"],
                amp=float(amp * self._master_amp),
                pan=float(pan) if pan is not None else params["pan"],
                filter_freq=float(self._filter_freq),
                filter_q=float(self._filter_q),
                align_zero_crossing=self._zero_crossing_align,
            )

            self._voice_idx += 1

        return True
    
    # --- Setters ---
    
    def set_trigger_rate(self, rate: float):
        """Set trigger rate with minimum of LATENT_HZ (21.5 Hz)."""
        self._trigger_rate = max(MIN_TRIGGER_RATE, float(rate))
        
    def set_trigger_jitter(self, jitter: float):
        self._trigger_jitter = float(np.clip(jitter, 0.0, 1.0))
        
    def set_master_amp(self, amp: float):
        self._master_amp = float(np.clip(amp, 0.0, 2.0))
        
    def set_pitch(self, pitch: float):
        self._pitch = float(np.clip(pitch, 0.25, 4.0))
        
    def set_pitch_spread(self, spread: float):
        self._pitch_spread = float(np.clip(spread, 0.0, 12.0))
        
    def set_grain_dur(self, dur: float):
        """Set grain duration (min ~0.01s, default ~0.0465s for latent-aligned)."""
        self._grain_dur = float(np.clip(dur, 0.01, 1.0))
        
    def set_grain_dur_spread(self, spread: float):
        self._grain_dur_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_position_spread(self, spread: float):
        self._position_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_pan(self, pan: float):
        self._pan = float(np.clip(pan, 0.0, 1.0))
        
    def set_pan_spread(self, spread: float):
        self._pan_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_filter_freq(self, freq: float):
        self._filter_freq = float(np.clip(freq, 20, 20000))
        
    def set_filter_q(self, q: float):
        self._filter_q = float(np.clip(q, 0.5, 10.0))
        
    def set_envelope(self, env_type: str):
        valid = {"hann", "hamming", "triangle", "trapezoid", "expodec", "rexpodec"}
        if env_type.lower() in valid:
            self._env_type = env_type.lower()
            
    def set_reverse_prob(self, prob: float):
        self._reverse_prob = float(np.clip(prob, 0.0, 1.0))

    def set_zero_crossing_align(self, enabled: bool):
        """Enable/disable zero-crossing grain alignment for click reduction."""
        self._zero_crossing_align = bool(enabled)

    def set_phase_coherence(self, enabled: bool):
        """Enable/disable phase coherence tracking (experimental)."""
        self._phase_coherence_enabled = bool(enabled)

    def reset_phase(self):
        """Reset all phase coherence managers."""
        for manager in self._phase_managers.values():
            manager.reset()

    def get_grain_params_snapshot(self) -> dict:
        """Get a thread-safe snapshot of grain parameters for scheduler use.

        Returns:
            dict with position_spread, grain_dur, grain_dur_spread, pan, pan_spread
        """
        with self._lock:
            return {
                "position_spread": self._position_spread,
                "grain_dur": self._grain_dur,
                "grain_dur_spread": self._grain_dur_spread,
                "pan": self._pan,
                "pan_spread": self._pan_spread,
            }

    def get_trigger_interval(self) -> float:
        base = 1.0 / self._trigger_rate
        if self._trigger_jitter > 0:
            jitter = (np.random.random() - 0.5) * 2 * self._trigger_jitter * base
            return max(0.01, base + jitter)
        return base
    
    @property
    def trigger_rate(self) -> float:
        return self._trigger_rate
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_state(self) -> dict:
        return {
            "trigger_rate": self._trigger_rate,
            "trigger_jitter": self._trigger_jitter,
            "master_amp": self._master_amp,
            "pitch": self._pitch,
            "pitch_spread": self._pitch_spread,
            "grain_dur": self._grain_dur,
            "grain_dur_spread": self._grain_dur_spread,
            "position_spread": self._position_spread,
            "pan": self._pan,
            "pan_spread": self._pan_spread,
            "filter_freq": self._filter_freq,
            "filter_q": self._filter_q,
            "envelope": self._env_type,
            "reverse_prob": self._reverse_prob,
            "zero_crossing_align": self._zero_crossing_align,
            "phase_coherence": self._phase_coherence_enabled,
        }


def preload_grain_buffers(manifest_path: str) -> Dict[int, np.ndarray]:
    """Preload grain audio buffers into memory."""
    manifest = load_grain_manifest(manifest_path)
    grain_paths = manifest["grain_paths"]
    buffers: Dict[int, np.ndarray] = {}

    for fid, path in enumerate(grain_paths):
        if path and os.path.isfile(path):
            buffers[fid] = load_wav(path)

    return buffers


# ============================================================================
# Multi-Stream Granular Synthesis Scheduler
# ============================================================================

# Resynthesis-optimized defaults
DEFAULT_NUM_STREAMS = 4         # Number of concurrent grain streams
DEFAULT_STREAM_GRAIN_DUR = 0.04  # 40ms grains for smooth resynthesis
DEFAULT_OVERLAP = 0.75          # 75% overlap = 10ms hop per stream
DEFAULT_POSITION_JITTER = 0.1   # Small position randomization (fraction of grain dur)
DEFAULT_DUR_JITTER = 0.05       # Small duration randomization (fraction)
DEFAULT_RATE_JITTER = 0.02      # Small rate randomization (fraction)


class GrainStream:
    """
    A single grain stream with independent timing and phase.

    Each stream fires grains at a regular interval, offset in time
    from other streams to create overlapping coverage. 
    A small random phase offset is added to avoid phase alignment artifacts.
    """

    def __init__(
        self,
        stream_id: int,
        phase_offset: float,    # 0-1, fraction of stream interval
        pan_center: float,      # Base pan position for this stream
    ):
        self.stream_id = stream_id
        self.phase_offset = phase_offset
        self.pan_center = pan_center
        self.time_until_trigger = 0.0

    def reset(self, interval: float):
        """Reset timing with phase offset applied and a small random phase jitter."""
        # Add small random phase offset to avoid phase alignment artifacts
        phase_jitter = np.random.uniform(0, 0.1)  # 0-0.1 fraction of interval
        self.time_until_trigger = (self.phase_offset + phase_jitter) * interval


class GrainScheduler:
    """
    Multi-stream granular synthesis scheduler for resynthesis-quality playback.

    Manages multiple concurrent grain streams that share a playhead position
    but trigger grains at staggered times with small randomizations.

    For resynthesis-quality playback:
    - Multiple streams ensure continuous audio coverage
    - High overlap prevents gaps during time-stretching
    - Small deviations on position/duration avoid repetition artifacts
    - When navigation moves slowly, grains overlap more = time-stretch
    - When navigation moves fast, grains advance quickly = fast-forward

    Usage:
        scheduler = GrainScheduler(grain_player, num_streams=4)
        scheduler.start()

        # In your main loop:
        while running:
            segment_idx = nav.pick_next_index()
            scheduler.set_playhead(segment_idx)
            time.sleep(nav.get_trigger_interval())
    """

    def __init__(
        self,
        grain_player: GrainPlayer,
        num_streams: int = DEFAULT_NUM_STREAMS,
        grain_dur: float = DEFAULT_STREAM_GRAIN_DUR,
        overlap: float = DEFAULT_OVERLAP,
        position_jitter: float = DEFAULT_POSITION_JITTER,
        dur_jitter: float = DEFAULT_DUR_JITTER,
        rate_jitter: float = DEFAULT_RATE_JITTER,
        stereo_spread: float = 0.3,  # How much streams spread in stereo field
        nav_speed: float = 1.0,  # Navigation speed multiplier
    ):
        """
        Args:
            grain_player: The GrainPlayer instance for triggering grains
            num_streams: Number of concurrent grain streams (more = smoother)
            grain_dur: Base grain duration in seconds
            overlap: Overlap ratio (0.5 = 50% overlap, 0.75 = 75%)
            position_jitter: Random position deviation (fraction of grain dur)
            dur_jitter: Random duration deviation (fraction of dur)
            rate_jitter: Random rate deviation (fraction of interval)
            stereo_spread: How much to spread streams across stereo field
            nav_speed: Navigation speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double)
        """
        self.player = grain_player
        self.num_streams = num_streams
        self._grain_dur = grain_dur
        self._overlap = overlap
        self._position_jitter = position_jitter
        self._dur_jitter = dur_jitter
        self._rate_jitter = rate_jitter
        self._stereo_spread = stereo_spread
        self._nav_speed = nav_speed

        # Computed timing
        self._update_timing()

        # Create streams with staggered phases and spread panning
        self._streams: List[GrainStream] = []
        self._create_streams()

        # Current playhead (segment index from navigation)
        self._playhead_segment = 0
        self._playhead_lock = threading.Lock()

        # Scheduler thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_interval = 0.002  # 2ms tick for responsive grain triggering

        # Amplitude per stream (divide by num_streams to avoid clipping)
        self._stream_amp = float(1.0 / np.sqrt(num_streams))

    def _update_timing(self):
        """Recompute timing parameters from grain_dur and overlap."""
        # Hop size = grain_dur * (1 - overlap)
        # e.g., 40ms grain with 75% overlap = 10ms hop
        self._hop_size = self._grain_dur * (1 - self._overlap)
        # Each stream fires at: 1 / hop_size Hz
        self._stream_interval = self._hop_size
        # Total grain rate across all streams
        self._total_rate = self.num_streams / self._hop_size

    def _create_streams(self):
        """Create grain streams with staggered timing and panning."""
        self._streams = []
        for i in range(self.num_streams):
            # Stagger phases evenly across the interval
            phase = i / self.num_streams
            # Spread panning across stereo field, centered on 0.5
            if self.num_streams > 1:
                pan = 0.5 + (i / (self.num_streams - 1) - 0.5) * self._stereo_spread
            else:
                pan = 0.5
            stream = GrainStream(i, phase, pan)
            stream.reset(self._stream_interval)
            self._streams.append(stream)

    def set_playhead(self, segment_idx: int):
        """Update the current playhead position (called from navigation)."""
        with self._playhead_lock:
            self._playhead_segment = segment_idx

    def get_playhead(self) -> int:
        """Get current playhead segment."""
        with self._playhead_lock:
            return self._playhead_segment

    def _trigger_grain_for_stream(self, stream: GrainStream):
        """Trigger a grain for the given stream with randomizations.

        Uses GrainPlayer's settings as base values for grain parameters,
        applying scheduler's jitter on top.
        """
        with self._playhead_lock:
            segment = self._playhead_segment

        # Get thread-safe parameter snapshot from player
        params = self.player.get_grain_params_snapshot()

        # Use player's position_spread setting as base
        position_spread = None
        player_pos_spread = params["position_spread"]
        if player_pos_spread > 0 or self._position_jitter > 0:
            # Combine player's spread with scheduler's jitter
            total_spread = max(player_pos_spread, self._position_jitter)
            position_spread = abs((np.random.random() - 0.5) * 2 * total_spread)

        # Use player's grain_dur setting as base duration
        dur = params["grain_dur"]
        # Apply spread from player's grain_dur_spread setting
        player_dur_spread = params["grain_dur_spread"]
        if player_dur_spread > 0 or self._dur_jitter > 0:
            # Combine player's spread with scheduler's jitter
            total_spread = max(player_dur_spread, self._dur_jitter)
            dur_factor = 1.0 + (np.random.random() - 0.5) * 2 * total_spread
            dur *= dur_factor
            dur = max(0.01, min(1.0, dur))

        # Use player's pan setting as base, with player's pan_spread
        pan = params["pan"]
        player_pan_spread = params["pan_spread"]
        if player_pan_spread > 0:
            pan_jitter = (np.random.random() - 0.5) * 2 * player_pan_spread
            pan = float(np.clip(pan + pan_jitter, 0.0, 1.0))

        # Trigger the grain with player's base settings
        self.player.trigger_grain(
            segment,
            amp=self._stream_amp,
            pan=pan,
            grain_dur=dur,
            position_spread=position_spread,
        )

    def _tick(self, dt: float):
        """
        Called every tick. Decrement stream timers and trigger grains when due.

        Args:
            dt: Time since last tick in seconds
        """
        for stream in self._streams:
            stream.time_until_trigger -= dt

            if stream.time_until_trigger <= 0:
                self._trigger_grain_for_stream(stream)

                # Reset timer with optional jitter
                interval = self._stream_interval
                if self._rate_jitter > 0:
                    rate_factor = 1.0 + (np.random.random() - 0.5) * 2 * self._rate_jitter
                    interval *= rate_factor
                stream.time_until_trigger += interval

                # Prevent drift accumulation
                if stream.time_until_trigger < 0:
                    stream.time_until_trigger = interval * 0.5

    def _scheduler_loop(self):
        """Main scheduler loop running in its own thread."""
        last_time = time.perf_counter()

        try:
            while self._running:
                current_time = time.perf_counter()
                dt = current_time - last_time
                last_time = current_time

                try:
                    self._tick(dt)
                except Exception as e:
                    # Log tick errors but continue - one bad tick shouldn't stop playback
                    print(f"[error] Scheduler tick error: {e}")
                    # Reset timing to prevent drift accumulation from error recovery
                    last_time = time.perf_counter()

                # Sleep for tick interval
                time.sleep(self._tick_interval)

        except Exception as e:
            print(f"[error] Scheduler loop fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[scheduler] Loop stopped")
            self._running = False

    def start(self):
        """Start the grain scheduler thread."""
        if self._running:
            return

        self._running = True

        # Reset all stream timers
        for stream in self._streams:
            stream.reset(self._stream_interval)

        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        print(f"[scheduler] Started {self.num_streams} streams @ {self._total_rate:.1f} grains/sec total")
        print(f"[scheduler] grain_dur={self._grain_dur*1000:.1f}ms, overlap={self._overlap*100:.0f}%, hop={self._hop_size*1000:.1f}ms")

    def stop(self):
        """Stop the grain scheduler thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)  # Increased timeout for clean shutdown
            if self._thread.is_alive():
                print("[warn] Scheduler thread did not stop within timeout")
            self._thread = None

    # --- Setters for runtime control ---

    def set_num_streams(self, n: int):
        """Set number of streams (requires restart to take effect)."""
        n = max(1, min(16, int(n)))
        if n != self.num_streams:
            self.num_streams = n
            self._stream_amp = float(1.0 / np.sqrt(n))
            self._create_streams()

    def set_grain_dur(self, dur: float):
        """Set grain duration."""
        self._grain_dur = float(np.clip(dur, 0.01, 0.2))
        self._update_timing()
        for stream in self._streams:
            stream.reset(self._stream_interval)

    def set_overlap(self, overlap: float):
        """Set overlap ratio (0-0.95)."""
        self._overlap = float(np.clip(overlap, 0.0, 0.95))
        self._update_timing()
        for stream in self._streams:
            stream.reset(self._stream_interval)

    def set_position_jitter(self, jitter: float):
        """Set position jitter (0-1)."""
        self._position_jitter = float(np.clip(jitter, 0.0, 1.0))

    def set_dur_jitter(self, jitter: float):
        """Set duration jitter (0-1)."""
        self._dur_jitter = float(np.clip(jitter, 0.0, 1.0))

    def set_rate_jitter(self, jitter: float):
        """Set rate jitter (0-1)."""
        self._rate_jitter = float(np.clip(jitter, 0.0, 1.0))

    def set_stereo_spread(self, spread: float):
        """Set stereo spread (0-1)."""
        self._stereo_spread = float(np.clip(spread, 0.0, 1.0))
        self._create_streams()

    def set_nav_speed(self, speed: float):
        """Set navigation speed multiplier (0.1-10.0). 1.0 = normal, <1 = slower, >1 = faster."""
        self._nav_speed = float(np.clip(speed, 0.1, 10.0))

    def get_nav_interval(self, base_interval: float) -> float:
        """
        Get navigation interval adjusted by nav_speed.

        Args:
            base_interval: Base interval from NavigationEngine

        Returns:
            Adjusted interval (shorter = faster navigation)
        """
        # Higher nav_speed = shorter interval = faster movement through corpus
        return base_interval / self._nav_speed

    @property
    def nav_speed(self) -> float:
        """Current navigation speed multiplier."""
        return self._nav_speed

    def get_state(self) -> dict:
        """Get current scheduler state."""
        return {
            "num_streams": self.num_streams,
            "grain_dur": self._grain_dur,
            "overlap": self._overlap,
            "position_jitter": self._position_jitter,
            "dur_jitter": self._dur_jitter,
            "rate_jitter": self._rate_jitter,
            "stereo_spread": self._stereo_spread,
            "nav_speed": self._nav_speed,
            "hop_size": self._hop_size,
            "total_rate": self._total_rate,
            "playhead": self._playhead_segment,
            "running": self._running,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def total_grain_rate(self) -> float:
        """Total grains per second across all streams."""
        return self._total_rate
