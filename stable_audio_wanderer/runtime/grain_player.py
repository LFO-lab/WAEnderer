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


def _make_envelope_table(env_type: str, size: int = 8192) -> pyo.PyoTableObject:
    """Create envelope table for grain shaping."""
    if env_type == "hann":
        return pyo.HannTable(size=size)
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
        
        # Looper for playback with start position control
        # mode=1 means one-shot (no loop)
        self._looper = pyo.Looper(
            table=self.table,
            dur=0.1,
            xfade=0,
            mode=1,  # One-shot
            xfadeshape=0,
            startfromloop=False,
            interp=4,
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
        
        # Filter
        self._filter = pyo.Biquad(
            self._output,
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
    ):
        """Trigger grain playback at specified position."""
        # Ensure all values are native Python floats for pyo compatibility
        start_sec = float(start_sec)
        dur_sec = float(dur_sec)
        pitch = float(pitch)
        amp = float(amp)
        pan = float(pan)
        filter_freq = float(filter_freq)
        filter_q = float(filter_q)

        # Don't call stop() - it causes clicks on voice stealing.
        # Instead, just set new parameters and retrigger.
        # The envelope will handle amplitude shaping.

        # Set looper parameters
        self._looper.setStart(max(0.0, start_sec))
        self._looper.setDur(dur_sec)
        self._looper.setPitch(pitch)
        self._looper.setMul(amp * self._mul)

        # Set envelope duration (accounting for pitch)
        env_dur = dur_sec / abs(pitch) if pitch != 0 else dur_sec
        self._env.setDur(env_dur)

        # Set filter
        self._filter.setFreq(filter_freq)
        self._filter.setQ(filter_q)

        # Set pan
        self._panner.setPan(pan)

        # Trigger playback (play() restarts from current position)
        self._looper.play()
        self._trig.play()
        
    def out(self, chnl: int = 0):
        """Send to output."""
        self._panner.out(chnl)
        return self


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
        
        self._lock = threading.Lock()
        
    def _load_tables(self):
        """Load grain buffers into pyo SndTables."""
        unique_files = set(int(fid) for fid in self.file_ids)
        for fid in unique_files:
            if fid < len(self.grain_paths) and self.grain_paths[fid]:
                path = self.grain_paths[fid]
                if os.path.isfile(path):
                    table = pyo.SndTable(path)
                    self._tables[fid] = table
                    self._table_durs[fid] = table.getDur()
                    print(f"[grain] Loaded table {fid}: {os.path.basename(path)} "
                          f"({table.getDur():.2f}s, {table.getSize()} samples)")
                    
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
        if self.server is not None:
            self.server.shutdown()
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

            # Ensure all values are native Python floats for pyo compatibility
            voice.trigger(
                start_sec=start_sec,
                dur_sec=params["dur"],
                pitch=params["pitch"],
                amp=float(amp * self._master_amp),
                pan=float(pan) if pan is not None else params["pan"],
                filter_freq=float(self._filter_freq),
                filter_q=float(self._filter_q),
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
        """Reset timing with phase offset applied."""
        self.time_until_trigger = self.phase_offset * interval


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
        """Trigger a grain for the given stream with randomizations."""
        with self._playhead_lock:
            segment = self._playhead_segment

        # Compute position jitter for this grain (does not modify player state)
        position_spread = None
        if self._position_jitter > 0:
            position_spread = abs((np.random.random() - 0.5) * 2 * self._position_jitter)

        # Compute duration with jitter for this grain (does not modify player state)
        dur = self._grain_dur
        if self._dur_jitter > 0:
            dur_factor = 1.0 + (np.random.random() - 0.5) * 2 * self._dur_jitter
            dur *= dur_factor

        # Apply pan with small jitter around stream's center
        pan = stream.pan_center
        pan_jitter = (np.random.random() - 0.5) * 0.1  # Small pan variation
        pan = float(np.clip(pan + pan_jitter, 0.0, 1.0))

        # Trigger the grain with scheduler-computed overrides
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

        while self._running:
            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time

            self._tick(dt)

            # Sleep for tick interval
            time.sleep(self._tick_interval)

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
            self._thread.join(timeout=1.0)
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
