"""
pyo-based granular synthesis player for corpus playback.
Uses Particle2 for advanced per-grain control with pitch, duration, filtering, and more.
"""
import os
import time
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pyo
except ImportError:
    raise ImportError("pyo is required for grain playback. Install with: pip install pyo")

from ..config import SR
from ..io.corpus_io import load_grain_manifest
from ..io.audio_io import load_wav


# Envelope table generators
def _make_envelope_table(env_type: str, size: int = 8192) -> pyo.TableRead:
    """Create envelope table for grain shaping."""
    if env_type == "hann":
        return pyo.HannTable(size=size)
    elif env_type == "hamming":
        return pyo.HammingTable(size=size)
    elif env_type == "triangle":
        return pyo.LinTable([(0, 0), (size // 2, 1), (size, 0)], size=size)
    elif env_type == "trapezoid":
        # Attack 10%, sustain 80%, release 10%
        a = size // 10
        return pyo.LinTable([(0, 0), (a, 1), (size - a, 1), (size, 0)], size=size)
    elif env_type == "expodec":
        # Exponential decay
        return pyo.ExpTable([(0, 1), (size, 0.001)], exp=5, size=size)
    elif env_type == "rexpodec":
        # Reverse exponential (attack)
        return pyo.ExpTable([(0, 0.001), (size, 1)], exp=5, size=size)
    else:
        # Default to Hann
        return pyo.HannTable(size=size)


class GranularVoice:
    """
    A granular synthesis voice using pyo's Granulator.
    Provides per-grain control over pitch, duration, position, and filtering.
    """
    
    def __init__(
        self,
        server: pyo.Server,
        table: pyo.SndTable,
        env_table: pyo.PyoTableObject,
        mul: float = 1.0,
    ):
        self.server = server
        self.table = table
        self.env_table = env_table
        self._mul = mul
        self._active = False
        
        # Position signal (normalized 0-1 in table)
        self._pos_sig = pyo.Sig(0.5)
        
        # Pitch signal
        self._pitch_sig = pyo.Sig(1.0)
        
        # Duration signal
        self._dur_sig = pyo.Sig(0.1)
        
        # Create Granulator for this voice
        # Note: pyo.Granulator uses pitch as a list or signal for detuning
        self.granulator = pyo.Granulator(
            table=self.table,
            env=self.env_table,
            pitch=self._pitch_sig,
            pos=self._pos_sig,
            dur=self._dur_sig,
            grains=8,  # Number of overlapping grains
            basedur=0.1,
            mul=mul,
        )
        
        # Pan for stereo positioning
        self._pan_sig = pyo.Sig(0.5)
        self.panner = pyo.Pan(self.granulator, outs=2, pan=self._pan_sig)
        
        # Optional lowpass filter
        self._filter_freq_sig = pyo.Sig(20000)
        self._filter_q = 1.0
        self.filter = pyo.Biquad(
            self.panner,
            freq=self._filter_freq_sig,
            q=self._filter_q,
            type=0,  # Lowpass
        )
        
    def set_position(self, pos: float):
        """Set grain read position (0-1 normalized to table size)."""
        self._pos_sig.setValue(float(np.clip(pos, 0.0, 1.0)))
        
    def set_pitch(self, pitch: float):
        """Set playback pitch ratio (1.0 = original)."""
        self._pitch_sig.setValue(float(np.clip(pitch, 0.1, 4.0)))
        
    def set_duration(self, dur: float):
        """Set grain duration in seconds."""
        self._dur_sig.setValue(float(np.clip(dur, 0.005, 1.0)))
        
    def set_pan(self, pan: float):
        """Set stereo pan position (0=left, 0.5=center, 1=right)."""
        self._pan_sig.setValue(float(np.clip(pan, 0.0, 1.0)))
        
    def set_filter_freq(self, freq: float):
        """Set lowpass filter cutoff frequency."""
        self._filter_freq_sig.setValue(float(np.clip(freq, 20, 20000)))
        
    def set_filter_q(self, q: float):
        """Set filter resonance."""
        self._filter_q = float(np.clip(q, 0.5, 20.0))
        self.filter.setQ(self._filter_q)
        
    def set_mul(self, mul: float):
        """Set amplitude multiplier."""
        self._mul = float(np.clip(mul, 0.0, 2.0))
        self.granulator.setMul(self._mul)
        
    def play(self):
        """Start granular playback."""
        self.granulator.play()
        self._active = True
        
    def stop(self):
        """Stop granular playback."""
        self.granulator.stop()
        self._active = False
        
    def out(self, chnl: int = 0):
        """Send to output (through filter)."""
        self.filter.out(chnl)
        return self
        
    def is_active(self) -> bool:
        return self._active


class GrainPlayer:
    """
    pyo-based granular synthesis player with full parameter control.
    
    Uses Granulator objects for continuous granular playback with:
    - Pitch control and randomization
    - Grain duration control
    - Position spread/jitter
    - Stereo panning and spread
    - Per-grain filtering
    - Multiple envelope shapes
    """
    
    def __init__(
        self,
        manifest_path: str,
        num_voices: int = 4,
        buffersize: int = 512,
        audio: str = "portaudio",
    ):
        """
        Initialize grain player.
        
        Args:
            manifest_path: Path to grain manifest NPZ file
            num_voices: Number of overlapping granular voices
            buffersize: Audio buffer size
            audio: Audio backend (portaudio, jack, coreaudio, etc.)
        """
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
        
        # pyo server (not started yet)
        self.server: Optional[pyo.Server] = None
        self._tables: Dict[int, pyo.SndTable] = {}  # file_id -> SndTable
        self._table_sizes: Dict[int, int] = {}  # file_id -> table size in samples
        self._voices: List[Tuple[int, GranularVoice]] = []  # (file_id, voice) pairs
        self._voice_idx = 0  # Round-robin voice selection
        
        # Envelope table (shared across voices)
        self._env_table: Optional[pyo.PyoTableObject] = None
        self._env_type = "hann"
        
        # Playback state - basic
        self._running = False
        self._trigger_rate = 10.0  # grains per second
        self._trigger_jitter = 0.0  # random variation in timing (0-1)
        self._master_amp = 1.0
        
        # Granular synthesis parameters
        self._pitch = 1.0              # Playback rate (1.0 = original pitch)
        self._pitch_spread = 0.0       # Random pitch variation in semitones (0-12)
        self._grain_dur = 0.1          # Grain duration in seconds
        self._grain_dur_spread = 0.0   # Duration randomization (0-1)
        self._position_spread = 0.0    # Position jitter in table (0-1)
        self._pan = 0.5                # Base pan position (0-1)
        self._pan_spread = 0.0         # Stereo spread (0-1)
        self._filter_freq = 20000      # Lowpass cutoff (Hz)
        self._filter_q = 1.0           # Filter resonance
        self._reverse_prob = 0.0       # Probability of reversed grains (0-1)
        
        # Current grain position (normalized 0-1)
        self._current_position = 0.5
        self._current_file_id = 0
        
        self._lock = threading.Lock()
        
    def _load_tables(self):
        """Load grain buffers into pyo SndTables."""
        unique_files = set(int(fid) for fid in self.file_ids)
        for fid in unique_files:
            if fid < len(self.grain_paths) and self.grain_paths[fid]:
                path = self.grain_paths[fid]
                if os.path.isfile(path):
                    # Load table
                    table = pyo.SndTable(path)
                    self._tables[fid] = table
                    self._table_sizes[fid] = table.getSize()
                    
    def _create_envelope(self):
        """Create the grain envelope table."""
        self._env_table = _make_envelope_table(self._env_type)
                    
    def _create_voices(self):
        """Create voice pool for polyphonic granular playback."""
        self._voices = []
        
        if not self._tables:
            return
            
        # Distribute voices across tables
        voices_per_table = max(1, self.num_voices // len(self._tables))
        
        for fid, table in self._tables.items():
            for _ in range(voices_per_table):
                voice = GranularVoice(
                    self.server,
                    table,
                    self._env_table,
                    mul=self._master_amp,
                )
                voice.out()
                self._voices.append((fid, voice))
        
        # Fill remaining slots with first table
        while len(self._voices) < self.num_voices and self._tables:
            fid = list(self._tables.keys())[0]
            voice = GranularVoice(
                self.server,
                self._tables[fid],
                self._env_table,
                mul=self._master_amp,
            )
            voice.out()
            self._voices.append((fid, voice))
            
        # Start all voices
        for _, voice in self._voices:
            voice.play()
    
    def boot(self):
        """Initialize and boot the pyo server."""
        if self.server is not None:
            return
            
        self.server = pyo.Server(
            sr=self.sr,
            nchnls=2,
            buffersize=self.buffersize,
            duplex=0,  # Output only
        )
        self.server.boot()
        
        # Create envelope table
        self._create_envelope()
        
        # Load grain tables
        self._load_tables()
        
        # Create voice pool
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
        for _, voice in self._voices:
            voice.stop()
        if self.server is not None:
            self.server.stop()
            
    def shutdown(self):
        """Shutdown the pyo server completely."""
        self.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server = None
        self._tables.clear()
        self._table_sizes.clear()
        self._voices.clear()
        
    def _compute_grain_params(self) -> dict:
        """Compute randomized grain parameters based on spread settings."""
        # Pitch with spread (in semitones)
        pitch = self._pitch
        if self._pitch_spread > 0:
            semitone_offset = (np.random.random() - 0.5) * 2 * self._pitch_spread
            pitch *= 2 ** (semitone_offset / 12.0)
        
        # Reverse probability
        if self._reverse_prob > 0 and np.random.random() < self._reverse_prob:
            pitch = -abs(pitch)
            
        # Duration with spread
        dur = self._grain_dur
        if self._grain_dur_spread > 0:
            dur_factor = 1.0 + (np.random.random() - 0.5) * 2 * self._grain_dur_spread
            dur *= dur_factor
            dur = max(0.005, min(1.0, dur))
            
        # Pan with spread
        pan = self._pan
        if self._pan_spread > 0:
            pan_offset = (np.random.random() - 0.5) * 2 * self._pan_spread
            pan = np.clip(pan + pan_offset, 0.0, 1.0)
            
        return {
            "pitch": pitch,
            "dur": dur,
            "pan": pan,
        }
        
    def trigger_grain(self, segment_idx: int, amp: float = 1.0, pan: float = None) -> bool:
        """
        Trigger grain for the given segment index.
        
        Args:
            segment_idx: Corpus segment index
            amp: Amplitude (0-1)
            pan: Stereo pan position (0=left, 0.5=center, 1=right), None uses internal setting
            
        Returns:
            True if grain was triggered, False if segment not found
        """
        if segment_idx not in self._segment_to_grain:
            return False
            
        grain_idx = self._segment_to_grain[segment_idx]
        file_id = int(self.file_ids[grain_idx])
        offset = int(self.offsets[grain_idx])
        length = int(self.lengths[grain_idx])
        
        if file_id not in self._tables:
            return False
            
        table_size = self._table_sizes.get(file_id, 1)
        
        with self._lock:
            # Find a voice for this file (round-robin among matching voices)
            matching_voices = [(i, fid, v) for i, (fid, v) in enumerate(self._voices) if fid == file_id]
            
            if not matching_voices:
                # Use any available voice if no matching ones
                if not self._voices:
                    return False
                idx = self._voice_idx % len(self._voices)
                _, voice = self._voices[idx]
            else:
                # Use round-robin among matching voices
                voice_pool_idx = self._voice_idx % len(matching_voices)
                _, _, voice = matching_voices[voice_pool_idx]
            
            # Compute position (normalized 0-1)
            base_pos = float(offset) / float(table_size) if table_size > 0 else 0.0
            
            # Add position spread
            if self._position_spread > 0:
                pos_jitter = (np.random.random() - 0.5) * 2 * self._position_spread * 0.1  # 10% max jitter
                base_pos = np.clip(base_pos + pos_jitter, 0.0, 1.0)
            
            # Get randomized parameters
            params = self._compute_grain_params()
            
            # Update voice parameters
            voice.set_position(base_pos)
            voice.set_pitch(params["pitch"])
            voice.set_duration(params["dur"])
            voice.set_pan(pan if pan is not None else params["pan"])
            voice.set_filter_freq(self._filter_freq)
            voice.set_filter_q(self._filter_q)
            voice.set_mul(amp * self._master_amp)
            
            # Store current state
            self._current_position = base_pos
            self._current_file_id = file_id
            
            self._voice_idx += 1
            
        return True
    
    # --- Basic controls ---
    
    def set_trigger_rate(self, rate: float):
        """Set grain trigger rate in grains per second."""
        self._trigger_rate = max(0.1, float(rate))
        
    def set_trigger_jitter(self, jitter: float):
        """Set random timing jitter (0-1)."""
        self._trigger_jitter = float(np.clip(jitter, 0.0, 1.0))
        
    def set_master_amp(self, amp: float):
        """Set master amplitude."""
        self._master_amp = float(np.clip(amp, 0.0, 2.0))
        for _, voice in self._voices:
            voice.set_mul(self._master_amp)
            
    # --- Granular synthesis controls ---
    
    def set_pitch(self, pitch: float):
        """Set base playback pitch ratio (0.25 to 4.0, 1.0 = original)."""
        self._pitch = float(np.clip(pitch, 0.25, 4.0))
        
    def set_pitch_spread(self, spread: float):
        """Set pitch randomization in semitones (0-12)."""
        self._pitch_spread = float(np.clip(spread, 0.0, 12.0))
        
    def set_grain_dur(self, dur: float):
        """Set grain duration in seconds (0.01 to 0.5)."""
        self._grain_dur = float(np.clip(dur, 0.01, 0.5))
        for _, voice in self._voices:
            voice.set_duration(self._grain_dur)
            
    def set_grain_dur_spread(self, spread: float):
        """Set grain duration randomization (0-1)."""
        self._grain_dur_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_position_spread(self, spread: float):
        """Set position jitter (0-1)."""
        self._position_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_pan(self, pan: float):
        """Set base pan position (0-1)."""
        self._pan = float(np.clip(pan, 0.0, 1.0))
        
    def set_pan_spread(self, spread: float):
        """Set stereo spread (0-1)."""
        self._pan_spread = float(np.clip(spread, 0.0, 1.0))
        
    def set_filter_freq(self, freq: float):
        """Set lowpass filter cutoff frequency (20-20000 Hz)."""
        self._filter_freq = float(np.clip(freq, 20, 20000))
        for _, voice in self._voices:
            voice.set_filter_freq(self._filter_freq)
            
    def set_filter_q(self, q: float):
        """Set filter resonance (0.5-10)."""
        self._filter_q = float(np.clip(q, 0.5, 10.0))
        for _, voice in self._voices:
            voice.set_filter_q(self._filter_q)
            
    def set_envelope(self, env_type: str):
        """
        Set grain envelope type.
        
        Args:
            env_type: One of 'hann', 'hamming', 'triangle', 'trapezoid', 'expodec', 'rexpodec'
        """
        valid_types = {"hann", "hamming", "triangle", "trapezoid", "expodec", "rexpodec"}
        if env_type.lower() in valid_types:
            self._env_type = env_type.lower()
            # Note: envelope changes require voice recreation for full effect
            # For now, store the preference for new voices
            
    def set_reverse_prob(self, prob: float):
        """Set probability of reversed grain playback (0-1)."""
        self._reverse_prob = float(np.clip(prob, 0.0, 1.0))
        
    def get_trigger_interval(self) -> float:
        """Get time between grain triggers, with optional jitter."""
        base_interval = 1.0 / self._trigger_rate
        if self._trigger_jitter > 0:
            jitter = (np.random.random() - 0.5) * 2 * self._trigger_jitter * base_interval
            return max(0.01, base_interval + jitter)
        return base_interval
    
    # --- Properties ---
    
    @property
    def trigger_rate(self) -> float:
        return self._trigger_rate
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def pitch(self) -> float:
        return self._pitch
    
    @property
    def pitch_spread(self) -> float:
        return self._pitch_spread
    
    @property
    def grain_dur(self) -> float:
        return self._grain_dur
    
    @property
    def grain_dur_spread(self) -> float:
        return self._grain_dur_spread
    
    @property
    def position_spread(self) -> float:
        return self._position_spread
    
    @property
    def pan(self) -> float:
        return self._pan
    
    @property
    def pan_spread(self) -> float:
        return self._pan_spread
    
    @property
    def filter_freq(self) -> float:
        return self._filter_freq
    
    @property
    def filter_q(self) -> float:
        return self._filter_q
    
    @property
    def envelope_type(self) -> str:
        return self._env_type
    
    @property
    def reverse_prob(self) -> float:
        return self._reverse_prob
    
    def get_state(self) -> dict:
        """Get current state of all granular parameters."""
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
    """
    Preload grain audio buffers into memory (alternative to pyo tables).
    
    Args:
        manifest_path: Path to grain manifest NPZ file
        
    Returns:
        Dictionary mapping file_id -> audio array [T, 2]
    """
    manifest = load_grain_manifest(manifest_path)
    grain_paths = manifest["grain_paths"]
    buffers: Dict[int, np.ndarray] = {}
    
    for fid, path in enumerate(grain_paths):
        if path and os.path.isfile(path):
            buffers[fid] = load_wav(path)
            
    return buffers
