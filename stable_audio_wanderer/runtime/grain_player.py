"""
pyo-based multi-voice grain player for corpus playback.
Reads pre-rendered grain buffers and triggers them based on navigation indices.
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


class GrainVoice:
    """A single grain voice that can play one grain at a time."""
    
    def __init__(self, server: pyo.Server, table: pyo.SndTable, mul: float = 1.0):
        self.server = server
        self.table = table
        self.mul = mul
        self._active = False
        
        # TableRead for playback with envelope
        self.player = pyo.TableRead(
            table=self.table,
            freq=self.table.getRate(),
            loop=False,
            mul=mul,
        ).stop()
        
        # Pan for stereo positioning
        self.panner = pyo.Pan(self.player, outs=2, pan=0.5)
        
    def trigger(self, start_sample: int, length: int, amp: float = 1.0, pan: float = 0.5):
        """Trigger grain playback from start_sample for length samples."""
        self.player.stop()
        
        # Set read position and amplitude
        freq = self.table.getRate()
        self.player.setFreq(freq)
        self.player.setMul(amp * self.mul)
        self.panner.setPan(pan)
        
        # Reset to start position and play
        # Note: TableRead reads from beginning; we'll use offset tables for precise positioning
        self.player.reset()
        self.player.play()
        self._active = True
        
    def stop(self):
        """Stop this voice."""
        self.player.stop()
        self._active = False
        
    def is_active(self) -> bool:
        """Check if voice is currently playing."""
        return self._active and self.player.isPlaying()
    
    def out(self, chnl: int = 0):
        """Send to output."""
        self.panner.out(chnl)
        return self


class GrainPlayer:
    """
    pyo-based multi-voice grain player with crossfade support.
    
    Plays pre-rendered grains from a manifest, supporting multiple
    overlapping voices for smooth transitions.
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
            num_voices: Number of overlapping voices for crossfade
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
        self._voices: List[GrainVoice] = []
        self._voice_idx = 0  # Round-robin voice selection
        
        # Playback state
        self._running = False
        self._trigger_rate = 10.0  # grains per second
        self._trigger_jitter = 0.0  # random variation in timing (0-1)
        self._master_amp = 1.0
        self._lock = threading.Lock()
        
    def _load_tables(self):
        """Load grain buffers into pyo SndTables."""
        unique_files = set(int(fid) for fid in self.file_ids)
        for fid in unique_files:
            if fid < len(self.grain_paths) and self.grain_paths[fid]:
                path = self.grain_paths[fid]
                if os.path.isfile(path):
                    # Load stereo table
                    self._tables[fid] = pyo.SndTable(path)
                    
    def _create_voices(self):
        """Create voice pool for polyphonic playback."""
        self._voices = []
        # Create voices for each loaded table
        for fid, table in self._tables.items():
            for _ in range(max(1, self.num_voices // max(len(self._tables), 1))):
                voice = GrainVoice(self.server, table, mul=self._master_amp)
                voice.out()
                self._voices.append((fid, voice))
        
        # If we have fewer voices than requested, add more for the first table
        while len(self._voices) < self.num_voices and self._tables:
            fid = list(self._tables.keys())[0]
            voice = GrainVoice(self.server, self._tables[fid], mul=self._master_amp)
            voice.out()
            self._voices.append((fid, voice))
    
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
        if self.server is not None:
            self.server.stop()
            
    def shutdown(self):
        """Shutdown the pyo server completely."""
        self.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server = None
        self._tables.clear()
        self._voices.clear()
        
    def trigger_grain(self, segment_idx: int, amp: float = 1.0, pan: float = 0.5) -> bool:
        """
        Trigger grain for the given segment index.
        
        Args:
            segment_idx: Corpus segment index
            amp: Amplitude (0-1)
            pan: Stereo pan position (0=left, 0.5=center, 1=right)
            
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
            
        with self._lock:
            # Find a voice for this file (round-robin among matching voices)
            matching_voices = [(i, v) for i, (fid, v) in enumerate(self._voices) if fid == file_id]
            if not matching_voices:
                # Use any available voice if no matching ones
                if not self._voices:
                    return False
                idx = self._voice_idx % len(self._voices)
                _, voice = self._voices[idx]
            else:
                # Use round-robin among matching voices
                voice_pool_idx = self._voice_idx % len(matching_voices)
                _, voice = matching_voices[voice_pool_idx]
            
            # Trigger the grain
            voice.trigger(offset, length, amp=amp * self._master_amp, pan=pan)
            self._voice_idx += 1
            
        return True
    
    def set_trigger_rate(self, rate: float):
        """Set grain trigger rate in grains per second."""
        self._trigger_rate = max(0.1, float(rate))
        
    def set_trigger_jitter(self, jitter: float):
        """Set random timing jitter (0-1)."""
        self._trigger_jitter = float(np.clip(jitter, 0.0, 1.0))
        
    def set_master_amp(self, amp: float):
        """Set master amplitude."""
        self._master_amp = float(np.clip(amp, 0.0, 2.0))
        
    def get_trigger_interval(self) -> float:
        """Get time between grain triggers, with optional jitter."""
        base_interval = 1.0 / self._trigger_rate
        if self._trigger_jitter > 0:
            jitter = (np.random.random() - 0.5) * 2 * self._trigger_jitter * base_interval
            return max(0.01, base_interval + jitter)
        return base_interval
    
    @property
    def trigger_rate(self) -> float:
        return self._trigger_rate
    
    @property
    def is_running(self) -> bool:
        return self._running


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
