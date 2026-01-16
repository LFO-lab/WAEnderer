"""
OSC server for real-time control of navigation and grain playback.
"""
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from typing import Optional


def run_server(
    nav,
    grain_player=None,
    ip: str = "127.0.0.1",
    port: int = 9000,
):
    """
    Start OSC server for navigation and grain playback control.
    
    OSC Messages:
        Navigation:
            /cursor x [y [z ...]]  - Set cursor position (coordinates in [0,1])
        
        Policy controls:
            /policy/width value       - Set width control (0-1)
            /policy/energy value      - Set energy control (0-1)
            /policy/gravity value     - Set gravity control (0-1)
            /policy/memory value      - Set memory control (0-1)
            /policy/coherence value   - Set coherence control (0-1)
            /policy/exploration value - Set exploration control (0-1)
            /policy/regime_bias value - Set regime bias (0-2)
            /policy/reset             - Reset policy state
        
        Grain controls (basic):
            /grain/rate value         - Set grain trigger rate (grains/sec)
            /grain/jitter value       - Set timing jitter (0-1)
            /grain/amp value          - Set master amplitude (0-2)
        
        Grain controls (synthesis):
            /grain/pitch value        - Set playback pitch ratio (0.25-4.0)
            /grain/pitch_spread value - Set pitch randomization in semitones (0-12)
            /grain/dur value          - Set grain duration in seconds (0.01-0.5)
            /grain/dur_spread value   - Set duration randomization (0-1)
            /grain/pos_spread value   - Set position jitter (0-1)
            /grain/pan value          - Set base pan position (0-1)
            /grain/pan_spread value   - Set stereo spread (0-1)
            /grain/filter_freq value  - Set lowpass filter cutoff (20-20000 Hz)
            /grain/filter_q value     - Set filter resonance (0.5-10)
            /grain/env type           - Set envelope type (hann, hamming, triangle, trapezoid, expodec, rexpodec)
            /grain/reverse value      - Set reverse probability (0-1)
        
        Legacy (mapped to grain/rate):
            /warp/speed value         - Alias for /grain/rate (multiplier)
    
    Args:
        nav: NavigationEngine instance
        grain_player: GrainPlayer instance (optional)
        ip: Server IP address
        port: Server port
    """
    dispatcher = Dispatcher()
    
    def _as_scalar(args):
        if len(args) == 0:
            return None
        try:
            return float(args[0])
        except Exception:
            return None
    
    def _as_string(args):
        if len(args) == 0:
            return None
        try:
            return str(args[0])
        except Exception:
            return None
    
    def _as_bool(args):
        if len(args) == 0:
            return None
        val = args[0]
        if isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("1", "true", "on", "yes"):
                return True
            if val_lower in ("0", "false", "off", "no"):
                return False
        try:
            return bool(int(val))
        except Exception:
            return None
    
    # --- Navigation cursor ---
    def on_cursor(addr, *coords):
        if len(coords) == 0:
            return
        arr = np.asarray(coords, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        nav.set_cursor_nd(arr)
    
    # --- Policy controls ---
    def on_width(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(width=v)
    
    def on_energy(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(energy=v)
    
    def on_gravity(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(gravity=v)
    
    def on_memory(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(memory=v)
    
    def on_coherence(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(coherence=v)
    
    def on_exploration(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(exploration=v)
    
    def on_regime_bias(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_policy_controls(regime_bias=v)
    
    def on_reset(addr, *vals):
        nav.reset_policy()
    
    # --- Grain controls (basic) ---
    def on_grain_rate(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            # Set on nav (affects trigger interval)
            nav.set_grain_rate(v)
            # Also set on grain_player if available
            if grain_player is not None:
                grain_player.set_trigger_rate(v)
    
    def on_grain_jitter(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_grain_jitter(v)
            if grain_player is not None:
                grain_player.set_trigger_jitter(v)
    
    def on_grain_amp(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_master_amp(v)
    
    # --- Grain controls (synthesis) ---
    def on_grain_pitch(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_pitch(v)
    
    def on_grain_pitch_spread(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_pitch_spread(v)
    
    def on_grain_dur(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_grain_dur(v)
    
    def on_grain_dur_spread(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_grain_dur_spread(v)
    
    def on_grain_pos_spread(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_position_spread(v)
    
    def on_grain_pan(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_pan(v)
    
    def on_grain_pan_spread(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_pan_spread(v)
    
    def on_grain_filter_freq(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_filter_freq(v)
    
    def on_grain_filter_q(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_filter_q(v)
    
    def on_grain_env(addr, *vals):
        v = _as_string(vals)
        if v is not None and grain_player is not None:
            grain_player.set_envelope(v)
    
    def on_grain_reverse(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and grain_player is not None:
            grain_player.set_reverse_prob(v)
    
    # --- Legacy warp mappings (redirect to grain controls) ---
    def on_warp_speed(addr, *vals):
        """Legacy: /warp/speed maps to grain rate multiplier."""
        v = _as_scalar(vals)
        if v is not None:
            # Interpret warp_speed as rate multiplier (1.0 = 10 grains/sec base)
            base_rate = 10.0
            rate = max(0.1, base_rate * v)
            nav.set_grain_rate(rate)
            if grain_player is not None:
                grain_player.set_trigger_rate(rate)
    
    # Register handlers
    dispatcher.map("/cursor", on_cursor)
    
    # Policy
    dispatcher.map("/policy/width", on_width)
    dispatcher.map("/policy/energy", on_energy)
    dispatcher.map("/policy/gravity", on_gravity)
    dispatcher.map("/policy/memory", on_memory)
    dispatcher.map("/policy/coherence", on_coherence)
    dispatcher.map("/policy/exploration", on_exploration)
    dispatcher.map("/policy/regime_bias", on_regime_bias)
    dispatcher.map("/policy/reset", on_reset)
    
    # Grain controls (basic)
    dispatcher.map("/grain/rate", on_grain_rate)
    dispatcher.map("/grain/jitter", on_grain_jitter)
    dispatcher.map("/grain/amp", on_grain_amp)
    
    # Grain controls (synthesis)
    dispatcher.map("/grain/pitch", on_grain_pitch)
    dispatcher.map("/grain/pitch_spread", on_grain_pitch_spread)
    dispatcher.map("/grain/dur", on_grain_dur)
    dispatcher.map("/grain/dur_spread", on_grain_dur_spread)
    dispatcher.map("/grain/pos_spread", on_grain_pos_spread)
    dispatcher.map("/grain/pan", on_grain_pan)
    dispatcher.map("/grain/pan_spread", on_grain_pan_spread)
    dispatcher.map("/grain/filter_freq", on_grain_filter_freq)
    dispatcher.map("/grain/filter_q", on_grain_filter_q)
    dispatcher.map("/grain/env", on_grain_env)
    dispatcher.map("/grain/reverse", on_grain_reverse)
    
    # Legacy warp (partial compatibility)
    dispatcher.map("/warp/speed", on_warp_speed)
    
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    
    print(f"OSC listening on {ip}:{port}")
    print("  /cursor d0 [d1 [d2 ...]] — set navigation cursor (values in [0..1])")
    print("  Policy: /policy/width, /energy, /gravity, /memory, /coherence, /exploration, /regime_bias (0..1), /policy/reset")
    print("  Grain (basic): /grain/rate (grains/sec), /grain/jitter (0..1), /grain/amp (0..2)")
    print("  Grain (synthesis):")
    print("    /grain/pitch (0.25-4), /grain/pitch_spread (0-12 semitones)")
    print("    /grain/dur (0.01-0.5s), /grain/dur_spread (0-1)")
    print("    /grain/pos_spread (0-1), /grain/pan (0-1), /grain/pan_spread (0-1)")
    print("    /grain/filter_freq (20-20000Hz), /grain/filter_q (0.5-10)")
    print("    /grain/env (hann|hamming|triangle|trapezoid|expodec|rexpodec)")
    print("    /grain/reverse (0-1 probability)")
    print("  Legacy: /warp/speed (maps to grain rate multiplier)")
    
    server.serve_forever()
