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
            /policy/width value    - Set width control (0-1)
            /policy/energy value   - Set energy control (0-1)
            /policy/gravity value  - Set gravity control (0-1)
            /policy/memory value   - Set memory control (0-1)
            /policy/reset          - Reset policy state
        
        Grain controls:
            /grain/rate value      - Set grain trigger rate (grains/sec)
            /grain/jitter value    - Set timing jitter (0-1)
            /grain/amp value       - Set master amplitude (0-1)
        
        Legacy (mapped to grain/rate):
            /warp/speed value      - Alias for /grain/rate
    
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
    
    def on_reset(addr, *vals):
        nav.reset_policy()
    
    # --- Grain controls ---
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
    dispatcher.map("/policy/reset", on_reset)
    
    # Grain controls
    dispatcher.map("/grain/rate", on_grain_rate)
    dispatcher.map("/grain/jitter", on_grain_jitter)
    dispatcher.map("/grain/amp", on_grain_amp)
    
    # Legacy warp (partial compatibility)
    dispatcher.map("/warp/speed", on_warp_speed)
    
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    
    print(f"OSC listening on {ip}:{port}")
    print("  /cursor d0 [d1 [d2 ...]] — set navigation cursor (values in [0..1])")
    print("  Policy: /policy/width, /energy, /gravity, /memory (0..1), /policy/reset")
    print("  Grain: /grain/rate (grains/sec), /grain/jitter (0..1), /grain/amp (0..1)")
    print("  Legacy: /warp/speed (maps to grain rate multiplier)")
    
    server.serve_forever()
