"""
OSC server for real-time control of navigation and decoder output.
"""
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def run_server(
    nav,
    decoder=None,
    ip: str = "127.0.0.1",
    port: int = 9000,
):
    """
    Start OSC server for navigation and decoder control.

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
            /policy/reset             - Reset policy state

        Decoder controls:
            /decoder/gain value        - Set output gain (0-2)
            /decoder/smoothing value   - Set crossfade smoothing (0-1)

    Args:
        nav: Navigation engine instance
        decoder: DecoderPlayer instance (optional)
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

    def on_reset(addr, *vals):
        nav.reset_policy()

    # --- Decoder controls ---
    def on_decoder_gain(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and decoder is not None:
            decoder.set_gain(v)

    def on_decoder_smoothing(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and decoder is not None:
            decoder.set_smoothing(v)

    # Register handlers
    dispatcher.map("/cursor", on_cursor)

    dispatcher.map("/policy/width", on_width)
    dispatcher.map("/policy/energy", on_energy)
    dispatcher.map("/policy/gravity", on_gravity)
    dispatcher.map("/policy/memory", on_memory)
    dispatcher.map("/policy/coherence", on_coherence)
    dispatcher.map("/policy/exploration", on_exploration)
    dispatcher.map("/policy/reset", on_reset)

    dispatcher.map("/decoder/gain", on_decoder_gain)
    dispatcher.map("/decoder/smoothing", on_decoder_smoothing)

    server = BlockingOSCUDPServer((ip, port), dispatcher)

    print(f"OSC listening on {ip}:{port}")
    print("  /cursor d0 [d1 [d2 ...]] — set navigation cursor (values in [0..1])")
    print("  Policy: /policy/width, /energy, /gravity, /memory, /coherence, /exploration (0..1), /policy/reset")
    print("  Decoder: /decoder/gain (0..2), /decoder/smoothing (0..1)")

    server.serve_forever()
