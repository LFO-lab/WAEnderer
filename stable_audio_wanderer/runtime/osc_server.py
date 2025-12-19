import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def run_server(player, ip="127.0.0.1", port=9000):
    """
    /cursor x [y [z ...]]  — coordonnées en [0,1], longueur variable
    """
    dispatcher = Dispatcher()

    def on_cursor(addr, *coords):
        if len(coords) == 0:
            return
        arr = np.asarray(coords, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        player.set_cursor_nd(arr)

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

    def on_width(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_policy_controls(width=v)

    def on_energy(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_policy_controls(energy=v)

    def on_gravity(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_policy_controls(gravity=v)

    def on_memory(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_policy_controls(memory=v)

    def on_reset(addr, *vals):
        player.reset_policy()

    def on_warp_speed(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_time_warp(speed=v)

    def on_warp_inertia(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_time_warp(inertia=v)

    def on_warp_jitter(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            player.set_time_warp(jitter=v)

    def on_warp_reverse(addr, *vals):
        v = _as_bool(vals)
        if v is not None:
            player.set_time_warp(allow_reverse=v)

    def on_warp_reset(addr, *vals):
        v = _as_scalar(vals)
        player.reset_time_warp(speed=v)

    dispatcher.map("/cursor", on_cursor)
    dispatcher.map("/policy/width", on_width)
    dispatcher.map("/policy/energy", on_energy)
    dispatcher.map("/policy/gravity", on_gravity)
    dispatcher.map("/policy/memory", on_memory)
    dispatcher.map("/policy/reset", on_reset)
    dispatcher.map("/warp/speed", on_warp_speed)
    dispatcher.map("/warp/inertia", on_warp_inertia)
    dispatcher.map("/warp/jitter", on_warp_jitter)
    dispatcher.map("/warp/reverse", on_warp_reverse)
    dispatcher.map("/warp/reset", on_warp_reset)
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC listening on {ip}:{port} — send /cursor d0 [d1 [d2 ...]] in [0..1]")
    print(
        "Policy controls: /policy/width, /policy/energy, /policy/gravity, "
        "/policy/memory (0..1), /policy/reset"
    )
    print("Time warp: /warp/speed, /warp/inertia, /warp/jitter, /warp/reverse (0/1), /warp/reset [speed]")
    server.serve_forever()
