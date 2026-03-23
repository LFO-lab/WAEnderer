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
    manual_controller=None,
    osc_debug: bool = False,
):
    """
    Start OSC server for navigation and decoder control.

    OSC Messages:
        Navigation:
            /cursor x [y [z ...]]  - Set cursor position (coordinates in [0,1])

        Random controls:
            /random/phrase_scale value
            /random/jump_rate value
            /random/timbre_lock value
            /random/drift value
            /random/repeat_avoid value
            /random/crossfile value
            /random/reset

        Reorganized controls:
            /reorganized/morph_len value
            /reorganized/jump_rate value
            /reorganized/timbre_lock value
            /reorganized/evolution value
            /reorganized/novelty value
            /reorganized/crossfile value
            /reorganized/reset

        Decoder controls:
            /decoder/gain value        - Set output gain (0-2)

        Manual controls:
            /manual/x value            - Set manual X control (0-1)
            /manual/y value            - Set manual Y control (0-1)
            /manual/z value            - Set manual Z control (0-1)
            /manual/w value            - Set manual W control (0-1)
            /manual/wander_k value     - Set manual wander neighborhood size (1-64)
            /manual/xyz x y z          - Set manual XYZ controls together (0-1 each)
            /manual/xyzw x y z w       - Set manual XYZW controls together (0-1 each)

    Args:
        nav: Navigation engine instance
        decoder: DecoderPlayer instance (optional)
        ip: Server IP address
        port: Server port
        manual_controller: Transport controller exposing set_manual_axis(axis, value)
        osc_debug: Print incoming OSC messages and handler outcomes.
    """
    dispatcher = Dispatcher()

    def _dbg(msg: str):
        if bool(osc_debug):
            print(f"[osc-debug] {msg}")

    def _as_scalar(args):
        if len(args) == 0:
            return None
        try:
            return float(args[0])
        except Exception:
            return None

    def _manual_control_dim() -> int:
        if manual_controller is None:
            return 3
        engine = getattr(manual_controller, "manual", None)
        try:
            return max(1, int(getattr(engine, "control_dim", 3)))
        except Exception:
            return 3

    # --- Navigation cursor ---
    def on_cursor(addr, *coords):
        if len(coords) == 0:
            _dbg(f"{addr} ignored (no args)")
            return
        arr = np.asarray(coords, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        # In manual mode, route /cursor to manual XYZ controls for convenience.
        if manual_controller is not None:
            selected_mode = str(getattr(manual_controller, "selected_mode", "random"))
            active_mode = str(getattr(manual_controller, "_active_mode", selected_mode))
            if selected_mode == "manual" or active_mode == "manual":
                set_all = getattr(manual_controller, "set_manual_faders", None)
                if set_all is not None:
                    control_dim = _manual_control_dim()
                    vec = arr[:control_dim]
                    if vec.shape[0] < control_dim:
                        vec = np.pad(vec, (0, control_dim - vec.shape[0]), mode="edge")
                    ok, msg = set_all(vec.tolist())
                    _dbg(f"{addr} -> manual {vec.tolist()} ({msg})")
                    return
        nav.set_cursor_nd(arr)
        _dbg(f"{addr} -> nav cursor {arr.tolist()}")

    # --- Random controls ---
    def on_random_phrase_scale(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(phrase_scale=v)
            _dbg(f"{addr} {v}")

    def on_random_jump_rate(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(jump_rate=v)
            _dbg(f"{addr} {v}")

    def on_random_timbre_lock(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(timbre_lock=v)
            _dbg(f"{addr} {v}")

    def on_random_drift(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(drift=v)
            _dbg(f"{addr} {v}")

    def on_random_repeat_avoid(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(repeat_avoid=v)
            _dbg(f"{addr} {v}")

    def on_random_crossfile(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_random_controls(crossfile=v)
            _dbg(f"{addr} {v}")

    def on_random_reset(addr, *vals):
        nav.set_policy_variant("random")
        nav.reset_policy()
        _dbg(f"{addr}")

    # --- Reorganized controls ---
    def on_reorganized_morph_len(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(morph_len=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_jump_rate(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(jump_rate=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_timbre_lock(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(timbre_lock=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_evolution(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(evolution=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_novelty(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(novelty=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_crossfile(addr, *vals):
        v = _as_scalar(vals)
        if v is not None:
            nav.set_reorganized_controls(crossfile=v)
            _dbg(f"{addr} {v}")

    def on_reorganized_reset(addr, *vals):
        nav.set_policy_variant("reorganized")
        nav.reset_policy()
        _dbg(f"{addr}")

    # --- Decoder controls ---
    def on_decoder_gain(addr, *vals):
        v = _as_scalar(vals)
        if v is not None and decoder is not None:
            decoder.set_gain(v)
            _dbg(f"{addr} {v}")

    # --- Manual controls ---
    def on_manual_x(addr, *vals):
        v = _as_scalar(vals)
        if v is None or manual_controller is None:
            _dbg(f"{addr} ignored (no controller or invalid value)")
            return
        setter = getattr(manual_controller, "set_manual_axis", None)
        if setter is None:
            return
        try:
            ok, msg = setter(0, v)
            if not ok:
                print(f"[osc] /manual/x error: {msg}")
            _dbg(f"{addr} {v} ({msg})")
        except Exception as exc:
            print(f"[osc] /manual/x exception: {exc}")

    def on_manual_y(addr, *vals):
        v = _as_scalar(vals)
        if v is None or manual_controller is None:
            _dbg(f"{addr} ignored (no controller or invalid value)")
            return
        setter = getattr(manual_controller, "set_manual_axis", None)
        if setter is None:
            return
        try:
            ok, msg = setter(1, v)
            if not ok:
                print(f"[osc] /manual/y error: {msg}")
            _dbg(f"{addr} {v} ({msg})")
        except Exception as exc:
            print(f"[osc] /manual/y exception: {exc}")

    def on_manual_z(addr, *vals):
        v = _as_scalar(vals)
        if v is None or manual_controller is None:
            _dbg(f"{addr} ignored (no controller or invalid value)")
            return
        setter = getattr(manual_controller, "set_manual_axis", None)
        if setter is None:
            return
        try:
            ok, msg = setter(2, v)
            if not ok:
                print(f"[osc] /manual/z error: {msg}")
            _dbg(f"{addr} {v} ({msg})")
        except Exception as exc:
            print(f"[osc] /manual/z exception: {exc}")

    def on_manual_w(addr, *vals):
        v = _as_scalar(vals)
        if v is None or manual_controller is None:
            _dbg(f"{addr} ignored (no controller or invalid value)")
            return
        setter = getattr(manual_controller, "set_manual_axis", None)
        if setter is None:
            return
        try:
            ok, msg = setter(3, v)
            if not ok:
                print(f"[osc] /manual/w error: {msg}")
            _dbg(f"{addr} {v} ({msg})")
        except Exception as exc:
            print(f"[osc] /manual/w exception: {exc}")

    def on_manual_xyz(addr, *vals):
        if manual_controller is None:
            _dbg(f"{addr} ignored (no controller)")
            return
        if len(vals) < 3:
            print("[osc] /manual/xyz expects 3 floats: x y z")
            return
        try:
            arr = np.asarray(vals[:3], dtype=np.float32)
            arr = np.clip(arr, 0.0, 1.0)
        except Exception as exc:
            print(f"[osc] /manual/xyz parse error: {exc}")
            return

        set_axis = getattr(manual_controller, "set_manual_axis", None)
        if set_axis is None:
            return
        for axis, value in enumerate(arr.tolist()):
            try:
                ok, msg = set_axis(axis, value)
                if not ok:
                    print(f"[osc] /manual/xyz axis {axis} error: {msg}")
                _dbg(f"{addr} axis={axis} value={value} ({msg})")
            except Exception as exc:
                print(f"[osc] /manual/xyz axis {axis} exception: {exc}")

    def on_manual_xyzw(addr, *vals):
        if manual_controller is None:
            _dbg(f"{addr} ignored (no controller)")
            return
        if len(vals) < 4:
            print("[osc] /manual/xyzw expects 4 floats: x y z w")
            return
        try:
            arr = np.asarray(vals[:4], dtype=np.float32)
            arr = np.clip(arr, 0.0, 1.0)
        except Exception as exc:
            print(f"[osc] /manual/xyzw parse error: {exc}")
            return

        set_axis = getattr(manual_controller, "set_manual_axis", None)
        if set_axis is None:
            return
        for axis, value in enumerate(arr.tolist()):
            try:
                ok, msg = set_axis(axis, value)
                if not ok:
                    print(f"[osc] /manual/xyzw axis {axis} error: {msg}")
                _dbg(f"{addr} axis={axis} value={value} ({msg})")
            except Exception as exc:
                print(f"[osc] /manual/xyzw axis {axis} exception: {exc}")

    def on_manual_wander_k(addr, *vals):
        v = _as_scalar(vals)
        if v is None or manual_controller is None:
            _dbg(f"{addr} ignored (no controller or invalid value)")
            return
        setter = getattr(manual_controller, "set_manual_wander_params", None)
        if setter is None:
            return
        try:
            k = int(max(1, min(64, round(v))))
            ok, msg = setter(k=k, speed=None)
            if not ok:
                print(f"[osc] /manual/wander_k error: {msg}")
            _dbg(f"{addr} {k} ({msg})")
        except Exception as exc:
            print(f"[osc] /manual/wander_k exception: {exc}")

    def on_unmapped(addr, *vals):
        _dbg(f"unmapped {addr} args={list(vals)}")

    # Register handlers
    dispatcher.map("/cursor", on_cursor)

    dispatcher.map("/random/phrase_scale", on_random_phrase_scale)
    dispatcher.map("/random/jump_rate", on_random_jump_rate)
    dispatcher.map("/random/timbre_lock", on_random_timbre_lock)
    dispatcher.map("/random/drift", on_random_drift)
    dispatcher.map("/random/repeat_avoid", on_random_repeat_avoid)
    dispatcher.map("/random/crossfile", on_random_crossfile)
    dispatcher.map("/random/reset", on_random_reset)

    dispatcher.map("/reorganized/morph_len", on_reorganized_morph_len)
    dispatcher.map("/reorganized/jump_rate", on_reorganized_jump_rate)
    dispatcher.map("/reorganized/timbre_lock", on_reorganized_timbre_lock)
    dispatcher.map("/reorganized/evolution", on_reorganized_evolution)
    dispatcher.map("/reorganized/novelty", on_reorganized_novelty)
    dispatcher.map("/reorganized/crossfile", on_reorganized_crossfile)
    dispatcher.map("/reorganized/reset", on_reorganized_reset)

    dispatcher.map("/decoder/gain", on_decoder_gain)
    dispatcher.map("/manual/x", on_manual_x)
    dispatcher.map("/manual/y", on_manual_y)
    dispatcher.map("/manual/z", on_manual_z)
    dispatcher.map("/manual/w", on_manual_w)
    dispatcher.map("/manual/wander_k", on_manual_wander_k)
    dispatcher.map("/manual/xyz", on_manual_xyz)
    dispatcher.map("/manual/xyzw", on_manual_xyzw)
    dispatcher.set_default_handler(on_unmapped)

    server = BlockingOSCUDPServer((ip, port), dispatcher)

    print(f"OSC listening on {ip}:{port}")
    print("  /cursor d0 [d1 [d2 ...]] — set navigation cursor (values in [0..1])")
    print(
        "  Random: /random/phrase_scale, /jump_rate, /timbre_lock, "
        "/drift, /repeat_avoid, /crossfile (0..1), /random/reset"
    )
    print(
        "  Reorganized: /reorganized/morph_len, /jump_rate, /timbre_lock, "
        "/evolution, /novelty, /crossfile (0..1), /reorganized/reset"
    )
    print("  Decoder: /decoder/gain (0..2), /decoder/smoothing (0..1)")
    print("  Manual: /manual/x, /manual/y, /manual/z, /manual/w (0..1), /manual/wander_k (1..64), /manual/xyz x y z, /manual/xyzw x y z w")
    if bool(osc_debug):
        print("  OSC debug: enabled (logs matched and unmatched OSC messages)")

    server.serve_forever()
