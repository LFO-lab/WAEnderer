"""Switchable general OSC input for the Web application's perform controller."""
import threading
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from .osc_server import create_dispatcher


class OscInput:
    _camera_opposites = {
        "left": "right", "right": "left", "over": "under", "under": "over",
        "forward": "backward", "backward": "forward",
    }

    def __init__(self, host="127.0.0.1", port=9001, debug=False):
        self.host, self.port, self.debug = host, port, debug
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._dispatcher = None
        self._server = None
        self._thread = None
        self._error = ""
        self._received = 0
        self._last_address = ""
        self._closed = False
        self._camera_revision = 0
        self._camera_updates = {}

    def _camera_message(self, address, values):
        if not address.startswith("/camera/"):
            return False
        key = address.removeprefix("/camera/")
        if key not in (*self._camera_opposites, "speed", "stop"):
            raise ValueError("unknown camera address")
        if key == "stop" and not values:
            value = 1.0
        else:
            if len(values) != 1:
                raise ValueError("expected one numeric value")
            value = float(values[0])
            if not math.isfinite(value):
                raise ValueError("camera value must be finite")
        if key != "speed" and value not in (0, 1):
            raise ValueError("camera switches expect 0 or 1")
        if key == "stop":
            if not value:
                return True
            updates = {direction: False for direction in self._camera_opposites}
        elif key == "speed":
            updates = {key: max(0.0, min(3.0, value))}
        else:
            updates = {key: bool(value)}
            if value:
                updates[self._camera_opposites[key]] = False
        self._camera_revision += 1
        for control, setting in updates.items():
            self._camera_updates[control] = {
                "revision": self._camera_revision, "value": setting}
        if self.debug:
            print(f"[osc-debug] {address} -> camera {updates}")
        return True

    def bind(self, controller):
        # Wait for any in-flight handler before the old controller is closed.
        with self._lock:
            self._dispatcher = None if controller is None or self._closed else create_dispatcher(
                controller.nav, controller.decoder, manual_controller=controller,
                osc_debug=self.debug)

    def _dispatch(self, address, *values):
        with self._lock:
            self._received += 1
            self._last_address = address
            try:
                if self._camera_message(address, values):
                    return
                if self._dispatcher is None:
                    return
                for handler in self._dispatcher.handlers_for_address(address):
                    handler.callback(address, *values)
            except Exception as exc:
                self._error = f"{address}: {exc}"
                print(f"[osc] {self._error}")

    def set_enabled(self, enabled):
        with self._lifecycle:
            if enabled:
                with self._lock:
                    if self._closed:
                        return
                    if self._server is not None:
                        return
                    dispatcher = Dispatcher()
                    dispatcher.set_default_handler(self._dispatch)
                    try:
                        server = BlockingOSCUDPServer((self.host, self.port), dispatcher)
                    except OSError as exc:
                        self._error = str(exc)
                        print(f"[osc] Could not listen on {self.host}:{self.port}: {exc}")
                        return
                    self._server = server
                    self._error = ""
                    self._thread = threading.Thread(
                        target=server.serve_forever, kwargs={"poll_interval": 0.05},
                        daemon=True, name="osc-input")
                    self._thread.start()
                    print(f"[osc] Listening on {server.server_address}")
            else:
                with self._lock:
                    server, thread = self._server, self._thread
                    self._server = self._thread = None
                    self._error = ""
                if server is not None:
                    # Do not hold the dispatch lock while waiting for the UDP thread.
                    server.shutdown()
                    server.server_close()
                    thread.join()
                    print("[osc] Input stopped")

    def handle_message(self, data):
        if data.get("type") != "osc_input":
            return False
        enabled = data.get("enabled")
        if isinstance(enabled, bool):
            self.set_enabled(enabled)
        return True

    def get_state(self):
        with self._lock:
            return {"osc_input": {
                "enabled": self._server is not None,
                "host": self.host,
                "port": self._server.server_address[1] if self._server else self.port,
                "ready": self._dispatcher is not None,
                "received": self._received,
                "last_address": self._last_address,
                "error": self._error,
                "camera_updates": {key: dict(update)
                                   for key, update in self._camera_updates.items()},
            }}

    def close(self):
        with self._lock:
            self._closed = True
        self.set_enabled(False)
        self.bind(None)
