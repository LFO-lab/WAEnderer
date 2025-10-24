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

    dispatcher.map("/cursor", on_cursor)
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC listening on {ip}:{port} — send /cursor d0 [d1 [d2 ...]] in [0..1]")
    server.serve_forever()
