import asyncio
import json
import socket
import threading
from types import SimpleNamespace

import pytest
from pythonosc.udp_client import SimpleUDPClient

from stable_audio_wanderer.runtime.osc_input import OscInput
from stable_audio_wanderer.runtime.ws_server import WSBroadcaster
from test_onnx_transport import _controller


def test_current_transport_manual_axes_and_wander_reset():
    controller, _, _ = _controller(initial_mode="manual")
    osc = OscInput()
    try:
        osc.bind(controller)
        osc._dispatch("/manual/x", 0.25)
        osc._dispatch("/manual/xyz", 0.1, 0.2, 0.3)
        assert controller.manual.faders.tolist() == pytest.approx([0.1, 0.2, 0.3])
        osc._dispatch("/manual/x", float("nan"))
        osc._dispatch("/cursor", "bad")
        assert controller.manual.faders.tolist() == pytest.approx([0.1, 0.2, 0.3])
        serial = controller._wander_reset_serial
        osc._dispatch("/wander/reset")
        assert controller._wander_reset_serial == serial + 1
        assert not osc.get_state()["osc_input"]["error"]
    finally:
        osc.close()
        controller.close()


def test_udp_routing_unbind_and_port_release():
    changed = threading.Event()
    values = []

    def set_faders(faders):
        values.append(faders)
        changed.set()
        return True, "updated"

    controller = SimpleNamespace(
        nav=SimpleNamespace(), decoder=SimpleNamespace(), selected_mode="manual",
        manual=SimpleNamespace(control_dim=3), set_manual_faders=set_faders)
    osc = OscInput(port=0)
    try:
        osc.bind(controller)
        osc.set_enabled(True)
        port = osc.get_state()["osc_input"]["port"]
        client = SimpleUDPClient("127.0.0.1", port)
        client.send_message("/cursor", [0.25, 0.5, 0.75])
        assert changed.wait(2)
        assert values[-1] == pytest.approx([0.25, 0.5, 0.75])
        assert osc.get_state()["osc_input"]["last_address"] == "/cursor"

        osc.bind(None)
        changed.clear()
        client.send_message("/cursor", [1, 1, 1])
        osc.set_enabled(False)  # Joins any in-flight handler.
        assert not changed.is_set()
        assert not osc.get_state()["osc_input"]["enabled"]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", port))
        osc.set_enabled(True)
        assert osc.get_state()["osc_input"]["enabled"]
    finally:
        osc.close()


def test_web_switch_reports_port_conflict_and_survives_perform_unbind():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        osc = OscInput(port=occupied.getsockname()[1])
        broadcaster = WSBroadcaster(service_message_handler=osc.handle_message,
                                    service_state_provider=osc.get_state)
        try:
            asyncio.run(broadcaster._handle_message(
                None, json.dumps({"type": "osc_input", "enabled": True})))
            state = json.loads(broadcaster._get_state_json())["osc_input"]
            assert not state["enabled"] and state["error"]
            broadcaster.unbind_nav_decoder()
            assert "osc_input" in json.loads(broadcaster._get_state_json())
        finally:
            osc.close()
