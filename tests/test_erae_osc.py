import socket
import time
from types import SimpleNamespace

import pytest
from stable_audio_wanderer.runtime.erae_osc import EraeOscServer
from stable_audio_wanderer.runtime.erae_protocol import encode, decode


class Controller:
    def __init__(self):
        self.selections = []
        self.window_calls = []
        self.Z_concat = SimpleNamespace(shape=(100, 256))
        self.latent_decoder = SimpleNamespace(supported_windows=(2, 4, 8, 32))
        self.nav = SimpleNamespace(set_cursor_index=self.selections.append)
        self.state = dict(running=True, mode='wander', valid=True, index=7,
                          generation=2, requested_window=8, active_window=4,
                          transition='staging', window_mode='fixed', error='')

    def get_erae_state(self):
        return self.state.copy()

    def set_decoder_window(self, value):
        self.window_calls.append(value)
        return True, 'accepted'


class Peer:
    def __init__(self, server):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.settimeout(.1)
        self.server = server

    def send(self, kind, *args):
        self.socket.sendto(encode(kind, *args), self.server.address)

    def receive(self, kind):
        end = time.monotonic() + 2
        while time.monotonic() < end:
            try:
                name, args = decode(self.socket.recvfrom(1400)[0])
            except socket.timeout:
                continue
            if name == kind:
                return args
        raise AssertionError('missing ' + kind)

    def subscribe(self):
        self.send('subscribe', 'test', self.socket.getsockname()[1])
        return self.receive('capabilities')


@pytest.fixture
def service():
    server = EraeOscServer(port=0).start()
    peer = Peer(server)
    try:
        yield server, peer
    finally:
        peer.socket.close()
        server.close()


def test_not_ready_then_late_binding_and_audio_cursor_without_browser(service):
    server, peer = service
    caps = peer.subscribe()
    assert caps[4:6] == (0, 0)
    state = peer.receive('state')
    assert state[4] == state[7] == 0
    controller = Controller()
    server.bind(controller)
    caps = peer.receive('capabilities')
    assert caps[4:] == (1, 100, 2, 4, 8, 32)
    state = peer.receive('state')
    assert state[8:12] == (7, 2, 8, 4)
    controller.state.update(index=15, running=False)
    while True:
        state = peer.receive('state')
        if state[8] == 15:
            break
    assert state[5] == 0  # held point must not imply playing


def test_commands_acknowledged_once_and_invalid_commands_rejected(service):
    server, peer = service
    c = Controller()
    server.bind(c)
    revision = peer.subscribe()[2]
    for _ in range(2):
        peer.send('select', 'test', 1, revision, 12)
        assert peer.receive('result')[3] == 1
    assert c.selections == [12]
    peer.send('select', 'test', 1, revision, 13)
    assert peer.receive('result')[3] == 0
    peer.send('window', 'test', 2, revision, 8)
    assert peer.receive('result')[3] == 1
    assert c.window_calls == [8]
    for seq, kind, rev, val in ((3, 'select', revision, 100), (4, 'window', revision, 3),
                               (5, 'select', 'old-corpus', 10), (0, 'select', revision, 10)):
        peer.send(kind, 'test', seq, rev, val)
        assert peer.receive('result')[3] == 0
    server.bind(Controller())
    peer.receive('capabilities')
    peer.send('select', 'test', 6, revision, 10)
    assert peer.receive('result')[3] == 0
    assert c.selections == [12]


def test_subscription_ownership_and_expiry():
    server = EraeOscServer(port=0, lease_seconds=.15).start()
    a, b = Peer(server), Peer(server)
    try:
        a.subscribe()
        b.send('subscribe', 'other', b.socket.getsockname()[1])
        with pytest.raises(socket.timeout):
            b.socket.recvfrom(1400)
        time.sleep(.1)
        b.send('subscribe', 'other', b.socket.getsockname()[1])
        assert b.receive('capabilities')[1] == 'other'
    finally:
        a.socket.close(); b.socket.close(); server.close()


def test_wire_schema_copy_matches_bridge_when_sibling_present():
    from pathlib import Path
    import stable_audio_wanderer.runtime.erae_protocol as protocol
    engine = Path(protocol.__file__)
    bridge = engine.parents[3] / 'WAEnderer_erae/waenderer_erae/osc_protocol.py'
    if not bridge.exists():
        pytest.skip('standalone engine checkout')
    assert engine.read_bytes() == bridge.read_bytes()


def test_lightweight_snapshot_never_uses_producer_index():
    import threading
    from stable_audio_wanderer.runtime.onnx_transport import OnnxTransportController
    c = OnnxTransportController.__new__(OnnxTransportController)
    c._lock = threading.Lock()
    c._running = threading.Event()
    c._running.set()
    c._active_mode = c.selected_mode = 'wander'
    c._requested_window = 8
    c._requested_generation = 4
    c._generation_windows = {2: 4, 4: 8}
    c._transport_error = None
    c._prebuffering = False
    c._window_mode = 'fixed'
    c.Z_concat = [None] * 100
    presentation = {'index': 7, 'generation': 2}
    c.decoder = SimpleNamespace(current_generation=2,
        get_presentation_state=lambda: presentation,
        get_state=lambda: {'transition_status': 'idle'})
    c.nav = SimpleNamespace(current_index=99)
    state = c.get_erae_state()
    assert state['index'] == 7
    assert state['active_window'] == 4 and state['requested_window'] == 8
    assert state['transition'] == 'staging'
    c._running.clear()
    assert not c.get_erae_state()['running']
    presentation.clear()
    assert c.get_erae_state()['index'] == -1
    assert not c.get_erae_state()['valid']


def test_real_bridge_loopback_rebind_restart_and_stale_recovery():
    import asyncio
    import sys
    from pathlib import Path
    bridge = Path(__file__).resolve().parents[2] / 'WAEnderer_erae'
    if not bridge.exists():
        pytest.skip('sibling bridge not present')
    sys.path.insert(0, str(bridge))
    from waenderer_erae.osc_client import OscClient

    async def wait_for(predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, 'timed out'
            await asyncio.sleep(.01)

    async def scenario():
        server = EraeOscServer(port=0).start()
        client = OscClient(server.address, renew=.1, stale=.2, log=lambda _: None)
        restarted = None
        try:
            await client.start(port=0)
            await wait_for(lambda: client.state.fresh(.2))
            assert not client.state.ready
            c = Controller()
            server.bind(c)
            await wait_for(lambda: client.state.ready and client.state.fresh(.2))
            seq = client.command('select', 23)
            await wait_for(lambda: seq in client.results)
            assert c.selections == [23]
            assert client.state.playback['index'] == 7
            assert client.results[seq][0] == 1
            old_revision = client.state.revision
            server.bind(Controller())
            await wait_for(lambda: client.state.revision != old_revision)
            assert not client.pending and not client.results
            old_session, address = server.session, server.address
            server.close()
            await asyncio.sleep(.3)
            assert not client.state.fresh(.2)
            with pytest.raises(RuntimeError):
                client.command('window', 8)
            restarted = EraeOscServer(port=address[1]).start()
            fresh_controller = Controller()
            restarted.bind(fresh_controller)
            await wait_for(lambda: client.state.session != old_session and client.state.fresh(.2))
            assert not fresh_controller.selections  # no replay
            seq = client.command('window', 32)
            await wait_for(lambda: seq in client.results)
            assert fresh_controller.window_calls == [32]
        finally:
            await client.close()
            server.close()
            if restarted:
                restarted.close()
    asyncio.run(scenario())


def test_cli_processes_against_live_loopback_service(tmp_path):
    import subprocess
    from pathlib import Path
    bridge = Path(__file__).resolve().parents[2] / 'WAEnderer_erae'
    python = bridge / '.venv/bin/python'
    if not python.exists():
        pytest.skip('bridge environment not installed')
    server = EraeOscServer(port=0).start()
    controller = Controller()
    server.bind(controller)
    reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reservation.bind(('127.0.0.1', 0))
    port = reservation.getsockname()[1]
    reservation.close()
    config = tmp_path / 'bridge.toml'
    config.write_text(f'engine_port = {server.address[1]}\nlisten_port = {port}\n')
    base = [str(python), '-m', 'waenderer_erae', '--config', str(config), '--dry-run']
    try:
        result = subprocess.run(base + ['run', '--seconds', '.3'], cwd=bridge,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert '"index": 7' in result.stdout
        # Let unsubscribe be processed before a new owner requests the lease.
        time.sleep(.05)
        result = subprocess.run(base + ['command', '--select', '12'], cwd=bridge,
                                capture_output=True, text=True, timeout=7)
        assert result.returncode == 0, result.stdout + result.stderr
        assert controller.selections == [12]
    finally:
        server.close()


def test_browser_full_geometry_selection_rejects_obsolete_binding(service):
    server, _ = service
    controller = Controller()
    server.bind(controller)
    old = server.revision
    assert server.select_from_view(server.session, old, 99)
    assert controller.selections == [99]
    assert not server.select_from_view(server.session, old, 100)
    server.bind(Controller())
    assert not server.select_from_view(server.session, old, 50)
    assert controller.selections == [99]
