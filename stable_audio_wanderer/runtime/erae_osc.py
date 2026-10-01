"""Optional, browser-independent Erae OSC service for the unified launcher."""
from collections import OrderedDict
import logging
import socket
import threading
import time
import uuid

from .erae_protocol import MAX_PACKET, MAX_SEQ, decode, encode

log = logging.getLogger(__name__)


class EraeOscServer:
    def __init__(self, host='127.0.0.1', port=9000, *, fps=30, lease_seconds=3):
        if not 1 <= fps <= 120 or not 0.1 <= lease_seconds <= 60:
            raise ValueError('invalid feedback rate or lease')
        self.interval = 1 / fps
        self.lease_seconds = lease_seconds
        self.session = uuid.uuid4().hex
        self.revision = ''
        self.controller = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._subscriber = None
        self._last_command = -1
        self._results = OrderedDict()
        self._sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind((host, port))
            self._socket.settimeout(min(self.interval, 0.1))
        except Exception:
            self._socket.close()
            raise
        self.address = self._socket.getsockname()

    def bind(self, controller):
        """Detach before closing/replacing a controller; serialized with dispatch."""
        with self._lock:
            self.controller = controller
            self.revision = uuid.uuid4().hex if controller is not None else ''
            # Keep command watermark/cache: old commands must not run on rebind.
            if self._subscriber:
                self._snapshot()

    def select_from_view(self, session, revision, index):
        """Revision-checked browser picking, serialized with corpus replacement."""
        with self._lock:
            if (session != self.session or revision != self.revision or self.controller is None
                    or type(index) is not int or not 0 <= index < self.controller.Z_concat.shape[0]):
                return False
            self.controller.nav.set_cursor_index(index)
            return True

    def start(self):
        if self._thread is not None:
            raise RuntimeError('OSC server already started')
        self._thread = threading.Thread(target=self._run, name='erae-osc', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self.controller = None
            self._subscriber = None
            self._socket.close()

    def _send(self, kind, *args):
        if self._subscriber:
            try:
                self._socket.sendto(encode(kind, *args), self._subscriber['reply'])
            except OSError as exc:
                log.debug('Erae feedback send failed: %s', exc)

    def _snapshot(self):
        sub = self._subscriber
        controller = self.controller
        ready = controller is not None
        count = int(controller.Z_concat.shape[0]) if ready else 0
        windows = tuple(int(t) for t in controller.latent_decoder.supported_windows) if ready else ()
        self._send('capabilities', self.session, sub['id'], self.revision, self._sequence, int(ready), count, *windows)
        self._state()

    def _state(self):
        if not self._subscriber:
            return
        if self._sequence >= MAX_SEQ:
            # Session rotation, never ambiguous integer wraparound.
            self.session = uuid.uuid4().hex
            self._sequence = 0
        self._sequence += 1
        values = dict(running=False, mode='', valid=False, index=-1, generation=-1,
                      requested_window=-1, active_window=-1, transition='not_ready',
                      window_mode='', error='')
        if self.controller is not None:
            try:
                values.update(self.controller.get_erae_state())
            except Exception:
                log.exception('Could not read Erae playback state')
                values.update(transition='error', error='playback state unavailable')
        self._send('state', self.session, self._subscriber['id'], self.revision,
                   self._sequence, int(self.controller is not None), int(values['running']),
                   values['mode'], int(values['valid']), values['index'], values['generation'],
                   values['requested_window'], values['active_window'], values['transition'],
                   values['window_mode'], str(values['error'])[:128])

    def _dispatch(self, kind, args, source):
        now = time.monotonic()
        sub = self._subscriber
        if sub and now - sub['seen'] > self.lease_seconds:
            self._subscriber = sub = None
        if kind == 'subscribe':
            client, port = args
            if sub and (sub['id'] != client or sub['source'] != source):
                return  # One live owner; do not steal it.
            if not sub:
                self._results.clear()
                self._last_command = -1
            self._subscriber = dict(id=client, source=source, reply=(source[0], port), seen=now)
            self._snapshot()
            return
        if not sub or args[0] != sub['id'] or source != sub['source']:
            return
        if kind == 'unsubscribe':
            self._subscriber = None
        elif kind == 'snapshot':
            self._snapshot()
        elif kind in ('select', 'window'):
            client, sequence, revision, value = args
            fingerprint = (kind, revision, value)
            if sequence in self._results:
                previous, result = self._results[sequence]
                if fingerprint != previous:
                    result = (0, value, 'sequence reused with different command')
            elif sequence <= self._last_command:
                result = (0, value, 'stale command sequence')
            else:
                self._last_command = sequence
                try:
                    if self.controller is None:
                        result = (0, value, 'engine not ready')
                    elif revision != self.revision:
                        result = (0, value, 'corpus revision mismatch')
                    elif kind == 'select':
                        if not 0 <= value < self.controller.Z_concat.shape[0]:
                            result = (0, value, 'frame out of range')
                        else:
                            self.controller.nav.set_cursor_index(value)
                            result = (1, value, 'selection accepted')
                    elif value not in self.controller.latent_decoder.supported_windows:
                        result = (0, value, 'unsupported window')
                    else:
                        ok, reason = self.controller.set_decoder_window(value)
                        result = (int(ok), value, str(reason)[:128])
                except Exception:
                    log.exception('Erae command failed')
                    result = (0, value, 'engine command failed')
                self._results[sequence] = (fingerprint, result)
                if len(self._results) > 128:
                    self._results.popitem(last=False)
            self._send('result', self.session, client, sequence, *result)

    def _run(self):
        next_state = time.monotonic()
        while not self._stop.is_set():
            try:
                packet, source = self._socket.recvfrom(MAX_PACKET + 1)
                kind, args = decode(packet)
                if kind in ('subscribe', 'snapshot', 'unsubscribe', 'select', 'window'):
                    with self._lock:
                        self._dispatch(kind, args, source)
            except socket.timeout:
                pass
            except ValueError:
                pass
            except OSError:
                if not self._stop.is_set():
                    log.exception('Erae OSC receive failed')
                break
            now = time.monotonic()
            if now >= next_state:
                with self._lock:
                    if self._subscriber and now - self._subscriber['seen'] > self.lease_seconds:
                        self._subscriber = None
                    self._state()
                next_state = now + self.interval
