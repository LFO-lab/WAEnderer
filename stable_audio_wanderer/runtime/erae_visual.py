"""Optional visual-state relay. All mutable relay state belongs to the WS loop."""
import asyncio
import json
import math
import time


def valid_view(view):
    if not isinstance(view, dict):
        return False
    for key in ('yaw', 'pitch', 'distance', 'width', 'height'):
        v = view.get(key)
        if type(v) not in (int, float) or not math.isfinite(v):
            return False
    if not (.08 <= view['distance'] <= 8 and 0 < view['width'] <= 32768
            and 0 < view['height'] <= 32768 and abs(view['yaw']) <= 1e9 and abs(view['pitch']) <= 1e9):
        return False
    for colours in ([view.get('colour_a')], [view.get('colour_b')], view.get('files')):
        if not isinstance(colours, list) or not 1 <= len(colours) <= 256:
            return False
        for rgb in colours:
            if not isinstance(rgb, list) or len(rgb) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 255 for v in rgb):
                return False
    return True


class VisualRelay:
    def __init__(self):
        self.identity = None
        self.geometry = ()
        self.peers = {}
        self.owner = None
        self.generation = 0
        self.view = None
        self.view_sequence = -1
        self.owner_seen = 0

    def bind(self, identity, packets=()):
        self.identity, self.geometry = identity, packets
        self.view, self.view_sequence = None, -1
        self.generation += 1
        for peer in self.peers.values():
            peer['queue'].clear()
            self.snapshot(peer)
        self.announce_owner()

    def queue(self, peer, topic, message):
        peer['queue'][topic] = message
        peer['event'].set()

    def broadcast(self, topic, message):
        for peer in self.peers.values():
            self.queue(peer, topic, message)

    def snapshot(self, peer):
        if self.identity:
            self.queue(peer, 'geometry', self.geometry)
            if self.view and time.monotonic()-self.owner_seen < 2:
                self.queue(peer, 'view', self.view)
        else:
            self.queue(peer, 'state', {'type': 'erae_unavailable'})

    def preview_subscription(self):
        enabled = any(peer['role'] == 'browser' for peer in self.peers.values())
        for peer in self.peers.values():
            if peer['role'] == 'bridge':
                self.queue(peer, 'preview_subscription', dict(type='erae_preview_subscription', enabled=enabled))

    def announce_owner(self):
        for ws, peer in self.peers.items():
            self.queue(peer, 'owner', dict(type='erae_owner', owner=self.generation,
                       available=self.owner is not None, mine=ws is self.owner))

    async def writer(self, ws, peer):
        try:
            while True:
                await peer['event'].wait()
                while peer['queue']:
                    topic = next(iter(peer['queue']))
                    message = peer['queue'].pop(topic)
                    if topic == 'geometry':
                        identity = self.identity
                        for packet in message:
                            if identity != self.identity or 'geometry' in peer['queue']:
                                break
                            await asyncio.wait_for(ws.send(packet), 2)
                    else:
                        await asyncio.wait_for(ws.send(json.dumps(message, allow_nan=False)), 2)
                peer['event'].clear()
        except Exception:
            # A slow/broken visual consumer must not block the broadcaster.
            await ws.close()

    def remove(self, ws):
        peer = self.peers.pop(ws, None)
        if peer:
            peer['task'].cancel()
            self.preview_subscription()
        if ws is self.owner:
            self.owner, self.view = None, None
            self.generation += 1
            self.announce_owner()

    async def handle(self, ws, data):
        kind = data.get('type', '')
        if not kind.startswith('erae_'):
            return False
        if kind == 'erae_subscribe':
            if data.get('version') != 1 or data.get('role') not in ('browser', 'bridge'):
                return True
            if ws not in self.peers:
                peer = dict(role=data['role'], queue={}, event=asyncio.Event(), last_preview=0, last_snapshot=0)
                self.peers[ws] = peer
                peer['task'] = asyncio.create_task(self.writer(ws, peer))
            self.snapshot(self.peers[ws])
            self.preview_subscription()
            self.announce_owner()
            return True
        peer = self.peers.get(ws)
        if not peer:
            return True
        if kind == 'erae_snapshot' and time.monotonic()-peer['last_snapshot'] >= 1:
            peer['last_snapshot'] = time.monotonic()
            self.snapshot(peer)
            self.announce_owner()
        elif kind == 'erae_claim' and peer['role'] == 'browser':
            expired = time.monotonic()-self.owner_seen >= 2
            if self.owner is None or self.owner is ws or data.get('takeover') is True or expired:
                if self.owner is not ws:
                    self.generation += 1
                    self.view, self.view_sequence = None, -1
                self.owner, self.owner_seen = ws, time.monotonic()
                self.announce_owner()
        elif kind == 'erae_release' and ws is self.owner:
            self.owner, self.view = None, None
            self.generation += 1
            self.announce_owner()
        elif kind == 'erae_view' and ws is self.owner and self.identity:
            if ((data.get('session'), data.get('corpus')) != self.identity
                    or data.get('owner') != self.generation or not valid_view(data.get('view'))):
                return True
            revision = data.get('revision')
            if type(revision) is not int or revision < 0 or revision < self.view_sequence:
                return True
            if revision == self.view_sequence and self.view and data['view'] != self.view['view']:
                return True
            self.view_sequence, self.owner_seen = revision, time.monotonic()
            self.view = dict(type='erae_view', session=self.identity[0], corpus=self.identity[1],
                             owner=self.generation, revision=revision, view=data['view'])
            self.broadcast('view', self.view)
        elif kind == 'erae_preview' and peer['role'] == 'bridge':
            if time.monotonic()-peer['last_preview'] < 1/30:
                return True
            pixels = data.get('pixels')
            if (not isinstance(pixels, list) or len(pixels) != 768 or any(
                not isinstance(p, list) or len(p) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in p)
                for p in pixels)):
                return True
            # Status text is rendered with textContent, never HTML.
            packet = dict(type='erae_preview', pixels=pixels, status=str(data.get('status', ''))[:256])
            peer['last_preview'] = time.monotonic()
            for target in self.peers.values():
                if target['role'] == 'browser':
                    self.queue(target, 'preview', packet)
        return True


def geometry_packets(session, corpus, points, colours, file_ids):
    """Build once on the setup thread, never in the audio or touch callback."""
    count = len(points)
    if not 0 < count <= 1_000_000 or len(file_ids) != count:
        raise ValueError('visual corpus must contain 1..1000000 points and matching file IDs')
    if colours is not None and len(colours) != count:
        raise ValueError('visual colour count mismatch')
    packets = [json.dumps(dict(type='erae_geometry_begin', version=1,
                              session=session, corpus=corpus, count=count))]
    for offset in range(0, count, 512):
        rows = [[i, *[float(v) for v in points[i][:3]],
                 None if colours is None else float(colours[i]), int(file_ids[i])]
                for i in range(offset, min(count, offset+512))]
        packets.append(json.dumps(dict(type='erae_geometry_chunk', offset=offset, rows=rows), allow_nan=False))
    packets.append(json.dumps(dict(type='erae_geometry_end')))
    return tuple(packets)
