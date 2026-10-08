import threading
from types import SimpleNamespace

import numpy as np
import pytest

from stable_audio_wanderer.cli.preprocess import compute_manual_navigation_features
from stable_audio_wanderer.runtime.audio_input import (
    AudioInputNavigation, CorpusMatcher, InputSettings, QueryResult,
)


def test_queries_use_corpus_fitted_transforms_and_raw_pitch_gate():
    rng = np.random.default_rng(4)
    raw_desc = rng.normal(size=(12, 35)).astype(np.float32)
    raw_desc[:, 33] = np.linspace(0, 1, 12)
    manual = compute_manual_navigation_features(raw_desc)
    mean = np.array([2, 5], dtype=np.float32)
    std = np.array([3, 7], dtype=np.float32)
    normalized = rng.normal(size=(12, 2)).astype(np.float32)
    corpus = dict(Z_concat=normalized, Z_mean=mean, Z_std=std,
                  file_offsets=[0, 12], latent_hz=10, paths=['source.wav'])
    matcher = CorpusMatcher(corpus, manual)
    for index in (0, 5, 11):
        assert matcher.query('descriptors', raw_desc[index])[0] == index
        assert matcher.query('latents', normalized[index] * std + mean)[0] == index
    with pytest.raises(ValueError):
        matcher.query('latents', [np.nan, 0])


def test_switch_dwell_silence_freshness_and_session_rejection():
    now = [10.0]
    nav = AudioInputNavigation(SimpleNamespace(settings=InputSettings()), clock=lambda: now[0])
    nav._stop.clear()
    nav._session = 2
    nav.level_db = -10
    nav.results['descriptors'] = QueryResult('descriptors', 2, 10, 4, .2, 8)
    assert nav.select(require_fresh=True) == 4
    nav.results['descriptors'] = QueryResult('descriptors', 2, 10, 5, .1, 8)
    now[0] += .05
    assert nav.select() == 4  # dwell
    nav.set_path('latents')
    assert nav.select() == 4  # no fallback query
    nav.results['latents'] = QueryResult('latents', 1, 10, 9, .1, 8)
    with pytest.raises(RuntimeError):
        nav.select(require_fresh=True)  # previous capture generation
    nav.results['latents'] = QueryResult('latents', 2, 10, 9, .1, 8)
    assert nav.select() == 9
    nav.level_db = -80
    nav.results['latents'] = QueryResult('latents', 2, 10, 2, .1, 8)
    assert nav.select() == 9  # silence holds
    now[0] = 13
    nav.level_db = -10
    assert nav.select() == 9  # stale holds
    nav.stop()
    assert nav.select() is None


def test_capture_ring_and_two_workers_consume_latest_windows(monkeypatch):
    import sounddevice as sd
    from test_onnx_transport import _wait_until
    seen = {'descriptors': [], 'latents': []}
    slow_entered, release = threading.Event(), threading.Event()

    def analyze(path, audio, rate):
        seen[path].append(audio[:, 0].copy())
        if path == 'latents' and len(seen[path]) == 1:
            slow_entered.set()
            assert release.wait(2)
        return 0, 0.0

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.closed = False
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            self.closed = True

    monkeypatch.setattr(sd, 'query_devices', lambda *a: dict(name='fake', default_samplerate=10))
    monkeypatch.setattr(sd, 'check_input_settings', lambda **kw: None)
    monkeypatch.setattr(sd, 'InputStream', Stream)
    nav = AudioInputNavigation(SimpleNamespace(settings=InputSettings(update_seconds=.01), analyze=analyze))
    nav.start()
    stream = nav._stream
    try:
        nav._capture(np.ones((10, 1), dtype=np.float32), 10, None, '')
        assert slow_entered.wait(2)
        _wait_until(lambda: len(seen['descriptors']) == 1)
        for value in range(2, 6):
            nav._capture(np.full((10, 1), value, dtype=np.float32), 10, None, '')
        _wait_until(lambda: len(seen['descriptors']) >= 2)
        release.set()
        _wait_until(lambda: len(seen['latents']) >= 2)
        assert np.all(seen['latents'][1] == 5)
        assert len(seen['latents']) == 2
        assert nav.results['latents'].session == nav._session
    finally:
        release.set()
        nav.close()
    assert stream.closed
    assert not nav._threads and not nav.results


def test_audio_input_transport_uses_source_windows_and_holds_across_switch():
    from test_onnx_transport import _controller, _wait_until
    controller, decoder, player = _controller()
    index = [5]
    closed = []
    controller.audio_input = SimpleNamespace(select=lambda **kwargs: index[0],
        state=lambda: {}, close=lambda: closed.append(True))
    assert controller.set_mode('audio_input')[0]
    assert not controller.set_window_controls({'mode': 'adaptive'})[0]
    request = controller._audio_input_request(1, 4)
    assert all(0 <= i < 6 for i in request.frame_indices)
    np.testing.assert_allclose(request.raw_latents,
        controller.Z_concat[list(request.frame_indices)] * controller.Z_std + controller.Z_mean)
    index[0] = None
    assert controller._audio_input_request(1, 4).frame_indices == request.frame_indices
    index[0] = 5
    assert controller.start()[0]
    _wait_until(lambda: len(player.writes) > 0)
    assert controller.nav.counter == 0
    controller.close()
    assert closed == [True]
