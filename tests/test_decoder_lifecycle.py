import gc
import json
import threading
from types import SimpleNamespace
import weakref

import pytest

from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
from stable_audio_wanderer.vae import decoder_factory as factory
from test_app_decoder_resource import _write_resource
from test_pipeline_onnx import CapturingBroadcaster
from test_onnx_transport import _controller, _wait_until


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    events, refs = [], []
    controls = {"load_failure": False, "setup_failure": False}
    def validate(path):
        events.append("validate")
        if path.endswith("invalid"):
            raise ValueError("incompatible corpus")
    monkeypatch.setattr("stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec", validate)
    def select(config, **kwargs):
        return factory.DecoderSelection(config.get("decoder_backend", "onnxruntime"),
                                        config.get("decoder_device", "cpu"),
                                        (config.get("revision", "first"),))
    monkeypatch.setattr(factory, "select_decoder", select)
    class Decoder:
        supported_windows = (2, 4)
        closed = False
        def __init__(self, selection):
            self.info = SimpleNamespace(backend=selection.backend, provider=selection.device,
                                        device=selection.device, vae_id="same_s")
        def close(self):
            if not self.closed:
                self.closed = True
                events.append("release:" + self.info.backend)
    def create(selection):
        events.append("load:" + selection.backend)
        if controls["load_failure"]:
            raise RuntimeError("model load failed")
        decoder = Decoder(selection)
        refs.append(weakref.ref(decoder))
        return decoder
    monkeypatch.setattr(factory, "create_decoder", create)
    pipeline = PipelineManager()
    broadcaster = CapturingBroadcaster()
    pipeline.set_broadcaster(broadcaster)
    def setup(path, decoder, config):
        events.append("setup:" + decoder.info.backend)
        if controls["setup_failure"]:
            raise RuntimeError("setup failed")
        return SimpleNamespace(close=lambda: events.append("drain"))
    pipeline.set_perform_setup_callback(setup)
    pipeline.set_perform_teardown_callback(lambda: events.append("detach"))
    def start(**config):
        pipeline.handle_message({"type": "pipeline_start_perform", "config": {"corpus_dir": str(corpus), **config}})
    def stop():
        pipeline.handle_message({"type": "pipeline_stop_perform"})
    yield SimpleNamespace(p=pipeline, b=broadcaster, start=start, stop=stop,
                          events=events, controls=controls, refs=refs, corpus=corpus)
    pipeline.close()


def test_stop_perform_drains_and_identical_restart_revalidates_and_reuses(lifecycle):
    x = lifecycle
    x.start()
    first = x.p._app_decoder
    assert x.p.phase == "perform"
    x.stop()
    assert x.p.phase == "idle" and x.events[-2:] == ["detach", "drain"]
    assert not first.closed
    assert x.b.messages[-1]["decoder"] is None
    x.start()
    assert x.p._app_decoder is first
    assert x.events.count("validate") == 2 and x.events.count("load:onnxruntime") == 1


@pytest.mark.parametrize("change", [{"decoder_backend": "pytorch", "decoder_device": "mps:0"},
                                  {"decoder_device": "cuda:1"}, {"revision": "new-weights"}])
def test_changed_selection_releases_before_loading_replacement(lifecycle, change):
    x = lifecycle
    x.start()
    first = x.p._app_decoder
    x.stop()
    x.events.clear()
    x.start(**change)
    assert first.closed and x.p._app_decoder is not first
    assert x.events.index("release:onnxruntime") < next(i for i,e in enumerate(x.events) if e.startswith("load:"))
    assert x.b.messages[-1]["decoder"]["backend"] == change.get("decoder_backend", "onnxruntime")


def test_reconfiguration_requires_exit_from_perform_even_when_audio_stopped(lifecycle):
    x = lifecycle
    x.start()
    decoder = x.p._app_decoder
    x.start(decoder_backend="pytorch")
    assert x.p._app_decoder is decoder and x.p.phase == "perform"
    assert "pipeline_stop_perform" in x.b.messages[-1]["error"]


@pytest.mark.parametrize("failure", ["load_failure", "setup_failure"])
def test_failed_replacement_is_idle_has_no_active_decoder_and_can_retry(lifecycle, failure):
    x = lifecycle
    x.start()
    x.stop()
    x.controls[failure] = True
    x.start(decoder_backend="pytorch", decoder_device="mps:0")
    assert x.p.phase == "idle" and x.p._app_decoder is None
    assert x.b.messages[-1]["decoder"] is None and "failed" in x.b.messages[-1]["error"]
    x.controls[failure] = False
    x.start()
    assert x.p.phase == "perform" and x.p._perform_error is None


def test_incompatible_corpus_is_rejected_even_with_cached_decoder(lifecycle):
    x = lifecycle
    x.start()
    x.stop()
    invalid = x.corpus.parent / "invalid"
    invalid.mkdir()
    x.start(corpus_dir=str(invalid))
    assert x.p.phase == "idle" and x.p._app_decoder is None
    assert "incompatible corpus" in x.b.messages[-1]["error"]


def test_preprocessing_model_is_released_before_loading_performance_decoder(lifecycle, monkeypatch):
    x = lifecycle
    monkeypatch.setattr(x.p, "_release_preprocessing_vae", lambda: x.events.append("release-encoder"))
    x.start(decoder_backend="pytorch", decoder_device="mps:0")
    assert x.events.index("release-encoder") < x.events.index("load:pytorch")


def test_repeated_switches_keep_only_one_decoder_and_shutdown_drops_it(lifecycle):
    x = lifecycle
    for index in range(12):
        x.start(decoder_backend="pytorch" if index % 2 else "onnxruntime")
        x.stop()
        gc.collect()
        assert sum(ref() is not None for ref in x.refs) == 1
    x.p.close()
    gc.collect()
    assert all(ref() is None for ref in x.refs)
    x.start()
    assert x.p.phase == "closed" and x.p._app_decoder is None


def test_stop_waits_for_inflight_decode_and_closed_controller_cannot_restart(lifecycle):
    x = lifecycle
    controller, decoder, player = _controller()
    x.p.set_perform_setup_callback(lambda *_: controller)
    x.start()
    # The controller's fake decoder blocks a real transport producer.
    decoder.block_once()
    controller.start()
    assert decoder.decode_entered.wait(2)
    stopped = threading.Event()
    thread = threading.Thread(target=lambda: (x.stop(), stopped.set()))
    thread.start()
    try:
        assert not stopped.wait(.05)
        decoder.decode_release.set()
        thread.join(2)
        assert stopped.is_set() and controller._decode_thread is None
        assert player.closed and not player.started and not player.buffers
        assert controller.start() == (False, "transport is closed")
    finally:
        decoder.decode_release.set()
        thread.join(2)


def test_drain_failure_keeps_model_owned_until_retry(lifecycle):
    x = lifecycle
    x.start()
    decoder = x.p._app_decoder
    x.p._perform_controller.close = lambda: (_ for _ in ()).throw(RuntimeError("cannot drain"))
    x.stop()
    assert x.p.phase == "error" and x.p._app_decoder is decoder and not decoder.closed
    x.p._perform_controller.close = lambda: None
    x.stop()
    assert x.p.phase == "idle"


def test_audio_stops_before_pending_candidate_finishes():
    controller, decoder, player = _controller()
    stopped = threading.Event()
    thread = None
    try:
        controller.start()
        _wait_until(lambda: player.started)
        decoder.block_once()
        controller.set_decoder_window(4)
        assert decoder.decode_entered.wait(2)
        thread = threading.Thread(target=lambda: (controller.close(), stopped.set()))
        thread.start()
        _wait_until(lambda: not player.started)
        assert not stopped.is_set()
        decoder.decode_release.set()
        thread.join(2)
        assert stopped.is_set() and not player.buffers
    finally:
        decoder.decode_release.set()
        if thread is not None:
            thread.join(2)
        controller.close()


def test_onnx_cache_identity_tracks_model_and_metadata_content(tmp_path):
    root = _write_resource(tmp_path)
    initial = factory.select_decoder({}, resource_dir=root)
    assert initial.backend == "onnxruntime" and initial.device == "cpu"
    (root / "same_s_decoder_dynamic.onnx").write_bytes(b"replacement")
    changed = factory.select_decoder({}, resource_dir=root)
    assert changed != initial
    info = json.loads((root / "decoder.json").read_text())
    info["supported_windows"] = [2,4]
    (root / "decoder.json").write_text(json.dumps(info))
    assert factory.select_decoder({}, resource_dir=root) != changed


@pytest.mark.parametrize("config", [{"decoder_backend": "unknown"},
                                  {"decoder_device": "cuda"},
                                  {"decoder_backend": "pytorch"}])
def test_factory_rejects_invalid_selection_without_fallback(config):
    with pytest.raises((ValueError, RuntimeError)):
        factory.select_decoder(config)
