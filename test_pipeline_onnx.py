from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

from stable_audio_wanderer.runtime.pipeline_server import PipelineManager


class CapturingBroadcaster:
    def __init__(self):
        self.messages = []
        self.event = threading.Event()

    def broadcast_pipeline_message(self, data):
        self.messages.append(dict(data))
        self.event.set()

    def wait_for(self, message_type, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in reversed(self.messages):
                if message.get("type") == message_type:
                    return message
            self.event.wait(timeout=0.02)
            self.event.clear()
        raise AssertionError(f"No {message_type} message arrived")


class FakeAppDecoder:
    supported_windows = (2, 4, 8, 16, 32)
    default_window = 2
    info = SimpleNamespace(
        backend="onnxruntime",
        provider="CPUExecutionProvider",
        vae_id="same_s",
        resource_path=Path("/app/resources/same_s"),
    )


def test_start_lazily_loads_and_reuses_app_decoder(tmp_path, monkeypatch):
    corpus_a = tmp_path / "corpus-a"
    corpus_b = tmp_path / "corpus-b"
    corpus_a.mkdir()
    corpus_b.mkdir()
    decoder = FakeAppDecoder()
    calls = []

    import stable_audio_wanderer.vae.onnx_decoder as decoder_module

    def fake_load(**kwargs):
        calls.append(("load", kwargs))
        return decoder

    def fake_validate(path):
        calls.append(("validate", str(path)))

    monkeypatch.setattr(decoder_module, "load_same_s_app_decoder", fake_load)
    monkeypatch.setattr(decoder_module, "validate_same_s_corpus", fake_validate)

    pipeline = PipelineManager(decoder_resource_dir="/release/decoder")
    broadcaster = CapturingBroadcaster()
    pipeline.set_broadcaster(broadcaster)
    setup_calls = []
    pipeline.set_perform_setup_callback(
        lambda corpus, selected_decoder, config: setup_calls.append(
            (corpus, selected_decoder, config)
        )
    )

    first_config = {"corpus_dir": str(corpus_a), "decoder_window": 4}
    pipeline.handle_message({"type": "pipeline_start_perform", "config": first_config})
    assert pipeline.phase == "perform"
    assert calls == [("load", {
        "corpus_path": str(corpus_a),
        "resource_dir": "/release/decoder",
    })]
    assert setup_calls == [(str(corpus_a), decoder, first_config)]

    pipeline.phase = "idle"
    second_config = {"corpus_dir": str(corpus_b), "decoder_window": 2}
    pipeline.handle_message({"type": "pipeline_start_perform", "config": second_config})
    assert calls[-1] == ("validate", str(corpus_b))
    assert len([call for call in calls if call[0] == "load"]) == 1


def test_start_surfaces_resource_or_corpus_failure(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    import stable_audio_wanderer.vae.onnx_decoder as decoder_module

    monkeypatch.setattr(
        decoder_module,
        "load_same_s_app_decoder",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("not a SAME-S corpus")),
    )
    pipeline = PipelineManager()
    broadcaster = CapturingBroadcaster()
    pipeline.set_broadcaster(broadcaster)
    setup_calls = []
    pipeline.set_perform_setup_callback(lambda *args: setup_calls.append(args))

    pipeline.handle_message({
        "type": "pipeline_start_perform",
        "config": {"corpus_dir": str(corpus_dir), "decoder_window": 2},
    })
    assert pipeline.phase == "idle"
    assert not setup_calls
    assert "not a SAME-S corpus" in broadcaster.messages[-1]["error"]


def test_start_rejects_unsupported_window(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    import stable_audio_wanderer.vae.onnx_decoder as decoder_module
    monkeypatch.setattr(
        decoder_module, "load_same_s_app_decoder", lambda **_kwargs: FakeAppDecoder()
    )
    pipeline = PipelineManager()
    broadcaster = CapturingBroadcaster()
    pipeline.set_broadcaster(broadcaster)
    pipeline.set_perform_setup_callback(lambda *_args: None)
    pipeline.handle_message({
        "type": "pipeline_start_perform",
        "config": {"corpus_dir": str(corpus_dir), "decoder_window": 64},
    })
    assert pipeline.phase == "idle"
    assert "unavailable" in broadcaster.messages[-1]["error"]


def test_vae_release_only_empties_the_cache_for_the_model_device(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
        cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class MpsModel:
        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type="mps"))

    pipeline = PipelineManager()
    pipeline._vae = MpsModel()
    pipeline._preprocess_result = {"vae": pipeline._vae}
    pipeline._release_preprocessing_vae()

    assert pipeline._vae is None
    assert pipeline._preprocess_result["vae"] is None
    assert calls == ["mps"]
