"""Compatibility and backend independence of the prepared decoder boundary."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stable_audio_wanderer.runtime.decoder_transport import DecoderTransportController
from stable_audio_wanderer.runtime.onnx_transport import OnnxTransportController
from stable_audio_wanderer.vae import decoder_contract as contract
from stable_audio_wanderer.vae import onnx_decoder
from test_app_decoder_resource import _write_resource
from test_onnx_decoder import FakeOrt, _write_bundle
from test_onnx_transport import _controller, _wait_until


def test_existing_imports_preserve_type_and_exception_identity():
    import stable_audio_wanderer.vae as vae

    assert OnnxTransportController is DecoderTransportController
    for name in ("DecodedAudioWindow", "DecoderWindowMetadata", "DecoderRuntimeError"):
        assert getattr(onnx_decoder, name) is getattr(contract, name)
        assert getattr(vae, name) is getattr(contract, name)


def test_contract_and_transport_import_without_inference_backends():
    # A fresh interpreter prevents a pre-imported dependency hiding coupling.
    script = """
import importlib.abc
import sys
class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'stable_audio_3'}:
            raise AssertionError('Unexpected inference dependency: ' + fullname)
sys.meta_path.insert(0, RejectBackends())
from stable_audio_wanderer.vae.decoder_contract import LatentDecoder
from stable_audio_wanderer.runtime.decoder_transport import DecoderTransportController
from stable_audio_wanderer.runtime.onnx_transport import OnnxTransportController
assert OnnxTransportController is DecoderTransportController
"""
    subprocess.run([sys.executable, "-c", script], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)


@pytest.mark.parametrize("packaging", ["bundle", "resource"])
def test_both_onnx_loaders_implement_shared_contract_without_changing_preflight(tmp_path, packaging):
    ort = FakeOrt()
    if packaging == "bundle":
        bundle, _manifest = _write_bundle(tmp_path)
        decoder = onnx_decoder.SameSOnnxDecoder(bundle, ort_module=ort)
        assert len(ort.run_inputs) == len(decoder.supported_windows)
    else:
        decoder = onnx_decoder.SameSAppOnnxDecoder(_write_resource(tmp_path), ort_module=ort)
        assert ort.run_inputs == []  # App startup must remain lazy.

    assert isinstance(decoder, contract.LatentDecoder)
    assert isinstance(decoder.info, contract.DecoderInfo)
    latents = np.arange(2 * 256, dtype=np.float32).reshape(2, 256) / 512
    before = latents.copy()
    decoded = decoder.decode(latents)
    assert isinstance(decoded, contract.DecodedAudioWindow)
    assert isinstance(decoded.metadata, contract.DecoderWindowMetadata)
    assert decoded.audio.shape == (8192, 2)
    assert decoded.audio.dtype == np.float32
    assert np.isfinite(decoded.audio).all()
    np.testing.assert_array_equal(latents, before)
    with pytest.raises(contract.DecoderRuntimeError, match="float32"):
        decoder.decode(latents.astype(np.float64))


def test_transport_runs_and_transitions_without_onnx_or_bundle_metadata():
    controller, decoder, player = _controller()
    # The test double provides the shared API, with no ONNX inheritance or
    # package/bundle metadata. It exercises the actual transport worker and OLA.
    decoder.info = SimpleNamespace(
        backend="test-native", provider="TestDevice", vae_id="same_s",
        model_path=Path("/models/test-native.weights"),
    )
    assert isinstance(decoder, contract.LatentDecoder)
    assert isinstance(decoder.info, contract.DecoderInfo)
    assert isinstance(controller, DecoderTransportController)
    try:
        assert controller.start()[0]
        _wait_until(lambda: player.started)
        state = controller.get_extra_state()["decoder"]
        assert state["backend"] == "test-native"
        assert state["provider"] == "TestDevice"
        assert state["model_path"] == "/models/test-native.weights"
        assert "bundle_path" not in state and "resource_path" not in state
        assert controller.set_decoder_window(4)[0]
        _wait_until(lambda: 4 in decoder.calls)
        _wait_until(lambda: player.pending_generation is not None)
        # FakePlayer has no hardware callback to commit the prepared crossfade.
        with player._lock:
            player.current_generation = player.pending_generation
            player.pending_generation = None
        _wait_until(lambda: controller.get_extra_state()["decoder"]["selected_window"] == 4)
        decoder.fail_once()
        assert controller.set_decoder_window(8)[0]
        _wait_until(lambda: controller.get_extra_state()["transport"]["error"] is not None)
        assert "Decoder runtime failure (test-native)" in controller.get_extra_state()["transport"]["error"]
    finally:
        controller.stop()
    assert not player.started


@pytest.mark.parametrize("packaging", ["bundle", "resource"])
def test_onnx_transport_keeps_existing_provenance_fields(packaging):
    controller, decoder, _player = _controller()
    decoder.info.model_path = Path("/models/decoder.onnx")
    if packaging == "resource":
        del decoder.info.bundle_path
        decoder.info.resource_path = Path("/app/resources/same_s")
    state = controller.get_extra_state()["decoder"]
    assert state["backend"] == "onnxruntime"
    assert state["provider"] == "CPUExecutionProvider"
    if packaging == "bundle":
        assert state["bundle_path"] == "/tmp/test.sawbundle"
        assert "model_path" not in state and "resource_path" not in state
    else:
        assert state["resource_path"] == "/app/resources/same_s"
        assert state["model_path"] == "/models/decoder.onnx"
        assert "bundle_path" not in state
