from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from stable_audio_wanderer.vae.decoder_contract import DecoderRuntimeError, LatentDecoder
from stable_audio_wanderer.vae import same_s_weights as weights_module
from stable_audio_wanderer.vae import torch_decoder as native


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 256)
        self.decoder = torch.nn.Linear(256, 2)
        self.calls = []
        self.failure = None
        self.busy = 0
        self.peak = 0
        self.delay = 0

    def decode_audio(self, latents, *, chunked):
        self.busy += 1
        self.peak = max(self.peak, self.busy)
        try:
            self.calls.append((tuple(latents.shape), chunked, torch.is_inference_mode_enabled()))
            time.sleep(self.delay)
            if self.failure == "exception":
                raise RuntimeError("unsupported native operation")
            audio = self.decoder(latents.transpose(1, 2)).transpose(1, 2).repeat_interleave(4096, dim=2)
            if self.failure == "dtype":
                audio = audio.double()
            elif self.failure == "shape":
                audio = audio[..., :-1]
            elif self.failure == "nan":
                audio[..., 0] = float("nan")
            elif self.failure == "not_tensor":
                return None
            return audio
        finally:
            self.busy -= 1


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fake_loader(monkeypatch):
    model = SmallModel()
    model.encoder = None
    files = weights_module.SameSWeights(Path("config.json"), Path("model.safetensors"), "config-hash", "weights-hash")
    monkeypatch.setattr(native, "resolve_same_s_weights", lambda **kwargs: files)
    monkeypatch.setattr(native, "load_same_s_model", lambda files: model)
    monkeypatch.setattr(native.package_metadata, "distribution", lambda name: types.SimpleNamespace(
        version="test", read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "test-revision"}})))
    return model


def test_prepared_decoder_contract_warms_every_window_and_preserves_global_device(fake_loader):
    from stable_audio_wanderer import config
    device_before = config.DEVICE
    decoder = native.SameSTorchDecoder(device="cpu")
    assert isinstance(decoder, LatentDecoder)
    assert config.DEVICE == device_before
    assert decoder.info.device == decoder.info.provider == "cpu"
    assert decoder.info.backend == "pytorch"
    assert decoder.info.source_revision == weights_module.SOURCE_REVISION
    assert decoder.info.library_revision == "test-revision"
    assert decoder.info.torch_version == str(torch.__version__)
    assert tuple(decoder.info.warmup_decode_ms) == tuple(range(2, 33, 2))
    assert [shape[-1] for shape, _, _ in fake_loader.calls] == list(range(2, 33, 2))
    assert all(not chunked and inference for _, chunked, inference in fake_loader.calls)
    assert not fake_loader.training and not any(p.requires_grad for p in fake_loader.parameters())
    latents = np.arange(4 * 256, dtype=np.float32).reshape(4, 256)[::2]
    latents.setflags(write=False)
    before = latents.copy()
    decoded = decoder.decode(latents)
    assert decoded.audio.shape == (8192, 2)
    assert decoded.audio.dtype == np.float32 and decoded.audio.flags.c_contiguous
    assert decoded.audio.flags.owndata and np.isfinite(decoded.audio).all()
    assert decoded.decode_time_ms > 0
    np.testing.assert_array_equal(latents, before)
    expected = before @ fake_loader.decoder.weight.detach().numpy().T + fake_loader.decoder.bias.detach().numpy()
    np.testing.assert_allclose(decoded.audio, np.repeat(expected, 4096, axis=0), rtol=1e-5, atol=1e-5)
    with pytest.raises(TypeError):
        decoder.info.windows[2] = None


@pytest.mark.parametrize("bad", [np.zeros((2,256),dtype=np.float64), np.zeros((2,255),dtype=np.float32),
                                np.zeros((3,256),dtype=np.float32), np.full((2,256),np.nan,dtype=np.float32)])
def test_invalid_input_is_rejected_before_inference(fake_loader, bad):
    decoder = native.SameSTorchDecoder(device="cpu")
    count = len(fake_loader.calls)
    with pytest.raises(DecoderRuntimeError):
        decoder.decode(bad)
    assert len(fake_loader.calls) == count


@pytest.mark.parametrize("failure", ["dtype", "shape", "nan", "not_tensor", "exception"])
def test_output_or_operation_failure_is_visible_without_fallback(fake_loader, failure):
    decoder = native.SameSTorchDecoder(device="cpu")
    fake_loader.failure = failure
    with pytest.raises(DecoderRuntimeError):
        decoder.decode(np.zeros((2,256),dtype=np.float32))
    with pytest.raises(weights_module.NativeDecoderLoadError, match="preparation failed"):
        native.SameSTorchDecoder(device="cpu")


def test_overlapping_calls_are_serialized_and_timing_includes_wait(fake_loader, monkeypatch):
    decoder = native.SameSTorchDecoder(device="cpu")
    fake_loader.delay = .025
    barrier = threading.Barrier(2)
    def decode():
        barrier.wait(timeout=2)
        return decoder.decode(np.zeros((2,256),dtype=np.float32))
    monkeypatch.setattr(torch, "manual_seed", lambda *_: pytest.fail("Runtime must not reset RNG"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: decode(), range(2)))
    assert fake_loader.peak == 1
    assert min(r.decode_time_ms for r in results) >= 25
    assert max(r.decode_time_ms for r in results) >= 50


@pytest.mark.parametrize("device", ["auto", "xpu", "cpu:1", "cuda:99", "mps:3"])
def test_invalid_or_unavailable_devices_fail_before_weights(device, monkeypatch):
    monkeypatch.setattr(native, "resolve_same_s_weights", lambda **_: pytest.fail("Weights loaded before device validation"))
    with pytest.raises(weights_module.NativeDecoderLoadError):
        native.SameSTorchDecoder(device=device)


def test_cuda_resolution_checks_index_and_mps_rejects_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert str(native._resolve_device(torch, "cuda")) == "cuda:1"
    assert str(native._resolve_device(torch, "cuda:0")) == "cuda:0"
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(weights_module.NativeDecoderLoadError, match="fallback"):
        native._resolve_device(torch, "mps")


def test_pinned_resolver_checks_revision_hash_and_offline_mode(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / weights_module.SOURCE_REVISION
    snapshot.mkdir(parents=True)
    for filename in ("model_config.json", "model.safetensors"):
        (snapshot / filename).write_bytes(b"fixture")
    expected = hashlib.sha256(b"fixture").hexdigest()
    monkeypatch.setattr(weights_module, "CONFIG_SHA256", expected)
    monkeypatch.setattr(weights_module, "WEIGHTS_SHA256", expected)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot / kwargs["filename"])
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    files = weights_module.resolve_same_s_weights()
    assert files.source_revision == weights_module.SOURCE_REVISION
    assert all(c["revision"] == files.source_revision and c["local_files_only"] for c in calls)
    (snapshot / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(weights_module.NativeDecoderLoadError, match="SHA-256"):
        weights_module.resolve_same_s_weights()


def test_native_reference_matches_packaged_decoder_identity_and_windows():
    path = Path(__file__).resolve().parents[1] / "stable_audio_wanderer/resources/same_s/decoder.json"
    info = json.loads(path.read_text())
    assert info["source_revision"] == weights_module.SOURCE_REVISION
    assert info["source_model"] == weights_module.SOURCE_MODEL
    assert info["supported_windows"] == list(native.SUPPORTED_WINDOWS)


def test_missing_native_library_is_an_actionable_load_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "stable_audio_3", None)
    monkeypatch.delitem(sys.modules, "stable_audio_3.factory", raising=False)
    files = weights_module.SameSWeights(tmp_path / "config", tmp_path / "weights", "", "")
    with pytest.raises(weights_module.NativeDecoderLoadError, match="compatible torch"):
        weights_module.load_same_s_model(files)


def test_unavailable_gpu_is_not_replaced_by_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(native, "resolve_same_s_weights", lambda **_: pytest.fail("Unexpected model load"))
    for device in ("cuda", "mps"):
        with pytest.raises(weights_module.NativeDecoderLoadError, match="unavailable"):
            native.SameSTorchDecoder(device=device)


@pytest.mark.parametrize("corruption", [None, "missing", "extra", "shape"])
def test_native_checkpoint_is_loaded_strictly_before_encoder_removal(tmp_path, monkeypatch, corruption):
    model = SmallModel()
    state = model.state_dict()
    if corruption == "missing":
        del state["decoder.bias"]
    elif corruption == "extra":
        state["unrecognized"] = torch.zeros(1)
    elif corruption == "shape":
        state["decoder.bias"] = torch.zeros(3)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {}, "sample_rate": 44100}))
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, str(checkpoint))
    package = types.ModuleType("stable_audio_3")
    factory = types.ModuleType("stable_audio_3.factory")
    factory.create_autoencoder_from_config = lambda *_: SmallModel()
    monkeypatch.setitem(sys.modules, "stable_audio_3", package)
    monkeypatch.setitem(sys.modules, "stable_audio_3.factory", factory)
    files = weights_module.SameSWeights(config, checkpoint, "config", "weights")
    if corruption:
        with pytest.raises(weights_module.NativeDecoderLoadError, match="strictly load"):
            weights_module.load_same_s_model(files)
    else:
        loaded = weights_module.load_same_s_model(files)
        assert loaded.encoder is None
        assert next(loaded.parameters()).device.type == "cpu"
        torch.testing.assert_close(loaded.decoder.weight, model.decoder.weight)
