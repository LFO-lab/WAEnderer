import json
import sys
import types

import numpy as np
import pytest

import bin.export_web_decoder as web_export
import bin.export_vst_bundle as vst_export


SOURCE_REVISION = "fbeb3dcf53a326e5682f38e22e7f740202d44232"


def test_release_export_stages_model_metadata_and_parity(tmp_path, monkeypatch):
    calls = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        model = kwargs["models_dir"] / "decoder" / "same_s_decoder_dynamic.onnx"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"onnx")
        report = kwargs["reports_dir"] / "decoder_parity.json"
        report.write_text('{"ok": true}\n', encoding="utf-8")
        return {"input_name": "latents", "output_name": "audio"}

    monkeypatch.setattr(web_export, "export_same_s_decoder_onnx", fake_export)
    metadata = web_export.export_web_decoder(
        tmp_path / "release", "same-s", source_revision=SOURCE_REVISION
    )

    assert metadata["supported_windows"] == [2, 4, 8, 16, 32]
    assert metadata["default_window"] == 2
    assert metadata["vae_id"] == "same_s"
    assert metadata["source_model"] == "stabilityai/SAME-S"
    assert metadata["source_revision"] == SOURCE_REVISION
    assert calls[0]["source_revision"] == SOURCE_REVISION
    assert len(metadata["model_sha256"]) == 64
    assert (tmp_path / "release" / "same_s_decoder_dynamic.onnx").read_bytes() == b"onnx"
    assert json.loads((tmp_path / "release" / "decoder.json").read_text()) == metadata
    samples = calls[0]["latent_samples"]
    assert tuple(samples) == (2, 4, 8, 16, 32)
    for window, sample in samples.items():
        assert sample.shape == (1, 256, window)
        assert sample.dtype == np.float32
        assert np.all(np.isfinite(sample))


def test_release_export_rejects_misleading_filesystem_placeholder(tmp_path):
    with pytest.raises(ValueError, match="registered model name 'same-s'"):
        web_export.export_web_decoder(
            tmp_path, "/path/to/same-s", source_revision=SOURCE_REVISION
        )


def test_release_export_requires_full_source_revision(tmp_path):
    with pytest.raises(ValueError, match="40-character Git commit SHA"):
        web_export.export_web_decoder(tmp_path, "same-s", source_revision="main")


def test_pinned_same_s_loader_uses_exact_revision(tmp_path, monkeypatch):
    calls = []
    snapshot = tmp_path / "snapshots" / SOURCE_REVISION
    snapshot.mkdir(parents=True)
    for filename in ("model_config.json", "model.safetensors"):
        (snapshot / filename).write_bytes(b"fixture")

    def fake_hf_hub_download(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(snapshot / filename)

    class FakeAutoencoder:
        def __init__(self):
            self.frozen = False

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            self.frozen = not enabled
            return self

    autoencoder = FakeAutoencoder()
    loading_utils = types.ModuleType("stable_audio_3.loading_utils")
    loading_utils.load_autoencoder = lambda config, ckpt, device: autoencoder
    stable_audio_3 = types.ModuleType("stable_audio_3")
    stable_audio_3.loading_utils = loading_utils
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "stable_audio_3", stable_audio_3)
    monkeypatch.setitem(sys.modules, "stable_audio_3.loading_utils", loading_utils)

    loaded = vst_export._load_pinned_same_s_autoencoder(SOURCE_REVISION)

    assert loaded is autoencoder
    assert autoencoder.frozen
    assert calls == [
        ("stabilityai/SAME-S", "model_config.json", SOURCE_REVISION),
        ("stabilityai/SAME-S", "model.safetensors", SOURCE_REVISION),
    ]
