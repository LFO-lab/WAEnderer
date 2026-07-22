import json

import numpy as np
import pytest

import bin.export_web_decoder as web_export


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
    metadata = web_export.export_web_decoder(tmp_path / "release", "same-s")

    assert metadata["supported_windows"] == [2, 4, 8, 16, 32]
    assert metadata["default_window"] == 2
    assert metadata["vae_id"] == "same_s"
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
        web_export.export_web_decoder(tmp_path, "/path/to/same-s")
