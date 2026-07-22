import json
from pathlib import Path

import numpy as np
import pytest

from stable_audio_wanderer.vae.onnx_decoder import (
    DecoderBundleError,
    SameSAppOnnxDecoder,
)
from test_onnx_decoder import FakeOrt, _write_corpus


def _write_resource(tmp_path, *, windows=(2, 4, 8, 16, 32), **updates):
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "same_s_decoder_dynamic.onnx").write_bytes(b"fake model")
    metadata = {
        "format_version": "same_s.web_decoder.v1",
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "vae_id": "same_s",
        "model": "same_s_decoder_dynamic.onnx",
        "input_name": "latents",
        "output_name": "audio",
        "sample_rate": 44100,
        "channels": 2,
        "latent_dim": 256,
        "samples_per_latent": 4096,
        "supported_windows": list(windows),
        "default_window": 2,
        "ola_mode": "full_overlap_add",
    }
    metadata.update(updates)
    (resource / "decoder.json").write_text(json.dumps(metadata), encoding="utf-8")
    return resource


def test_app_resource_load_is_cpu_only_and_does_not_warm_all_windows(tmp_path):
    resource = _write_resource(tmp_path)
    corpus = _write_corpus(tmp_path)
    ort = FakeOrt()
    decoder = SameSAppOnnxDecoder(resource, corpus_path=corpus, ort_module=ort)

    assert decoder.supported_windows == (2, 4, 8, 16, 32)
    assert decoder.default_window == 2
    assert ort.run_inputs == []
    _path, options, providers = ort.session_args
    assert providers == ["CPUExecutionProvider"]
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1

    decoded = decoder.decode(np.zeros((2, 256), dtype=np.float32))
    assert decoded.audio.shape == (8192, 2)
    assert len(ort.run_inputs) == 1


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"vae_id": "stable_audio_open"}, "vae_id"),
        ({"supported_windows": [4, 8]}, "default_window"),
        ({"supported_windows": [2, 3]}, "sorted subset"),
        ({"model": "../escape.onnx"}, "local .onnx"),
    ],
)
def test_bad_resource_metadata_fails_before_session(tmp_path, updates, match):
    resource = _write_resource(tmp_path, **updates)
    ort = FakeOrt()
    with pytest.raises(DecoderBundleError, match=match):
        SameSAppOnnxDecoder(resource, ort_module=ort)
    assert ort.session_args is None


def test_non_same_s_corpus_fails_before_resource_load(tmp_path):
    resource = _write_resource(tmp_path)
    corpus = _write_corpus(tmp_path, vae_id="stable_audio_open")
    ort = FakeOrt()
    with pytest.raises(DecoderBundleError, match="incompatible"):
        SameSAppOnnxDecoder(resource, corpus_path=corpus, ort_module=ort)
    assert ort.session_args is None


def test_missing_resource_is_visible(tmp_path):
    with pytest.raises(DecoderBundleError, match="metadata is missing"):
        SameSAppOnnxDecoder(tmp_path / "absent", ort_module=FakeOrt())
