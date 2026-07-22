import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from stable_audio_wanderer.vae.onnx_decoder import (
    DecoderBundleError,
    DecoderRuntimeError,
    SameSOnnxDecoder,
    preflight_same_s_decoder_bundle,
)


class _NodeArg:
    def __init__(self, name, shape, node_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = node_type


class _SessionOptions:
    pass


class _GraphOptimizationLevel:
    ORT_ENABLE_BASIC = object()


class FakeOrt:
    SessionOptions = _SessionOptions
    GraphOptimizationLevel = _GraphOptimizationLevel

    def __init__(self):
        self.input_name = "latents"
        self.output_name = "audio"
        self.input_shape = [1, 256, "latent_time"]
        self.output_shape = [1, 2, "audio_time"]
        self.input_type = "tensor(float)"
        self.output_type = "tensor(float)"
        self.providers = ["CPUExecutionProvider"]
        self.output_dtype = np.float32
        self.output_length_delta = 0
        self.non_finite = False
        self.runtime_error = None
        self.session_args = None
        self.run_inputs = []

    def InferenceSession(self, model_path, *, sess_options, providers):
        self.session_args = (model_path, sess_options, providers)
        return self

    def get_inputs(self):
        return [_NodeArg(self.input_name, self.input_shape, self.input_type)]

    def get_outputs(self):
        return [_NodeArg(self.output_name, self.output_shape, self.output_type)]

    def get_providers(self):
        return list(self.providers)

    def run(self, output_names, inputs):
        if self.runtime_error is not None:
            raise self.runtime_error
        model_input = np.asarray(inputs[self.input_name])
        self.run_inputs.append(model_input.copy())
        window = model_input.shape[2]
        length = window * 4096 + self.output_length_delta
        output = np.zeros((1, 2, length), dtype=self.output_dtype)
        output[:, 0, :] = 0.25
        output[:, 1, :] = -0.5
        if self.non_finite:
            output[0, 0, 0] = np.nan
        return [output]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_entry(window, model_hash):
    hop = (window + 1) // 2
    return {
        "path": "models/decoder/same_s_decoder_dynamic.onnx",
        "sha256": model_hash,
        "latent_window": window,
        "dynamic_latent_window": True,
        "input_name": "latents",
        "output_name": "audio",
        "input_shape": [1, 256, window],
        "output_shape": [1, 2, window * 4096],
        "output_samples": window * 4096,
        "samples_per_latent": 4096,
        "latent_hop": hop,
        "audio_hop_samples": hop * 4096,
        "ola_mode": "full_overlap_add",
    }


def _write_bundle(tmp_path, windows=(2, 4, 8)):
    bundle = tmp_path / "fixture.sawbundle"
    model = bundle / "models" / "decoder" / "same_s_decoder_dynamic.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake dynamic SAME-S model")
    model_hash = _hash(model)
    arrays_dir = bundle / "arrays"
    arrays_dir.mkdir()
    fixture_arrays = {
        "Z_concat": np.zeros((3, 256), dtype=np.float32),
        "Z_mean": np.zeros((256,), dtype=np.float32),
        "Z_std": np.ones((256,), dtype=np.float32),
        "file_offsets": np.asarray([0, 3], dtype=np.int64),
        "frame_file_ids": np.zeros((3,), dtype=np.int32),
    }
    array_entries = {}
    for name, array in fixture_arrays.items():
        path = arrays_dir / f"{name}.npy"
        np.save(path, array)
        array_entries[name] = {
            "path": f"arrays/{path.name}",
            "sha256": _hash(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    manifest = {
        "bundle_format_version": "sawbundle.v0.mvp",
        "vae": {
            "vae_id": "same_s",
            "sample_rate": 44100,
            "latent_hz": 44100 / 4096,
            "latent_dim": 256,
            "channels": 2,
        },
        "models": {
            "decoder": {
                "backend": "onnxruntime",
                "provider_baseline": "CPUExecutionProvider",
                "vae_id": "same_s",
                "input_name": "latents",
                "output_name": "audio",
                "path": "models/decoder/same_s_decoder_dynamic.onnx",
                "sha256": model_hash,
                "dynamic_latent_window": True,
                "validated_latent_windows": list(windows),
                "min_latent_window": min(windows),
                "max_latent_window": max(windows),
                "samples_per_latent": 4096,
                "ola_mode": "full_overlap_add",
                "windows": {
                    str(window): _window_entry(window, model_hash) for window in windows
                },
            }
        },
        "arrays": array_entries,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, manifest


def _rewrite_manifest(bundle, manifest):
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_corpus(
    tmp_path, *, vae_id="same_s", sr=44100, dim=256, latent_hz=44100 / 4096
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    np.savez(
        corpus_dir / "corpus.npz",
        vae_id=np.asarray(vae_id),
        sr=np.asarray(sr, dtype=np.int32),
        latent_hz=np.asarray(latent_hz, dtype=np.float32),
        Z_concat=np.zeros((3, dim), dtype=np.float32),
        Z_mean=np.zeros((dim,), dtype=np.float32),
        Z_std=np.ones((dim,), dtype=np.float32),
        file_offsets=np.asarray([0, 3], dtype=np.int64),
        frame_file_ids=np.zeros((3,), dtype=np.int32),
    )
    return corpus_dir


def test_preflight_configures_cpu_session_warms_every_window_and_decodes(tmp_path):
    bundle, _manifest = _write_bundle(tmp_path)
    corpus = _write_corpus(tmp_path)
    ort = FakeOrt()

    decoder = preflight_same_s_decoder_bundle(
        bundle,
        corpus_path=corpus,
        ort_module=ort,
    )

    assert decoder.supported_windows == (2, 4, 8)
    assert decoder.default_window == 2
    assert [sample.shape[2] for sample in ort.run_inputs] == [2, 4, 8]
    _model_path, options, providers = ort.session_args
    assert providers == ["CPUExecutionProvider"]
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.graph_optimization_level is _GraphOptimizationLevel.ORT_ENABLE_BASIC

    latents = np.arange(4 * 256, dtype=np.float32).reshape(4, 256)
    decoded = decoder.decode(latents)

    assert ort.run_inputs[-1].shape == (1, 256, 4)
    assert np.array_equal(ort.run_inputs[-1], latents.T[None, :, :])
    assert decoded.audio.shape == (4 * 4096, 2)
    assert decoded.audio.dtype == np.float32
    assert np.all(decoded.audio[:, 0] == np.float32(0.25))
    assert np.all(decoded.audio[:, 1] == np.float32(-0.5))
    assert decoded.metadata.latent_window == 4
    assert decoded.metadata.latent_hop == 2
    assert decoded.metadata.audio_hop_samples == 8192
    assert decoded.metadata.audio_window_samples == 16384
    assert decoded.decode_time_ms >= 0.0


def test_preflight_rejects_same_s_corpus_from_a_different_bundle(tmp_path):
    bundle, _manifest = _write_bundle(tmp_path)
    corpus = _write_corpus(tmp_path)
    corpus_path = corpus / "corpus.npz"
    with np.load(corpus_path, allow_pickle=False) as original:
        values = {name: original[name] for name in original.files}
    values["Z_concat"] = values["Z_concat"].copy()
    values["Z_concat"][1, 7] = np.float32(0.125)
    np.savez(corpus_path, **values)

    with pytest.raises(DecoderBundleError, match="does not match"):
        SameSOnnxDecoder(bundle, corpus_path=corpus, ort_module=FakeOrt())


def test_preflight_warms_every_supported_same_s_window(tmp_path):
    bundle, _manifest = _write_bundle(tmp_path, windows=(2, 4, 8, 16, 32))
    ort = FakeOrt()

    decoder = SameSOnnxDecoder(bundle, ort_module=ort)

    assert decoder.supported_windows == (2, 4, 8, 16, 32)
    assert [sample.shape[2] for sample in ort.run_inputs] == [2, 4, 8, 16, 32]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda manifest: manifest.__setitem__("bundle_format_version", "future"),
            "Unsupported bundle format",
        ),
        (
            lambda manifest: manifest["vae"].__setitem__("vae_id", "stable_audio_open"),
            "manifest.vae.vae_id",
        ),
        (
            lambda manifest: manifest["vae"].__setitem__("latent_hz", 21.5),
            "manifest.vae.latent_hz",
        ),
        (
            lambda manifest: manifest["models"]["decoder"].__setitem__(
                "dynamic_latent_window", False
            ),
            "dynamic_latent_window",
        ),
        (
            lambda manifest: manifest["models"]["decoder"]["windows"]["4"].__setitem__(
                "latent_hop", 1
            ),
            "latent_hop",
        ),
        (
            lambda manifest: manifest["models"]["decoder"]["windows"]["4"].__setitem__(
                "input_name", "wrong"
            ),
            "I/O names",
        ),
        (
            lambda manifest: manifest["models"]["decoder"]["windows"].pop("2"),
            "default T2",
        ),
    ],
)
def test_manifest_contract_fails_closed(tmp_path, mutation, match):
    bundle, manifest = _write_bundle(tmp_path)
    mutation(manifest)
    if "2" not in manifest["models"]["decoder"]["windows"]:
        manifest["models"]["decoder"]["validated_latent_windows"] = [4, 8]
        manifest["models"]["decoder"]["min_latent_window"] = 4
    _rewrite_manifest(bundle, manifest)

    with pytest.raises(DecoderBundleError, match=match):
        SameSOnnxDecoder(bundle, ort_module=FakeOrt())


def test_model_hash_mismatch_fails_before_session_creation(tmp_path):
    bundle, _manifest = _write_bundle(tmp_path)
    (bundle / "models" / "decoder" / "same_s_decoder_dynamic.onnx").write_bytes(b"tampered")
    ort = FakeOrt()

    with pytest.raises(DecoderBundleError, match="SHA-256 mismatch"):
        SameSOnnxDecoder(bundle, ort_module=ort)

    assert ort.session_args is None


@pytest.mark.parametrize(
    "configure,match",
    [
        (lambda ort: setattr(ort, "input_shape", [1, 256, 8]), "must be dynamic"),
        (lambda ort: setattr(ort, "providers", ["CoreMLExecutionProvider"]), "must use only"),
        (lambda ort: setattr(ort, "output_type", "tensor(double)"), "float32"),
        (lambda ort: setattr(ort, "output_length_delta", -1), "expected"),
        (lambda ort: setattr(ort, "non_finite", True), "non-finite"),
    ],
)
def test_session_and_warmup_contract_fail_closed(tmp_path, configure, match):
    bundle, _manifest = _write_bundle(tmp_path)
    ort = FakeOrt()
    configure(ort)

    with pytest.raises(DecoderBundleError, match=match):
        SameSOnnxDecoder(bundle, ort_module=ort)


@pytest.mark.parametrize(
    "vae_id,sr,dim,latent_hz,match",
    [
        ("stable_audio_open", 44100, 256, 44100 / 4096, "incompatible"),
        ("same_s", 48000, 256, 44100 / 4096, "sample rate"),
        ("same_s", 44100, 256, 21.5, "latent_hz"),
        ("same_s", 44100, 64, 44100 / 4096, "Z_concat"),
    ],
)
def test_incompatible_corpus_fails_before_session_creation(
    tmp_path, vae_id, sr, dim, latent_hz, match
):
    bundle, _manifest = _write_bundle(tmp_path)
    corpus = _write_corpus(
        tmp_path, vae_id=vae_id, sr=sr, dim=dim, latent_hz=latent_hz
    )
    ort = FakeOrt()

    with pytest.raises(DecoderBundleError, match=match):
        SameSOnnxDecoder(bundle, corpus_path=corpus, ort_module=ort)

    assert ort.session_args is None


def test_runtime_rejects_bad_latents_and_does_not_fallback(tmp_path):
    bundle, _manifest = _write_bundle(tmp_path)
    ort = FakeOrt()
    decoder = SameSOnnxDecoder(bundle, ort_module=ort)

    with pytest.raises(DecoderRuntimeError, match="float32"):
        decoder.decode(np.zeros((2, 256), dtype=np.float64))
    with pytest.raises(DecoderRuntimeError, match="unavailable"):
        decoder.decode(np.zeros((16, 256), dtype=np.float32))

    bad = np.zeros((2, 256), dtype=np.float32)
    bad[0, 0] = np.inf
    with pytest.raises(DecoderRuntimeError, match="non-finite"):
        decoder.decode(bad)

    ort.runtime_error = RuntimeError("synthetic ORT failure")
    with pytest.raises(DecoderRuntimeError, match="synthetic ORT failure"):
        decoder.decode(np.zeros((2, 256), dtype=np.float32))


@pytest.mark.skipif(
    not os.environ.get("SAW_REAL_SAME_S_BUNDLE"),
    reason="set SAW_REAL_SAME_S_BUNDLE for opt-in real ONNX preflight",
)
def test_opt_in_real_same_s_bundle():
    bundle = Path(os.environ["SAW_REAL_SAME_S_BUNDLE"])
    corpus = os.environ.get("SAW_REAL_SAME_S_CORPUS")
    decoder = SameSOnnxDecoder(bundle, corpus_path=corpus)
    decoded = decoder.decode(np.zeros((decoder.default_window, 256), dtype=np.float32))

    assert decoded.audio.shape == (decoder.default_window * 4096, 2)
    assert np.all(np.isfinite(decoded.audio))
