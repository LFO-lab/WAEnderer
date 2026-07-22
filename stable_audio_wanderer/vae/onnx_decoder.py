"""Fail-closed SAME-S decoder runtime for prepared ``.sawbundle`` bundles.

The Web performance pipeline uses this module as a decoder-only boundary.  It
accepts raw (denormalized) latent frames in ``[T, 256]`` layout and returns
stereo PCM in the repository's usual ``[samples, channels]`` layout.  Model
loading is intentionally lazy with respect to :mod:`onnxruntime` so the
standalone Torch performance path does not acquire an ONNX dependency merely
by importing the VAE package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import resources
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import numpy as np


BUNDLE_FORMAT_VERSION = "sawbundle.v0.mvp"
BACKEND = "onnxruntime"
PROVIDER = "CPUExecutionProvider"
VAE_ID = "same_s"
SAMPLE_RATE = 44_100
CHANNELS = 2
LATENT_DIM = 256
SAMPLES_PER_LATENT = 4_096
LATENT_HZ = SAMPLE_RATE / SAMPLES_PER_LATENT
OLA_MODE = "full_overlap_add"
DEFAULT_WINDOW = 2
ALLOWED_WINDOWS = (2, 4, 8, 16, 32)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class DecoderBundleError(RuntimeError):
    """Raised when a bundle or corpus fails decoder preflight."""


class DecoderRuntimeError(RuntimeError):
    """Raised when ONNX Runtime fails after successful preflight."""


@dataclass(frozen=True)
class DecoderWindowMetadata:
    """Validated timing and shape metadata for one latent window."""

    latent_window: int
    latent_dim: int
    sample_rate: int
    channels: int
    samples_per_latent: int
    audio_window_samples: int
    latent_hop: int
    audio_hop_samples: int
    ola_mode: str


@dataclass(frozen=True)
class DecodedAudioWindow:
    """One full decoder output and the metadata needed by streaming OLA."""

    audio: np.ndarray
    metadata: DecoderWindowMetadata
    decode_time_ms: float


@dataclass(frozen=True)
class DecoderBundleInfo:
    """Immutable result of manifest, corpus, session, and warm-up preflight."""

    bundle_path: Path
    model_path: Path
    backend: str
    provider: str
    vae_id: str
    input_name: str
    output_name: str
    supported_windows: tuple[int, ...]
    default_window: int
    windows: Mapping[int, DecoderWindowMetadata]
    warmup_decode_ms: Mapping[int, float]


@dataclass(frozen=True)
class DecoderResourceInfo:
    """Metadata for the app-owned SAME-S decoder resource."""

    resource_path: Path
    model_path: Path
    backend: str
    provider: str
    vae_id: str
    input_name: str
    output_name: str
    supported_windows: tuple[int, ...]
    default_window: int
    windows: Mapping[int, DecoderWindowMetadata]
    warmup_decode_ms: Mapping[int, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DecoderBundleError(f"{label} must be a JSON object")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderBundleError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderBundleError(f"{label} must be an integer") from exc
    if parsed != value:
        raise DecoderBundleError(f"{label} must be an integer")
    return parsed


def _resolve_hashed_file(
    bundle_path: Path,
    entry: Mapping[str, Any],
    label: str,
) -> Path:
    rel_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise DecoderBundleError(f"{label} is missing a manifest path")
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        raise DecoderBundleError(f"{label} is missing a valid SHA-256 hash")

    relative = Path(rel_path)
    if relative.is_absolute():
        raise DecoderBundleError(f"{label} path must be relative to the bundle")

    bundle_root = bundle_path.resolve()
    resolved = (bundle_root / relative).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise DecoderBundleError(f"{label} path escapes the bundle") from exc

    if not resolved.is_file():
        raise DecoderBundleError(f"{label} file is missing: {rel_path}")
    if _sha256(resolved).lower() != expected_hash.lower():
        raise DecoderBundleError(f"{label} SHA-256 mismatch")
    return resolved


def _read_scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data:
        raise DecoderBundleError(f"Corpus is missing required metadata {key!r}")
    value = np.asarray(data[key]).reshape(-1)
    if value.size != 1:
        raise DecoderBundleError(f"Corpus metadata {key!r} must be scalar")
    item = value[0]
    if hasattr(item, "item"):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


def validate_same_s_corpus(corpus_path: Union[str, Path]) -> Path:
    """Validate that a corpus can supply raw SAME-S decoder latents.

    ``corpus_path`` may name either the corpus directory or ``corpus.npz``.
    This check deliberately reads only numeric/string fields, with pickle
    loading disabled.
    """

    path = Path(corpus_path).expanduser()
    if path.is_dir():
        path = path / "corpus.npz"
    if not path.is_file():
        raise DecoderBundleError(f"Corpus file is missing: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            vae_id = str(_read_scalar(data, "vae_id"))
            sample_rate = _require_int(_read_scalar(data, "sr"), "corpus sr")
            latent_hz = float(_read_scalar(data, "latent_hz"))
            if vae_id != VAE_ID:
                raise DecoderBundleError(
                    f"Corpus VAE {vae_id!r} is incompatible; expected {VAE_ID!r}"
                )
            if sample_rate != SAMPLE_RATE:
                raise DecoderBundleError(
                    f"Corpus sample rate is {sample_rate}; expected {SAMPLE_RATE}"
                )
            if not np.isfinite(latent_hz) or not np.isclose(
                latent_hz, LATENT_HZ, rtol=0.0, atol=1.0e-7
            ):
                raise DecoderBundleError(
                    f"Corpus latent_hz is {latent_hz}; expected {LATENT_HZ}"
                )

            z_concat = np.asarray(data["Z_concat"])
            z_mean = np.asarray(data["Z_mean"])
            z_std = np.asarray(data["Z_std"])
            if z_concat.ndim != 2 or z_concat.shape[1] != LATENT_DIM:
                raise DecoderBundleError(
                    f"Corpus Z_concat must have shape [frames,{LATENT_DIM}], "
                    f"got {z_concat.shape}"
                )
            if z_concat.shape[0] < 1:
                raise DecoderBundleError("Corpus contains no latent frames")
            for name, array in (("Z_concat", z_concat), ("Z_mean", z_mean), ("Z_std", z_std)):
                if array.dtype != np.float32:
                    raise DecoderBundleError(f"Corpus {name} must be float32, got {array.dtype}")
                if not np.all(np.isfinite(array)):
                    raise DecoderBundleError(f"Corpus {name} contains non-finite values")
            if z_mean.shape not in ((LATENT_DIM,), (1, LATENT_DIM)):
                raise DecoderBundleError(
                    f"Corpus Z_mean must contain {LATENT_DIM} values, got {z_mean.shape}"
                )
            if z_std.shape not in ((LATENT_DIM,), (1, LATENT_DIM)):
                raise DecoderBundleError(
                    f"Corpus Z_std must contain {LATENT_DIM} values, got {z_std.shape}"
                )
            if np.any(z_std <= 0.0):
                raise DecoderBundleError("Corpus Z_std must be strictly positive")
            if "channels" in data:
                channels = _require_int(_read_scalar(data, "channels"), "corpus channels")
                if channels != CHANNELS:
                    raise DecoderBundleError(
                        f"Corpus has {channels} channels; expected stereo ({CHANNELS})"
                    )
    except DecoderBundleError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise DecoderBundleError(f"Could not validate corpus {path}: {exc}") from exc

    return path.resolve()


def _validate_manifest(bundle_path: Path) -> tuple[
    Path,
    str,
    str,
    dict[int, DecoderWindowMetadata],
    dict[str, Path],
]:
    if not bundle_path.is_dir():
        raise DecoderBundleError(f"Decoder bundle directory is missing: {bundle_path}")
    if bundle_path.suffix != ".sawbundle":
        raise DecoderBundleError("Decoder bundle path must end in .sawbundle")

    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.is_file():
        raise DecoderBundleError(f"Bundle manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecoderBundleError(f"Could not read bundle manifest: {exc}") from exc
    manifest = _require_mapping(manifest, "manifest")

    if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise DecoderBundleError(
            f"Unsupported bundle format {manifest.get('bundle_format_version')!r}; "
            f"expected {BUNDLE_FORMAT_VERSION!r}"
        )

    vae = _require_mapping(manifest.get("vae"), "manifest.vae")
    expected_vae_fields = {
        "vae_id": VAE_ID,
        "sample_rate": SAMPLE_RATE,
        "latent_dim": LATENT_DIM,
        "channels": CHANNELS,
    }
    for key, expected in expected_vae_fields.items():
        actual = vae.get(key)
        if actual != expected:
            raise DecoderBundleError(
                f"manifest.vae.{key} is {actual!r}; expected {expected!r}"
            )
    try:
        manifest_latent_hz = float(vae.get("latent_hz"))
    except (TypeError, ValueError) as exc:
        raise DecoderBundleError("manifest.vae.latent_hz is missing or invalid") from exc
    if not np.isfinite(manifest_latent_hz) or not np.isclose(
        manifest_latent_hz, LATENT_HZ, rtol=0.0, atol=1.0e-7
    ):
        raise DecoderBundleError(
            f"manifest.vae.latent_hz is {manifest_latent_hz!r}; expected {LATENT_HZ}"
        )

    models = _require_mapping(manifest.get("models"), "manifest.models")
    decoder = _require_mapping(models.get("decoder"), "manifest.models.decoder")
    if decoder.get("backend") != BACKEND:
        raise DecoderBundleError("Decoder backend must be onnxruntime")
    if decoder.get("provider_baseline") != PROVIDER:
        raise DecoderBundleError("Decoder provider_baseline must be CPUExecutionProvider")
    if decoder.get("vae_id") != VAE_ID:
        raise DecoderBundleError("Decoder vae_id must be same_s")
    if decoder.get("dynamic_latent_window") is not True:
        raise DecoderBundleError("SAME-S decoder must declare dynamic_latent_window=true")
    if decoder.get("ola_mode") != OLA_MODE:
        raise DecoderBundleError(f"Decoder ola_mode must be {OLA_MODE}")
    if _require_int(decoder.get("samples_per_latent"), "decoder samples_per_latent") != SAMPLES_PER_LATENT:
        raise DecoderBundleError(
            f"Decoder samples_per_latent must be {SAMPLES_PER_LATENT}"
        )

    input_name = decoder.get("input_name")
    output_name = decoder.get("output_name")
    if not isinstance(input_name, str) or not input_name:
        raise DecoderBundleError("Decoder input_name is missing")
    if not isinstance(output_name, str) or not output_name:
        raise DecoderBundleError("Decoder output_name is missing")

    model_path = _resolve_hashed_file(bundle_path, decoder, "decoder model")
    if model_path.suffix.lower() != ".onnx":
        raise DecoderBundleError("Decoder model path must name an .onnx file")

    arrays = _require_mapping(manifest.get("arrays"), "manifest.arrays")
    array_paths: dict[str, Path] = {}
    for name, raw_entry in arrays.items():
        entry = _require_mapping(raw_entry, f"manifest.arrays.{name}")
        array_path = _resolve_hashed_file(bundle_path, entry, f"array {name}")
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DecoderBundleError(f"Could not read bundle array {name}: {exc}") from exc
        if entry.get("dtype") != str(array.dtype):
            raise DecoderBundleError(f"Bundle array {name} dtype does not match its manifest")
        if entry.get("shape") != list(array.shape):
            raise DecoderBundleError(f"Bundle array {name} shape does not match its manifest")
        array_paths[str(name)] = array_path

    required_corpus_arrays = (
        "Z_concat",
        "Z_mean",
        "Z_std",
        "file_offsets",
        "frame_file_ids",
    )
    missing_arrays = [name for name in required_corpus_arrays if name not in array_paths]
    if missing_arrays:
        raise DecoderBundleError(
            f"Bundle is missing corpus compatibility arrays: {missing_arrays}"
        )

    parity_path = decoder.get("parity_report")
    parity_hash = decoder.get("parity_report_sha256")
    if parity_path is not None or parity_hash is not None:
        _resolve_hashed_file(
            bundle_path,
            {"path": parity_path, "sha256": parity_hash},
            "decoder parity report",
        )

    raw_windows = _require_mapping(decoder.get("windows"), "decoder windows")
    windows: dict[int, DecoderWindowMetadata] = {}
    for raw_window, raw_entry in raw_windows.items():
        try:
            window = int(raw_window)
        except (TypeError, ValueError) as exc:
            raise DecoderBundleError(f"Invalid decoder window key {raw_window!r}") from exc
        if str(window) != str(raw_window):
            raise DecoderBundleError(f"Decoder window key must be canonical: {raw_window!r}")
        if window not in ALLOWED_WINDOWS:
            raise DecoderBundleError(
                f"Unsupported SAME-S decoder window T{window}; allowed: {ALLOWED_WINDOWS}"
            )
        entry = _require_mapping(raw_entry, f"decoder window T{window}")
        entry_path = _resolve_hashed_file(bundle_path, entry, f"decoder window T{window}")
        if entry_path != model_path:
            raise DecoderBundleError(f"Decoder window T{window} does not reference the dynamic model")
        if entry.get("dynamic_latent_window") is not True:
            raise DecoderBundleError(f"Decoder window T{window} must be dynamic")
        if _require_int(entry.get("latent_window"), f"T{window} latent_window") != window:
            raise DecoderBundleError(f"Decoder window T{window} latent_window mismatch")
        if entry.get("input_name") != input_name or entry.get("output_name") != output_name:
            raise DecoderBundleError(f"Decoder window T{window} I/O names do not match the model")
        if entry.get("input_shape") != [1, LATENT_DIM, window]:
            raise DecoderBundleError(
                f"Decoder window T{window} input_shape must be [1,{LATENT_DIM},{window}]"
            )
        audio_samples = window * SAMPLES_PER_LATENT
        if entry.get("output_shape") != [1, CHANNELS, audio_samples]:
            raise DecoderBundleError(
                f"Decoder window T{window} output_shape must be [1,{CHANNELS},{audio_samples}]"
            )
        if _require_int(entry.get("output_samples"), f"T{window} output_samples") != audio_samples:
            raise DecoderBundleError(f"Decoder window T{window} output_samples mismatch")
        if _require_int(entry.get("samples_per_latent"), f"T{window} samples_per_latent") != SAMPLES_PER_LATENT:
            raise DecoderBundleError(f"Decoder window T{window} samples_per_latent mismatch")
        latent_hop = (window + 1) // 2
        if _require_int(entry.get("latent_hop"), f"T{window} latent_hop") != latent_hop:
            raise DecoderBundleError(f"Decoder window T{window} latent_hop must be {latent_hop}")
        audio_hop = latent_hop * SAMPLES_PER_LATENT
        if _require_int(entry.get("audio_hop_samples"), f"T{window} audio_hop_samples") != audio_hop:
            raise DecoderBundleError(
                f"Decoder window T{window} audio_hop_samples must be {audio_hop}"
            )
        if entry.get("ola_mode") != OLA_MODE:
            raise DecoderBundleError(f"Decoder window T{window} ola_mode must be {OLA_MODE}")
        windows[window] = DecoderWindowMetadata(
            latent_window=window,
            latent_dim=LATENT_DIM,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            samples_per_latent=SAMPLES_PER_LATENT,
            audio_window_samples=audio_samples,
            latent_hop=latent_hop,
            audio_hop_samples=audio_hop,
            ola_mode=OLA_MODE,
        )

    supported = tuple(sorted(windows))
    if DEFAULT_WINDOW not in windows:
        raise DecoderBundleError("SAME-S bundle must offer the default T2 decoder window")
    declared_windows = decoder.get("validated_latent_windows")
    if not isinstance(declared_windows, list) or tuple(declared_windows) != supported:
        raise DecoderBundleError(
            "validated_latent_windows must exactly match the sorted decoder window entries"
        )
    if _require_int(decoder.get("min_latent_window"), "decoder min_latent_window") != supported[0]:
        raise DecoderBundleError("Decoder min_latent_window does not match its windows")
    if _require_int(decoder.get("max_latent_window"), "decoder max_latent_window") != supported[-1]:
        raise DecoderBundleError("Decoder max_latent_window does not match its windows")

    return model_path, input_name, output_name, windows, array_paths


def _validate_corpus_matches_bundle(
    corpus_path: Union[str, Path], array_paths: Mapping[str, Path]
) -> None:
    resolved_corpus = validate_same_s_corpus(corpus_path)
    try:
        with np.load(resolved_corpus, allow_pickle=False) as corpus:
            for name in (
                "Z_concat",
                "Z_mean",
                "Z_std",
                "file_offsets",
                "frame_file_ids",
            ):
                if name not in corpus:
                    raise DecoderBundleError(
                        f"Corpus is missing compatibility array {name!r}"
                    )
                corpus_array = np.asarray(corpus[name])
                bundle_array = np.load(
                    array_paths[name], mmap_mode="r", allow_pickle=False
                )
                if (
                    corpus_array.dtype != bundle_array.dtype
                    or corpus_array.shape != bundle_array.shape
                    or not np.array_equal(corpus_array, bundle_array)
                ):
                    raise DecoderBundleError(
                        f"Corpus array {name} does not match the selected decoder bundle"
                    )
    except DecoderBundleError:
        raise
    except (OSError, ValueError) as exc:
        raise DecoderBundleError(
            f"Could not compare corpus with decoder bundle: {exc}"
        ) from exc


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise DecoderBundleError(
            "SAME-S Web decoding requires onnxruntime; no Torch fallback is available"
        ) from exc
    return ort


def _is_dynamic_dimension(value: Any) -> bool:
    return value is None or not isinstance(value, (int, np.integer))


class SameSOnnxDecoder:
    """Preflighted, CPU-only decoder for one dynamic SAME-S ONNX model."""

    def __init__(
        self,
        bundle_path: Union[str, Path],
        *,
        corpus_path: Optional[Union[str, Path]] = None,
        ort_module: Any = None,
    ) -> None:
        resolved_bundle = Path(bundle_path).expanduser().resolve()
        model_path, input_name, output_name, windows, array_paths = _validate_manifest(
            resolved_bundle
        )
        if corpus_path is not None:
            _validate_corpus_matches_bundle(corpus_path, array_paths)

        ort = ort_module if ort_module is not None else _load_onnxruntime()
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=[PROVIDER],
            )
        except Exception as exc:
            raise DecoderBundleError(f"Could not create CPU ONNX decoder session: {exc}") from exc

        self._session = session
        self._input_name = input_name
        self._output_name = output_name
        self._windows = windows
        self._validate_session_contract()

        warmup_times: dict[int, float] = {}
        for window in sorted(windows):
            sample = np.zeros((1, LATENT_DIM, window), dtype=np.float32)
            _output, elapsed_ms = self._run_onnx(sample, window, preflight=True)
            warmup_times[window] = elapsed_ms

        immutable_windows = MappingProxyType(dict(windows))
        self.info = DecoderBundleInfo(
            bundle_path=resolved_bundle,
            model_path=model_path,
            backend=BACKEND,
            provider=PROVIDER,
            vae_id=VAE_ID,
            input_name=input_name,
            output_name=output_name,
            supported_windows=tuple(sorted(windows)),
            default_window=DEFAULT_WINDOW,
            windows=immutable_windows,
            warmup_decode_ms=MappingProxyType(warmup_times),
        )

    @property
    def supported_windows(self) -> tuple[int, ...]:
        return self.info.supported_windows

    @property
    def default_window(self) -> int:
        return self.info.default_window

    def metadata_for(self, window: int) -> DecoderWindowMetadata:
        try:
            return self._windows[int(window)]
        except (KeyError, TypeError, ValueError) as exc:
            raise DecoderRuntimeError(
                f"Decoder window T{window} is unavailable; supported: {self.supported_windows}"
            ) from exc

    def _validate_session_contract(self) -> None:
        try:
            inputs = list(self._session.get_inputs())
            outputs = list(self._session.get_outputs())
            providers = list(self._session.get_providers())
        except Exception as exc:
            raise DecoderBundleError(f"Could not inspect ONNX decoder session: {exc}") from exc

        if providers != [PROVIDER]:
            raise DecoderBundleError(
                f"ONNX decoder must use only {PROVIDER}; session reported {providers}"
            )
        if len(inputs) != 1 or len(outputs) != 1:
            raise DecoderBundleError("ONNX decoder must expose exactly one input and one output")
        input_meta, output_meta = inputs[0], outputs[0]
        if input_meta.name != self._input_name or output_meta.name != self._output_name:
            raise DecoderBundleError(
                "ONNX model I/O names do not match the bundle manifest"
            )
        if input_meta.type != "tensor(float)" or output_meta.type != "tensor(float)":
            raise DecoderBundleError("ONNX decoder I/O must be float32 tensor(float)")

        input_shape = list(input_meta.shape)
        output_shape = list(output_meta.shape)
        if (
            len(input_shape) != 3
            or input_shape[0] != 1
            or input_shape[1] != LATENT_DIM
            or not _is_dynamic_dimension(input_shape[2])
        ):
            raise DecoderBundleError(
                f"ONNX input must be dynamic float32 [1,{LATENT_DIM},T], got {input_shape}"
            )
        if (
            len(output_shape) != 3
            or output_shape[0] != 1
            or output_shape[1] != CHANNELS
            or not _is_dynamic_dimension(output_shape[2])
        ):
            raise DecoderBundleError(
                f"ONNX output must be dynamic float32 [1,{CHANNELS},T*{SAMPLES_PER_LATENT}], "
                f"got {output_shape}"
            )

    def _run_onnx(
        self,
        model_input: np.ndarray,
        window: int,
        *,
        preflight: bool,
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        try:
            result = self._session.run(
                [self._output_name],
                {self._input_name: model_input},
            )
        except Exception as exc:
            error_type = DecoderBundleError if preflight else DecoderRuntimeError
            phase = "warm-up" if preflight else "runtime"
            raise error_type(f"ONNX decoder {phase} failed for T{window}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        error_type = DecoderBundleError if preflight else DecoderRuntimeError
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise error_type(f"ONNX decoder returned an invalid result for T{window}")
        output = np.asarray(result[0])
        expected_shape = (1, CHANNELS, window * SAMPLES_PER_LATENT)
        if output.dtype != np.float32:
            raise error_type(
                f"ONNX decoder output for T{window} must be float32, got {output.dtype}"
            )
        if output.shape != expected_shape:
            raise error_type(
                f"ONNX decoder output for T{window} has shape {output.shape}; "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(output)):
            raise error_type(f"ONNX decoder output for T{window} contains non-finite values")
        return output, elapsed_ms

    def decode(self, raw_latents: np.ndarray) -> DecodedAudioWindow:
        """Decode raw float32 latents with shape ``[T, 256]``.

        The latent window itself selects the validated manifest entry.  No
        padding, window prediction, backend fallback, or dtype coercion occurs.
        """

        latents = np.asarray(raw_latents)
        if latents.dtype != np.float32:
            raise DecoderRuntimeError(
                f"Raw SAME-S latents must be float32, got {latents.dtype}"
            )
        if latents.ndim != 2 or latents.shape[1] != LATENT_DIM:
            raise DecoderRuntimeError(
                f"Raw SAME-S latents must have shape [T,{LATENT_DIM}], got {latents.shape}"
            )
        window = int(latents.shape[0])
        metadata = self.metadata_for(window)
        if not np.all(np.isfinite(latents)):
            raise DecoderRuntimeError("Raw SAME-S latents contain non-finite values")

        model_input = np.ascontiguousarray(latents.T[None, :, :], dtype=np.float32)
        output, elapsed_ms = self._run_onnx(model_input, window, preflight=False)
        audio = np.ascontiguousarray(output[0].T, dtype=np.float32)
        return DecodedAudioWindow(
            audio=audio,
            metadata=metadata,
            decode_time_ms=elapsed_ms,
        )


class SameSAppOnnxDecoder(SameSOnnxDecoder):
    """Lazy Web decoder loaded from the application package resources.

    Unlike :class:`SameSOnnxDecoder`, this path has no corpus or bundle
    coupling and does not warm every window. Release preparation owns the
    exhaustive Torch/ONNX parity check.
    """

    RESOURCE_PACKAGE = "stable_audio_wanderer.resources.same_s"
    METADATA_NAME = "decoder.json"

    def __init__(
        self,
        resource_dir: Optional[Union[str, Path]] = None,
        *,
        corpus_path: Optional[Union[str, Path]] = None,
        ort_module: Any = None,
    ) -> None:
        if corpus_path is not None:
            validate_same_s_corpus(corpus_path)

        if resource_dir is None:
            try:
                resource_root = resources.files(self.RESOURCE_PACKAGE)
                metadata_text = resource_root.joinpath(self.METADATA_NAME).read_text(
                    encoding="utf-8"
                )
                resolved_resource = Path(str(resource_root)).resolve()
            except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
                raise DecoderBundleError(
                    "Packaged SAME-S decoder resource is missing; rebuild the release "
                    "with the Web decoder asset"
                ) from exc
        else:
            resolved_resource = Path(resource_dir).expanduser().resolve()
            metadata_path = resolved_resource / self.METADATA_NAME
            try:
                metadata_text = metadata_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise DecoderBundleError(
                    f"SAME-S decoder metadata is missing: {metadata_path}"
                ) from exc

        try:
            metadata = _require_mapping(json.loads(metadata_text), "decoder resource")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DecoderBundleError(f"Could not read SAME-S decoder metadata: {exc}") from exc

        expected = {
            "format_version": "same_s.web_decoder.v1",
            "backend": BACKEND,
            "provider": PROVIDER,
            "vae_id": VAE_ID,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "latent_dim": LATENT_DIM,
            "samples_per_latent": SAMPLES_PER_LATENT,
            "ola_mode": OLA_MODE,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise DecoderBundleError(
                    f"Decoder resource {key} is {metadata.get(key)!r}; expected {value!r}"
                )
        input_name = metadata.get("input_name")
        output_name = metadata.get("output_name")
        model_name = metadata.get("model")
        if not all(isinstance(value, str) and value for value in (
            input_name, output_name, model_name
        )):
            raise DecoderBundleError("Decoder resource model and I/O names are required")
        if Path(model_name).name != model_name or not model_name.endswith(".onnx"):
            raise DecoderBundleError("Decoder resource model must be a local .onnx filename")
        model_path = resolved_resource / model_name
        if not model_path.is_file():
            raise DecoderBundleError(f"Packaged SAME-S ONNX model is missing: {model_path}")

        raw_windows = metadata.get("supported_windows")
        if not isinstance(raw_windows, list):
            raise DecoderBundleError("Decoder resource supported_windows must be a list")
        supported = tuple(_require_int(value, "supported window") for value in raw_windows)
        if supported != tuple(sorted(set(supported))) or any(
            value not in ALLOWED_WINDOWS for value in supported
        ):
            raise DecoderBundleError(
                f"Decoder resource windows must be a sorted subset of {ALLOWED_WINDOWS}"
            )
        default_window = _require_int(metadata.get("default_window"), "default_window")
        if default_window not in supported:
            raise DecoderBundleError("Decoder resource default_window is unavailable")

        windows = {
            window: DecoderWindowMetadata(
                latent_window=window,
                latent_dim=LATENT_DIM,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                samples_per_latent=SAMPLES_PER_LATENT,
                audio_window_samples=window * SAMPLES_PER_LATENT,
                latent_hop=(window + 1) // 2,
                audio_hop_samples=((window + 1) // 2) * SAMPLES_PER_LATENT,
                ola_mode=OLA_MODE,
            )
            for window in supported
        }

        ort = ort_module if ort_module is not None else _load_onnxruntime()
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=[PROVIDER]
            )
        except Exception as exc:
            raise DecoderBundleError(f"Could not create CPU ONNX decoder session: {exc}") from exc

        self._session = session
        self._input_name = input_name
        self._output_name = output_name
        self._windows = windows
        self._validate_session_contract()
        immutable_windows = MappingProxyType(dict(windows))
        self.info = DecoderResourceInfo(
            resource_path=resolved_resource,
            model_path=model_path,
            backend=BACKEND,
            provider=PROVIDER,
            vae_id=VAE_ID,
            input_name=input_name,
            output_name=output_name,
            supported_windows=supported,
            default_window=default_window,
            windows=immutable_windows,
            warmup_decode_ms=MappingProxyType({}),
        )


def preflight_same_s_decoder_bundle(
    bundle_path: Union[str, Path],
    *,
    corpus_path: Optional[Union[str, Path]] = None,
    ort_module: Any = None,
) -> SameSOnnxDecoder:
    """Construct and fully warm a fail-closed SAME-S decoder runtime."""

    return SameSOnnxDecoder(
        bundle_path,
        corpus_path=corpus_path,
        ort_module=ort_module,
    )


def load_same_s_app_decoder(
    *,
    corpus_path: Optional[Union[str, Path]] = None,
    resource_dir: Optional[Union[str, Path]] = None,
    ort_module: Any = None,
) -> SameSAppOnnxDecoder:
    """Load the packaged SAME-S decoder on demand for Web performance."""

    return SameSAppOnnxDecoder(
        resource_dir=resource_dir,
        corpus_path=corpus_path,
        ort_module=ort_module,
    )


__all__ = [
    "BACKEND",
    "PROVIDER",
    "DecoderBundleError",
    "DecoderBundleInfo",
    "DecoderResourceInfo",
    "DecodedAudioWindow",
    "DecoderRuntimeError",
    "DecoderWindowMetadata",
    "SameSOnnxDecoder",
    "SameSAppOnnxDecoder",
    "load_same_s_app_decoder",
    "preflight_same_s_decoder_bundle",
    "validate_same_s_corpus",
]
