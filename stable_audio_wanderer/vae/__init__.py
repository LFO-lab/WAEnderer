"""VAE adapter subsystem — pluggable encode/decode backends."""
from .base import VAEAdapter, VAEInfo
from .onnx_decoder import (
    DecoderBundleError,
    DecoderBundleInfo,
    DecoderResourceInfo,
    DecodedAudioWindow,
    DecoderRuntimeError,
    DecoderWindowMetadata,
    SameSOnnxDecoder,
    SameSAppOnnxDecoder,
    load_same_s_app_decoder,
    preflight_same_s_decoder_bundle,
    validate_same_s_corpus,
)
from .registry import list_vaes, load_vae_adapter, register_vae

# Import adapters subpackage to trigger registration of all known VAEs.
from . import adapters  # noqa: F401

__all__ = [
    "VAEAdapter",
    "VAEInfo",
    "DecoderBundleError",
    "DecoderBundleInfo",
    "DecoderResourceInfo",
    "DecodedAudioWindow",
    "DecoderRuntimeError",
    "DecoderWindowMetadata",
    "SameSOnnxDecoder",
    "SameSAppOnnxDecoder",
    "list_vaes",
    "load_vae_adapter",
    "load_same_s_app_decoder",
    "preflight_same_s_decoder_bundle",
    "register_vae",
    "validate_same_s_corpus",
]
