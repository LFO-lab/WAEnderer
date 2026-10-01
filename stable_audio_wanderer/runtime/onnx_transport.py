"""Compatibility import for the former ONNX-specific Web transport.

The same controller is now shared by any decoder implementing LatentDecoder.
Existing callers keep the original class name without a second implementation.
"""

from .decoder_transport import DecoderTransportController

OnnxTransportController = DecoderTransportController

__all__ = ["OnnxTransportController"]
