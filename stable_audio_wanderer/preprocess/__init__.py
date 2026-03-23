"""Preprocessing helpers."""

from .silence import SilenceTrimConfig, SilenceTrimResult, trim_silent_frames

__all__ = [
    "SilenceTrimConfig",
    "SilenceTrimResult",
    "trim_silent_frames",
]
