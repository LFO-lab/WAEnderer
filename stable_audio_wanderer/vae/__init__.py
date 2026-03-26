"""VAE adapter subsystem — pluggable encode/decode backends."""
from .base import VAEAdapter, VAEInfo
from .registry import list_vaes, load_vae_adapter, register_vae

# Import adapters subpackage to trigger registration of all known VAEs.
from . import adapters  # noqa: F401

__all__ = [
    "VAEAdapter",
    "VAEInfo",
    "list_vaes",
    "load_vae_adapter",
    "register_vae",
]
