"""VAE adapter registry — maps vae_id strings to loader factories."""
from typing import Callable, Dict, List

from .base import VAEAdapter, VAEInfo

_REGISTRY: Dict[str, dict] = {}


def register_vae(vae_id: str, info: VAEInfo, loader: Callable[..., VAEAdapter]):
    """Register a VAE adapter factory.

    Args:
        vae_id: Unique identifier (e.g. "stable_audio_open").
        info: Static metadata (displayed in GUI dropdown before loading).
        loader: Callable that returns a loaded VAEAdapter.  Accepts **kwargs
                (e.g. weight_path for models that need a local file).
    """
    _REGISTRY[vae_id] = {"info": info, "loader": loader}


def list_vaes() -> List[VAEInfo]:
    """Return metadata for all registered VAEs (for GUI dropdown)."""
    return [entry["info"] for entry in _REGISTRY.values()]


def load_vae_adapter(vae_id: str, **kwargs) -> VAEAdapter:
    """Instantiate and load a VAE adapter by registry ID.

    Extra kwargs are forwarded to the adapter's loader (e.g. weight_path).
    """
    entry = _REGISTRY.get(vae_id)
    if entry is None:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown VAE: {vae_id!r}. Available: {available}")
    return entry["loader"](**kwargs)
