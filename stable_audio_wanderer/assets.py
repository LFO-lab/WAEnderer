"""Locate canonical checkout assets or their installed package copy."""
from importlib import resources
from pathlib import Path


def web_directory() -> Path:
    # Editable installs must follow Web edits even if a previous build left
    # a packaged copy in the source tree.
    checkout = Path(__file__).resolve().parents[1] / 'web'
    if (checkout / 'index.html').is_file():
        return checkout
    packaged = Path(str(resources.files('stable_audio_wanderer.resources') / 'web'))
    if (packaged / 'index.html').is_file():
        return packaged
    raise FileNotFoundError('Web assets missing; reinstall the application wheel')
