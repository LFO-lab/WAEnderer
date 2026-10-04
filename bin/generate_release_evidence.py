"""Compatibility entry point for the installed generate_release_evidence command."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_audio_wanderer.cli import generate_release_evidence as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
