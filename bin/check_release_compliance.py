"""Compatibility entry point for the installed check_release_compliance command."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_audio_wanderer.cli import check_release_compliance as _implementation

if __name__ == '__main__':
    if not any(arg == '--project-root' or arg.startswith('--project-root=') for arg in sys.argv[1:]):
        sys.argv.extend(['--project-root', str(Path(__file__).resolve().parents[1])])
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
