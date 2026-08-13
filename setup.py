import importlib.util
import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


_COMPLIANCE_PATH = Path(__file__).parent / "stable_audio_wanderer" / "release_compliance.py"
_COMPLIANCE_SPEC = importlib.util.spec_from_file_location(
    "_saw_release_compliance", _COMPLIANCE_PATH
)
if _COMPLIANCE_SPEC is None or _COMPLIANCE_SPEC.loader is None:
    raise RuntimeError(f"Could not load release compliance module: {_COMPLIANCE_PATH}")
_COMPLIANCE_MODULE = importlib.util.module_from_spec(_COMPLIANCE_SPEC)
_COMPLIANCE_SPEC.loader.exec_module(_COMPLIANCE_MODULE)
validate_model_release = _COMPLIANCE_MODULE.validate_model_release


class ReleaseBuildPy(build_py):
    """Require the untracked Web decoder when producing a release artifact."""

    def run(self):
        if os.environ.get("SAW_RELEASE_BUILD") == "1":
            validate_model_release(Path(__file__).parent)
        super().run()

setup(
    cmdclass={"build_py": ReleaseBuildPy},
)
