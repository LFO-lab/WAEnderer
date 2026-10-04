import importlib.util
import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


_COMPLIANCE_PATH = Path(__file__).parent / "stable_audio_wanderer" / "release_compliance.py"
_COMPLIANCE_SPEC = importlib.util.spec_from_file_location(
    "_saw_release_compliance", _COMPLIANCE_PATH
)
if _COMPLIANCE_SPEC is None or _COMPLIANCE_SPEC.loader is None:
    raise RuntimeError(f"Could not load release compliance module: {_COMPLIANCE_PATH}")
_COMPLIANCE_MODULE = importlib.util.module_from_spec(_COMPLIANCE_SPEC)
_COMPLIANCE_SPEC.loader.exec_module(_COMPLIANCE_MODULE)
validate_model_release = _COMPLIANCE_MODULE.validate_model_release


def model_payload(path):
    """Ordinary builds never pick up private exports from resource directories."""
    parts = Path(path).parts
    if 'resources' not in parts or 'web' in parts:
        return False
    return Path(path).suffix not in ('.py', '.md', '.txt') and Path(path).name != 'even_window_validation.json'


def include_payload(path):
    return not model_payload(path) or (
        os.environ.get('SAW_RELEASE_BUILD') == '1' and 'same_s' in Path(path).parts
    )


class ReleaseBuildPy(build_py):
    """Stage canonical Web assets and opt-in SAME-S release resources."""

    def find_data_files(self, package, src_dir):
        return [path for path in super().find_data_files(package, src_dir) if include_payload(path)]

    def run(self):
        if os.environ.get("SAW_RELEASE_BUILD") == "1":
            validate_model_release(Path(__file__).parent)
        # Reusing build/ must not leak an earlier model-bearing build.
        resources_dir = Path(self.build_lib) / 'stable_audio_wanderer/resources'
        if resources_dir.exists():
            for path in resources_dir.rglob('*'):
                if path.is_file() and not include_payload(path):
                    path.unlink()
        super().run()
        web = Path(__file__).parent / 'web'
        if not (web / 'index.html').is_file():
            raise RuntimeError('Canonical Web assets missing from source distribution')
        target = resources_dir / 'web'
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(web, target, ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))


class ReleaseSdist(sdist):
    def run(self):
        if os.environ.get('SAW_RELEASE_BUILD') == '1':
            validate_model_release(Path(__file__).parent)
        super().run()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, [path for path in files if include_payload(path)])

setup(
    cmdclass={"build_py": ReleaseBuildPy, "sdist": ReleaseSdist},
)
