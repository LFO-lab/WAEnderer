import os
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class ReleaseBuildPy(build_py):
    """Require the untracked Web decoder when producing a release artifact."""

    def run(self):
        resource_dir = Path(__file__).parent / "stable_audio_wanderer" / "resources" / "same_s"
        required = ("same_s_decoder_dynamic.onnx", "decoder.json", "decoder_parity.json")
        missing = [name for name in required if not (resource_dir / name).is_file()]
        if os.environ.get("SAW_RELEASE_BUILD") == "1" and missing:
            raise RuntimeError(
                "Release build is missing SAME-S decoder resources: " + ", ".join(missing)
            )
        super().run()

setup(
    name="stable_audio_wanderer",
    version="0.1.0",
    packages=find_packages(),
    package_data={"stable_audio_wanderer.resources.same_s": ["*.onnx", "*.json", "README.md"]},
    cmdclass={"build_py": ReleaseBuildPy},
    install_requires=[],
)
