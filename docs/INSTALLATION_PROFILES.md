# Installation profiles

`pyproject.toml` owns the base package dependencies. `requirements.txt` installs the package with its `export` and `native-same-s` extras instead of maintaining a competing list. A plain package installation supplies the application and prepared ONNX runtime; it does not imply that optional native SAME-S/EAR libraries or checkpoints are installed.

From the repository's Python root, choose a profile and honor normal dependency resolution:

| Profile | Command | Additional requirements |
| --- | --- | --- |
| Runtime / prepared ONNX | `python -m pip install -r requirements-runtime.txt` | Matching prepared decoder and corpus supplied separately |
| Default: SAME-S and ONNX export | `python -m pip install -r requirements.txt` | Git; pinned Stable Audio 3 and Torch/Torchaudio 2.7.1; model weights separately |
| Focused native SAME-S | `python -m pip install -r requirements-same-s-native.txt` | Pinned Stable Audio 3 commit, Torch/Torchaudio 2.7.1 |
| Optional Stable Audio Open | `python -m pip install -r requirements-stable-audio-open-native.txt` | Hugging Face access conditions and `hf auth login` |
| Native EAR and export | `python -m pip install -r requirements-ear-native.txt` | DAC 1.0.0, pinned audio-tools revision, einops, protobuf 4.25.x |
| Both native families | `python -m pip install -r requirements-native.txt` | Union of the two source profiles |

Use a fresh environment for a native profile; its name and path are unrestricted. Run `python -m pip check`. Never use `--no-deps` to force a conflicting installation. Python 3.12 is the existing SAME-S reference platform; this does not establish runtime support on every Python version, OS or GPU.

The EAR audio-tools revision declares `protobuf>=3.19.6,<5` (excluding 4.24.0); the installed ONNX dependency declares `protobuf>=4.25.1`. The profile uses their overlapping range. The earlier claim that these dependencies necessarily conflict was based on an older audio-tools release.

`requirements-same-s-native-macos.lock` and `requirements-ear-export.lock` preserve separate tested historical environments. The latter is an environment snapshot, not a portable installer for the application: it does not itself install the project. The combined source profile resolved successfully (122 packages, Python 3.12 on the current macOS host), including Torch/Torchaudio 2.7.1 and protobuf 4.25.9. It still requires clean installation and real-model validation; no combined lock or universal qualification is claimed.

Model assets remain separate: EAR needs its checkpoint and repository, SAME-S/Stable Audio Open need their pinned cached/downloaded weights, and ONNX needs a prepared artifact. These paths are user configuration, not package dependencies or predefined virtual environments.

Native worker modules are Python package files and are included by setuptools package discovery. Workers launch with `python -m stable_audio_wanderer.vae.native_worker` using the current interpreter by default. They do not assume a checkout path, `.venv` directory or site-packages working directory. Optional external interpreter overrides require this package to be installed in that interpreter too.

The portability check built a wheel, confirmed the three worker modules are included, extracted packaged Python files into a separate directory and successfully launched worker discovery there with the default interpreter. The [check receipt](installation_portability_checks.json) records this result. It verifies package inclusion and checkout independence; it does not substitute for a fresh dependency installation or model/hardware qualification.

The default Stable Audio Open + SAME-S profile also resolved successfully for
Python 3.12 on macOS (72 packages) after adding the shared `native-same-s` extra.
The development lockfile was refreshed successfully. These are dependency
resolution checks, not a fresh install or model-load qualification.

The default now uses SAME-S only, with automatic public-weight download on first
Encode and no login prerequisite. Stable Audio Open is an optional installation;
EAR is separate as well. Earlier combined-profile receipts remain historical.
