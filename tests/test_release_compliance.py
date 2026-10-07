import hashlib
import json
from pathlib import Path

import pytest

from bin.generate_release_evidence import (
    _apply_license_overrides,
    _license_overrides,
)
from stable_audio_wanderer.release_compliance import (
    check_dependency_license_report,
    check_public_tree,
    validate_model_release,
)


PROJECT_ROOT = Path(__file__).parent.parent


def test_public_tree_has_required_license_evidence():
    assert check_public_tree(PROJECT_ROOT) == []


def test_model_release_validates_provenance_and_digest(tmp_path):
    resource = tmp_path / "stable_audio_wanderer" / "resources" / "same_s"
    resource.mkdir(parents=True)
    model = resource / "same_s_decoder_dynamic.onnx"
    model.write_bytes(b"model")
    (resource / "decoder_parity.json").write_text("{}\n", encoding="utf-8")
    (resource / "MODEL_NOTICE.md").write_text(
        "This Stability AI Model is licensed under the Stability AI Community License\n"
        "Powered by Stability AI\n"
        "not licensed under the\nApache License 2.0\n",
        encoding="utf-8",
    )
    (resource / "UPSTREAM_NOTICE.txt").write_text("upstream\n", encoding="utf-8")

    for relative in (
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "licenses/STABILITY_AI_COMMUNITY_LICENSE.md",
        "licenses/GEMMA_TERMS_OF_USE.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("license\n", encoding="utf-8")

    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    metadata = {
        "source_model": "stabilityai/SAME-S",
        "source_url": "https://huggingface.co/stabilityai/SAME-S",
        "source_revision": "a" * 40,
        "source_license": "Stability AI Community License",
        "conversion_description": "Exported decoder to ONNX.",
        "model_sha256": digest,
    }
    metadata_path = resource / "decoder.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    validate_model_release(tmp_path)

    metadata["model_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="model_sha256"):
        validate_model_release(tmp_path)


def test_dependency_license_policy_flags_unknown_and_strong_copyleft(tmp_path):
    report = {
        "packages": [
            {"name": "permissive", "version": "1", "license": "MIT"},
            {"name": "mystery", "version": "2", "license": "NOASSERTION"},
            {"name": "server", "version": "3", "license": "AGPL-3.0-only"},
            {"name": "weak", "version": "4", "license": "LGPL-2.1-or-later"},
            {
                "name": "runtime-exception",
                "version": "5",
                "license": "GPL-3.0-or-later WITH GCC-exception-3.1",
            },
        ]
    }
    path = tmp_path / "licenses.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    issues = check_dependency_license_report(path)
    assert any("mystery==2" in issue for issue in issues)
    assert any("server==3" in issue for issue in issues)
    assert not any("weak==4" in issue for issue in issues)
    assert not any("runtime-exception==5" in issue for issue in issues)


def test_dependency_license_override_is_exact_version_and_evidence_backed():
    overrides = _license_overrides(
        PROJECT_ROOT / "licenses" / "dependency-license-overrides.json"
    )
    exact = {
        "cuda-toolkit": {
            "name": "cuda-toolkit",
            "version": "13.0.3.0",
            "license": "NOASSERTION",
        }
    }
    future = {
        "cuda-toolkit": {
            "name": "cuda-toolkit",
            "version": "13.0.4.0",
            "license": "NOASSERTION",
        }
    }
    asserted = {
        "cuda-toolkit": {
            "name": "cuda-toolkit",
            "version": "13.0.3.0",
            "license": "Upstream asserted license",
        }
    }

    _apply_license_overrides(exact, overrides)
    _apply_license_overrides(future, overrides)
    _apply_license_overrides(asserted, overrides)

    assert exact["cuda-toolkit"]["license"] == (
        "NVIDIA Software License Agreement and CUDA Supplement"
    )
    assert exact["cuda-toolkit"]["license_evidence"].startswith("https://docs.nvidia.com/")
    assert future["cuda-toolkit"]["license"] == "NOASSERTION"
    assert asserted["cuda-toolkit"]["license"] == "Upstream asserted license"


def test_same_s_license_override_preserves_policy(tmp_path):
    overrides = _license_overrides(
        PROJECT_ROOT / "licenses" / "dependency-license-overrides.json"
    )
    packages = {
        "stable-audio-3": {
            "name": "stable-audio-3", "version": "0.1.0", "license": "NOASSERTION"
        }
    }
    _apply_license_overrides(packages, overrides)
    package = packages["stable-audio-3"]
    assert package["license"] == "MIT"
    assert package["license_evidence"] == (
        "https://github.com/Stability-AI/stable-audio-3/blob/"
        "779434a908193105335fd8d833418603625b2859/LICENSE"
    )
    report = tmp_path / "licenses.json"
    report.write_text(json.dumps({"packages": list(packages.values())}))
    assert check_dependency_license_report(report) == []

    for version, license_value in [("0.2.0", "NOASSERTION"), ("0.1.0", "AGPL-3.0-only")]:
        packages = {"stable-audio-3": {
            "name": "stable-audio-3", "version": version, "license": license_value
        }}
        _apply_license_overrides(packages, overrides)
        assert packages["stable-audio-3"]["license"] == license_value
        report.write_text(json.dumps({"packages": list(packages.values())}))
        assert len(check_dependency_license_report(report)) == 1
