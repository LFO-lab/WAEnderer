"""Release licensing checks shared by packaging, tests, and CI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


REQUIRED_PROVENANCE = (
    "source_model",
    "source_url",
    "source_revision",
    "source_license",
    "conversion_description",
    "model_sha256",
)

PINNED_FILE_HASHES = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "licenses/STABILITY_AI_COMMUNITY_LICENSE.md": "0bb4b5e635903088dbb5bbcc3c104e1298cc12f1989a4ff1b172f33893609b33",
    "licenses/GEMMA_TERMS_OF_USE.md": "3b65bc46747423037f7ca57af25b885a7be421948d7a81ca02b0cb038177ec3d",
    "licenses/p5.js-LGPL-2.1.txt": "eb2f25f8f165d5f20682c1ce7e99ce69144ecfd80fbed8c17dfd327ecd8b6cb6",
    "stable_audio_wanderer/resources/same_s/UPSTREAM_NOTICE.txt": "66f856d7da72797f528fca46b7c80634ab481f917bfe020960e123d84b19f75f",
    "web/vendor/p5/LICENSE.txt": "eb2f25f8f165d5f20682c1ce7e99ce69144ecfd80fbed8c17dfd327ecd8b6cb6",
    "web/vendor/p5/p5.min.js": "726ac96626b93f5bcaff83a910b6c60d3a9728f063e0eb73b5d0819ffc356915",
}

REQUIRED_PUBLIC_FILES = (
    "NOTICE",
    "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md",
    "licenses/dependency-license-overrides.json",
    "licenses/README.md",
    "stable_audio_wanderer/resources/same_s/MODEL_NOTICE.md",
    "docs/media/RIGHTS.md",
    *PINNED_FILE_HASHES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_release(project_root: Path, *, model_bundle: Path | None = None) -> None:
    """Fail closed when a model-bearing distribution lacks license evidence."""

    if model_bundle is not None:
        from .model_distribution import validate_bundle
        validate_bundle(model_bundle)
        return

    resource_dir = project_root / "stable_audio_wanderer" / "resources" / "same_s"
    required = (
        resource_dir / "same_s_decoder_dynamic.onnx",
        resource_dir / "decoder.json",
        resource_dir / "decoder_parity.json",
        resource_dir / "MODEL_NOTICE.md",
        resource_dir / "UPSTREAM_NOTICE.txt",
        project_root / "LICENSE",
        project_root / "NOTICE",
        project_root / "THIRD_PARTY_NOTICES.md",
        project_root / "licenses" / "STABILITY_AI_COMMUNITY_LICENSE.md",
        project_root / "licenses" / "GEMMA_TERMS_OF_USE.md",
    )
    missing = [str(path.relative_to(project_root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Release build is missing required resources: " + ", ".join(missing))

    try:
        metadata = json.loads((resource_dir / "decoder.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Release decoder metadata is unreadable: {exc}") from exc

    missing_fields = [key for key in REQUIRED_PROVENANCE if not metadata.get(key)]
    if missing_fields:
        raise RuntimeError(
            "Release decoder metadata is missing provenance: " + ", ".join(missing_fields)
        )
    if metadata["source_model"] != "stabilityai/SAME-S":
        raise RuntimeError("Release decoder source_model must be 'stabilityai/SAME-S'")
    if metadata["source_license"] != "Stability AI Community License":
        raise RuntimeError("Release decoder source_license is incorrect")
    if not re.fullmatch(r"[0-9a-f]{40}", str(metadata["source_revision"])):
        raise RuntimeError("Release decoder source_revision must be a full Git commit SHA")
    expected_hash = str(metadata["model_sha256"]).lower()
    actual_hash = sha256(resource_dir / "same_s_decoder_dynamic.onnx")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or expected_hash != actual_hash:
        raise RuntimeError("Release decoder model_sha256 does not match the packaged ONNX file")

    notice = (resource_dir / "MODEL_NOTICE.md").read_text(encoding="utf-8")
    for required_text in (
        "This Stability AI Model is licensed under the Stability AI Community License",
        "Powered by Stability AI",
        "not licensed under the\nApache License 2.0",
    ):
        if required_text not in notice:
            raise RuntimeError(f"Release model notice is missing required text: {required_text}")


def _missing_text(path: Path, snippets: Iterable[str]) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [snippet for snippet in snippets if snippet not in text]


def check_public_tree(project_root: Path, *, check_history: bool = False) -> list[str]:
    """Return human-readable public-release compliance failures."""

    issues: list[str] = []
    for relative in REQUIRED_PUBLIC_FILES:
        if not (project_root / relative).is_file():
            issues.append(f"missing required file: {relative}")

    for relative, expected in PINNED_FILE_HASHES.items():
        path = project_root / relative
        if path.is_file() and sha256(path) != expected:
            issues.append(f"authoritative or vendored file changed: {relative}")

    required_text = {
        "README.md": (
            "Apache License 2.0",
            "Stability AI Community License",
            "Powered by Stability AI",
        ),
        "NOTICE": (
            "This Stability AI Model is licensed under the Stability AI Community License",
            "Gemma is provided under and subject to the Gemma Terms of Use",
        ),
        "docs/index.html": ("Powered by Stability AI", "THIRD_PARTY_NOTICES.md"),
        "web/index.html": ("Powered by Stability AI", "vendor/p5/p5.min.js"),
        "stable_audio_wanderer/resources/same_s/MODEL_NOTICE.md": (
            "Powered by Stability AI",
            "derived model artifact",
        ),
    }
    for relative, snippets in required_text.items():
        for snippet in _missing_text(project_root / relative, snippets):
            issues.append(f"{relative} is missing required text: {snippet}")

    paper = project_root / "smalley-spectromorphology.pdf"
    if paper.exists():
        issues.append("copyrighted Smalley PDF is present in the public tree")

    model_path = (
        project_root
        / "stable_audio_wanderer"
        / "resources"
        / "same_s"
        / "same_s_decoder_dynamic.onnx"
    )
    if model_path.is_file():
        try:
            validate_model_release(project_root)
        except RuntimeError as exc:
            issues.append(str(exc))

    if check_history:
        result = subprocess.run(
            ["git", "rev-list", "--objects", "--all"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            issues.append("could not inspect Git history for removed restricted files")
        elif "smalley-spectromorphology.pdf" in result.stdout:
            issues.append("copyrighted Smalley PDF remains in Git history")

    return issues


def check_dependency_license_report(report_path: Path) -> list[str]:
    """Flag unresolved or high-risk package licenses in a generated report."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    strong_copyleft = re.compile(r"(?<!L)GPL|AGPL", re.IGNORECASE)
    for package in report.get("packages", []):
        name = str(package.get("name", "unknown"))
        version = str(package.get("version", "unknown"))
        license_value = str(package.get("license", "NOASSERTION"))
        if license_value.upper() in {"NOASSERTION", "UNKNOWN", ""}:
            issues.append(f"dependency has no asserted license: {name}=={version}")
        elif (
            len(license_value) <= 512
            and strong_copyleft.search(license_value)
            and "GCC-exception" not in license_value
        ):
            issues.append(
                f"dependency needs manual strong-copyleft review: "
                f"{name}=={version} ({license_value})"
            )
    return issues
