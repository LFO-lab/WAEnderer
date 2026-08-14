#!/usr/bin/env python3
"""Generate a CycloneDX SBOM and license report for the active environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import tomllib
from typing import Any
from urllib.parse import quote
import uuid


def _license_value(dist: metadata.Distribution) -> str:
    expression = dist.metadata.get("License-Expression")
    if expression:
        return expression.strip()
    value = dist.metadata.get("License")
    if value and value.strip() and value.strip().upper() != "UNKNOWN":
        return re.sub(r"\s+", " ", value.strip())
    classifiers = [
        item.removeprefix("License :: ")
        for item in dist.metadata.get_all("Classifier", [])
        if item.startswith("License :: ")
    ]
    return " AND ".join(classifiers) if classifiers else "NOASSERTION"


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_names(lock_file: Path | None) -> set[str] | None:
    if lock_file is None or not lock_file.is_file():
        return None
    lock = tomllib.loads(lock_file.read_text(encoding="utf-8"))
    return {_normalized_name(str(package["name"])) for package in lock.get("package", [])}


def _license_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[str, dict[str, str]] = {}
    for item in payload.get("overrides", []):
        name = _normalized_name(str(item["name"]))
        version = str(item["version"])
        key = f"{name}=={version}"
        overrides[key] = {
            "license": str(item["license"]),
            "license_evidence": str(item["evidence_url"]),
            "license_note": str(item["reason"]),
        }
    return overrides


def _apply_license_overrides(
    packages: dict[str, dict[str, str]], overrides: dict[str, dict[str, str]]
) -> None:
    for key, package in packages.items():
        override = overrides.get(f"{key}=={package['version']}")
        if package["license"] == "NOASSERTION" and override is not None:
            package.update(override)


def _packages(
    lock_file: Path | None = None,
    overrides_file: Path | None = Path("licenses/dependency-license-overrides.json"),
) -> list[dict[str, str]]:
    locked = _locked_names(lock_file)
    packages: dict[str, dict[str, str]] = {}
    search_paths = list(sys.path)
    if lock_file is not None:
        search_paths.insert(0, str(lock_file.resolve().parent))
    for dist in metadata.distributions(path=search_paths):
        name = dist.metadata.get("Name") or "unknown"
        key = _normalized_name(name)
        if locked is not None and key not in locked:
            continue
        candidate = {
            "name": name,
            "version": dist.version,
            "license": _license_value(dist),
        }
        existing = packages.get(key)
        if existing is None or (
            existing["license"] == "NOASSERTION"
            and candidate["license"] != "NOASSERTION"
        ):
            packages[key] = candidate

    _apply_license_overrides(packages, _license_overrides(overrides_file))
    return [packages[key] for key in sorted(packages)]


def generate(
    output_dir: Path,
    lock_file: Path | None = Path("uv.lock"),
    overrides_file: Path | None = Path("licenses/dependency-license-overrides.json"),
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = _packages(lock_file, overrides_file)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    components: list[dict[str, Any]] = []
    for package in packages:
        name = package["name"]
        version = package["version"]
        license_entry = {"name": package["license"]}
        if package.get("license_evidence"):
            license_entry["url"] = package["license_evidence"]
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{quote(name.lower())}@{quote(version)}",
                "licenses": [{"license": license_entry}],
            }
        )

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "WÆnderer release evidence generator",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
    }
    report = {
        "generated_at": timestamp,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lock_file": str(lock_file) if lock_file and lock_file.is_file() else None,
        "package_count": len(packages),
        "packages": packages,
    }

    sbom_path = output_dir / "sbom.cdx.json"
    report_path = output_dir / "licenses.json"
    sbom_path.write_text(json.dumps(sbom, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sbom_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/release-evidence"),
        help="Directory for sbom.cdx.json and licenses.json.",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("uv.lock"),
        help="Restrict evidence to distributions present in this lockfile.",
    )
    parser.add_argument(
        "--license-overrides",
        type=Path,
        default=Path("licenses/dependency-license-overrides.json"),
        help="Reviewed exact-version evidence for packages with incomplete metadata.",
    )
    args = parser.parse_args()
    sbom, licenses = generate(args.output_dir, args.lock_file, args.license_overrides)
    print(f"Wrote {sbom}")
    print(f"Wrote {licenses}")


if __name__ == "__main__":
    main()
