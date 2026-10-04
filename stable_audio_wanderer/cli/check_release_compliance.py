#!/usr/bin/env python3
"""Check that the repository or release tree carries required license evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_audio_wanderer.release_compliance import (
    check_dependency_license_report,
    check_public_tree,
    validate_model_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check-history",
        action="store_true",
        help="Also fail if restricted files remain in any Git revision.",
    )
    parser.add_argument(
        "--license-report",
        type=Path,
        help="Also reject unknown or strong-copyleft entries in licenses.json.",
    )
    parser.add_argument('--model-bundle', type=Path, help='Verify an explicitly assembled decoder delivery')
    args = parser.parse_args()
    root = args.project_root.resolve()
    issues = check_public_tree(root, check_history=args.check_history)
    if args.license_report:
        issues.extend(check_dependency_license_report(args.license_report.resolve()))
    if args.model_bundle:
        try:
            validate_model_release(root, model_bundle=args.model_bundle)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(str(exc))
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print("Release compliance checks passed.")


if __name__ == "__main__":
    main()
