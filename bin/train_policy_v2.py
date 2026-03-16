#!/usr/bin/env python3
"""
Compatibility wrapper for reorganized model training.

Deprecated: use `bin/train_policy.py --navigation_mode reorganized`.
"""
import os
import subprocess
import sys


def main():
    train_policy_script = os.path.join(os.path.dirname(__file__), "train_policy.py")
    forwarded_args = list(sys.argv[1:])
    if "--navigation_mode" not in forwarded_args:
        forwarded_args.extend(["--navigation_mode", "reorganized"])
    cmd = [sys.executable, train_policy_script, *forwarded_args]
    print(
        "[warn] bin/train_policy_v2.py is deprecated. "
        "Forwarding to train_policy.py --navigation_mode reorganized."
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
