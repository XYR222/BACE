#!/usr/bin/env python3
"""Fail closed when formal BACE outputs are not on the intended filesystem."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


GIB = 1024**3


def check_storage(
    output_root: Path,
    expected_prefix: Path,
    minimum_free_bytes: int,
) -> dict:
    resolved_root = output_root.resolve(strict=True)
    resolved_prefix = expected_prefix.resolve(strict=True)
    try:
        resolved_root.relative_to(resolved_prefix)
    except ValueError as error:
        raise ValueError(
            f"output root {resolved_root} is outside {resolved_prefix}"
        ) from error

    usage = shutil.disk_usage(resolved_root)
    payload = {
        "ok": usage.free >= minimum_free_bytes,
        "configured_output_root": str(output_root.absolute()),
        "resolved_output_root": str(resolved_root),
        "expected_prefix": str(resolved_prefix),
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
    }
    if not payload["ok"]:
        raise ValueError(
            f"only {usage.free} bytes free at {resolved_root}; "
            f"at least {minimum_free_bytes} required"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-prefix", required=True, type=Path)
    parser.add_argument("--minimum-free-gib", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        payload = check_storage(
            args.output_root,
            args.expected_prefix,
            args.minimum_free_gib * GIB,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
