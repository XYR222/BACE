#!/usr/bin/env python3
"""Validate cgroup task usage recorded during a multi-step H100 run."""

import argparse
import csv
import json
from pathlib import Path


def analyze(rows: list[dict[str, str]], max_pids: int) -> dict:
    numeric = [row for row in rows if row.get("pids_current", "").isdigit()]
    if not numeric:
        raise ValueError("no numeric cgroup PID samples")

    values = [int(row["pids_current"]) for row in numeric]
    return {
        "ok": max(values) <= max_pids,
        "sample_count": len(values),
        "first_pids_current": values[0],
        "last_pids_current": values[-1],
        "peak_pids_current": max(values),
        "configured_max_pids": max_pids,
        "cgroup_pids_limit": numeric[0].get("pids_limit"),
        "pids_sources": sorted(
            {
                row.get("pids_source", "legacy") or "legacy"
                for row in numeric
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--max-pids", type=int, default=12000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.samples.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        payload = analyze(rows, args.max_pids)
    except ValueError as error:
        raise SystemExit(f"{error} in {args.samples}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["ok"]:
        raise SystemExit(
            f"cgroup task usage peaked at {payload['peak_pids_current']}, "
            f"above safety limit {args.max_pids}"
        )


if __name__ == "__main__":
    main()
