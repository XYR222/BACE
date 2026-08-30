#!/usr/bin/env python3
"""Safely prune old global_step checkpoints across separate trainer processes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


GIB = 1024 ** 3


def checkpoint_steps(root: Path) -> list[tuple[int, Path]]:
    steps = []
    if not root.exists():
        return steps
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("global_step_"):
            continue
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        steps.append((step, path))
    return sorted(steps)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def recommend_keep(root: Path, reserve_bytes: int = 10 * GIB) -> dict:
    root = root.resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"checkpoint tracker not found: {tracker}")
    latest = int(tracker.read_text().strip())
    latest_path = root / f"global_step_{latest}"
    if not latest_path.is_dir():
        raise FileNotFoundError(f"tracked checkpoint is missing: {latest_path}")
    checkpoint_bytes = directory_size(latest_path)
    free_bytes = shutil.disk_usage(root).free
    keep = 2 if free_bytes >= checkpoint_bytes + reserve_bytes else 1
    return {
        "root": str(root),
        "latest": latest,
        "checkpoint_bytes": checkpoint_bytes,
        "free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "recommended_keep": keep,
    }


def prune(root: Path, keep: int, dry_run: bool = False, require_bace_state: bool = True) -> dict:
    if keep < 1 or keep > 2:
        raise ValueError("keep must be 1 or 2")
    root = root.resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"checkpoint tracker not found: {tracker}")
    latest = int(tracker.read_text().strip())
    entries = checkpoint_steps(root)
    entry_by_step = dict(entries)
    if latest not in entry_by_step:
        raise FileNotFoundError(f"tracked checkpoint global_step_{latest} is missing")
    latest_path = entry_by_step[latest]
    required = [latest_path / "actor", latest_path / "data.pt"]
    # Pure GiGPO checkpoints intentionally have no BACE collector state.
    # Keep the historical strict check as the default for BACE runs.
    if require_bace_state:
        required.append(latest_path / "bace_collector_state.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"latest checkpoint is incomplete: {missing}")

    committed_entries = [(step, path) for step, path in entries if step <= latest]
    retained = [step for step, _ in committed_entries[-keep:]]
    if latest not in retained:
        raise RuntimeError("latest tracked checkpoint would not be retained")
    removed = []
    for step, path in entries:
        if step in retained:
            continue
        if path.parent != root or not path.name.startswith("global_step_"):
            raise RuntimeError(f"refusing unsafe checkpoint deletion: {path}")
        removed.append(str(path))
        if not dry_run:
            shutil.rmtree(path)
    return {"root": str(root), "latest": latest, "retained": retained, "removed": removed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--keep", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recommend", action="store_true")
    parser.add_argument("--allow-missing-bace-state", action="store_true")
    parser.add_argument("--reserve-gib", type=int, default=10)
    args = parser.parse_args()
    if args.recommend:
        result = recommend_keep(args.checkpoint_root, args.reserve_gib * GIB)
    else:
        if args.keep is None:
            parser.error("--keep is required unless --recommend is used")
        result = prune(args.checkpoint_root, args.keep, args.dry_run,
                       require_bace_state=not args.allow_missing_bace_state)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
