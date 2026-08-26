#!/usr/bin/env python3
"""Validate that natural ALFWorld tasks advance across training steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _reset_keys_by_task(step_dir: Path) -> dict[int, str]:
    roots_path = step_dir / "roots.jsonl"
    if not roots_path.is_file():
        raise RuntimeError(f"missing roots artifact: {roots_path}")

    keys_by_task: dict[int, set[str]] = {}
    with roots_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task_index = int(row["task_batch_index"])
            reset_key = str(row["environment_reset_key"])
            keys_by_task.setdefault(task_index, set()).add(reset_key)

    if not keys_by_task:
        raise RuntimeError(f"no natural roots found in {roots_path}")
    ambiguous = {
        task_index: sorted(keys)
        for task_index, keys in keys_by_task.items()
        if len(keys) != 1
    }
    if ambiguous:
        raise RuntimeError(
            f"a task used multiple natural game files in one step: {ambiguous}"
        )
    return {task_index: next(iter(keys)) for task_index, keys in keys_by_task.items()}


def validate_rotation(artifact_root: Path, first_step: int, second_step: int) -> dict:
    first = _reset_keys_by_task(artifact_root / f"step_{first_step:08d}")
    second = _reset_keys_by_task(artifact_root / f"step_{second_step:08d}")
    if first.keys() != second.keys():
        raise RuntimeError(
            f"task indices changed between steps: {sorted(first)} != {sorted(second)}"
        )

    unchanged = [index for index in sorted(first) if first[index] == second[index]]
    if unchanged:
        raise RuntimeError(
            "natural ALFWorld workers remained pinned across steps for task indices "
            f"{unchanged}"
        )

    return {
        "ok": True,
        "first_step": first_step,
        "second_step": second_step,
        "task_count": len(first),
        "changed_task_count": len(first),
        "first_reset_keys": first,
        "second_reset_keys": second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--second-step", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_rotation(args.artifact_root, args.first_step, args.second_step)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
