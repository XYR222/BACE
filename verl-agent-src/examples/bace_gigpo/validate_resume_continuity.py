#!/usr/bin/env python3
"""Validate that a new trainer process resumed all step-coupled BACE state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def history_from_checkpoint(checkpoint: dict) -> dict[str, list[float]]:
    families = checkpoint["collector"]["competence_history"]["families"]
    return {
        family: [
            float(values["decayed_successes"]),
            float(values["decayed_failures"]),
        ]
        for family, values in families.items()
    }


def find_trace_step(artifact_root: Path, step: int) -> Path:
    matches = []
    for summary_path in artifact_root.glob("step_*/summary.json"):
        summary = read_json(summary_path)
        if int(summary.get("step", -1)) == step and summary.get("status") == "complete":
            matches.append(summary_path.parent)
    if not matches:
        raise FileNotFoundError(f"no complete trace for step {step} under {artifact_root}")
    return sorted(matches)[-1]


def validate(checkpoint_root: Path, artifact_root: Path, first_step: int, second_step: int):
    errors = []
    first_dir = checkpoint_root / f"global_step_{first_step}"
    second_dir = checkpoint_root / f"global_step_{second_step}"
    for step_dir in (first_dir, second_dir):
        for relative in ("actor", "data.pt", "bace_collector_state.json"):
            if not (step_dir / relative).exists():
                errors.append(f"missing {step_dir / relative}")
        actor_dir = step_dir / "actor"
        for pattern, component in (
            ("model_world_size_*_rank_*.pt", "model"),
            ("optim_world_size_*_rank_*.pt", "optimizer"),
            ("extra_state_world_size_*_rank_*.pt", "scheduler/RNG"),
        ):
            if actor_dir.is_dir() and not list(actor_dir.glob(pattern)):
                errors.append(f"missing actor {component} shards in {actor_dir}")

    first_state = read_json(first_dir / "bace_collector_state.json")
    second_state = read_json(second_dir / "bace_collector_state.json")
    if int(first_state.get("global_step", -1)) != first_step:
        errors.append("first collector state has the wrong global_step")
    if int(second_state.get("global_step", -1)) != second_step:
        errors.append("second collector state has the wrong global_step")
    if int(first_state.get("collector", {}).get("current_step", -1)) != first_step:
        errors.append("first collector current_step does not match checkpoint")
    if int(second_state.get("collector", {}).get("current_step", -1)) != second_step:
        errors.append("second collector current_step does not match checkpoint")

    trace_dir = find_trace_step(artifact_root, second_step)
    updates = read_jsonl(trace_dir / "family_history_updates.jsonl")
    if len(updates) != 1:
        errors.append(f"expected one family history update at step {second_step}, got {len(updates)}")
        history_before = None
    else:
        history_before = {
            family: [float(value[0]), float(value[1])]
            for family, value in updates[0]["history_before"].items()
        }
    checkpoint_history_after = history_from_checkpoint(first_state)
    if history_before != checkpoint_history_after:
        errors.append("step-2 history_before differs from step-1 checkpoint history_after")

    return {
        "ok": not errors,
        "errors": errors,
        "first_step": first_step,
        "second_step": second_step,
        "checkpoint_history_after_step_1": checkpoint_history_after,
        "trace_history_before_step_2": history_before,
        "trace_step_dir": str(trace_dir),
        "restored_components": {
            "actor": not any("actor" in error for error in errors),
            "dataloader": not any("data.pt" in error for error in errors),
            "global_step": not any("global_step" in error for error in errors),
            "family_history": history_before == checkpoint_history_after,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--second-step", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        args.checkpoint_root.resolve(),
        args.artifact_root.resolve(),
        args.first_step,
        args.second_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
