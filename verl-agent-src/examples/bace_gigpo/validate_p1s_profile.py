#!/usr/bin/env python3
"""Validate the execution-only invariants of a P1-S profile run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_summary(artifact_dir: Path) -> tuple[Path, dict]:
    summaries = sorted(artifact_dir.glob("step_*/summary.json"))
    if not summaries:
        raise SystemExit(f"No step summary found below {artifact_dir}")
    path = summaries[-1]
    return path, json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--pool-mode", choices=("dedicated", "main_reuse"), required=True)
    parser.add_argument(
        "--root-active", choices=("true", "false"), default="false"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path, summary = latest_summary(args.artifact_dir)
    metrics = summary.get("bace_metrics", {})
    requested = int(metrics.get("requested", 0))
    capacity = int(metrics.get("branch_replay_capacity", 0))
    cohorts = int(
        metrics.get(
            "branch_capacity_cohorts",
            metrics.get("branch_execution_waves", 0),
        )
    )
    expected_main_reuse = args.pool_mode == "main_reuse"
    observed_main_reuse = bool(metrics.get("branch_main_pool_reuse", 0))
    expected_root_active = args.root_active == "true"
    observed_root_active = bool(metrics.get("root_active_executor_enabled", 0))
    errors = []
    if requested <= 0:
        errors.append("profile did not exercise a branch")
    if capacity <= 0:
        errors.append("branch replay capacity was not recorded")
    if cohorts <= 0:
        errors.append("branch capacity cohort count was not recorded")
    if observed_main_reuse != expected_main_reuse:
        errors.append(
            f"branch pool mode mismatch: expected main_reuse={expected_main_reuse}, "
            f"observed {observed_main_reuse}"
        )
    if observed_root_active != expected_root_active:
        errors.append(
            f"root executor mismatch: expected active={expected_root_active}, "
            f"observed {observed_root_active}"
        )
    if expected_root_active:
        logical = int(metrics.get("root_logical_trajectories", 0))
        active_rows = int(metrics.get("root_active_sequences", 0))
        dense_rows = int(metrics.get("root_dense_equivalent_sequences", 0))
        submitted_rows = int(metrics.get("root_model_rows_submitted", 0))
        padding_rows = int(metrics.get("root_padding_rows_submitted", 0))
        if logical <= 0 or active_rows <= 0 or dense_rows <= 0:
            errors.append("active-root accounting did not record executed roots")
        if active_rows > dense_rows:
            errors.append("active-root rows exceed their dense equivalent")
        if submitted_rows != active_rows + padding_rows:
            errors.append(
                "active-root submitted rows do not equal active plus padding rows"
            )
    if float(metrics.get("branch_selected_worker_execution", 0)) != 1.0:
        errors.append("selected-worker branch execution was not active")
    if float(metrics.get("branch_slot_collisions", 0)) != 0.0:
        errors.append("branch physical slot collision was recorded")
    if requested <= capacity and cohorts != 1:
        errors.append(
            f"{requested} requests fit capacity {capacity}, but used {cohorts} cohorts"
        )

    report = {
        "ok": not errors,
        "pool_mode": args.pool_mode,
        "root_active": expected_root_active,
        "summary": str(summary_path),
        "requested": requested,
        "replay_capacity": capacity,
        "capacity_cohorts": cohorts,
        "branch_suffix_generation_seconds": metrics.get(
            "branch_suffix_generation_seconds"
        ),
        "root_active_sequences": metrics.get("root_active_sequences"),
        "root_dense_equivalent_sequences": metrics.get(
            "root_dense_equivalent_sequences"
        ),
        "root_rows_avoided": metrics.get("root_rows_avoided"),
        "root_padding_rows_submitted": metrics.get(
            "root_padding_rows_submitted"
        ),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
