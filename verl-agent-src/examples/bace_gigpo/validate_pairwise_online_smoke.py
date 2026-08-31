#!/usr/bin/env python3
"""Validate the online C0/C1/C2 smoke invariants from BACE artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--arm", choices=("c0", "c1", "c2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps = []
    errors = []
    for step_dir in sorted(args.artifact_root.glob("step_*")):
        summary_path = step_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        metrics = summary.get("bace_metrics", {})
        step_errors = []
        if summary.get("status") != "complete":
            step_errors.append("summary_status_not_complete")
        for key, expected in {
            "batch_erv_exact": 1.0,
            "staged_root_batching_packed": 1.0,
            "branch_selected_worker_execution": 1.0,
        }.items():
            if float(metrics.get(key, 0.0)) != expected:
                step_errors.append(f"{key}!={expected}")
        if abs(float(metrics.get("final_roots_mean", 0.0)) + float(
            metrics.get("final_branches_mean", 0.0)
        ) - 8.0) > 1e-9:
            step_errors.append("mean_leaf_budget_not_8")

        rounds = jsonl(step_dir / "acquisition_rounds.jsonl")
        active_rounds = [item for item in rounds if item.get("requests")]
        requested = float(metrics.get("requested", 0.0))
        # Each record is emitted once by coordinator.build_round_requests(),
        # before any branch execution.  A late round can legitimately contain
        # branches of a single still-active task, so task-count alone is not a
        # valid sequential-execution detector.
        if not active_rounds and requested > 0:
            step_errors.append("missing_global_acquisition_round_record")

        if args.arm == "c0":
            if float(metrics.get("pairwise_fixed", 0.0)) != 0.0 or float(
                metrics.get("pairwise_stopping", 0.0)
            ) != 0.0:
                step_errors.append("c0_pairwise_flag")
        elif args.arm == "c1":
            if float(metrics.get("pairwise_fixed", 0.0)) != 1.0 or float(
                metrics.get("pairwise_stopping", 0.0)
            ) != 0.0:
                step_errors.append("c1_pairwise_flags")
            if float(metrics.get("fallback_root_count", 0.0)) != 0.0:
                step_errors.append("c1_has_fallback_roots")
            if requested > 0 and float(metrics.get("pairwise_branch_rounds", 0.0)) < 1.0:
                step_errors.append("c1_has_no_pairwise_round")
        else:
            if float(metrics.get("pairwise_fixed", 0.0)) != 0.0 or float(
                metrics.get("pairwise_stopping", 0.0)
            ) != 1.0:
                step_errors.append("c2_pairwise_flags")
            checks = jsonl(step_dir / "capacity_checks.jsonl")
            if requested > 0 and not any(
                check.get("lazy_pairwise_capacity") for check in checks
            ):
                step_errors.append("c2_missing_lazy_capacity_record")
            if requested > 0 and float(metrics.get("pairwise_branch_rounds", 0.0)) < 1.0:
                step_errors.append("c2_has_no_pairwise_round")

        steps.append({
            "step": summary.get("step"),
            "ok": not step_errors,
            "errors": step_errors,
            "pairwise_rounds": metrics.get("pairwise_branch_rounds"),
            "requested": requested,
            "fallback_root_count": metrics.get("fallback_root_count", 0.0),
            "global_round_records": len(active_rounds),
            "time": {
                key: metrics.get(key)
                for key in (
                    "time/initial_root", "time/capacity_correction",
                    "time/branch_generation", "time/fallback_root",
                    "time/acquisition_compute", "time/rollout_total",
                )
            },
        })
        errors.extend(f"step={summary.get('step')}:{error}" for error in step_errors)

    if not steps:
        errors.append("no_complete_step_artifacts")
    report = {"ok": not errors, "arm": args.arm, "steps": steps, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
