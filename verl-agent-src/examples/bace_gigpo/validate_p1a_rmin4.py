#!/usr/bin/env python3
"""Validate the one-step conservative R_min=4 diagnostic smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = sorted(args.artifact_dir.glob("step_*/summary.json"))
    if not summaries:
        raise SystemExit(f"No step summary found below {args.artifact_dir}")
    summary_path = summaries[-1]
    summary = json.loads(summary_path.read_text())
    metrics = summary.get("bace_metrics", {})
    errors: list[str] = []

    def number(key: str) -> float:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite metric: {key}")
            return float("nan")
        return float(value)

    planned_roots = number("planned_roots_mean")
    planned_branches = number("planned_branches_mean")
    final_roots = number("final_roots_mean")
    final_branches = number("final_branches_mean")
    requested = number("requested")
    initial_capacity = number("initial_information_capacity_mean")
    initial_deficit = number("initial_quota_deficit_mean")
    normalized_deficit = number("normalized_initial_quota_deficit_mean")

    if number("quota_policy_conservative_rmin4") != 1.0:
        errors.append("conservative R_min=4 policy flag was not active")
    if planned_roots < 4.0 or planned_branches > 4.0:
        errors.append(
            f"planned quota violates R_min=4/Q_max=4: R={planned_roots}, Q={planned_branches}"
        )
    if not math.isclose(final_roots + final_branches, 8.0, abs_tol=1e-6):
        errors.append(f"final R+Q is not 8: R={final_roots}, Q={final_branches}")
    if requested <= 0:
        errors.append("smoke did not exercise branch replay")
    if min(initial_capacity, initial_deficit, normalized_deficit) < 0.0:
        errors.append("initial capacity/deficit accounting contains a negative value")

    for key in (
        "batch_erv_exact",
        "staged_root_batching_packed",
        "branch_selected_worker_execution",
        "branch_main_pool_reuse",
        "root_active_executor_enabled",
    ):
        if number(key) != 1.0:
            errors.append(f"required execution flag is not active: {key}")

    for key in (
        "root_generated_tokens",
        "branch_generated_tokens",
        "root_trainable_tokens",
        "branch_trainable_tokens",
        "corrections_total",
        "correction_rate_normalized",
        "root_generation_seconds_total",
    ):
        if number(key) < 0.0:
            errors.append(f"metric must be non-negative: {key}")

    report = {
        "ok": not errors,
        "summary": str(summary_path),
        "quota_policy": "conservative_rmin4",
        "planned_roots_mean": planned_roots,
        "planned_branches_mean": planned_branches,
        "final_roots_mean": final_roots,
        "final_branches_mean": final_branches,
        "requested": requested,
        "initial_information_capacity_mean": initial_capacity,
        "initial_quota_deficit_mean": initial_deficit,
        "normalized_initial_quota_deficit_mean": normalized_deficit,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
