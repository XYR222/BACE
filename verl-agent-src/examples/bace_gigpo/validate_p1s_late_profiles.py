#!/usr/bin/env python3
"""Validate late-checkpoint S0-S3 profiles and select S1 or S3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "s0": {"main_reuse": False, "root_active": False},
    "s1": {"main_reuse": True, "root_active": False},
    "s2": {"main_reuse": False, "root_active": True},
    "s3": {"main_reuse": True, "root_active": True},
}


def latest_summary(path: Path) -> tuple[Path, dict]:
    summaries = sorted(path.glob("step_*/summary.json"))
    if not summaries:
        raise ValueError(f"No summary.json found below {path}")
    summary_path = summaries[-1]
    return summary_path, json.loads(summary_path.read_text())


def number(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value if value is not None else default)


def reduction(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return 1.0 - after / before


def main() -> None:
    parser = argparse.ArgumentParser()
    for label in EXPECTED:
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=151)
    parser.add_argument("--minimum-root-gain", type=float, default=0.02)
    parser.add_argument("--pair-regression-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    profiles: dict[str, dict] = {}
    for label, expected in EXPECTED.items():
        artifact_dir: Path = getattr(args, label)
        try:
            summary_path, summary = latest_summary(artifact_dir)
        except (ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label}: {error}")
            continue
        metrics = summary.get("bace_metrics", {})
        observed_main_reuse = bool(number(metrics, "branch_main_pool_reuse"))
        observed_root_active = bool(number(metrics, "root_active_executor_enabled"))
        requested = int(number(metrics, "branch_requests_total", metrics.get("requested", 0)))
        cohorts = int(number(metrics, "branch_capacity_cohorts"))
        active_rows = number(metrics, "root_active_sequences")
        dense_rows = number(metrics, "root_dense_equivalent_sequences")
        submitted_rows = number(metrics, "root_model_rows_submitted")
        padding_rows = number(metrics, "root_padding_rows_submitted")
        profile = {
            "summary": str(summary_path),
            "status": summary.get("status"),
            "step": int(summary.get("step", -1)),
            "main_reuse": observed_main_reuse,
            "root_active": observed_root_active,
            "requested": requested,
            "capacity_cohorts": cohorts,
            "branch_suffix_generation_seconds": number(
                metrics, "branch_suffix_generation_seconds"
            ),
            "root_generation_seconds": number(
                metrics, "planned_root_generation_seconds"
            ),
            "root_active_sequences": active_rows,
            "root_dense_equivalent_sequences": dense_rows,
            "root_rows_avoided": number(metrics, "root_rows_avoided"),
            "root_model_rows_submitted": submitted_rows,
            "root_padding_rows_submitted": padding_rows,
        }
        profiles[label] = profile
        if profile["status"] != "complete":
            errors.append(f"{label}: summary status is not complete")
        if profile["step"] != args.expected_step:
            errors.append(
                f"{label}: expected resumed step {args.expected_step}, "
                f"got {profile['step']}"
            )
        if observed_main_reuse != expected["main_reuse"]:
            errors.append(f"{label}: branch pool mode mismatch")
        if observed_root_active != expected["root_active"]:
            errors.append(f"{label}: root executor mode mismatch")
        if requested <= 0 or cohorts <= 0:
            errors.append(f"{label}: branch execution was not exercised")
        if number(metrics, "branch_selected_worker_execution") != 1.0:
            errors.append(f"{label}: selected-worker branch execution was not active")
        if number(metrics, "branch_slot_collisions") != 0.0:
            errors.append(f"{label}: physical branch slot collision recorded")
        if number(metrics, "batch_erv_exact") != 1.0:
            errors.append(f"{label}: Exact Batch-ERV flag is missing")
        if number(metrics, "staged_root_batching_packed") != 1.0:
            errors.append(f"{label}: packed-root flag is missing")
        if expected["root_active"]:
            if active_rows <= 0 or dense_rows <= 0 or active_rows > dense_rows:
                errors.append(f"{label}: invalid active-root accounting")
            if submitted_rows != active_rows + padding_rows:
                errors.append(f"{label}: submitted rows != active rows + padding")

    comparisons: dict[str, float | None] = {}
    selected_mode = "s1"
    if not errors and set(profiles) == set(EXPECTED):
        s0, s1, s2, s3 = (profiles[label] for label in EXPECTED)
        if s1["capacity_cohorts"] > s0["capacity_cohorts"]:
            errors.append("S1 used more branch capacity cohorts than S0")
        if s3["capacity_cohorts"] > s2["capacity_cohorts"]:
            errors.append("S3 used more branch capacity cohorts than S2")
        comparisons = {
            "s1_branch_gain": reduction(
                s0["branch_suffix_generation_seconds"],
                s1["branch_suffix_generation_seconds"],
            ),
            "s3_branch_gain": reduction(
                s2["branch_suffix_generation_seconds"],
                s3["branch_suffix_generation_seconds"],
            ),
            "s2_root_gain": reduction(
                s0["root_generation_seconds"], s2["root_generation_seconds"]
            ),
            "s3_root_gain": reduction(
                s1["root_generation_seconds"], s3["root_generation_seconds"]
            ),
        }
        root_gains = [
            gain
            for gain in (comparisons["s2_root_gain"], comparisons["s3_root_gain"])
            if gain is not None
        ]
        avoided = s2["root_rows_avoided"] + s3["root_rows_avoided"]
        meaningful_root_gain = (
            len(root_gains) == 2
            and sum(root_gains) / len(root_gains) >= args.minimum_root_gain
            and min(root_gains) >= -args.pair_regression_tolerance
        )
        if avoided > 0 and meaningful_root_gain:
            selected_mode = "s3"
        else:
            reason = []
            if avoided <= 0:
                reason.append("active profiles contained no early-finished root rows")
            if not meaningful_root_gain:
                reason.append("paired root wall-clock gain did not pass the threshold")
            warnings.append("S2 production gain not established: " + "; ".join(reason))

    report = {
        "ok": not errors,
        "selected_mode": selected_mode if not errors else None,
        "selection_policy": {
            "minimum_root_gain": args.minimum_root_gain,
            "pair_regression_tolerance": args.pair_regression_tolerance,
            "fallback": "s1",
        },
        "profiles": profiles,
        "comparisons": comparisons,
        "warnings": warnings,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(selected_mode + "\n")


if __name__ == "__main__":
    main()
