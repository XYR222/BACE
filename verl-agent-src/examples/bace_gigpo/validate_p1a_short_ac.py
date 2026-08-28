#!/usr/bin/env python3
"""Validate and summarize the paired step145->150 P1-A mechanism diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_SOURCE_STEP = 148


def load_run(
    artifact_dir: Path, expected_policy_flag: str, source_step: int
) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    summaries: list[dict] = []
    observed_steps: list[int] = []
    expected_steps = list(range(source_step + 1, 151))
    for path in sorted(artifact_dir.glob("step_*/summary.json")):
        payload = json.loads(path.read_text())
        step = int(payload.get("step", int(path.parent.name.split("_")[-1])))
        if step not in expected_steps:
            continue
        observed_steps.append(step)
        summaries.append(payload)
        if payload.get("status") != "complete":
            errors.append(f"incomplete artifact: {path}")
        metrics = payload.get("bace_metrics", {})
        if float(metrics.get(expected_policy_flag, 0.0)) != 1.0:
            errors.append(f"{path}: missing policy flag {expected_policy_flag}")
        if float(metrics.get("batch_erv_exact", 0.0)) != 1.0:
            errors.append(f"{path}: Exact Batch-ERV was not active")
        if float(metrics.get("staged_root_batching_packed", 0.0)) != 1.0:
            errors.append(f"{path}: packed root generation was not active")
        if float(metrics.get("branch_selected_worker_execution", 0.0)) != 1.0:
            errors.append(f"{path}: selected-worker branch execution was not active")
        if float(metrics.get("branch_main_pool_reuse", 0.0)) != 1.0:
            errors.append(f"{path}: main rollout pool reuse was not active")
        roots = float(metrics.get("final_roots_mean", float("nan")))
        branches = float(metrics.get("final_branches_mean", float("nan")))
        if not math.isclose(roots + branches, 8.0, abs_tol=1e-6):
            errors.append(f"{path}: final R+Q is not 8 ({roots}+{branches})")
    if observed_steps != expected_steps:
        errors.append(f"expected steps {expected_steps}, observed {observed_steps}")
    return summaries, errors


def aggregate(summaries: list[dict]) -> dict:
    keys = (
        "planned_branches_total",
        "final_branches_total",
        "corrections_total",
        "correction_rate_normalized",
        "normalized_initial_quota_deficit_mean",
        "correction_depth_p90",
        "root_generation_seconds_total",
        "branch_suffix_generation_seconds",
        "root_generated_tokens",
        "branch_generated_tokens",
        "root_trainable_tokens",
        "branch_trainable_tokens",
    )
    result: dict[str, float] = {}
    for key in keys:
        values = [
            float(summary.get("bace_metrics", {}).get(key, 0.0))
            for summary in summaries
        ]
        result[f"{key}_sum"] = sum(values)
        result[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
    result["steps"] = [int(summary.get("step", 0)) for summary in summaries]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--source-step", type=int, default=DEFAULT_SOURCE_STEP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a_summaries, a_errors = load_run(
        args.a, "quota_policy_current", args.source_step
    )
    c_summaries, c_errors = load_run(
        args.c, "quota_policy_conservative_rmin4", args.source_step
    )
    errors = [f"A: {error}" for error in a_errors]
    errors.extend(f"C: {error}" for error in c_errors)

    if not args.migration.is_file():
        migration = None
        errors.append(f"missing C migration audit: {args.migration}")
    else:
        migration = json.loads(args.migration.read_text())
        expected_diff = {"min_natural_roots": {"saved": 2, "current": 4}}
        if migration.get("migration_policy") != "diagnostic_rmin2_to4":
            errors.append("C migration policy is not diagnostic_rmin2_to4")
        if migration.get("allowed_diff") != expected_diff:
            errors.append(
                f"C migration contains a non-whitelisted diff: {migration.get('allowed_diff')!r}"
            )
        if int(migration.get("source_global_step", -1)) != args.source_step:
            errors.append(
                "C migration did not originate at global step "
                f"{args.source_step}"
            )

    report = {
        "ok": not errors,
        "interpretation_scope": "mechanism_and_profile_only",
        "a": aggregate(a_summaries),
        "c": aggregate(c_summaries),
        "migration": migration,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
