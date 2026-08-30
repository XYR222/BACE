#!/usr/bin/env python3
"""Step 1: exact offline threshold sensitivity for Pairwise-Stopping.

This intentionally reads the same acquisition trace and production Exact
Batch-ERV implementation as A1.  It is CPU-only and evaluates each threshold
with posterior-predictive Beta-Binomial outcome enumeration; it never clips a
historical selected plan after the fact.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import a1_pairwise_feedback_audit as a1


DEFAULT_OUTPUT = a1.REPO_ROOT / "analysis_outputs/Pairwise_Threshold_Sensitivity"
THRESHOLDS = (0.005, 0.0075, 0.010, 0.015)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=a1.DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--max-branches-per-anchor", type=int, default=2)
    parser.add_argument("--pair-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-min", type=int, default=1)
    parser.add_argument("--step-max", type=int, default=150)
    return parser.parse_args()


def p2_policy(
    posteriors, used, q, *, threshold, max_per_anchor, pair_size, seed, prefix,
    round_index=1, memo=None,
):
    """Expected P2 result under C=1 single-branch and C=0 one-way fallback."""
    if q <= 0:
        return {"value": 0.0, "executed": 0.0, "fallback": 0.0, "rounds": 0.0}
    memo = {} if memo is None else memo
    key = (round_index, a1.state_signature(posteriors, used, q))
    if key in memo:
        return memo[key]
    engine, designs, capacity = a1.design_state(
        posteriors, used, threshold=threshold,
        max_per_anchor=max_per_anchor, seed=seed,
    )
    if capacity == 0:
        result = {"value": 0.0, "executed": 0.0, "fallback": float(q), "rounds": 0.0}
        memo[key] = result
        return result
    k = min(pair_size, q, capacity)
    plan = a1.solve_plan(
        posteriors, used, k, threshold=threshold,
        max_per_anchor=max_per_anchor, seed=seed,
        seed_parts=prefix + ("p2", round_index, key[1]),
    )
    if plan is None:
        raise AssertionError("positive Pairwise capacity failed to yield an Exact plan")
    before = a1.utility(posteriors)
    result = {"value": 0.0, "executed": 0.0, "fallback": 0.0, "rounds": 0.0}
    for probability, successes, totals in a1.enumerate_outcomes(posteriors, plan.selections, engine):
        updated = a1.apply_count_outcome(posteriors, successes, totals)
        next_used = dict(used)
        for anchor, count in Counter(anchor for anchor, _ in plan.selections).items():
            next_used[anchor] = next_used.get(anchor, 0) + count
        future = p2_policy(
            updated, next_used, q-k, threshold=threshold,
            max_per_anchor=max_per_anchor, pair_size=pair_size, seed=seed,
            prefix=prefix, round_index=round_index+1, memo=memo,
        )
        result["value"] += probability * (a1.utility(updated) - before + future["value"])
        result["executed"] += probability * (k + future["executed"])
        result["fallback"] += probability * future["fallback"]
        result["rounds"] += probability * (1 + future["rounds"])
    memo[key] = result
    return result


def post_pair_pressures(posteriors, q, *, threshold, max_per_anchor, seed, prefix):
    """Exact probability that the first P2 round leaves zero/short capacity."""
    used = {anchor: 0 for anchor in posteriors}
    engine, designs, capacity = a1.design_state(
        posteriors, used, threshold=threshold,
        max_per_anchor=max_per_anchor, seed=seed,
    )
    if capacity == 0:
        return 1.0, 1.0
    k = min(2, q, capacity)
    plan = a1.solve_plan(
        posteriors, used, k, threshold=threshold,
        max_per_anchor=max_per_anchor, seed=seed,
        seed_parts=prefix + ("pressure",),
    )
    if plan is None or q == k:
        return 0.0, 0.0
    zero = shortfall = 0.0
    for probability, successes, totals in a1.enumerate_outcomes(posteriors, plan.selections, engine):
        updated = a1.apply_count_outcome(posteriors, successes, totals)
        next_used = dict(used)
        for anchor, count in Counter(anchor for anchor, _ in plan.selections).items():
            next_used[anchor] += count
        _, _, remaining_capacity = a1.design_state(
            updated, next_used, threshold=threshold,
            max_per_anchor=max_per_anchor, seed=seed,
        )
        remaining = q-k
        zero += probability * float(remaining_capacity == 0)
        shortfall += probability * float(remaining_capacity < remaining)
    return zero, shortfall


def weak_mass(posteriors, tau, *, max_per_anchor, seed):
    """Current tau=.005 marginal slots removed by a higher threshold."""
    if tau <= 0.005:
        return 0.0, 0.0, 0.0, 0.0
    _, designs, _ = a1.design_state(
        posteriors, {anchor: 0 for anchor in posteriors}, threshold=0.005,
        max_per_anchor=max_per_anchor, seed=seed,
    )
    all_slots = weak_slots = 0
    all_value = weak_value = 0.0
    for design in designs.values():
        for size, delta in design.delta_by_size.items():
            if delta + 1e-12 >= 0.005:
                all_slots += 1
                all_value += float(delta)
                if delta < tau - 1e-12:
                    weak_slots += 1
                    weak_value += float(delta)
    return weak_slots, all_slots, weak_value, all_value


def phase(step):
    return "early" if step <= 50 else "middle" if step <= 100 else "late"


def run(args):
    corrupt = []
    rows = []
    for step_dir in sorted(args.artifact_root.glob("step_*")):
        step = int(step_dir.name.split("_")[-1])
        if not args.step_min <= step <= args.step_max:
            continue
        loaded = a1.load_step(step_dir, corrupt)
        requests_by_task = defaultdict(list)
        for request in loaded["requests"]:
            requests_by_task[request["task_id"]].append(request)
        for task_id, topology in loaded["families"].items():
            diagnostic = loaded["diagnostics"].get(task_id)
            if not diagnostic or diagnostic.get("status") != "PLANNED" or not diagnostic.get("anchors"):
                continue
            q = int(diagnostic["branch_quota"])
            if q < 3:
                continue
            posteriors = a1.posteriors_from_diagnostic(diagnostic)
            prefix = (step, str(diagnostic.get("decision_task_key", task_id)))
            base_full = float(diagnostic["global_optimal_value"])
            actual = requests_by_task.get(task_id, [])
            for tau in sorted(set(args.thresholds)):
                _, _, capacity = a1.design_state(
                    posteriors, {anchor: 0 for anchor in posteriors}, threshold=tau,
                    max_per_anchor=args.max_branches_per_anchor, seed=args.seed,
                )
                p2 = p2_policy(
                    posteriors, {anchor: 0 for anchor in posteriors}, q,
                    threshold=tau, max_per_anchor=args.max_branches_per_anchor,
                    pair_size=args.pair_size, seed=args.seed, prefix=prefix,
                )
                zero, shortfall = post_pair_pressures(
                    posteriors, q, threshold=tau,
                    max_per_anchor=args.max_branches_per_anchor, seed=args.seed, prefix=prefix,
                )
                weak_slots, all_slots, weak_value, all_value = weak_mass(
                    posteriors, tau, max_per_anchor=args.max_branches_per_anchor, seed=args.seed,
                )
                realized = {}
                if len(actual) == q and all(item["branch_id"] in loaded["outcomes"] for item in actual):
                    realized = a1.realized_audit(
                        posteriors, actual, loaded["outcomes"], q=q, threshold=tau,
                        max_per_anchor=args.max_branches_per_anchor, seed=args.seed, seed_prefix=prefix,
                    )
                # Without counterfactual extra root rollouts, Q-C is the exact
                # number of branch slots that an eager controller must convert
                # at least once; it is reported as a lower bound, not faked as
                # physical root-wave count.
                gap = max(0, q-capacity)
                rows.append({
                    "threshold": tau, "step": step, "phase": phase(step),
                    "task_id": task_id, "task_family": topology.get("task_family", "unknown"),
                    "Q": q, "num_anchors": len(posteriors),
                    "initial_capacity": capacity,
                    "initial_pair_feasible": capacity >= 2,
                    "initial_full_feasible": capacity >= q,
                    "eager_capacity_gap": gap,
                    "eager_correction_rounds_lower_bound": gap,
                    "pairwise_replan_changed_realized": realized.get("replan_changed"),
                    "post_pair_zero_capacity_probability": zero,
                    "post_pair_shortfall_probability": shortfall,
                    "expected_executed_branches": p2["executed"],
                    "expected_unexecuted_branches": q-p2["executed"],
                    "expected_fallback_roots": p2["fallback"],
                    "expected_pair_rounds": p2["rounds"],
                    "p2_expected_berv_value": p2["value"],
                    "full_005_reference_value": base_full,
                    "retained_berv_value": p2["value"] / (base_full + 1e-12),
                    "weak_branch_slots": weak_slots,
                    "base_positive_slots": all_slots,
                    "weak_branch_ratio": weak_slots / all_slots if all_slots else 0.0,
                    "weak_branch_value": weak_value,
                    "base_positive_value": all_value,
                    "weak_value_ratio": weak_value / all_value if all_value else 0.0,
                })
    return pd.DataFrame(rows), corrupt


METRICS = [
    "initial_pair_feasible", "initial_full_feasible", "eager_capacity_gap",
    "eager_correction_rounds_lower_bound", "pairwise_replan_changed_realized",
    "post_pair_zero_capacity_probability", "post_pair_shortfall_probability",
    "expected_executed_branches", "expected_unexecuted_branches",
    "expected_fallback_roots", "expected_pair_rounds", "p2_expected_berv_value",
    "retained_berv_value", "weak_branch_ratio", "weak_value_ratio",
]


def aggregate(frame, keys):
    return frame.groupby(keys, as_index=False)[METRICS].mean()


def write_report(out, frame, summary):
    low = summary[(summary.retained_berv_value >= .97) & (summary.threshold > .005)]
    high = summary[(summary.retained_berv_value >= .90) & (summary.threshold > .005)]
    conservative = float(low.threshold.max()) if not low.empty else 0.005
    aggressive = float(high.threshold.max()) if not high.empty else conservative
    decision = {
        "tau_low_star": conservative,
        "tau_high_star": aggressive,
        "rule": "largest evaluated tau retaining >=97% / >=90% of BERV value",
    }
    (out / "online_candidates.json").write_text(json.dumps(decision, indent=2) + "\n")
    rows = summary.to_markdown(index=False, floatfmt=".6g")
    report = f"""# Pairwise Threshold Sensitivity（Step 1）

## Result

This exact CPU-only audit evaluates `{len(frame)}` task-threshold states from the completed 150-step BACE trace.  It recomputes local designs, threshold capacity, P2 K=2 selection, Beta-Binomial outcomes, C=1 single-branch execution, and C=0 one-way fallback for every threshold.

Recommended offline candidates under the mechanical retention screen are:

- conservative `tau_low* = {conservative:g}` (retain >=97%);
- aggressive `tau_high* = {aggressive:g}` (retain >=90%).

These are candidates for Step-100 continuation controls, not claims about validation improvement.

## Overall threshold table

{rows}

## Important boundary

`eager_correction_rounds_lower_bound = max(Q-C_tau, 0)` is an exact lower bound on the number of slot conversions an eager Full-Batch controller must make after the observed root support.  Counterfactual *physical* root-wave counts above this bound require fresh roots not present in the historical trace at higher thresholds; this analysis deliberately does not invent those outcomes.  The Step 4 online P2 implementation measures actual correction/fallback waves.

## Outputs

- `threshold_task_audit.parquet`: task-level exact calculations.
- `threshold_summary.csv`, `threshold_phase_summary.csv`, `threshold_family_summary.csv`, `threshold_q_summary.csv`.
- `online_candidates.json`: mechanical candidate screen.
"""
    (out / "report.md").write_text(report)


def write_figures(out: Path, summary: pd.DataFrame) -> None:
    """Write dependency-free SVG trend figures required by the Step-1 audit."""
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    series = {
        "retained_berv_value.svg": ("retained_berv_value", "Retained BERV value"),
        "expected_execution.svg": ("expected_executed_branches", "Expected executed branches"),
        "stopping_pressure.svg": (
            "post_pair_shortfall_probability", "Post-pair shortfall probability",
        ),
        "fallback_roots.svg": ("expected_fallback_roots", "Expected fallback roots"),
        "weak_mass.svg": ("weak_branch_ratio", "Weak branch-slot ratio"),
    }
    width, height, left, bottom = 760, 420, 75, 65
    xs = [float(value) for value in summary["threshold"]]
    x_min, x_max = min(xs), max(xs)
    x_span = max(x_max - x_min, 1e-12)
    for filename, (column, label) in series.items():
        ys = [float(value) for value in summary[column]]
        y_min, y_max = min(ys), max(ys)
        y_pad = max((y_max - y_min) * 0.08, 1e-9)
        lo, hi = y_min - y_pad, y_max + y_pad
        def px(x): return left + (x - x_min) / x_span * (width - left - 30)
        def py(y): return height - bottom - (y - lo) / (hi - lo) * (height - bottom - 45)
        points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
        circles = "".join(
            f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="#1769aa" />'
            for x, y in zip(xs, ys)
        )
        labels = "".join(
            f'<text x="{px(x):.2f}" y="{height - 35}" text-anchor="middle" '
            f'font-size="12">{x:g}</text>' for x in xs
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="28" font-family="sans-serif" font-size="18" font-weight="bold">{label}</text>
<line x1="{left}" y1="45" x2="{left}" y2="{height-bottom}" stroke="#333"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="#333"/>
<text x="10" y="{height/2}" font-family="sans-serif" font-size="12" transform="rotate(-90 10 {height/2})">{label}</text>
<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">tau_BERV</text>
<polyline fill="none" stroke="#1769aa" stroke-width="2.5" points="{points}"/>{circles}{labels}
<text x="{left}" y="{height-bottom+18}" font-family="sans-serif" font-size="11">{lo:.5g}</text>
<text x="{left}" y="58" font-family="sans-serif" font-size="11">{hi:.5g}</text>
</svg>'''
        (figures / filename).write_text(svg)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, corrupt = run(args)
    if frame.empty:
        raise RuntimeError("No usable Q>=3 acquisition records")
    frame.to_parquet(args.output_dir / "threshold_task_audit.parquet", index=False)
    summary = aggregate(frame, ["threshold"])
    summary.to_csv(args.output_dir / "threshold_summary.csv", index=False)
    aggregate(frame, ["threshold", "phase"]).to_csv(args.output_dir / "threshold_phase_summary.csv", index=False)
    aggregate(frame, ["threshold", "task_family"]).to_csv(args.output_dir / "threshold_family_summary.csv", index=False)
    aggregate(frame, ["threshold", "Q"]).to_csv(args.output_dir / "threshold_q_summary.csv", index=False)
    config = vars(args) | {"artifact_root": str(args.artifact_root), "output_dir": str(args.output_dir), "thresholds": sorted(set(args.thresholds))}
    (args.output_dir / "resolved_analysis_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output_dir / "data_coverage_report.json").write_text(json.dumps({"rows": len(frame), "corrupted_records": corrupt}, indent=2) + "\n")
    write_report(args.output_dir, frame, summary)
    write_figures(args.output_dir, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
