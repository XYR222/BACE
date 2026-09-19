#!/usr/bin/env python3
"""Offline Search BACE diagnostics from a root trace and GiGPO train log."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
import re

from scipy.special import betainc

from recipe.bace_gigpo.anchor_index import AnchorIndex
from recipe.bace_gigpo.batch_erv import ExactBatchErvEngine
from recipe.bace_gigpo.posterior import initialize_local_posterior
from recipe.bace_gigpo.types import RootEvent, RootEventLog


ROOT_FIELDS = {field.name for field in fields(RootEventLog)}
EVENT_FIELDS = {field.name for field in fields(RootEvent)}
STEP_PATTERN = re.compile(
    r"training/global_step:(?P<step>[0-9.]+).*?episode/success_rate:(?P<rate>[0-9.]+)"
)


def load_roots(path: Path) -> list[RootEventLog]:
    roots = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload["events"] = tuple(
            RootEvent(**{key: value for key, value in event.items() if key in EVENT_FIELDS})
            for event in payload["events"]
        )
        roots.append(RootEventLog(**{
            key: value for key, value in payload.items() if key in ROOT_FIELDS
        }))
    return roots


def build_designs(roots: list[RootEventLog], threshold: float):
    index = AnchorIndex(
        roots,
        tie_break_identity_mode="stable_v1",
        anchor_similarity_enabled=True,
        anchor_similarity_threshold=0.9,
        allow_initial_search_anchor=True,
    )
    roots_by_task: dict[str, list[RootEventLog]] = defaultdict(list)
    for root in roots:
        roots_by_task[root.task_id].append(root)
    engine = ExactBatchErvEngine(
        max_branches_per_anchor=2, threshold=threshold, seed=0
    )
    designs = {}
    for task_id in index.ordered_task_ids():
        task_roots = roots_by_task[task_id]
        successes = sum(root.won for root in task_roots)
        task_mean = (0.2 + successes) / (2.0 + len(task_roots))
        task_designs = {}
        for anchor in index.anchors_for_task(task_id):
            posteriors = {}
            for action, origins in anchor.origins_by_action.items():
                outcomes = [index.root_for(origin.root_id).won for origin in origins]
                posteriors[action] = initialize_local_posterior(
                    task_mean, 2.0, outcomes
                )
            task_designs[anchor.anchor_id] = engine.design_anchor(
                anchor.anchor_id, posteriors
            )
        designs[task_id] = task_designs
    return index, designs


def threshold_diagnostics(roots: list[RootEventLog], thresholds: list[float]):
    base_index, base_designs = build_designs(roots, 0.0)
    base_value = sum(
        design.values_by_size[design.capacity]
        for designs in base_designs.values() for design in designs.values()
    )
    base_capacity = sum(
        design.capacity
        for designs in base_designs.values() for design in designs.values()
    )
    rows = []
    for threshold in thresholds:
        _, designs_by_task = build_designs(roots, threshold)
        capacities = {
            task_id: sum(design.capacity for design in designs.values())
            for task_id, designs in designs_by_task.items()
        }
        retained_value = sum(
            design.values_by_size[design.capacity]
            for designs in designs_by_task.values() for design in designs.values()
        )
        rows.append({
            "threshold": threshold,
            "total_capacity": sum(capacities.values()),
            "capacity_retention": (
                sum(capacities.values()) / base_capacity if base_capacity else None
            ),
            "erv_value_retention": retained_value / base_value if base_value else None,
            "tasks_with_capacity_ge_1": sum(value >= 1 for value in capacities.values()),
            "tasks_with_capacity_ge_2": sum(value >= 2 for value in capacities.values()),
            "tasks_with_capacity_ge_3": sum(value >= 3 for value in capacities.values()),
            "capacity_by_task": capacities,
        })
    marginals = [
        {"task_id": task_id, "anchor_id": anchor_id, "size": size, "delta": delta}
        for task_id, designs in base_designs.items()
        for anchor_id, design in designs.items()
        for size, delta in design.delta_by_size.items()
    ]
    return base_index, rows, sorted(marginals, key=lambda row: row["delta"])


def parse_success_curve(path: Path) -> dict[int, float]:
    curve = {}
    for match in STEP_PATTERN.finditer(path.read_text(errors="replace")):
        curve[int(float(match.group("step")))] = float(match.group("rate"))
    return dict(sorted(curve.items()))


def competence_diagnostics(
    curve: dict[int, float], thresholds: list[float], max_step: int, batch_episodes: int
):
    total_budget = 5
    tasks_per_step = batch_episodes // total_budget
    results = []
    for threshold in thresholds:
        successes = 0.0
        failures = 0.0
        quotas = {}
        readiness_by_step = {}
        for step, rate in curve.items():
            if step > max_step:
                break
            alpha = 0.2 + successes
            beta = 1.8 + failures
            concentration = alpha + beta
            strength = min(8.0, max(2.0, 0.1 * concentration))
            mean = alpha / concentration
            posterior_alpha = mean * strength
            posterior_beta = (1.0 - mean) * strength
            readiness = float(1.0 - betainc(
                posterior_alpha, posterior_beta, threshold
            ))
            quota = min(3, max(0, int(math.floor(3 * readiness + 0.5))))
            quotas[step] = quota
            readiness_by_step[step] = readiness
            # Family history receives only the natural roots retained by BACE.
            # Use the GiGPO rate as the counterfactual success probability for
            # those roots, while closing the loop between Q and evidence size.
            natural_episodes = tasks_per_step * (total_budget - quota)
            successes = 0.8 * successes + rate * natural_episodes
            failures = 0.8 * failures + (1.0 - rate) * natural_episodes
        counts = Counter(quotas.values())
        positive = [step for step, quota in quotas.items() if quota > 0]
        blocks = []
        for start in range(1, max_step + 1, 50):
            steps = [step for step in quotas if start <= step <= min(start + 49, max_step)]
            if not steps:
                continue
            blocks.append({
                "steps": f"{start}-{min(start + 49, max_step)}",
                "mean_success_rate": sum(curve[step] for step in steps) / len(steps),
                "mean_planned_branches": sum(quotas[step] for step in steps) / len(steps),
                "zero_branch_fraction": sum(quotas[step] == 0 for step in steps) / len(steps),
            })
        results.append({
            "competence_threshold": threshold,
            "observed_steps": len(quotas),
            "first_positive_branch_step": min(positive) if positive else None,
            "quota_counts": {str(key): counts.get(key, 0) for key in range(4)},
            "mean_planned_branches": sum(quotas.values()) / len(quotas) if quotas else None,
            "planned_branch_slots": sum(quotas.values()),
            "planned_natural_root_slots": sum(total_budget - quota for quota in quotas.values()),
            "planned_branch_episodes": sum(quotas.values()) * tasks_per_step,
            "planned_natural_root_episodes": sum(total_budget - quota for quota in quotas.values()) * tasks_per_step,
            "zero_branch_fraction": counts.get(0, 0) / len(quotas) if quotas else None,
            "readiness_step_1": readiness_by_step.get(1),
            "readiness_final": readiness_by_step.get(max(quotas)) if quotas else None,
            "blocks": blocks,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    roots = load_roots(args.roots)
    old_index = AnchorIndex(
        roots,
        tie_break_identity_mode="stable_v1",
        anchor_similarity_enabled=True,
        anchor_similarity_threshold=0.9,
    )
    new_index, threshold_rows, marginals = threshold_diagnostics(
        roots, [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
    )
    old_ids = set(old_index._anchors)
    new_ids = set(new_index._anchors)
    step_zero_events = [
        event for root in roots for event in root.events if event.step_index == 0
    ]
    curve = parse_success_curve(args.trainer_log)
    payload = {
        "schema_version": "search-bace-offline-diagnostic-v1",
        "inputs": {
            "roots": str(args.roots),
            "trainer_log": str(args.trainer_log),
            "root_count": len(roots),
            "task_count": len({root.task_id for root in roots}),
            "reference_curve_steps": len(curve),
            "competence_proxy_max_step": args.max_step,
        },
        "initial_anchor": {
            "step_zero_occurrences": len(step_zero_events),
            "step_zero_format_valid": sum(event.action_format_valid for event in step_zero_events),
            "step_zero_search_candidates": sum(
                event.action_identity_kind == "valid" for event in step_zero_events
            ),
            "step_zero_terminal": sum(
                event.action_identity_kind == "terminal" for event in step_zero_events
            ),
            "old_anchor_count": len(old_ids),
            "new_anchor_count": len(new_ids),
            "new_initial_anchor_count": len(new_ids - old_ids),
            "new_initial_anchor_ids": sorted(new_ids - old_ids),
        },
        "batch_erv_threshold": {
            "rows": threshold_rows,
            "marginal_deltas": marginals,
        },
        "competence_threshold": competence_diagnostics(
            curve,
            [
                0.30, 0.35, 0.40, 0.42, 0.43, 0.44, 0.45, 0.46,
                0.47, 0.475, 0.48, 0.49, 0.50, 0.51, 0.525, 0.55,
            ],
            args.max_step,
            256 * 5,
        ),
        "limitations": [
            "ERV uses one eight-task cold-policy Search smoke trace.",
            "Competence uses GiGPO natural-rollout success rates as a counterfactual proxy; a BACE policy and its natural-root subset can diverge.",
            "Format-invalid step-0 outputs remain excluded even when initial Search anchors are enabled.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
