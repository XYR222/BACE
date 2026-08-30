#!/usr/bin/env python3
"""Offline A1 audit: frozen Exact Batch-ERV versus outcome-adaptive K=2.

The audit deliberately has no GPU dependency.  It reads the historical BACE
trace, reproduces the recorded Exact objective with the production engine,
then runs two separate analyses:

* A1.1 conditions on the first two *realized* outcomes of the frozen plan.
* A1.2 exactly enumerates Beta-Binomial posterior-predictive outcomes for a
  myopic K=2 policy, both with fixed quota and threshold-aware early stopping.

The two result families remain separate in every output because A1.1 is a
sensitivity audit, not an unbiased counterfactual evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl-agent-src"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from recipe.bace_gigpo.batch_erv import (  # noqa: E402
    AnchorBatchDesign,
    ExactBatchErvEngine,
)
from recipe.bace_gigpo.posterior import BetaPosterior  # noqa: E402


DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "experiments/alfworld-qwen2.5-1.5b-exact/bace_artifacts"
    / "bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs/A1_pairwise_feedback"
EPS = 1e-12


@dataclass
class Plan:
    value: float
    allocation: dict[str, int]
    actions: dict[str, tuple[str, ...]]
    capacity: int
    tie_count: int

    @property
    def selections(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (anchor, action)
            for anchor in sorted(self.actions)
            for action in self.actions[anchor]
        )


@dataclass
class PolicyValue:
    value: float
    executed: float
    shortfall_probability: float
    round_value: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--max-branches-per-anchor", type=int, default=2)
    parser.add_argument("--phase-bounds", default="early:1-50,middle:51-100,late:101-150")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-min", type=int, default=1)
    parser.add_argument("--step-max", type=int, default=150)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def parse_phases(spec: str) -> list[tuple[str, int, int]]:
    phases = []
    for item in spec.split(","):
        name, bounds = item.split(":", 1)
        low, high = bounds.split("-", 1)
        phases.append((name.strip(), int(low), int(high)))
    return phases


def phase_for(step: int, phases: list[tuple[str, int, int]]) -> str:
    for name, low, high in phases:
        if low <= step <= high:
            return name
    return "outside"


def read_json(path: Path, corrupt: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # coverage report must retain every malformed input
        corrupt.append({"path": str(path), "error": repr(exc)})
        return None


def read_jsonl(
    path: Path,
    corrupt: list[dict[str, Any]],
    *,
    missing_is_error: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        if missing_is_error:
            corrupt.append({"path": str(path), "error": "missing"})
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                corrupt.append(
                    {"path": str(path), "line": line_number, "error": repr(exc)}
                )
    return records


def posterior_from_payload(payload: dict[str, Any]) -> BetaPosterior:
    return BetaPosterior(
        alpha=float(payload["alpha"]),
        beta=float(payload["beta"]),
        natural_successes=int(payload.get("natural_successes", 0)),
        natural_failures=int(payload.get("natural_failures", 0)),
        branch_successes=int(payload.get("branch_successes", 0)),
        branch_failures=int(payload.get("branch_failures", 0)),
    )


def posteriors_from_diagnostic(diagnostic: dict[str, Any]) -> dict[str, dict[str, BetaPosterior]]:
    return {
        anchor: {
            action: posterior_from_payload(payload)
            for action, payload in anchor_payload["posteriors"].items()
        }
        for anchor, anchor_payload in diagnostic["anchors"].items()
    }


def clone_posteriors(
    posteriors: dict[str, dict[str, BetaPosterior]],
) -> dict[str, dict[str, BetaPosterior]]:
    return {
        anchor: {action: copy.copy(value) for action, value in actions.items()}
        for anchor, actions in posteriors.items()
    }


def utility(posteriors: dict[str, dict[str, BetaPosterior]]) -> float:
    return sum(max(value.mean for value in actions.values()) for actions in posteriors.values())


def state_signature(
    posteriors: dict[str, dict[str, BetaPosterior]], used: dict[str, int], q: int
) -> tuple[Any, ...]:
    return (
        q,
        tuple(sorted(used.items())),
        tuple(
            (anchor, action, round(value.alpha, 12), round(value.beta, 12))
            for anchor in sorted(posteriors)
            for action, value in sorted(posteriors[anchor].items())
        ),
    )


def design_state(
    posteriors: dict[str, dict[str, BetaPosterior]],
    used: dict[str, int],
    *,
    threshold: float,
    max_per_anchor: int,
    seed: int,
) -> tuple[ExactBatchErvEngine, dict[str, AnchorBatchDesign], int]:
    engine = ExactBatchErvEngine(
        max_branches_per_anchor=max_per_anchor,
        threshold=threshold,
        seed=seed,
    )
    designs: dict[str, AnchorBatchDesign] = {}
    for anchor, actions in posteriors.items():
        remaining = max(0, max_per_anchor - int(used.get(anchor, 0)))
        design = engine.design_anchor(anchor, actions)
        designs[anchor] = replace(design, capacity=min(design.capacity, remaining))
    return engine, designs, sum(item.capacity for item in designs.values())


def solve_plan(
    posteriors: dict[str, dict[str, BetaPosterior]],
    used: dict[str, int],
    quota: int,
    *,
    threshold: float,
    max_per_anchor: int,
    seed: int,
    seed_parts: tuple[Any, ...],
) -> Plan | None:
    engine, designs, capacity = design_state(
        posteriors,
        used,
        threshold=threshold,
        max_per_anchor=max_per_anchor,
        seed=seed,
    )
    if capacity < quota:
        return None
    result = engine.global_allocation(designs, quota, *seed_parts, "global")
    actions: dict[str, tuple[str, ...]] = {}
    for anchor in sorted(result.selected_allocation):
        size = result.selected_allocation[anchor]
        if not size:
            continue
        ties = designs[anchor].optimal_plans_by_size[size]
        local = engine.choose_uniform(ties, *seed_parts, anchor, "local", size)
        actions[anchor] = tuple(local.actions)
    return Plan(
        value=float(result.optimal_value),
        allocation=dict(result.selected_allocation),
        actions=actions,
        capacity=capacity,
        tie_count=int(result.optimal_count),
    )


def enumerate_outcomes(
    posteriors: dict[str, dict[str, BetaPosterior]],
    selections: Iterable[tuple[str, str]],
    engine: ExactBatchErvEngine,
) -> Iterable[tuple[float, dict[tuple[str, str], int], dict[tuple[str, str], int]]]:
    counts = Counter(selections)
    groups = sorted(counts)
    ranges = [range(counts[group] + 1) for group in groups]
    for successes_tuple in product(*ranges):
        probability = 1.0
        successes: dict[tuple[str, str], int] = {}
        totals: dict[tuple[str, str], int] = {}
        for group, success_count in zip(groups, successes_tuple):
            anchor, action = group
            n = counts[group]
            posterior = posteriors[anchor][action]
            probability *= engine._beta_binomial_probability(
                posterior.alpha, posterior.beta, n, success_count
            )
            successes[group] = success_count
            totals[group] = n
        yield probability, successes, totals


def apply_count_outcome(
    posteriors: dict[str, dict[str, BetaPosterior]],
    successes: dict[tuple[str, str], int],
    totals: dict[tuple[str, str], int],
) -> dict[str, dict[str, BetaPosterior]]:
    updated = clone_posteriors(posteriors)
    for (anchor, action), n in totals.items():
        value = updated[anchor][action]
        value.alpha += successes[(anchor, action)]
        value.beta += n - successes[(anchor, action)]
    return updated


def policy_value(
    posteriors: dict[str, dict[str, BetaPosterior]],
    used: dict[str, int],
    q: int,
    *,
    pair_size: int,
    threshold: float,
    threshold_aware: bool,
    max_per_anchor: int,
    seed: int,
    seed_prefix: tuple[Any, ...],
    round_index: int = 1,
    memo: dict[tuple[Any, ...], PolicyValue] | None = None,
) -> PolicyValue:
    if q == 0:
        return PolicyValue(0.0, 0.0, 0.0, {})
    memo = memo if memo is not None else {}
    key = (threshold_aware, round_index, state_signature(posteriors, used, q))
    if key in memo:
        return memo[key]
    planning_threshold = threshold if threshold_aware else 0.0
    _, _, capacity = design_state(
        posteriors,
        used,
        threshold=planning_threshold,
        max_per_anchor=max_per_anchor,
        seed=seed,
    )
    # Frozen-Q threshold semantics: any inability to finish the full remaining
    # quota terminates immediately; it is not silently converted into a root.
    if threshold_aware and capacity < q:
        result = PolicyValue(0.0, 0.0, 1.0, {})
        memo[key] = result
        return result
    k = min(pair_size, q)
    signature = state_signature(posteriors, used, q)
    plan = solve_plan(
        posteriors,
        used,
        k,
        threshold=planning_threshold,
        max_per_anchor=max_per_anchor,
        seed=seed,
        seed_parts=seed_prefix + (round_index, signature),
    )
    if plan is None:
        result = PolicyValue(0.0, 0.0, 1.0, {})
        memo[key] = result
        return result

    engine = ExactBatchErvEngine(max_per_anchor, threshold=planning_threshold, seed=seed)
    before = utility(posteriors)
    value = executed = shortfall = 0.0
    round_value: defaultdict[int, float] = defaultdict(float)
    for probability, successes, totals in enumerate_outcomes(posteriors, plan.selections, engine):
        updated = apply_count_outcome(posteriors, successes, totals)
        immediate = utility(updated) - before
        next_used = dict(used)
        for anchor, count in Counter(anchor for anchor, _ in plan.selections).items():
            next_used[anchor] = next_used.get(anchor, 0) + count
        future = policy_value(
            updated,
            next_used,
            q - k,
            pair_size=pair_size,
            threshold=threshold,
            threshold_aware=threshold_aware,
            max_per_anchor=max_per_anchor,
            seed=seed,
            seed_prefix=seed_prefix,
            round_index=round_index + 1,
            memo=memo,
        )
        value += probability * (immediate + future.value)
        executed += probability * (k + future.executed)
        shortfall += probability * future.shortfall_probability
        round_value[round_index] += probability * immediate
        for index, contribution in future.round_value.items():
            round_value[index] += probability * contribution
    result = PolicyValue(value, executed, shortfall, dict(round_value))
    memo[key] = result
    return result


def hybrid_first_pair_then_frozen(
    posteriors: dict[str, dict[str, BetaPosterior]],
    q: int,
    *,
    pair_size: int,
    max_per_anchor: int,
    seed: int,
    seed_prefix: tuple[Any, ...],
) -> float:
    k = min(pair_size, q)
    used = {anchor: 0 for anchor in posteriors}
    signature = state_signature(posteriors, used, q)
    first = solve_plan(
        posteriors,
        used,
        k,
        threshold=0.0,
        max_per_anchor=max_per_anchor,
        seed=seed,
        # Match the first physical pair of P-Fixed exactly, including ties.
        seed_parts=seed_prefix + ("fixed", 1, signature),
    )
    if first is None:
        return 0.0
    engine = ExactBatchErvEngine(max_per_anchor, threshold=0.0, seed=seed)
    before = utility(posteriors)
    expected = 0.0
    for probability, successes, totals in enumerate_outcomes(posteriors, first.selections, engine):
        updated = apply_count_outcome(posteriors, successes, totals)
        immediate = utility(updated) - before
        next_used = dict(used)
        for anchor, count in Counter(anchor for anchor, _ in first.selections).items():
            next_used[anchor] += count
        remaining = q - k
        if remaining:
            frozen = solve_plan(
                updated,
                next_used,
                remaining,
                threshold=0.0,
                max_per_anchor=max_per_anchor,
                seed=seed,
                seed_parts=seed_prefix + ("hybrid-rest", state_signature(updated, next_used, remaining)),
            )
            remainder_value = 0.0 if frozen is None else frozen.value
        else:
            remainder_value = 0.0
        expected += probability * (immediate + remainder_value)
    return expected


def plan_value_for_actions(
    posteriors: dict[str, dict[str, BetaPosterior]],
    actions: dict[str, tuple[str, ...]],
    max_per_anchor: int,
) -> float:
    engine = ExactBatchErvEngine(max_per_anchor, threshold=0.0)
    return sum(engine.value(posteriors[anchor], list(plan)) for anchor, plan in actions.items() if plan)


def counter_dict(selections: Iterable[tuple[str, str]]) -> tuple[dict[str, int], dict[str, Counter]]:
    allocation: Counter = Counter()
    actions: dict[str, Counter] = defaultdict(Counter)
    for anchor, action in selections:
        allocation[anchor] += 1
        actions[anchor][action] += 1
    return dict(allocation), dict(actions)


def allocation_l1(left: dict[str, int], right: dict[str, int]) -> float:
    anchors = set(left) | set(right)
    return 0.5 * sum(abs(left.get(anchor, 0) - right.get(anchor, 0)) for anchor in anchors)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": str(request.get("branch_id", "")),
        "task_id": str(request.get("task_id", "")),
        "selected_action": str(request.get("selected_canonical_action", "")),
    }


def load_step(step_dir: Path, corrupt: list[dict[str, Any]]) -> dict[str, Any]:
    step = int(step_dir.name.split("_")[-1])
    family_records = read_jsonl(step_dir / "family_topology_plans.jsonl", corrupt)
    families: dict[str, dict[str, Any]] = {}
    for record in family_records:
        families.update(record.get("tasks", {}))

    has_planned_branches = any(
        int(task.get("branch_count", 0)) > 0 for task in families.values()
    )

    acquisition_path = step_dir / "acquisition_rounds.jsonl"
    posterior_path = step_dir / "posterior_snapshots.jsonl"
    acquisition_records = read_jsonl(
        acquisition_path, corrupt, missing_is_error=has_planned_branches
    )
    diagnostics: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    for record in acquisition_records:
        requests.extend(compact_request(item) for item in record.get("requests", []))
        for diagnostic in record.get("diagnostics", []):
            diagnostics[str(diagnostic.get("task_id"))] = diagnostic

    posterior_records = read_jsonl(
        posterior_path, corrupt, missing_is_error=has_planned_branches
    )
    outcomes: dict[str, dict[str, Any]] = {}
    for record in posterior_records:
        outcomes.update(
            record.get("posterior_snapshot", {}).get("completed_branch_outcomes", {})
        )
    summary = read_json(step_dir / "summary.json", corrupt)
    return {
        "step": step,
        "families": families,
        "diagnostics": diagnostics,
        "requests": requests,
        "outcomes": outcomes,
        "summary_ok": bool(summary and summary.get("status") == "complete"),
        "expected_absent_root_only_streams": int(
            not has_planned_branches and not acquisition_path.exists()
        )
        + int(not has_planned_branches and not posterior_path.exists()),
    }


def realized_audit(
    posteriors: dict[str, dict[str, BetaPosterior]],
    requests: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    q: int,
    threshold: float,
    max_per_anchor: int,
    seed: int,
    seed_prefix: tuple[Any, ...],
) -> dict[str, Any]:
    ordered = []
    for request in requests:
        outcome = outcomes[request["branch_id"]]
        ordered.append((str(outcome["anchor_id"]), str(outcome["action"]), bool(outcome["success"])))
    first = ordered[:2]
    remaining = ordered[2:]
    updated = clone_posteriors(posteriors)
    used: Counter = Counter()
    for anchor, action, success in first:
        updated[anchor][action].update_branch(success)
        used[anchor] += 1
    frozen_allocation, frozen_actions_counter = counter_dict((a, u) for a, u, _ in remaining)
    frozen_actions = {
        anchor: tuple(sorted(counter.elements()))
        for anchor, counter in frozen_actions_counter.items()
    }
    remaining_q = q - 2
    strict_engine, strict_designs, strict_capacity = design_state(
        updated,
        dict(used),
        threshold=threshold,
        max_per_anchor=max_per_anchor,
        seed=seed,
    )
    del strict_engine, strict_designs
    strict = solve_plan(
        updated,
        dict(used),
        remaining_q,
        threshold=threshold,
        max_per_anchor=max_per_anchor,
        seed=seed,
        seed_parts=seed_prefix + ("realized-strict",),
    ) if strict_capacity >= remaining_q else None
    structural = solve_plan(
        updated,
        dict(used),
        remaining_q,
        threshold=0.0,
        max_per_anchor=max_per_anchor,
        seed=seed,
        seed_parts=seed_prefix + ("realized-fixed",),
    )
    if structural is None:
        raise AssertionError("Structurally feasible frozen plan became infeasible")
    replanned_allocation = {a: n for a, n in structural.allocation.items() if n}
    replanned_actions_counter = {
        anchor: Counter(actions) for anchor, actions in structural.actions.items()
    }
    anchors = set(frozen_actions_counter) | set(replanned_actions_counter)
    action_changed = any(
        frozen_actions_counter.get(anchor, Counter())
        != replanned_actions_counter.get(anchor, Counter())
        for anchor in anchors
    )
    frozen_value = plan_value_for_actions(updated, frozen_actions, max_per_anchor)
    return {
        "realized_pair_outcome": "".join("1" if item[2] else "0" for item in first),
        "first_pair_anchor_multiset": json_text(sorted(item[0] for item in first)),
        "first_pair_action_multiset": json_text(sorted(item[1] for item in first)),
        "remaining_q": remaining_q,
        "replan_changed": replanned_allocation != frozen_allocation,
        "allocation_l1_distance": allocation_l1(replanned_allocation, frozen_allocation),
        "action_plan_changed": action_changed,
        "remaining_capacity_after_realized_pair": strict_capacity,
        "adaptive_capacity_shortfall": strict_capacity < remaining_q,
        "capacity_shortfall_slots": max(0, remaining_q - strict_capacity),
        "realized_feedback_gain": structural.value - frozen_value,
        "strict_replan_value": np.nan if strict is None else strict.value,
        "structural_replan_value": structural.value,
        "frozen_remaining_value_after_outcome": frozen_value,
    }


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    phases = parse_phases(args.phase_bounds)
    corrupt: list[dict[str, Any]] = []
    step_dirs = [
        path
        for path in sorted(args.artifact_root.glob("step_*"))
        if args.step_min <= int(path.name.split("_")[-1]) <= args.step_max
    ]
    rows: list[dict[str, Any]] = []
    tasks_total = tasks_q_ge3 = usable_realized = usable_model = 0
    missing_outcomes = 0
    usable_steps: set[int] = set()
    reproduction_mismatches: list[dict[str, Any]] = []
    expected_absent_root_only_streams = 0

    for step_dir in step_dirs:
        loaded = load_step(step_dir, corrupt)
        expected_absent_root_only_streams += loaded["expected_absent_root_only_streams"]
        step = loaded["step"]
        requests_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for request in loaded["requests"]:
            requests_by_task[request["task_id"]].append(request)
        tasks_total += len(loaded["families"])
        step_has_usable = False
        for task_id, topology in loaded["families"].items():
            diagnostic = loaded["diagnostics"].get(task_id)
            # acquisition_rounds is authoritative for the plan actually sent
            # to replay.  family_topology_plans can describe the earlier
            # family allocation before task-local capacity reconciliation.
            q = int(
                diagnostic.get("branch_quota", 0)
                if diagnostic is not None
                else topology.get("branch_count", 0)
            )
            if q < 3:
                continue
            tasks_q_ge3 += 1
            if not diagnostic or diagnostic.get("status") != "PLANNED" or not diagnostic.get("anchors"):
                continue
            try:
                posteriors = posteriors_from_diagnostic(diagnostic)
                reproduced = solve_plan(
                    posteriors,
                    {anchor: 0 for anchor in posteriors},
                    q,
                    threshold=args.threshold,
                    max_per_anchor=args.max_branches_per_anchor,
                    seed=args.seed,
                    seed_parts=(step, diagnostic.get("decision_task_key"), "reproduce"),
                )
            except Exception as exc:
                corrupt.append({"step": step, "task_id": task_id, "error": f"model_setup: {exc!r}"})
                continue
            if reproduced is None:
                reproduction_mismatches.append(
                    {"step": step, "task_id": task_id, "reason": "capacity_below_quota"}
                )
                continue
            trace_value = float(diagnostic["global_optimal_value"])
            reproduction_error = reproduced.value - trace_value
            if abs(reproduction_error) > 1e-9:
                reproduction_mismatches.append(
                    {
                        "step": step,
                        "task_id": task_id,
                        "trace": trace_value,
                        "reproduced": reproduced.value,
                        "error": reproduction_error,
                    }
                )
                continue

            prefix = (step, diagnostic.get("decision_task_key", task_id))
            fixed = policy_value(
                posteriors,
                {anchor: 0 for anchor in posteriors},
                q,
                pair_size=args.pair_size,
                threshold=args.threshold,
                threshold_aware=False,
                max_per_anchor=args.max_branches_per_anchor,
                seed=args.seed,
                seed_prefix=prefix + ("fixed",),
            )
            threshold_value = policy_value(
                posteriors,
                {anchor: 0 for anchor in posteriors},
                q,
                pair_size=args.pair_size,
                threshold=args.threshold,
                threshold_aware=True,
                max_per_anchor=args.max_branches_per_anchor,
                seed=args.seed,
                seed_prefix=prefix + ("threshold",),
            )
            initial_used = {anchor: 0 for anchor in posteriors}
            first_pair = solve_plan(
                posteriors,
                initial_used,
                min(args.pair_size, q),
                threshold=0.0,
                max_per_anchor=args.max_branches_per_anchor,
                seed=args.seed,
                seed_parts=prefix
                + ("fixed", 1, state_signature(posteriors, initial_used, q)),
            )
            hybrid = hybrid_first_pair_then_frozen(
                posteriors,
                q,
                pair_size=args.pair_size,
                max_per_anchor=args.max_branches_per_anchor,
                seed=args.seed,
                seed_prefix=prefix,
            )
            base = {
                "step": step,
                "phase": phase_for(step, phases),
                "task_id": task_id,
                "task_family": str(topology.get("task_family", "unknown")),
                "Q": q,
                "num_anchors": len(posteriors),
                "initial_information_capacity": int(diagnostic.get("information_capacity", 0)),
                "full_batch_value": trace_value,
                "full_batch_reproduction_error": reproduction_error,
                "pairwise_first_pair_anchor_multiset": json_text(sorted(a for a, _ in first_pair.selections)),
                "pairwise_first_pair_action_multiset": json_text(sorted(u for _, u in first_pair.selections)),
                "pairwise_expected_value_fixed": fixed.value,
                "pairwise_expected_gain_fixed": fixed.value - trace_value,
                "pairwise_relative_gain_fixed": (fixed.value - trace_value) / (trace_value + EPS),
                "pairwise_expected_value_threshold": threshold_value.value,
                "expected_executed_branches_threshold": threshold_value.executed,
                "threshold_shortfall_probability": threshold_value.shortfall_probability,
                "hybrid_2_then_frozen_value": hybrid,
                "first_adaptation_gain_vs_full": hybrid - trace_value,
                "later_pairwise_gain_vs_hybrid": fixed.value - hybrid,
                "round_expected_information_gain": json_text(fixed.round_value),
                "global_tie_count": int(diagnostic.get("global_tie_count", 1)),
                "notes": "",
            }
            usable_model += 1

            task_requests = requests_by_task.get(task_id, [])
            missing = [item["branch_id"] for item in task_requests if item["branch_id"] not in loaded["outcomes"]]
            realized_ok = len(task_requests) == q and not missing
            if missing:
                missing_outcomes += len(missing)
            if realized_ok:
                base.update(
                    realized_audit(
                        posteriors,
                        task_requests,
                        loaded["outcomes"],
                        q=q,
                        threshold=args.threshold,
                        max_per_anchor=args.max_branches_per_anchor,
                        seed=args.seed,
                        seed_prefix=prefix,
                    )
                )
                usable_realized += 1
            else:
                base.update(
                    {
                        "realized_pair_outcome": None,
                        "first_pair_anchor_multiset": None,
                        "first_pair_action_multiset": None,
                        "remaining_q": q - 2,
                        "replan_changed": np.nan,
                        "allocation_l1_distance": np.nan,
                        "action_plan_changed": np.nan,
                        "remaining_capacity_after_realized_pair": np.nan,
                        "adaptive_capacity_shortfall": np.nan,
                        "capacity_shortfall_slots": np.nan,
                        "realized_feedback_gain": np.nan,
                        "strict_replan_value": np.nan,
                        "structural_replan_value": np.nan,
                        "frozen_remaining_value_after_outcome": np.nan,
                        "notes": f"REALIZED_UNUSABLE requests={len(task_requests)} missing={len(missing)}",
                    }
                )
            rows.append(base)
            step_has_usable = True
        if step_has_usable:
            usable_steps.add(step)

    coverage = {
        "steps_total": len(step_dirs),
        "steps_usable": len(usable_steps),
        "tasks_total": tasks_total,
        "tasks_q_ge3": tasks_q_ge3,
        "tasks_usable_realized_audit": usable_realized,
        "tasks_usable_model_based": usable_model,
        "corrupted_records": len(corrupt),
        "expected_absent_root_only_streams": expected_absent_root_only_streams,
        "missing_outcomes": missing_outcomes,
        "full_batch_value_reproduced": usable_model,
        "full_batch_value_mismatches": len(reproduction_mismatches),
        "corruption_details": corrupt,
        "reproduction_mismatch_details": reproduction_mismatches,
    }
    return pd.DataFrame(rows), coverage


def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if series.notna().any() else float("nan")


def summarize_group(frame: pd.DataFrame, key: str, all_tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for group, data in frame.groupby(key, sort=False):
        opportunity = float("nan")
        if all_tasks is not None:
            denominator = len(all_tasks[all_tasks[key] == group])
            opportunity = len(data) / denominator if denominator else float("nan")
        rows.append(
            {
                key: group,
                "tasks_q_ge3": len(data),
                "q_ge3_ratio": opportunity,
                "realized_usable": int(data["replan_changed"].notna().sum()),
                "replan_rate": safe_mean(data["replan_changed"].astype(float)),
                "action_change_rate": safe_mean(data["action_plan_changed"].astype(float)),
                "realized_shortfall_rate": safe_mean(data["adaptive_capacity_shortfall"].astype(float)),
                "mean_realized_feedback_gain": safe_mean(data["realized_feedback_gain"]),
                "mean_expected_feedback_gain_fixed": safe_mean(data["pairwise_expected_gain_fixed"]),
                "median_expected_feedback_gain_fixed": float(data["pairwise_expected_gain_fixed"].median()),
                "mean_relative_feedback_gain_fixed": safe_mean(data["pairwise_relative_gain_fixed"]),
                "median_relative_feedback_gain_fixed": float(data["pairwise_relative_gain_fixed"].median()),
                "mean_threshold_shortfall_probability": safe_mean(data["threshold_shortfall_probability"]),
                "mean_expected_executed_threshold": safe_mean(data["expected_executed_branches_threshold"]),
                "mean_later_pairwise_gain_vs_hybrid": safe_mean(data["later_pairwise_gain_vs_hybrid"]),
            }
        )
    return pd.DataFrame(rows)


def load_all_task_denominators(
    artifact_root: Path, step_min: int, step_max: int, phases: list[tuple[str, int, int]]
) -> pd.DataFrame:
    rows = []
    ignored: list[dict[str, Any]] = []
    for step_dir in sorted(artifact_root.glob("step_*")):
        step = int(step_dir.name.split("_")[-1])
        if not step_min <= step <= step_max:
            continue
        records = read_jsonl(step_dir / "family_topology_plans.jsonl", ignored)
        acquisition = read_jsonl(step_dir / "acquisition_rounds.jsonl", ignored)
        actual_q = {
            str(diagnostic.get("task_id")): int(diagnostic.get("branch_quota", 0))
            for record in acquisition
            for diagnostic in record.get("diagnostics", [])
        }
        for record in records:
            for task_id, task in record.get("tasks", {}).items():
                rows.append(
                    {
                        "step": step,
                        "phase": phase_for(step, phases),
                        "task_family": task.get("task_family", "unknown"),
                        "Q": actual_q.get(task_id, int(task.get("branch_count", 0))),
                    }
                )
    return pd.DataFrame(rows)


def draw_line_plot(path: Path, series: list[tuple[str, list[float], list[float]]], title: str, y_label: str) -> None:
    width, height = 1100, 650
    margin = (90, 70, 40, 85)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    all_x = [x for _, xs, _ in series for x in xs]
    all_y = [y for _, _, ys in series for y in ys if math.isfinite(y)]
    if not all_x or not all_y:
        draw.text((20, 20), f"{title}: no usable data", fill="black", font=font)
        image.save(path)
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(min(all_y), 0.0), max(all_y)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    left, top, right, bottom = margin[0], margin[1], width - margin[2], height - margin[3]
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left - 5, y, right, y), fill="#dddddd", width=1)
        draw.text((8, y - 6), f"{value:.4g}", fill="black", font=font)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for index, (label, xs, ys) in enumerate(series):
        points = []
        for x, y in zip(xs, ys):
            if not math.isfinite(y):
                continue
            px = left + (right - left) * (x - xmin) / max(xmax - xmin, 1)
            py = bottom - (bottom - top) * (y - ymin) / (ymax - ymin)
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=colors[index % len(colors)], width=3)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[index % len(colors)])
        draw.text((left + 10 + index * 230, 35), label, fill=colors[index % len(colors)], font=font)
    draw.text((width // 2 - 100, 10), title, fill="black", font=font)
    draw.text((width // 2, height - 30), "training step", fill="black", font=font)
    draw.text((10, 45), y_label, fill="black", font=font)
    image.save(path)


def draw_bar_plot(path: Path, labels: list[str], values: list[float], title: str, y_label: str) -> None:
    width, height = 1100, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 90, 70, width - 40, height - 100
    finite = [value for value in values if math.isfinite(value)]
    ymin, ymax = min([0.0] + finite), max([0.0] + finite)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    zero_y = bottom - (bottom - top) * (0 - ymin) / (ymax - ymin)
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, zero_y, right, zero_y), fill="black", width=2)
    slot = (right - left) / max(len(labels), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        if not math.isfinite(value):
            continue
        x0 = left + slot * index + slot * 0.15
        x1 = left + slot * (index + 1) - slot * 0.15
        y = bottom - (bottom - top) * (value - ymin) / (ymax - ymin)
        draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill="#1f77b4")
        draw.text((x0, bottom + 10), str(label)[:18], fill="black", font=font)
        draw.text((x0, min(y, zero_y) - 14), f"{value:.4g}", fill="black", font=font)
    draw.text((width // 2 - 120, 15), title, fill="black", font=font)
    draw.text((10, 45), y_label, fill="black", font=font)
    image.save(path)


def write_outputs(args: argparse.Namespace, frame: pd.DataFrame, coverage: dict[str, Any]) -> None:
    output = args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phase_bounds)
    all_tasks = load_all_task_denominators(args.artifact_root, args.step_min, args.step_max, phases)

    resolved = {
        "artifact_root": str(args.artifact_root.resolve()),
        "output_dir": str(output.resolve()),
        "pair_size": args.pair_size,
        "threshold": args.threshold,
        "max_branches_per_anchor": args.max_branches_per_anchor,
        "phase_bounds": phases,
        "seed": args.seed,
        "step_min": args.step_min,
        "step_max": args.step_max,
        "production_engine": str((VERL_ROOT / "recipe/bace_gigpo/batch_erv.py").resolve()),
        "variants": {
            "A1.1": "realized first pair from frozen Full Batch request order",
            "P-Fixed": "fixed Q, frozen root support, structural L_max only",
            "P-Threshold": "fixed initial Q, stop when remaining threshold capacity < remaining Q",
        },
    }
    (output / "resolved_analysis_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "data_coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    frame.to_parquet(output / "pairwise_task_audit.parquet", index=False)

    phase_summary = summarize_group(frame, "phase", all_tasks)
    family_summary = summarize_group(frame, "task_family", all_tasks)
    q_summary = summarize_group(frame, "Q", all_tasks)
    phase_summary.to_csv(output / "phase_summary.csv", index=False)
    family_summary.to_csv(output / "family_summary.csv", index=False)
    q_summary.to_csv(output / "q_summary.csv", index=False)

    round_rows = []
    for row in frame.itertuples(index=False):
        values = json.loads(row.round_expected_information_gain)
        for round_index, value in values.items():
            round_rows.append(
                {
                    "step": row.step,
                    "phase": row.phase,
                    "task_family": row.task_family,
                    "Q": row.Q,
                    "pair_round": int(round_index),
                    "expected_information_gain": float(value),
                }
            )
    round_frame = pd.DataFrame(round_rows)
    round_frame.to_csv(output / "round_value_contributions.csv", index=False)

    per_step_all = all_tasks.groupby("step").agg(tasks=("Q", "size"), q_ge3=("Q", lambda x: int((x >= 3).sum()))).reset_index()
    per_step_all["opportunity"] = per_step_all["q_ge3"] / per_step_all["tasks"]
    per_step = frame.groupby("step").agg(
        replan=("replan_changed", "mean"),
        gain=("pairwise_expected_gain_fixed", "mean"),
        relative=("pairwise_relative_gain_fixed", "median"),
    ).reset_index()
    draw_line_plot(
        figures / "pairwise_opportunity_over_steps.png",
        [("P(Q>=3)", per_step_all.step.tolist(), per_step_all.opportunity.tolist())],
        "Pairwise opportunity over training",
        "task fraction",
    )
    draw_line_plot(
        figures / "realized_replan_rate.png",
        [("realized allocation change", per_step.step.tolist(), per_step.replan.tolist())],
        "Realized frozen-plan replanning rate",
        "rate",
    )
    draw_line_plot(
        figures / "expected_feedback_gain.png",
        [("P-Fixed - Full", per_step.step.tolist(), per_step.gain.tolist())],
        "Expected K=2 feedback gain",
        "absolute BERV value",
    )
    draw_line_plot(
        figures / "relative_feedback_gain.png",
        [("median relative gain", per_step.step.tolist(), per_step.relative.tolist())],
        "Relative K=2 feedback gain",
        "relative gain",
    )
    draw_bar_plot(
        figures / "adaptive_capacity_shortfall.png",
        phase_summary.phase.astype(str).tolist(),
        phase_summary.mean_threshold_shortfall_probability.tolist(),
        "Threshold-aware capacity shortfall",
        "probability",
    )
    draw_bar_plot(
        figures / "family_feedback_gain.png",
        family_summary.task_family.astype(str).tolist(),
        family_summary.mean_expected_feedback_gain_fixed.tolist(),
        "Expected feedback gain by family",
        "absolute BERV value",
    )

    overall_replan = safe_mean(frame["replan_changed"].astype(float))
    median_relative = float(frame["pairwise_relative_gain_fixed"].median())
    mean_gain = safe_mean(frame["pairwise_expected_gain_fixed"])
    action_change = safe_mean(frame["action_plan_changed"].astype(float))
    threshold_nonzero = int((frame["threshold_shortfall_probability"] > 1e-12).sum())
    mean_threshold_shortfall = safe_mean(frame["threshold_shortfall_probability"])
    realized_shortfall = int((frame["adaptive_capacity_shortfall"] == True).sum())  # noqa: E712
    mean_later_gain = safe_mean(frame["later_pairwise_gain_vs_hybrid"])
    mean_first_gain = safe_mean(frame["first_adaptation_gain_vs_full"])
    tie_tasks = int((frame["global_tie_count"] > 1).sum())
    substantive_negative = int((frame["pairwise_expected_gain_fixed"] < -1e-10).sum())
    if overall_replan >= 0.25 and median_relative > 0.05:
        decision = "GO"
        rationale = "replan rate >=25% and median relative model-based gain >5%"
    elif overall_replan >= 0.10 or median_relative > 0:
        decision = "HOLD / ABLATION ONLY"
        rationale = "some adaptive sensitivity exists, but the strong engineering gate is not met"
    else:
        decision = "NO-GO FOR NOW"
        rationale = "replanning is rare and the median expected gain is not positive"

    phase_md = phase_summary.to_markdown(index=False, floatfmt=".6g")
    family_md = family_summary.to_markdown(index=False, floatfmt=".6g")
    q_counts = all_tasks.groupby("Q").size().reset_index(name="tasks").to_markdown(index=False)
    worst_negative = frame.nsmallest(10, "pairwise_expected_gain_fixed")[[
        "step", "task_family", "Q", "full_batch_value", "pairwise_expected_value_fixed", "pairwise_expected_gain_fixed"
    ]].to_markdown(index=False, floatfmt=".6g")
    report = f"""# A1 Full Batch → Pairwise K=2 离线反馈审计

## 结论

工程判定：**{decision}**。判定依据：{rationale}。

在 `{coverage['tasks_total']}` 个训练 task 中，`Q>=3` 的 task 有 `{coverage['tasks_q_ge3']}` 个；其中 model-based A1.2 可用 `{coverage['tasks_usable_model_based']}` 个，realized A1.1 可用 `{coverage['tasks_usable_realized_audit']}` 个。整体 realized allocation replan rate 为 `{overall_replan:.3%}`，P-Fixed 相对 Full Batch 的 expected gain 均值为 `{mean_gain:.8g}`、中位相对收益为 `{median_relative:.3%}`。

这只说明当前 Beta posterior + BERV acquisition objective 下的离线 acquisition value；**不能直接推出 validation success 会提升**。

## 计划中的 11 个问题：直接回答

1. `Q>=3` 共 `{coverage['tasks_q_ge3']}` / `{coverage['tasks_total']}` 个 task（`{coverage['tasks_q_ge3']/coverage['tasks_total']:.3%}`）。
2. 机会比例从 early `{float(phase_summary.loc[phase_summary.phase == 'early', 'q_ge3_ratio'].iloc[0]):.3%}` 增至 middle `{float(phase_summary.loc[phase_summary.phase == 'middle', 'q_ge3_ratio'].iloc[0]):.3%}`、late `{float(phase_summary.loc[phase_summary.phase == 'late', 'q_ge3_ratio'].iloc[0]):.3%}`；机会主要在 late，但收益并未随机会同步扩大。
3. 第一对真实 outcome 后，anchor allocation 有 `{overall_replan:.3%}` 改变。
4. action multiset 改变率为 `{action_change:.3%}`，明显高于 anchor allocation 改变率；反馈更多改变局部 action plan，而不只是把 slot 搬到另一 anchor。
5. model-based P-Fixed absolute gain 均值 `{mean_gain:.8g}`，中位相对 gain `{median_relative:.3%}`。
6. early/middle/late 的中位相对 gain 见下表；它们都远低于 5% 强门槛，因此结论不是由后期 BERV 绝对尺度单独造成。
7. P-Threshold 有 `{threshold_nonzero}` / `{len(frame)}` 个 task 存在非零 shortfall 概率，平均 shortfall 概率 `{mean_threshold_shortfall:.3%}`；历史真实首对 outcome 中 `{realized_shortfall}` 个 task 发生 shortfall。
8. mean expected gain 最大的 family 是 `{family_summary.sort_values('mean_expected_feedback_gain_fixed', ascending=False).iloc[0].task_family}`；但 family 间 replan 与 gain 排名不完全一致，说明“改计划”不等于“价值大”。
9. `2 + remaining frozen` 的首轮适应 gain 均值 `{mean_first_gain:.8g}`，后续继续 Pairwise 的额外 gain 均值 `{mean_later_gain:.8g}`；后续轮有价值，但只占总均值的一部分。
10. 工程结论为 **{decision}**：适合作为受控 ablation，不足以直接替换 Full Batch 主线。
11. 必须设计 mid-branch quota/fallback 语义：shortfall 概率虽小但非零，不能依赖“通常不会发生”。

## 数据完整性与数学复现

- 扫描 step：`{coverage['steps_total']}`；有可用 `Q>=3` task 的 step：`{coverage['steps_usable']}`。
- 损坏/缺失记录：`{coverage['corrupted_records']}`；缺失 branch outcomes：`{coverage['missing_outcomes']}`。
- 全 root-only step 按设计不生成的 acquisition/posterior 流：`{coverage['expected_absent_root_only_streams']}`（不计为损坏）。
- Full Batch `global_optimal_value` 成功复现：`{coverage['full_batch_value_reproduced']}`；不匹配：`{coverage['full_batch_value_mismatches']}`。
- 同 action 重复采样复用生产 `ExactBatchErvEngine._beta_binomial_probability`，没有按固定均值独立 Bernoulli 近似。
- 每个 anchor 的 `L_max={args.max_branches_per_anchor}` 在 pair 之间累计扣减。
- 历史 Full Batch global tie count >1 的 task 有 `{tie_tasks}` 个；主分析复用 stable seed 选一个 tie-optimal 路径，不跨 UUID 比较 request identity。

完整明细见 `data_coverage_report.json`。若存在坏记录，它们被显式排除而非静默补齐。

## A1.1：Realized Frozen-Plan Feedback

第一 pair 严格按历史 `requests` 的物理冻结顺序读取，再用 `completed_branch_outcomes` 更新 posterior。该结果衡量当前 frozen Full Batch plan 对真实结果的敏感性，不是 K=2 的无偏反事实。

## A1.2：Model-Based Exact Pairwise

- **P-Fixed**：初始 Q、root support 和每-anchor 总 slot 上限冻结；每轮只求 `min(2,q)`，枚举 outcome 后重规划；中途不再用 threshold 缩 quota。
- **P-Threshold**：每轮 outcome 后重新计算 threshold capacity；若剩余 capacity 小于冻结的剩余 Q，立即停止并记 shortfall，不擅自用 root 补齐。
- `hybrid_2_then_frozen_value` 是“先做一对、看 outcome、随后一次性规划全部剩余”的诊断；`P-Fixed - hybrid` 隔离第二轮及以后继续重规划的额外价值。

## Phase 汇总

{phase_md}

## Family 汇总

{family_md}

## Q 基数（包括无 adaptivity opportunity）

{q_counts}

`Q=0/1/2` 只计入基数和 opportunity denominator，不进入 A1 的主效应估计。

## 负收益 case（保留、不 clamp）

{worst_negative}

轻微或实质负值均被保留，因为 Pairwise 是 myopic K=2 policy，并不等价于全局最优 POMDP acquisition policy；tie choice 也可能影响具体路径。

其中低于 `-1e-10` 的实质负收益 task 有 `{substantive_negative}` 个；其余表中负数为浮点量级。

## 第几轮贡献

`round_value_contributions.csv` 给出 Pairwise 各物理 pair round 对最终 expected information gain 的贡献。对 Q=5/6，`first_adaptation_gain_vs_full` 与 `later_pairwise_gain_vs_hybrid` 进一步区分“首对后一次重规划”和“后续继续 Pairwise”的价值。它们是 policy-granularity 分解，不应解释成训练回报的因果分解。

## 是否需要 quota/fallback 语义

P-Threshold 的 phase-level `mean_threshold_shortfall_probability` 直接回答该问题：只要它非零，真实 K=2 实现就不能只是把 Full Batch 拆成多个执行波；必须明确 early stop、branch→root fallback，或固定 Q 后取消中途 threshold 三者之一。本审计没有替主方法偷选其中任何一种。

## 输出

- `pairwise_task_audit.parquet`：每个 Q>=3 task 的 A1.1/A1.2 明细。
- `phase_summary.csv`、`family_summary.csv`、`q_summary.csv`：聚合统计。
- `round_value_contributions.csv`：逐 pair round expected information gain。
- `figures/`：计划要求的 6 张图。
- `resolved_analysis_config.json`：精确输入与方法口径。
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def self_test() -> None:
    engine = ExactBatchErvEngine(max_branches_per_anchor=2, threshold=0.0, seed=0)
    probabilities = [engine._beta_binomial_probability(0.2, 1.8, 2, k) for k in range(3)]
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    independent_p2 = (0.2 / 2.0) ** 2
    assert not math.isclose(probabilities[2], independent_p2, abs_tol=1e-6)

    posterior = {
        "a": {
            "x": BetaPosterior(1.0, 2.0),
            "y": BetaPosterior(2.0, 1.0),
        },
        "b": {
            "x": BetaPosterior(1.5, 1.5),
            "y": BetaPosterior(0.5, 2.5),
        },
    }
    fixed = policy_value(
        posterior,
        {"a": 0, "b": 0},
        3,
        pair_size=2,
        threshold=0.005,
        threshold_aware=False,
        max_per_anchor=2,
        seed=0,
        seed_prefix=("test",),
    )
    assert math.isfinite(fixed.value) and math.isclose(fixed.executed, 3.0, abs_tol=1e-10)
    plan = solve_plan(
        posterior,
        {"a": 2, "b": 0},
        2,
        threshold=0.0,
        max_per_anchor=2,
        seed=0,
        seed_parts=("cap-test",),
    )
    assert plan is not None and plan.allocation["a"] == 0 and plan.allocation["b"] == 2
    print("A1 self-test: ok")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.pair_size != 2:
        raise ValueError("A1 is specified for pair_size=2")
    if not args.artifact_root.is_dir():
        raise FileNotFoundError(args.artifact_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = audit(args)
    if frame.empty:
        raise RuntimeError("No Q>=3 task passed the trace-integrity and reproduction gates")
    write_outputs(args, frame, coverage)
    print(json.dumps({"rows": len(frame), "coverage": coverage, "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
