from __future__ import annotations

import hashlib
import itertools
import math
import random
import time
from dataclasses import dataclass

import numpy as np

from .posterior import BetaPosterior


@dataclass(frozen=True)
class BatchPlanValue:
    """Exact value of an unordered local branch plan."""

    actions: tuple[str, ...]
    value: float


@dataclass(frozen=True)
class AnchorBatchDesign:
    anchor_id: str
    values_by_size: dict[int, float]
    optimal_plans_by_size: dict[int, tuple[BatchPlanValue, ...]]
    all_plans_by_size: dict[int, tuple[BatchPlanValue, ...]]
    delta_by_size: dict[int, float]
    capacity: int


@dataclass(frozen=True)
class _DPChoice:
    """A compressed tie-optimal predecessor for one DP state."""

    branches_for_anchor: int
    previous_quota: int
    previous_count: int


@dataclass(frozen=True)
class _DPState:
    best_value: float
    optimal_count: int
    choices: tuple[_DPChoice, ...]


@dataclass(frozen=True)
class GlobalAllocationResult:
    """One seeded sample from the exact set of optimal global allocations.

    The complete tie set is intentionally represented by ``optimal_count`` and
    compressed DP predecessors instead of being materialized in memory.
    """

    optimal_value: float
    optimal_count: int
    selected_allocation: dict[str, int]
    solver: str
    num_anchors: int
    branch_quota: int
    total_information_capacity: int
    reachable_state_count: int
    solver_wall_time_ms: float


def _stable_seed(global_seed: int, *parts: object) -> int:
    payload = repr((int(global_seed),) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


class ExactBatchErvEngine:
    """Deterministic Beta-Binomial finite-sum Batch-ERV engine."""

    def __init__(
        self,
        max_branches_per_anchor: int = 2,
        threshold: float = 0.0,
        tie_abs_tolerance: float = 1e-12,
        tie_rel_tolerance: float = 1e-10,
        seed: int = 0,
    ):
        if max_branches_per_anchor < 1:
            raise ValueError("max_branches_per_anchor must be positive")
        if threshold < 0:
            raise ValueError("Batch-ERV threshold must be non-negative")
        if tie_abs_tolerance < 0 or tie_rel_tolerance < 0:
            raise ValueError("Batch-ERV tie tolerances must be non-negative")
        self.max_branches_per_anchor = int(max_branches_per_anchor)
        self.threshold = float(threshold)
        self.tie_abs_tolerance = float(tie_abs_tolerance)
        self.tie_rel_tolerance = float(tie_rel_tolerance)
        self.seed = int(seed)

    def tied(self, left: float, right: float) -> bool:
        return abs(left - right) <= self.tie_abs_tolerance + self.tie_rel_tolerance * max(
            abs(left), abs(right)
        )

    @staticmethod
    def _beta_binomial_probability(alpha: float, beta: float, n: int, k: int) -> float:
        if not 0 <= k <= n:
            return 0.0
        log_probability = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + math.lgamma(alpha + k)
            + math.lgamma(beta + n - k)
            - math.lgamma(alpha + beta + n)
            - math.lgamma(alpha)
            - math.lgamma(beta)
            + math.lgamma(alpha + beta)
        )
        return math.exp(log_probability)

    def value(
        self,
        posteriors: dict[str, BetaPosterior],
        actions: tuple[str, ...] | list[str],
    ) -> float:
        if len(posteriors) < 2:
            raise ValueError("Batch-ERV requires at least two observed actions")
        unknown = set(actions).difference(posteriors)
        if unknown:
            raise ValueError(f"Batch plan contains unknown actions: {sorted(unknown)}")

        ordered_actions = sorted(posteriors)
        counts = {action: actions.count(action) for action in ordered_actions}
        current_best = max(posteriors[action].mean for action in ordered_actions)
        sampled_actions = [action for action in ordered_actions if counts[action] > 0]
        if not sampled_actions:
            return 0.0

        expected_best = 0.0
        outcome_ranges = [range(counts[action] + 1) for action in sampled_actions]
        for successes in itertools.product(*outcome_ranges):
            probability = 1.0
            updated_means = {
                action: posteriors[action].mean for action in ordered_actions
            }
            for action, success_count in zip(sampled_actions, successes):
                posterior = posteriors[action]
                sample_count = counts[action]
                probability *= self._beta_binomial_probability(
                    posterior.alpha, posterior.beta, sample_count, success_count
                )
                updated_means[action] = (
                    posterior.alpha + success_count
                ) / (posterior.alpha + posterior.beta + sample_count)
            expected_best += probability * max(updated_means.values())

        # The theoretical value is non-negative; clamp only floating-point noise.
        result = expected_best - current_best
        if result < 0 and self.tied(result, 0.0):
            return 0.0
        return max(0.0, float(result))

    def design_anchor(
        self,
        anchor_id: str,
        posteriors: dict[str, BetaPosterior],
    ) -> AnchorBatchDesign:
        actions = sorted(posteriors)
        if len(actions) < 2:
            raise ValueError("Structural anchors require at least two observed actions")

        all_plans: dict[int, tuple[BatchPlanValue, ...]] = {
            0: (BatchPlanValue(actions=(), value=0.0),)
        }
        optimal_plans: dict[int, tuple[BatchPlanValue, ...]] = {0: all_plans[0]}
        values = {0: 0.0}
        deltas = {}
        capacity = 0

        for size in range(1, self.max_branches_per_anchor + 1):
            evaluated = tuple(
                BatchPlanValue(plan, self.value(posteriors, plan))
                for plan in itertools.combinations_with_replacement(actions, size)
            )
            best_value = max(item.value for item in evaluated)
            ties = tuple(item for item in evaluated if self.tied(item.value, best_value))
            all_plans[size] = evaluated
            optimal_plans[size] = ties
            values[size] = float(best_value)
            marginal = max(0.0, float(best_value - values[size - 1]))
            deltas[size] = marginal
            if capacity == size - 1 and (
                marginal > self.threshold or self.tied(marginal, self.threshold)
            ):
                capacity = size

        return AnchorBatchDesign(
            anchor_id=anchor_id,
            values_by_size=values,
            optimal_plans_by_size=optimal_plans,
            all_plans_by_size=all_plans,
            delta_by_size=deltas,
            capacity=capacity,
        )

    def choose_uniform(self, choices: tuple | list, *seed_parts: object):
        if not choices:
            raise ValueError("Cannot choose from an empty tie-optimal set")
        rng = np.random.default_rng(_stable_seed(self.seed, *seed_parts))
        return choices[int(rng.integers(len(choices)))]

    def _validate_global_problem(
        self,
        designs: dict[str, AnchorBatchDesign],
        branch_quota: int,
    ) -> tuple[list[str], int]:
        if branch_quota < 0:
            raise ValueError("branch_quota must be non-negative")
        anchor_ids = sorted(designs)
        capacity = sum(designs[anchor_id].capacity for anchor_id in anchor_ids)
        if capacity < branch_quota:
            raise ValueError(
                f"Exact Batch-ERV capacity {capacity} is below branch quota {branch_quota}"
            )
        for anchor_id in anchor_ids:
            design = designs[anchor_id]
            if design.capacity < 0:
                raise ValueError(f"Anchor {anchor_id} has negative capacity {design.capacity}")
            if design.capacity > self.max_branches_per_anchor:
                raise ValueError(
                    f"Anchor {anchor_id} capacity {design.capacity} exceeds configured "
                    f"maximum {self.max_branches_per_anchor}"
                )
            for size in range(min(design.capacity, branch_quota) + 1):
                if size not in design.values_by_size:
                    raise ValueError(
                        f"Anchor {anchor_id} is missing Batch-ERV value for size {size}"
                    )
                if not math.isfinite(float(design.values_by_size[size])):
                    raise ValueError(
                        f"Anchor {anchor_id} has non-finite Batch-ERV value for size {size}"
                    )
        return anchor_ids, capacity

    def _global_allocations_cartesian_reference(
        self,
        designs: dict[str, AnchorBatchDesign],
        branch_quota: int,
        *,
        max_states: int = 1_000_000,
    ) -> tuple[float, tuple[dict[str, int], ...]]:
        """Brute-force correctness oracle for small tests only.

        Production code must use :meth:`global_allocation`.  The hard guard
        prevents an accidental return of the exponential allocator regression.
        """

        anchor_ids, _ = self._validate_global_problem(designs, branch_quota)
        cartesian_state_count = math.prod(
            designs[anchor_id].capacity + 1 for anchor_id in anchor_ids
        )
        if cartesian_state_count > max_states:
            raise RuntimeError(
                "Cartesian reference solver disabled: "
                f"state space = {cartesian_state_count} > {max_states}. "
                "Use quota-aware exact DP."
            )
        candidates = []
        ranges = [range(designs[anchor_id].capacity + 1) for anchor_id in anchor_ids]
        for sizes in itertools.product(*ranges):
            if sum(sizes) != branch_quota:
                continue
            allocation = dict(zip(anchor_ids, sizes))
            value = sum(
                designs[anchor_id].values_by_size[size]
                for anchor_id, size in allocation.items()
            )
            candidates.append((float(value), allocation))
        if not candidates:
            if branch_quota == 0:
                return 0.0, ({},)
            raise ValueError("No feasible exact Batch-ERV global allocation")
        best_value = max(value for value, _ in candidates)
        ties = tuple(allocation for value, allocation in candidates if self.tied(value, best_value))
        return best_value, ties

    @staticmethod
    def _sample_count_weighted_choice(
        choices: tuple[_DPChoice, ...], rng: random.Random
    ) -> _DPChoice:
        total = sum(choice.previous_count for choice in choices)
        if total <= 0:
            raise AssertionError("Tie-optimal DP predecessor count must be positive")
        # random.Random.randrange accepts arbitrary-size Python ints, unlike
        # numpy Generator.integers, whose upper bound is limited to int64.
        draw = rng.randrange(total)
        cumulative = 0
        for choice in choices:
            cumulative += choice.previous_count
            if draw < cumulative:
                return choice
        raise AssertionError("Count-weighted DP backtracking failed")

    def global_allocation(
        self,
        designs: dict[str, AnchorBatchDesign],
        branch_quota: int,
        *seed_parts: object,
    ) -> GlobalAllocationResult:
        """Solve global Exact Batch-ERV allocation with quota-aware exact DP.

        Runtime is O(A * Q * L_max).  Numerically tie-optimal complete
        allocations are sampled uniformly using exact path counts and seeded,
        count-weighted backtracking; the tie set is never materialized.
        """

        started = time.perf_counter()
        anchor_ids, total_capacity = self._validate_global_problem(
            designs, branch_quota
        )

        if branch_quota == 0:
            return GlobalAllocationResult(
                optimal_value=0.0,
                optimal_count=1,
                selected_allocation={anchor_id: 0 for anchor_id in anchor_ids},
                solver="quota_aware_exact_dp",
                num_anchors=len(anchor_ids),
                branch_quota=0,
                total_information_capacity=total_capacity,
                reachable_state_count=1,
                solver_wall_time_ms=(time.perf_counter() - started) * 1000.0,
            )

        rows: list[list[_DPState | None]] = [
            [None for _ in range(branch_quota + 1)]
            for _ in range(len(anchor_ids) + 1)
        ]
        rows[0][0] = _DPState(best_value=0.0, optimal_count=1, choices=())
        reachable_state_count = 1

        for index, anchor_id in enumerate(anchor_ids, start=1):
            design = designs[anchor_id]
            effective_capacity = min(
                design.capacity, self.max_branches_per_anchor, branch_quota
            )
            for used_quota in range(branch_quota + 1):
                candidates: list[tuple[float, _DPChoice]] = []
                for size in range(min(effective_capacity, used_quota) + 1):
                    previous_quota = used_quota - size
                    previous = rows[index - 1][previous_quota]
                    if previous is None:
                        continue
                    value = previous.best_value + float(design.values_by_size[size])
                    candidates.append((
                        value,
                        _DPChoice(
                            branches_for_anchor=size,
                            previous_quota=previous_quota,
                            previous_count=previous.optimal_count,
                        ),
                    ))
                if not candidates:
                    continue
                best_value = max(value for value, _ in candidates)
                tied_choices = tuple(
                    choice
                    for value, choice in candidates
                    if self.tied(value, best_value)
                )
                rows[index][used_quota] = _DPState(
                    best_value=float(best_value),
                    optimal_count=sum(
                        choice.previous_count for choice in tied_choices
                    ),
                    choices=tied_choices,
                )
                reachable_state_count += 1

        final_state = rows[len(anchor_ids)][branch_quota]
        if final_state is None:
            raise ValueError("No feasible exact Batch-ERV global allocation")

        rng = random.Random(_stable_seed(self.seed, *seed_parts))
        selected_allocation: dict[str, int] = {}
        remaining_quota = branch_quota
        for index in range(len(anchor_ids), 0, -1):
            state = rows[index][remaining_quota]
            if state is None or not state.choices:
                raise AssertionError("Reachable DP state has no optimal predecessor")
            choice = self._sample_count_weighted_choice(state.choices, rng)
            selected_allocation[anchor_ids[index - 1]] = choice.branches_for_anchor
            remaining_quota = choice.previous_quota
        if remaining_quota != 0 or sum(selected_allocation.values()) != branch_quota:
            raise AssertionError("Exact DP returned an allocation with the wrong quota")

        # Restore deterministic anchor order after reverse backtracking.
        selected_allocation = {
            anchor_id: selected_allocation[anchor_id] for anchor_id in anchor_ids
        }
        return GlobalAllocationResult(
            optimal_value=final_state.best_value,
            optimal_count=final_state.optimal_count,
            selected_allocation=selected_allocation,
            solver="quota_aware_exact_dp",
            num_anchors=len(anchor_ids),
            branch_quota=branch_quota,
            total_information_capacity=total_capacity,
            reachable_state_count=reachable_state_count,
            solver_wall_time_ms=(time.perf_counter() - started) * 1000.0,
        )
