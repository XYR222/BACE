from __future__ import annotations

import hashlib
import itertools
import math
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

    def global_allocations(
        self,
        designs: dict[str, AnchorBatchDesign],
        branch_quota: int,
    ) -> tuple[float, tuple[dict[str, int], ...]]:
        if branch_quota < 0:
            raise ValueError("branch_quota must be non-negative")
        anchor_ids = sorted(designs)
        capacity = sum(designs[anchor_id].capacity for anchor_id in anchor_ids)
        if capacity < branch_quota:
            raise ValueError(
                f"Exact Batch-ERV capacity {capacity} is below branch quota {branch_quota}"
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
