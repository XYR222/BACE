from __future__ import annotations

from dataclasses import dataclass
import math

from .anchor_index import AnchorIndex
from .batch_erv import AnchorBatchDesign, ExactBatchErvEngine
from .competence import CompetenceHistory, CompetencePosterior
from .posterior import PosteriorEngine, initialize_local_posterior
from .types import RootEventLog


@dataclass(frozen=True)
class TaskTopology:
    task_id: str
    task_family: str
    posterior: CompetencePosterior
    readiness: float
    planned_branch_count: int
    final_root_count: int
    final_branch_count: int
    effective_anchor_count: int
    family_prior_mean: float | None = None
    family_controller_strength: float | None = None
    correction_count: int = 0
    information_capacity: int | None = None
    global_batch_value: float | None = None


@dataclass(frozen=True)
class TopologyPlan:
    roots: tuple[RootEventLog, ...]
    tasks: dict[str, TaskTopology]

    @property
    def branch_quota_by_task(self) -> dict[str, int]:
        return {task_id: task.final_branch_count for task_id, task in self.tasks.items()}

    @property
    def posterior_mean_by_task(self) -> dict[str, float]:
        return {task_id: task.posterior.mean for task_id, task in self.tasks.items()}


@dataclass
class StagedTaskState:
    task_id: str
    task_family: str
    posterior: CompetencePosterior
    readiness: float
    planned_branch_count: int
    root_count: int
    branch_count: int
    effective_anchor_count: int = 0


class DynamicTopologyPlanner:
    """Pilot planning and sequential root-side capacity correction."""

    def __init__(
        self,
        history: CompetenceHistory,
        total_budget: int,
        pilot_roots: int,
        competence_threshold: float,
        max_branches_per_anchor: int,
        local_prior_strength: float,
        erv_threshold: float,
        erv_temperature: float,
        erv_mc_samples: int,
        seed: int = 0,
        invalid_action_mode: str = "strict_identity",
    ):
        if not 1 <= pilot_roots <= total_budget:
            raise ValueError("pilot_roots must be in [1, total_budget]")
        self.history = history
        self.total_budget = total_budget
        self.pilot_roots = pilot_roots
        self.competence_threshold = competence_threshold
        self.max_branches_per_anchor = max_branches_per_anchor
        self.local_prior_strength = local_prior_strength
        self.erv_threshold = erv_threshold
        self.engine = PosteriorEngine(erv_mc_samples, erv_temperature, seed)
        self.invalid_action_mode = invalid_action_mode

    def _effective_anchor_count(self, roots: list[RootEventLog], prior_mean: float) -> int:
        index = AnchorIndex(roots, invalid_action_mode=self.invalid_action_mode)
        effective = 0
        task_id = roots[0].task_id
        for anchor in index.anchors_for_task(task_id):
            posteriors = {}
            for action, origins in anchor.origins_by_action.items():
                outcomes = [index.root_for(origin.root_id).won for origin in origins]
                posteriors[action] = initialize_local_posterior(
                    prior_mean, self.local_prior_strength, outcomes
                )
            if self.engine.evaluate(posteriors).utility >= self.erv_threshold:
                effective += 1
        return effective

    def plan(self, candidate_roots: list[RootEventLog]) -> TopologyPlan:
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in candidate_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)

        selected_roots = []
        task_plans = {}
        for task_id, roots in roots_by_task.items():
            if len(roots) < self.total_budget:
                raise ValueError(
                    f"Task {task_id} has {len(roots)} candidate roots, expected at least {self.total_budget}"
                )
            roots = roots[: self.total_budget]
            family = roots[0].task_family
            prior = self.history.prior(family)
            posterior = self.history.posterior(
                prior, [root.won for root in roots[: self.pilot_roots]]
            )
            readiness = posterior.probability_above(self.competence_threshold)
            planned_branches = self.history.planned_branch_quota(
                self.total_budget, self.pilot_roots, readiness
            )
            branch_count = planned_branches
            root_count = self.total_budget - branch_count
            effective_anchor_count = 0
            while True:
                revealed = roots[:root_count]
                effective_anchor_count = self._effective_anchor_count(revealed, posterior.mean)
                capacity = self.max_branches_per_anchor * effective_anchor_count
                if branch_count == 0 or capacity >= branch_count:
                    break
                root_count += 1
                branch_count -= 1
            if root_count + branch_count != self.total_budget:
                raise AssertionError("Capacity correction violated the terminal-leaf budget")
            selected_roots.extend(roots[:root_count])
            task_plans[task_id] = TaskTopology(
                task_id=task_id,
                task_family=family,
                posterior=posterior,
                readiness=readiness,
                planned_branch_count=planned_branches,
                final_root_count=root_count,
                final_branch_count=branch_count,
                effective_anchor_count=effective_anchor_count,
            )
        return TopologyPlan(roots=tuple(selected_roots), tasks=task_plans)

    def initialize_staged(self, pilot_roots: list[RootEventLog]) -> dict[str, StagedTaskState]:
        """Plan initial root targets from pilot evidence only."""
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in pilot_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)

        states = {}
        for task_id, roots in roots_by_task.items():
            if len(roots) != self.pilot_roots:
                raise ValueError(
                    f"Task {task_id} has {len(roots)} pilot roots, expected {self.pilot_roots}"
                )
            family = roots[0].task_family
            prior = self.history.prior(family)
            posterior = self.history.posterior(prior, [root.won for root in roots])
            readiness = posterior.probability_above(self.competence_threshold)
            planned_branches = self.history.planned_branch_quota(
                self.total_budget, self.pilot_roots, readiness
            )
            states[task_id] = StagedTaskState(
                task_id=task_id,
                task_family=family,
                posterior=posterior,
                readiness=readiness,
                planned_branch_count=planned_branches,
                root_count=self.total_budget - planned_branches,
                branch_count=planned_branches,
            )
        return states

    def correct_staged_capacity(
        self,
        collected_roots: list[RootEventLog],
        states: dict[str, StagedTaskState],
    ) -> set[str]:
        """Convert at most one branch slot per deficient task into a root slot."""
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in collected_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)

        need_more_roots = set()
        for task_id, state in states.items():
            roots = roots_by_task.get(task_id, [])
            if len(roots) < state.root_count:
                raise ValueError(
                    f"Task {task_id} has {len(roots)} roots, requires {state.root_count} before correction"
                )
            revealed = roots[: state.root_count]
            state.effective_anchor_count = self._effective_anchor_count(
                revealed, state.posterior.mean
            )
            capacity = self.max_branches_per_anchor * state.effective_anchor_count
            if state.branch_count > 0 and capacity < state.branch_count:
                state.root_count += 1
                state.branch_count -= 1
                need_more_roots.add(task_id)
            if state.root_count + state.branch_count != self.total_budget:
                raise AssertionError("Capacity correction violated the terminal-leaf budget")
        return need_more_roots

    def finalize_staged(
        self,
        collected_roots: list[RootEventLog],
        states: dict[str, StagedTaskState],
    ) -> TopologyPlan:
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in collected_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)

        selected_roots = []
        tasks = {}
        for task_id, state in states.items():
            roots = roots_by_task.get(task_id, [])
            if len(roots) != state.root_count:
                raise ValueError(
                    f"Task {task_id} generated {len(roots)} roots, expected exactly {state.root_count}"
                )
            capacity = self.max_branches_per_anchor * state.effective_anchor_count
            if state.branch_count > 0 and capacity < state.branch_count:
                raise AssertionError("Staged topology finalized before capacity correction converged")
            if state.root_count + state.branch_count != self.total_budget:
                raise AssertionError("Staged topology violated the terminal-leaf budget")
            selected_roots.extend(roots)
            tasks[task_id] = TaskTopology(
                task_id=task_id,
                task_family=state.task_family,
                posterior=state.posterior,
                readiness=state.readiness,
                planned_branch_count=state.planned_branch_count,
                final_root_count=state.root_count,
                final_branch_count=state.branch_count,
                effective_anchor_count=state.effective_anchor_count,
            )
        return TopologyPlan(roots=tuple(selected_roots), tasks=tasks)

    def update_history(self, plan: TopologyPlan) -> None:
        outcomes_by_family: dict[str, list[bool]] = {}
        for root in plan.roots:
            outcomes_by_family.setdefault(root.task_family, []).append(root.won)
        self.history.update(outcomes_by_family)


@dataclass
class ExactBatchTaskState:
    task_id: str
    task_family: str
    family_posterior: CompetencePosterior
    readiness: float
    planned_branch_count: int
    root_count: int
    branch_count: int
    current_posterior: CompetencePosterior | None = None
    structural_anchor_count: int = 0
    information_capacity: int = 0
    correction_count: int = 0
    anchor_designs: dict[str, AnchorBatchDesign] | None = None


class ExactBatchTopologyPlanner:
    """No-pilot lagged-family topology with exact Batch-ERV capacity correction."""

    def __init__(
        self,
        history: CompetenceHistory,
        total_budget: int,
        min_natural_roots: int,
        competence_threshold: float,
        max_branches_per_anchor: int,
        local_prior_strength: float,
        batch_erv_threshold: float,
        tie_abs_tolerance: float,
        tie_rel_tolerance: float,
        seed: int = 0,
        invalid_action_mode: str = "strict_identity",
    ):
        if not 1 <= min_natural_roots <= total_budget:
            raise ValueError("min_natural_roots must be in [1, total_budget]")
        self.history = history
        self.total_budget = int(total_budget)
        self.min_natural_roots = int(min_natural_roots)
        self.competence_threshold = float(competence_threshold)
        self.local_prior_strength = float(local_prior_strength)
        self.invalid_action_mode = invalid_action_mode
        self.engine = ExactBatchErvEngine(
            max_branches_per_anchor=max_branches_per_anchor,
            threshold=batch_erv_threshold,
            tie_abs_tolerance=tie_abs_tolerance,
            tie_rel_tolerance=tie_rel_tolerance,
            seed=seed,
        )

    def initialize(self, task_families: dict[str, str]) -> dict[str, ExactBatchTaskState]:
        """Plan roots from lagged family history without reading current outcomes."""
        states = {}
        flexible_budget = self.total_budget - self.min_natural_roots
        for task_id, family in task_families.items():
            prior = self.history.prior(family)
            family_posterior = CompetencePosterior(alpha=prior.alpha, beta=prior.beta)
            readiness = family_posterior.probability_above(self.competence_threshold)
            planned_branches = int(math.floor(flexible_budget * readiness + 0.5))
            planned_branches = min(flexible_budget, max(0, planned_branches))
            states[task_id] = ExactBatchTaskState(
                task_id=task_id,
                task_family=family,
                family_posterior=family_posterior,
                readiness=readiness,
                planned_branch_count=planned_branches,
                root_count=self.total_budget - planned_branches,
                branch_count=planned_branches,
            )
        return states

    def _assess_task(
        self,
        roots: list[RootEventLog],
        state: ExactBatchTaskState,
    ) -> None:
        if len(roots) != state.root_count:
            raise ValueError(
                f"Task {state.task_id} has {len(roots)} roots, expected {state.root_count}"
            )
        successes = sum(root.won for root in roots)
        state.current_posterior = CompetencePosterior(
            alpha=state.family_posterior.alpha + successes,
            beta=state.family_posterior.beta + len(roots) - successes,
        )
        index = AnchorIndex(roots, invalid_action_mode=self.invalid_action_mode)
        designs = {}
        for anchor in index.anchors_for_task(state.task_id):
            posteriors = {}
            for action, origins in anchor.origins_by_action.items():
                outcomes = [index.root_for(origin.root_id).won for origin in origins]
                posteriors[action] = initialize_local_posterior(
                    state.current_posterior.mean, self.local_prior_strength, outcomes
                )
            designs[anchor.anchor_id] = self.engine.design_anchor(anchor.anchor_id, posteriors)
        state.anchor_designs = designs
        state.structural_anchor_count = len(designs)
        state.information_capacity = sum(design.capacity for design in designs.values())

    def correct_capacity(
        self,
        collected_roots: list[RootEventLog],
        states: dict[str, ExactBatchTaskState],
    ) -> set[str]:
        """Assess exact marginal capacity and convert at most one slot per task."""
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in collected_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)
        need_more_roots = set()
        for task_id, state in states.items():
            self._assess_task(roots_by_task.get(task_id, []), state)
            if state.branch_count > 0 and state.information_capacity < state.branch_count:
                state.root_count += 1
                state.branch_count -= 1
                state.correction_count += 1
                need_more_roots.add(task_id)
            if state.root_count + state.branch_count != self.total_budget:
                raise AssertionError("Exact capacity correction violated the leaf budget")
        return need_more_roots

    def finalize(
        self,
        collected_roots: list[RootEventLog],
        states: dict[str, ExactBatchTaskState],
    ) -> TopologyPlan:
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in collected_roots:
            roots_by_task.setdefault(root.task_id, []).append(root)
        tasks = {}
        selected_roots = []
        for task_id, state in states.items():
            roots = roots_by_task.get(task_id, [])
            self._assess_task(roots, state)
            if state.information_capacity < state.branch_count:
                raise AssertionError("Exact topology finalized before capacity converged")
            selected_roots.extend(roots)
            tasks[task_id] = TaskTopology(
                task_id=task_id,
                task_family=state.task_family,
                posterior=state.current_posterior,
                readiness=state.readiness,
                planned_branch_count=state.planned_branch_count,
                final_root_count=state.root_count,
                final_branch_count=state.branch_count,
                effective_anchor_count=state.structural_anchor_count,
                family_prior_mean=state.family_posterior.mean,
                family_controller_strength=(
                    state.family_posterior.alpha + state.family_posterior.beta
                ),
                correction_count=state.correction_count,
                information_capacity=state.information_capacity,
            )
        return TopologyPlan(roots=tuple(selected_roots), tasks=tasks)

    def update_history(self, plan: TopologyPlan) -> None:
        outcomes_by_family: dict[str, list[bool]] = {}
        for root in plan.roots:
            outcomes_by_family.setdefault(root.task_family, []).append(root.won)
        self.history.update(outcomes_by_family)
