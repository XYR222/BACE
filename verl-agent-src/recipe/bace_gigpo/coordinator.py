from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

import numpy as np

from gigpo.core_gigpo import to_hashable

from .anchor_index import AnchorIndex
from .batch_erv import ExactBatchErvEngine
from .posterior import BetaPosterior, PosteriorEngine, initialize_local_posterior
from .types import ReplayRequest, RootEventLog


def _make_request(index, task_id, branch_idx, anchor, action, origin):
    root = index.root_for(origin.root_id)
    event = index.event_for(origin.occurrence_id)
    prefix = root.events[: event.step_index]
    return ReplayRequest(
        request_id=str(uuid.uuid4()),
        task_id=task_id,
        branch_id=f"{task_id}:branch:{branch_idx}",
        origin_occurrence_id=origin.occurrence_id,
        environment_reset_key=root.environment_reset_key,
        task_description=root.task_description,
        task_batch_index=root.task_batch_index,
        target_turn=event.step_index,
        parsed_action_prefix=tuple(item.parsed_environment_action for item in prefix),
        prefix_observations=tuple(item.pre_action_observation for item in prefix),
        expected_anchor_key=anchor.anchor_key,
        expected_observation=event.pre_action_observation,
        expected_action_set=event.admissible_actions,
        selected_canonical_action=action,
        copied_response_token_ids=event.response_token_ids,
        copied_raw_model_response=event.raw_model_response,
        copied_response_loss_mask=event.response_loss_mask,
        copied_old_log_probs=event.old_log_probs,
        original_prompt_token_ids=event.prompt_token_ids,
        remaining_horizon=event.remaining_horizon,
        copied_parsed_environment_action=event.parsed_environment_action,
        copied_action_identity=event.action_identity or event.canonical_action,
        copied_action_identity_kind=event.action_identity_kind,
        copied_action_environment_valid=event.action_environment_valid,
        expected_post_action_observation=event.post_action_observation,
        expected_immediate_reward=event.reward,
        expected_post_action_done=event.done,
    )


class FixedTopologyCoordinator:
    """Stage-2 random acquisition with a fixed natural-root/branch budget."""

    def __init__(self, root_count: int = 3, branch_count: int = 1, seed: int = 0,
                 invalid_action_mode: str = "strict_identity"):
        if root_count < 2 or branch_count < 1:
            raise ValueError("Fixed topology requires at least two roots and one branch")
        self.root_count = root_count
        self.branch_count = branch_count
        self.rng = random.Random(seed)
        self.invalid_action_mode = invalid_action_mode

    def build_requests(self, roots: list[RootEventLog]) -> tuple[list[ReplayRequest], dict[str, str]]:
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in roots:
            roots_by_task.setdefault(root.task_id, []).append(root)
        for task_id, task_roots in roots_by_task.items():
            if len(task_roots) != self.root_count:
                raise ValueError(f"Task {task_id} has {len(task_roots)} roots, expected {self.root_count}")

        index = AnchorIndex(roots, invalid_action_mode=self.invalid_action_mode)
        requests = []
        skipped = {}
        for task_id in sorted(roots_by_task):
            anchors = index.anchors_for_task(task_id)
            if not anchors:
                skipped[task_id] = "NO_STRUCTURAL_ANCHOR"
                continue
            for branch_idx in range(self.branch_count):
                anchor = self.rng.choice(anchors)
                action = self.rng.choice(anchor.observed_action_ids)
                origin = self.rng.choice(anchor.origins_by_action[action])
                requests.append(_make_request(index, task_id, branch_idx, anchor, action, origin))
        return requests, skipped


@dataclass
class AcquisitionTaskState:
    task_id: str
    branch_quota: int
    posteriors: dict[str, dict[str, BetaPosterior]]
    branch_count_by_anchor: dict[str, int] = field(default_factory=dict)
    requested_slots: int = 0
    completed_branches: int = 0


class ExpectedErvCoordinator:
    """Fixed-topology, sequential ERV acquisition over frozen natural support."""

    def __init__(
        self,
        root_count: int,
        branch_count: int,
        max_branches_per_anchor: int,
        prior_strength: float,
        erv_threshold: float,
        temperature: float,
        mc_samples: int,
        seed: int = 0,
        invalid_action_mode: str = "strict_identity",
    ):
        self.root_count = root_count
        self.branch_count = branch_count
        self.max_branches_per_anchor = max_branches_per_anchor
        self.prior_strength = prior_strength
        self.erv_threshold = erv_threshold
        self.rng = np.random.default_rng(seed)
        self.engine = PosteriorEngine(mc_samples=mc_samples, temperature=temperature, seed=seed)
        self.invalid_action_mode = invalid_action_mode
        self.index = None
        self.states: dict[str, AcquisitionTaskState] = {}
        self._selection_by_request: dict[str, tuple[str, str]] = {}
        self.last_round_diagnostics: list[dict[str, object]] = []

    def initialize(
        self,
        roots: list[RootEventLog],
        branch_quota_by_task: dict[str, int] | None = None,
        prior_mean_by_task: dict[str, float] | None = None,
    ) -> dict[str, str]:
        roots_by_task: dict[str, list[RootEventLog]] = {}
        for root in roots:
            roots_by_task.setdefault(root.task_id, []).append(root)
        self.index = AnchorIndex(roots, invalid_action_mode=self.invalid_action_mode)
        self.states = {}
        self.last_round_diagnostics = []
        skipped = {}
        for task_id, task_roots in roots_by_task.items():
            if branch_quota_by_task is None and len(task_roots) != self.root_count:
                raise ValueError(f"Task {task_id} has {len(task_roots)} roots, expected {self.root_count}")
            successes = sum(root.won for root in task_roots)
            prior_mean = (
                prior_mean_by_task[task_id]
                if prior_mean_by_task is not None
                else (1.0 + successes) / (2.0 + len(task_roots))
            )
            branch_quota = (
                branch_quota_by_task[task_id]
                if branch_quota_by_task is not None
                else self.branch_count
            )
            if branch_quota == 0:
                continue
            anchor_posteriors = {}
            for anchor in self.index.anchors_for_task(task_id):
                action_posteriors = {}
                for action, origins in anchor.origins_by_action.items():
                    outcomes = [self.index.root_for(origin.root_id).won for origin in origins]
                    action_posteriors[action] = initialize_local_posterior(
                        prior_mean, self.prior_strength, outcomes
                    )
                acquisition = self.engine.evaluate(action_posteriors)
                if acquisition.utility >= self.erv_threshold:
                    anchor_posteriors[anchor.anchor_id] = action_posteriors
            if not anchor_posteriors:
                skipped[task_id] = "NO_EFFECTIVE_ANCHOR"
                continue
            self.states[task_id] = AcquisitionTaskState(
                task_id=task_id,
                branch_quota=branch_quota,
                posteriors=anchor_posteriors,
                branch_count_by_anchor={anchor_id: 0 for anchor_id in anchor_posteriors},
            )
        return skipped

    def build_round_requests(self) -> list[ReplayRequest]:
        if self.index is None:
            raise RuntimeError("Coordinator must be initialized before acquisition")
        requests = []
        self.last_round_diagnostics = []
        anchors_by_id = {anchor.anchor_id: anchor for anchor in self.index._anchors.values()}
        for task_id in sorted(self.states):
            state = self.states[task_id]
            if state.requested_slots >= state.branch_quota:
                continue
            candidates = []
            candidate_diagnostics = []
            for anchor_id, posteriors in state.posteriors.items():
                if state.branch_count_by_anchor[anchor_id] >= self.max_branches_per_anchor:
                    continue
                acquisition = self.engine.evaluate(posteriors)
                total_evidence = sum(posterior.evidence for posterior in posteriors.values())
                candidates.append((acquisition.utility, acquisition.regret, -total_evidence, anchor_id, acquisition))
                candidate_diagnostics.append({
                    "anchor_id": anchor_id,
                    "utility": acquisition.utility,
                    "regret": acquisition.regret,
                    "erv_by_action": acquisition.erv_by_action,
                    "probability_by_action": acquisition.probability_by_action,
                    "posteriors": {
                        action_id: {
                            "alpha": posterior.alpha,
                            "beta": posterior.beta,
                            "natural_successes": posterior.natural_successes,
                            "natural_failures": posterior.natural_failures,
                            "branch_successes": posterior.branch_successes,
                            "branch_failures": posterior.branch_failures,
                        }
                        for action_id, posterior in posteriors.items()
                    },
                })
            if not candidates:
                continue
            _, _, _, anchor_id, acquisition = max(candidates, key=lambda item: item[:3] + (item[3],))
            anchor = anchors_by_id[anchor_id]
            actions = sorted(acquisition.probability_by_action)
            probabilities = [acquisition.probability_by_action[action] for action in actions]
            action = str(self.rng.choice(actions, p=probabilities))
            origins = anchor.origins_by_action[action]
            origin = origins[int(self.rng.integers(len(origins)))]
            request = _make_request(self.index, task_id, state.requested_slots, anchor, action, origin)
            state.requested_slots += 1
            self._selection_by_request[request.request_id] = (anchor_id, action)
            requests.append(request)
            selected = next(item for item in candidate_diagnostics if item["anchor_id"] == anchor_id)
            self.last_round_diagnostics.append({
                "task_id": task_id,
                "request_id": request.request_id,
                "selected_anchor_id": anchor_id,
                "selected_action_id": action,
                "candidates": candidate_diagnostics,
                "selected": selected,
            })
        return requests

    def update_from_branch(self, request: ReplayRequest, success: bool, suffix_pairs=()) -> None:
        state = self.states[request.task_id]
        anchor_id, action = self._selection_by_request.pop(request.request_id)
        state.posteriors[anchor_id][action].update_branch(success)
        state.branch_count_by_anchor[anchor_id] += 1
        state.completed_branches += 1

        frozen_pairs = {
            (candidate_anchor_id, candidate_action): posterior
            for candidate_anchor_id, posteriors in state.posteriors.items()
            for candidate_action, posterior in posteriors.items()
        }
        seen = {(anchor_id, action)}
        anchors_by_key = {
            to_hashable(anchor.anchor_key): anchor.anchor_id
            for anchor in self.index.anchors_for_task(request.task_id)
            if anchor.anchor_id in state.posteriors
        }
        for observation, suffix_action in suffix_pairs:
            suffix_anchor_id = anchors_by_key.get(to_hashable(observation))
            pair = (suffix_anchor_id, str(suffix_action))
            if suffix_anchor_id is not None and pair in frozen_pairs and pair not in seen:
                frozen_pairs[pair].update_branch(success)
                seen.add(pair)

    def adopt_retry_request(self, old_request: ReplayRequest, new_request: ReplayRequest) -> None:
        """Move the pending selection bookkeeping to a retry request id."""
        selection = self._selection_by_request.pop(old_request.request_id, None)
        if selection is not None:
            self._selection_by_request[new_request.request_id] = selection

    def acquisition_snapshot(self, task_id: str) -> dict[str, object]:
        state = self.states[task_id]
        return {
            anchor_id: self.engine.evaluate(posteriors)
            for anchor_id, posteriors in state.posteriors.items()
        }

    def posterior_snapshot(self) -> dict[str, object]:
        """Return posterior counters without consuming the ERV RNG."""
        return {
            task_id: {
                anchor_id: {
                    action_id: {
                        "alpha": posterior.alpha,
                        "beta": posterior.beta,
                        "natural_successes": posterior.natural_successes,
                        "natural_failures": posterior.natural_failures,
                        "branch_successes": posterior.branch_successes,
                        "branch_failures": posterior.branch_failures,
                    }
                    for action_id, posterior in posteriors.items()
                }
                for anchor_id, posteriors in state.posteriors.items()
            }
            for task_id, state in self.states.items()
        }


class ExactBatchErvCoordinator:
    """Freeze support, solve the exact joint plan, then emit one parallel round."""

    def __init__(
        self,
        max_branches_per_anchor: int,
        prior_strength: float,
        threshold: float,
        tie_abs_tolerance: float = 1e-12,
        tie_rel_tolerance: float = 1e-10,
        seed: int = 0,
        invalid_action_mode: str = "strict_identity",
    ):
        self.prior_strength = float(prior_strength)
        self.invalid_action_mode = invalid_action_mode
        self.engine = ExactBatchErvEngine(
            max_branches_per_anchor=max_branches_per_anchor,
            threshold=threshold,
            tie_abs_tolerance=tie_abs_tolerance,
            tie_rel_tolerance=tie_rel_tolerance,
            seed=seed,
        )
        self.policy_update_id = 0
        self.index = None
        self._pending_requests = []
        self._selection_by_request = {}
        self._posterior_by_task = {}
        self._completed_outcomes = {}
        self.last_round_diagnostics = []

    def set_policy_update_id(self, policy_update_id: int) -> None:
        self.policy_update_id = int(policy_update_id)

    @staticmethod
    def _posterior_payload(posteriors):
        return {
            action: {
                "alpha": posterior.alpha,
                "beta": posterior.beta,
                "mean": posterior.mean,
                "natural_successes": posterior.natural_successes,
                "natural_failures": posterior.natural_failures,
            }
            for action, posterior in posteriors.items()
        }

    @staticmethod
    def _design_payload(design):
        return {
            "anchor_id": design.anchor_id,
            "values_by_size": design.values_by_size,
            "delta_by_size": design.delta_by_size,
            "capacity": design.capacity,
            "all_plans_by_size": {
                size: [
                    {"actions": plan.actions, "value": plan.value}
                    for plan in plans
                ]
                for size, plans in design.all_plans_by_size.items()
            },
            "tie_optimal_plans_by_size": {
                size: [
                    {"actions": plan.actions, "value": plan.value}
                    for plan in plans
                ]
                for size, plans in design.optimal_plans_by_size.items()
            },
            "tie_optimal_plan_count_by_size": {
                size: len(plans)
                for size, plans in design.optimal_plans_by_size.items()
            },
        }

    def initialize(
        self,
        roots: list[RootEventLog],
        branch_quota_by_task: dict[str, int] | None = None,
        prior_mean_by_task: dict[str, float] | None = None,
    ) -> dict[str, str]:
        if branch_quota_by_task is None or prior_mean_by_task is None:
            raise ValueError(
                "Exact Batch-ERV requires topology branch quotas and current-instance prior means"
            )
        roots_by_task = {}
        for root in roots:
            roots_by_task.setdefault(root.task_id, []).append(root)
        self.index = AnchorIndex(roots, invalid_action_mode=self.invalid_action_mode)
        anchors_by_id = self.index._anchors
        self._pending_requests = []
        self._selection_by_request = {}
        self._posterior_by_task = {}
        self._completed_outcomes = {}
        self.last_round_diagnostics = []
        skipped = {}

        for task_id in sorted(roots_by_task):
            quota = int(branch_quota_by_task[task_id])
            if quota == 0:
                self.last_round_diagnostics.append({
                    "task_id": task_id,
                    "branch_quota": 0,
                    "global_optimal_value": 0.0,
                    "selected_allocation": {},
                    "status": "ROOT_ONLY_TOPOLOGY",
                })
                continue
            posteriors_by_anchor = {}
            designs = {}
            for anchor in self.index.anchors_for_task(task_id):
                posteriors = {}
                for action, origins in anchor.origins_by_action.items():
                    outcomes = [self.index.root_for(origin.root_id).won for origin in origins]
                    posteriors[action] = initialize_local_posterior(
                        prior_mean_by_task[task_id], self.prior_strength, outcomes
                    )
                posteriors_by_anchor[anchor.anchor_id] = posteriors
                designs[anchor.anchor_id] = self.engine.design_anchor(
                    anchor.anchor_id, posteriors
                )
            total_capacity = sum(design.capacity for design in designs.values())
            if total_capacity < quota:
                raise ValueError(
                    f"Task {task_id} exact capacity {total_capacity} is below frozen quota {quota}"
                )
            global_result = self.engine.global_allocation(
                designs,
                quota,
                self.policy_update_id,
                task_id,
                "global_allocation",
            )
            allocation = global_result.selected_allocation
            self._posterior_by_task[task_id] = posteriors_by_anchor
            selected_local_plans = {}
            branch_idx = 0
            for anchor_id in sorted(allocation):
                size = allocation[anchor_id]
                if size == 0:
                    continue
                local_ties = designs[anchor_id].optimal_plans_by_size[size]
                local_plan = self.engine.choose_uniform(
                    local_ties,
                    self.policy_update_id,
                    task_id,
                    anchor_id,
                    "local_plan",
                    size,
                )
                selected_local_plans[anchor_id] = {
                    "size": size,
                    "actions": local_plan.actions,
                    "value": local_plan.value,
                    "tie_count": len(local_ties),
                }
                anchor = anchors_by_id[anchor_id]
                for local_index, action in enumerate(local_plan.actions):
                    origins = anchor.origins_by_action[action]
                    origin = self.engine.choose_uniform(
                        origins,
                        self.policy_update_id,
                        task_id,
                        anchor_id,
                        action,
                        "origin",
                        local_index,
                    )
                    request = _make_request(
                        self.index, task_id, branch_idx, anchor, action, origin
                    )
                    self._selection_by_request[request.request_id] = (anchor_id, action)
                    self._pending_requests.append(request)
                    branch_idx += 1
            if branch_idx != quota:
                raise AssertionError(
                    f"Task {task_id} emitted {branch_idx} branches for quota {quota}"
                )
            self.last_round_diagnostics.append({
                "task_id": task_id,
                "branch_quota": quota,
                "current_instance_prior_mean": prior_mean_by_task[task_id],
                "information_capacity": total_capacity,
                "anchors": {
                    anchor_id: {
                        "posteriors": self._posterior_payload(posteriors_by_anchor[anchor_id]),
                        "design": self._design_payload(design),
                    }
                    for anchor_id, design in designs.items()
                },
                "global_allocation": {
                    "solver": global_result.solver,
                    "num_anchors": global_result.num_anchors,
                    "branch_quota": global_result.branch_quota,
                    "total_information_capacity": (
                        global_result.total_information_capacity
                    ),
                    "reachable_state_count": global_result.reachable_state_count,
                    "optimal_value": global_result.optimal_value,
                    "optimal_tie_count": global_result.optimal_count,
                    "selected_allocation": allocation,
                    "solver_wall_time_ms": global_result.solver_wall_time_ms,
                },
                "global_optimal_value": global_result.optimal_value,
                "global_tie_count": global_result.optimal_count,
                "selected_allocation": allocation,
                "selected_local_plans": selected_local_plans,
                "status": "PLANNED",
            })
        return skipped

    def build_round_requests(self) -> list[ReplayRequest]:
        requests = self._pending_requests
        self._pending_requests = []
        return requests

    def update_from_branch(self, request: ReplayRequest, success: bool, suffix_pairs=()) -> None:
        selection = self._selection_by_request.pop(request.request_id, None)
        if selection is None:
            return
        anchor_id, action = selection
        posterior = self._posterior_by_task[request.task_id][anchor_id][action]
        posterior.update_branch(success)
        self._completed_outcomes[request.branch_id] = {
            "task_id": request.task_id,
            "anchor_id": anchor_id,
            "action": action,
            "success": bool(success),
            "suffix_pair_count": len(suffix_pairs),
        }

    def adopt_retry_request(self, old_request: ReplayRequest, new_request: ReplayRequest) -> None:
        selection = self._selection_by_request.pop(old_request.request_id, None)
        if selection is not None:
            self._selection_by_request[new_request.request_id] = selection

    def posterior_snapshot(self) -> dict[str, object]:
        return {
            "frozen_acquisition_posteriors": {
                task_id: {
                    anchor_id: self._posterior_payload(posteriors)
                    for anchor_id, posteriors in anchors.items()
                }
                for task_id, anchors in self._posterior_by_task.items()
            },
            "completed_branch_outcomes": self._completed_outcomes,
        }
