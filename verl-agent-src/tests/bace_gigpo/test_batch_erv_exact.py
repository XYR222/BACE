from dataclasses import replace

import pytest

from recipe.bace_gigpo.batch_erv import ExactBatchErvEngine
from recipe.bace_gigpo.competence import CompetenceHistory
from recipe.bace_gigpo.coordinator import ExactBatchErvCoordinator
from recipe.bace_gigpo.posterior import BetaPosterior
from recipe.bace_gigpo.topology import ExactBatchTopologyPlanner
from tests.bace_gigpo.test_anchor_and_replay import root
from tests.bace_gigpo.test_invalid_action_identity import (
    invalid_event,
    root_with_event,
    valid_event,
)


def won(item):
    return replace(item, terminal_reward=1.0, won=True)


def engine(threshold=0.0, seed=17):
    return ExactBatchErvEngine(
        max_branches_per_anchor=2,
        threshold=threshold,
        tie_abs_tolerance=1e-12,
        tie_rel_tolerance=1e-10,
        seed=seed,
    )


def test_exact_one_sample_erv_matches_closed_form():
    posteriors = {"a": BetaPosterior(1.0, 1.0), "b": BetaPosterior(1.0, 1.0)}
    assert engine().value(posteriors, ("a",)) == pytest.approx(1.0 / 12.0)


def test_exact_two_sample_beta_binomial_enumeration_and_ties():
    posteriors = {"a": BetaPosterior(1.0, 1.0), "b": BetaPosterior(1.0, 1.0)}
    design = engine().design_anchor("z", posteriors)

    assert design.values_by_size[1] == pytest.approx(1.0 / 12.0)
    assert design.values_by_size[2] == pytest.approx(1.0 / 12.0)
    assert design.delta_by_size[2] == pytest.approx(0.0, abs=1e-12)
    assert {plan.actions for plan in design.optimal_plans_by_size[1]} == {("a",), ("b",)}
    assert {plan.actions for plan in design.optimal_plans_by_size[2]} == {
        ("a", "a"),
        ("a", "b"),
        ("b", "b"),
    }
    # Threshold equality is valid, so a zero second marginal remains a slot at tau=0.
    assert design.capacity == 2


def test_global_allocation_is_exact_and_seeded_ties_are_reproducible():
    posteriors = {"a": BetaPosterior(1.0, 1.0), "b": BetaPosterior(1.0, 1.0)}
    exact = engine(seed=9)
    designs = {
        "z1": exact.design_anchor("z1", posteriors),
        "z2": exact.design_anchor("z2", posteriors),
    }
    value, ties = exact.global_allocations(designs, branch_quota=1)
    first = exact.choose_uniform(ties, 4, "task", "global")
    second = exact.choose_uniform(ties, 4, "task", "global")

    assert value == pytest.approx(1.0 / 12.0)
    assert len(ties) == 2
    assert first == second
    assert sum(first.values()) == 1


def make_exact_planner(history, budget=4):
    return ExactBatchTopologyPlanner(
        history=history,
        total_budget=budget,
        min_natural_roots=2,
        competence_threshold=0.5,
        max_branches_per_anchor=2,
        local_prior_strength=2.0,
        batch_erv_threshold=0.0,
        tie_abs_tolerance=1e-12,
        tie_rel_tolerance=1e-10,
        seed=5,
    )


def test_exact_topology_has_no_pilot_and_uses_only_lagged_history():
    history = CompetenceHistory(forgetting=0.9, transfer_fraction=0.1)
    planner = make_exact_planner(history)
    before = planner.initialize({"task-1": "pick_and_place"})["task-1"]
    history.update({"pick_and_place": [True] * 20})
    after = planner.initialize({"task-2": "pick_and_place"})["task-2"]

    assert before.planned_branch_count == 1
    assert (before.root_count, before.branch_count) == (3, 1)
    assert after.planned_branch_count == 2
    assert (after.root_count, after.branch_count) == (2, 2)


def test_exact_capacity_correction_is_one_way_and_preserves_budget():
    history = CompetenceHistory()
    history.update({"pick_and_place": [True] * 20})
    planner = make_exact_planner(history)
    states = planner.initialize({"task-1": "pick_and_place"})
    state = states["task-1"]
    initial = [won(root("r1", "open fridge")), won(root("r2", "open fridge"))]

    assert planner.correct_capacity(initial, states) == {"task-1"}
    assert (state.root_count, state.branch_count, state.correction_count) == (3, 1, 1)

    corrected = initial + [root("r3", "go to table")]
    assert planner.correct_capacity(corrected, states) == set()
    plan = planner.finalize(corrected, states)
    task = plan.tasks["task-1"]
    assert task.final_root_count + task.final_branch_count == 4
    assert task.information_capacity >= task.final_branch_count


def test_exact_coordinator_emits_all_branches_in_one_frozen_batch():
    roots = [
        won(root("r1", "open fridge")),
        root("r2", "go to table"),
        root("r3", "open fridge"),
    ]
    coordinator = ExactBatchErvCoordinator(
        max_branches_per_anchor=2,
        prior_strength=2.0,
        threshold=0.0,
        seed=11,
    )
    coordinator.set_policy_update_id(7)
    assert coordinator.initialize(
        roots,
        branch_quota_by_task={"task-1": 2},
        prior_mean_by_task={"task-1": 0.5},
    ) == {}

    requests = coordinator.build_round_requests()
    assert len(requests) == 2
    assert coordinator.build_round_requests() == []
    assert all(request.task_id == "task-1" for request in requests)
    assert len(coordinator.last_round_diagnostics) == 1
    diagnostic = coordinator.last_round_diagnostics[0]
    assert diagnostic["global_tie_count"] >= 1
    assert sum(diagnostic["selected_allocation"].values()) == 2


def test_exact_strict_identity_support_includes_valid_and_invalid_natural_edges():
    roots = [
        root_with_event("r1", invalid_event("r1:1", "go to cabinet 5")),
        root_with_event("r2", valid_event("r2:1", "open fridge")),
    ]
    coordinator = ExactBatchErvCoordinator(
        max_branches_per_anchor=2,
        prior_strength=2.0,
        threshold=0.0,
        seed=3,
        invalid_action_mode="strict_identity",
    )
    coordinator.initialize(
        roots,
        branch_quota_by_task={"task-1": 1},
        prior_mean_by_task={"task-1": 0.5},
    )

    anchor = next(iter(coordinator.last_round_diagnostics[0]["anchors"].values()))
    assert set(anchor["posteriors"]) == {
        "invalid::go to cabinet 5",
        "valid::open fridge",
    }
