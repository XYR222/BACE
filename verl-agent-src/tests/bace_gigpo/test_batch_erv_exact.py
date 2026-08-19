from dataclasses import replace
import math
import random

import pytest

from recipe.bace_gigpo.batch_erv import AnchorBatchDesign, ExactBatchErvEngine
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


def allocation_design(anchor_id, values, capacity=None):
    if capacity is None:
        capacity = max(values)
    return AnchorBatchDesign(
        anchor_id=anchor_id,
        values_by_size=dict(values),
        optimal_plans_by_size={},
        all_plans_by_size={},
        delta_by_size={},
        capacity=capacity,
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
    first = exact.global_allocation(designs, 1, 4, "task", "global")
    second = exact.global_allocation(designs, 1, 4, "task", "global")

    assert first.optimal_value == pytest.approx(1.0 / 12.0)
    assert first.optimal_count == 2
    assert first.selected_allocation == second.selected_allocation
    assert sum(first.selected_allocation.values()) == 1
    assert first.solver == "quota_aware_exact_dp"


def test_global_dp_matches_cartesian_reference_for_small_random_problems():
    exact = engine(seed=23)
    rng = random.Random(101)
    for num_anchors in range(1, 7):
        for fixture_index in range(20):
            designs = {}
            for anchor_index in range(num_anchors):
                capacity = rng.randrange(3)
                values = {0: 0.0}
                for size in range(1, capacity + 1):
                    values[size] = rng.random()
                anchor_id = f"z{anchor_index:02d}"
                designs[anchor_id] = allocation_design(anchor_id, values, capacity)
            total_capacity = sum(item.capacity for item in designs.values())
            for quota in range(min(4, total_capacity) + 1):
                reference_value, reference_ties = (
                    exact._global_allocations_cartesian_reference(designs, quota)
                )
                result = exact.global_allocation(
                    designs, quota, num_anchors, fixture_index, quota
                )
                assert result.optimal_value == pytest.approx(reference_value)
                assert result.optimal_count == len(reference_ties)
                assert result.selected_allocation in reference_ties


def test_global_dp_tie_sampling_is_uniform_over_complete_allocations():
    exact = engine(seed=29)
    designs = {
        f"z{index}": allocation_design(f"z{index}", {0: 0.0, 1: 1.0}, 1)
        for index in range(3)
    }
    counts = {anchor_id: 0 for anchor_id in designs}
    for sample_seed in range(3000):
        result = exact.global_allocation(designs, 1, sample_seed)
        selected = [
            anchor_id
            for anchor_id, size in result.selected_allocation.items()
            if size == 1
        ]
        assert len(selected) == 1
        counts[selected[0]] += 1
    assert all(abs(count - 1000) < 120 for count in counts.values())


def test_global_dp_handles_zero_quota_mixed_capacity_and_infeasibility():
    exact = engine()
    designs = {
        "z0": allocation_design("z0", {0: 0.0}, 0),
        "z1": allocation_design("z1", {0: 0.0, 1: 0.2}, 1),
        "z2": allocation_design("z2", {0: 0.0, 1: 0.3, 2: 0.4}, 2),
    }
    zero = exact.global_allocation(designs, 0, "zero")
    assert zero.optimal_value == 0.0
    assert zero.optimal_count == 1
    assert zero.selected_allocation == {"z0": 0, "z1": 0, "z2": 0}

    mixed = exact.global_allocation(designs, 2, "mixed")
    assert mixed.optimal_value == pytest.approx(0.5)
    assert mixed.selected_allocation == {"z0": 0, "z1": 1, "z2": 1}

    with pytest.raises(ValueError, match="below branch quota"):
        exact.global_allocation(designs, 4, "infeasible")


def test_incident_scale_q1_regression_is_linear_and_reference_is_guarded():
    exact = engine(seed=31)
    designs = {}
    for index in range(27):
        capacity = 0 if index == 0 else 2
        values = {0: 0.0}
        if capacity:
            values.update({1: float(index % 5), 2: float(index % 5) + 0.25})
        anchor_id = f"anchor-{index:02d}"
        designs[anchor_id] = allocation_design(anchor_id, values, capacity)

    result = exact.global_allocation(designs, 1, "step-3-regression")
    best_value = max(design.values_by_size.get(1, -math.inf) for design in designs.values())
    best_anchor_count = sum(
        design.capacity >= 1 and exact.tied(design.values_by_size[1], best_value)
        for design in designs.values()
    )
    assert result.optimal_value == best_value
    assert result.optimal_count == best_anchor_count
    assert sum(result.selected_allocation.values()) == 1
    assert result.solver_wall_time_ms < 1000.0
    with pytest.raises(RuntimeError, match="Cartesian reference solver disabled"):
        exact._global_allocations_cartesian_reference(designs, 1)


def test_large_global_dp_does_not_materialize_huge_tie_set():
    exact = engine(seed=37)
    designs = {
        f"z{index:03d}": allocation_design(
            f"z{index:03d}", {0: 0.0, 1: 1.0}, 1
        )
        for index in range(100)
    }
    result = exact.global_allocation(designs, 6, "large-tie")
    assert result.optimal_value == pytest.approx(6.0)
    assert result.optimal_count == math.comb(100, 6)
    assert sum(result.selected_allocation.values()) == 6
    assert result.reachable_state_count <= 1 + 100 * 7
    assert result.solver_wall_time_ms < 1000.0


def test_large_q6_capacity_two_global_dp_is_fast():
    exact = engine(seed=41)
    designs = {
        f"z{index:03d}": allocation_design(
            f"z{index:03d}", {0: 0.0, 1: 0.1, 2: 0.15}, 2
        )
        for index in range(100)
    }
    result = exact.global_allocation(designs, 6, "large-capacity-two")
    assert result.optimal_value == pytest.approx(0.6)
    assert sum(result.selected_allocation.values()) == 6
    assert result.solver_wall_time_ms < 1000.0


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
    assert diagnostic["global_allocation"]["solver"] == "quota_aware_exact_dp"
    assert diagnostic["global_allocation"]["branch_quota"] == 2
    assert "tie_optimal_global_allocations" not in diagnostic


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
