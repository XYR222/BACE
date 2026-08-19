from dataclasses import replace
import json

import pytest

from recipe.bace_gigpo.competence import CompetenceHistory
from recipe.bace_gigpo.topology import DynamicTopologyPlanner, ExactBatchTopologyPlanner
from tests.bace_gigpo.test_anchor_and_replay import root


def make_planner(history, budget=4):
    return DynamicTopologyPlanner(
        history=history,
        total_budget=budget,
        pilot_roots=2,
        competence_threshold=0.5,
        max_branches_per_anchor=2,
        local_prior_strength=2.0,
        erv_threshold=0.0,
        erv_temperature=0.1,
        erv_mc_samples=512,
        seed=7,
    )


def won_root(root_id, action):
    return replace(root(root_id, action), terminal_reward=1.0, won=True)


def test_competence_history_is_lagged_decayed_and_bounded():
    history = CompetenceHistory(forgetting=0.5, transfer_fraction=1.0, max_strength=4.0)
    initial = history.prior("heat")
    assert initial.mean == 0.5
    assert initial.strength == 2.0
    history.update({"heat": [True, True, False]})
    updated = history.prior("heat")
    assert updated.mean == 3.0 / 5.0
    assert updated.strength == 4.0
    history.update({"heat": []})
    assert history.snapshot()["heat"] == (1.0, 0.5)


def test_cold_start_prior_plans_zero_branches_for_exact_main_configuration():
    history = CompetenceHistory(
        base_alpha=0.2,
        base_beta=1.8,
        forgetting=0.8,
        transfer_fraction=0.1,
        min_strength=2.0,
        max_strength=8.0,
    )
    planner = ExactBatchTopologyPlanner(
        history=history,
        total_budget=8,
        min_natural_roots=2,
        competence_threshold=0.5,
        max_branches_per_anchor=2,
        local_prior_strength=2.0,
        batch_erv_threshold=0.005,
        tie_abs_tolerance=1e-12,
        tie_rel_tolerance=1e-10,
    )

    state = planner.initialize({"task-1": "pick_and_place"})["task-1"]

    assert state.family_posterior.alpha == pytest.approx(0.2)
    assert state.family_posterior.beta == pytest.approx(1.8)
    assert state.readiness == pytest.approx(0.05205601166667073)
    assert state.planned_branch_count == 0
    assert (state.root_count, state.branch_count) == (8, 0)


def test_competence_history_state_round_trip_and_validation():
    history = CompetenceHistory(
        base_alpha=0.2,
        base_beta=1.8,
        forgetting=0.8,
        transfer_fraction=0.1,
        min_strength=2.0,
        max_strength=8.0,
    )
    history.update({"heat": [True, False, False], "cool": [True]})
    state = json.loads(json.dumps(history.state_dict()))
    restored = CompetenceHistory(
        base_alpha=0.2,
        base_beta=1.8,
        forgetting=0.8,
        transfer_fraction=0.1,
        min_strength=2.0,
        max_strength=8.0,
    )
    restored.load_state_dict(state)
    assert restored.snapshot() == history.snapshot()

    mismatched = CompetenceHistory(base_alpha=1.0, base_beta=1.0)
    with pytest.raises(ValueError, match="parameter mismatch"):
        mismatched.load_state_dict(state)

    state["families"]["heat"]["decayed_failures"] = -1
    with pytest.raises(ValueError, match="invalid competence history counts"):
        restored.load_state_dict(state)


def test_high_competence_allocates_branches_when_capacity_exists():
    roots = [
        won_root("r1", "open fridge"),
        won_root("r2", "go to table"),
        root("r3", "open fridge"),
        root("r4", "go to table"),
    ]
    plan = make_planner(CompetenceHistory()).plan(roots)
    task = plan.tasks["task-1"]
    assert task.planned_branch_count == 2
    assert task.final_root_count == 2
    assert task.final_branch_count == 2
    assert task.final_root_count + task.final_branch_count == 4


def test_capacity_correction_converts_one_branch_slot_to_root():
    roots = [
        won_root("r1", "open fridge"),
        won_root("r2", "open fridge"),
        root("r3", "go to table"),
        root("r4", "go to table"),
    ]
    plan = make_planner(CompetenceHistory()).plan(roots)
    task = plan.tasks["task-1"]
    assert task.planned_branch_count == 2
    assert task.final_root_count == 3
    assert task.final_branch_count == 1
    assert len(plan.roots) == 3


def test_low_competence_keeps_full_root_budget():
    roots = [
        root("r1", "open fridge"),
        root("r2", "go to table"),
        root("r3", "open fridge"),
        root("r4", "go to table"),
    ]
    plan = make_planner(CompetenceHistory()).plan(roots)
    task = plan.tasks["task-1"]
    assert task.planned_branch_count == 0
    assert task.final_root_count == 4
    assert task.final_branch_count == 0


def test_staged_high_competence_stops_at_required_roots():
    planner = make_planner(CompetenceHistory())
    pilots = [
        won_root("r1", "open fridge"),
        won_root("r2", "go to table"),
    ]
    states = planner.initialize_staged(pilots)
    state = states["task-1"]
    assert state.root_count == 2
    assert state.branch_count == 2

    assert planner.correct_staged_capacity(pilots, states) == set()
    plan = planner.finalize_staged(pilots, states)
    task = plan.tasks["task-1"]
    assert len(plan.roots) == 2
    assert task.final_root_count == 2
    assert task.final_branch_count == 2
    assert task.final_root_count + task.final_branch_count == 4


def test_staged_capacity_correction_requests_roots_one_at_a_time():
    planner = make_planner(CompetenceHistory())
    pilots = [
        won_root("r1", "open fridge"),
        won_root("r2", "open fridge"),
    ]
    states = planner.initialize_staged(pilots)

    assert planner.correct_staged_capacity(pilots, states) == {"task-1"}
    state = states["task-1"]
    assert (state.root_count, state.branch_count) == (3, 1)

    roots = pilots + [root("r3", "go to table")]
    assert planner.correct_staged_capacity(roots, states) == set()
    plan = planner.finalize_staged(roots, states)
    task = plan.tasks["task-1"]
    assert len(plan.roots) == 3
    assert task.final_root_count == 3
    assert task.final_branch_count == 1


def test_staged_low_competence_requests_full_root_budget():
    planner = make_planner(CompetenceHistory())
    pilots = [root("r1", "open fridge"), root("r2", "go to table")]
    states = planner.initialize_staged(pilots)
    state = states["task-1"]
    assert state.root_count == 4
    assert state.branch_count == 0

    roots = pilots + [root("r3", "open fridge"), root("r4", "go to table")]
    assert planner.correct_staged_capacity(roots, states) == set()
    plan = planner.finalize_staged(roots, states)
    assert len(plan.roots) == 4
