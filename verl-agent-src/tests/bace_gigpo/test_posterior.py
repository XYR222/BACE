from dataclasses import replace

import numpy as np

from recipe.bace_gigpo.coordinator import ExpectedErvCoordinator
from recipe.bace_gigpo.posterior import BetaPosterior, PosteriorEngine, initialize_local_posterior
from tests.bace_gigpo.test_anchor_and_replay import root


def test_beta_initialization_and_branch_update():
    posterior = initialize_local_posterior(0.25, 2.0, [True, False, False])
    assert posterior.alpha == 1.5
    assert posterior.beta == 3.5
    assert posterior.natural_successes == 1
    posterior.update_branch(True)
    assert posterior.alpha == 2.5
    assert posterior.branch_successes == 1


def test_erv_distribution_is_normalized_and_utility_matches_expectation():
    engine = PosteriorEngine(mc_samples=4096, temperature=0.1, seed=3)
    result = engine.evaluate(
        {
            "a": BetaPosterior(2.0, 2.0),
            "b": BetaPosterior(5.0, 1.0),
            "c": BetaPosterior(1.0, 5.0),
        }
    )
    probabilities = np.array(list(result.probability_by_action.values()))
    erv = np.array(list(result.erv_by_action.values()))
    assert result.regret >= 0
    assert np.all(erv >= 0)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(result.utility, np.dot(probabilities, erv))


def test_sequential_coordinator_freezes_support_and_deduplicates_suffix_evidence():
    roots = [
        replace(root("r1", "open fridge"), terminal_reward=1.0, won=True),
        root("r2", "go to table"),
        root("r3", "open fridge"),
    ]
    coordinator = ExpectedErvCoordinator(
        root_count=3,
        branch_count=2,
        max_branches_per_anchor=2,
        prior_strength=2.0,
        erv_threshold=0.0,
        temperature=0.1,
        mc_samples=1024,
        seed=4,
    )
    assert coordinator.initialize(roots) == {}
    state = coordinator.states["task-1"]
    support_before = {
        anchor_id: frozenset(posteriors)
        for anchor_id, posteriors in state.posteriors.items()
    }
    request = coordinator.build_round_requests()[0]
    selected_action = request.selected_canonical_action
    other_action = next(action for action in ("open fridge", "go to table") if action != selected_action)
    coordinator.update_from_branch(
        request,
        success=True,
        suffix_pairs=[(request.expected_anchor_key, other_action)] * 2,
    )

    assert support_before == {
        anchor_id: frozenset(posteriors)
        for anchor_id, posteriors in state.posteriors.items()
    }
    branch_evidence = {
        action: posterior.branch_successes + posterior.branch_failures
        for posteriors in state.posteriors.values()
        for action, posterior in posteriors.items()
    }
    assert branch_evidence[selected_action] == 1
    assert branch_evidence[other_action] == 1
    assert len(coordinator.build_round_requests()) == 1
