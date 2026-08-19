import numpy as np
import pytest
import torch

from gigpo import core_gigpo
from recipe.bace_gigpo.advantage import compute_bace_gigpo_advantage


def _inputs():
    return {
        "token_level_rewards": torch.tensor(
            [
                [0.0, 0.0],
                [-0.1, 0.0],
                [1.0, 0.0],
                [0.9, 0.0],
                [0.5, 0.0],
                [-0.2, 0.0],
            ]
        ),
        "step_rewards": torch.tensor([0.0, -0.1, 1.0, 0.9, 0.5, -0.2]),
        "response_mask": torch.ones((6, 2)),
        "anchor_obs": np.array(["a", "b", "a", "b", "a", "c"], dtype=object),
        "task_ids": np.array(["g"] * 6, dtype=object),
        "traj_ids": np.array(["t0", "t0", "t1", "t2", "t2", "t2"], dtype=object),
        "action_ids": np.array(["x", "y", "x", "y", "z", "z"], dtype=object),
    }


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
@pytest.mark.parametrize("compute_cross_steps", [True, False])
def test_occurrence_credit_matches_gigpo(mode, compute_cross_steps):
    inputs = _inputs()
    advantages, returns, components = compute_bace_gigpo_advantage(
        **inputs,
        mode=mode,
        compute_mean_std_cross_steps=compute_cross_steps,
    )
    expected, expected_returns = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=inputs["token_level_rewards"],
        step_rewards=inputs["step_rewards"],
        response_mask=inputs["response_mask"],
        anchor_obs=inputs["anchor_obs"],
        index=inputs["task_ids"],
        traj_index=inputs["traj_ids"],
        mode=mode,
        compute_mean_std_cross_steps=compute_cross_steps,
    )

    assert torch.allclose(advantages, expected, atol=1e-7)
    assert torch.allclose(returns, expected_returns, atol=1e-7)
    assert torch.allclose(components["occurrence"], advantages[:, 0], atol=1e-7)


def test_invalid_action_penalty_in_token_rewards_reaches_macro_advantage():
    advantages, _, components = compute_bace_gigpo_advantage(
        token_level_rewards=torch.tensor([[0.0], [-0.1]]),
        step_rewards=torch.zeros(2),
        response_mask=torch.ones((2, 1)),
        anchor_obs=np.array(["a", "b"], dtype=object),
        task_ids=np.array(["g", "g"], dtype=object),
        traj_ids=np.array(["t0", "t1"], dtype=object),
        step_advantage_w=0.0,
        mode="mean_std_norm",
    )

    expected = torch.tensor([0.7071068, -0.7071068])
    assert torch.allclose(components["macro"], expected, atol=1e-5)
    assert torch.allclose(advantages[:, 0], expected, atol=1e-5)


def test_mean_std_norm_uses_gigpo_sample_standard_deviation():
    _, _, components = compute_bace_gigpo_advantage(
        token_level_rewards=torch.tensor([[0.0], [1.0], [2.0]]),
        step_rewards=torch.zeros(3),
        response_mask=torch.ones((3, 1)),
        anchor_obs=np.array(["a", "b", "c"], dtype=object),
        task_ids=np.array(["g", "g", "g"], dtype=object),
        traj_ids=np.array(["t0", "t1", "t2"], dtype=object),
        step_advantage_w=0.0,
        mode="mean_std_norm",
    )

    assert torch.allclose(components["macro"], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5)


def test_cross_step_option_controls_trajectory_deduplication():
    kwargs = {
        "token_level_rewards": torch.tensor([[0.0], [0.0], [2.0]]),
        "step_rewards": torch.zeros(3),
        "response_mask": torch.ones((3, 1)),
        "anchor_obs": np.array(["a", "b", "c"], dtype=object),
        "task_ids": np.array(["g", "g", "g"], dtype=object),
        "traj_ids": np.array(["t0", "t0", "t1"], dtype=object),
        "step_advantage_w": 0.0,
        "mode": "mean_norm",
    }
    _, _, occurrence_weighted = compute_bace_gigpo_advantage(
        **kwargs, compute_mean_std_cross_steps=True
    )
    _, _, trajectory_weighted = compute_bace_gigpo_advantage(
        **kwargs, compute_mean_std_cross_steps=False
    )

    assert torch.allclose(
        occurrence_weighted["macro"], torch.tensor([-2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0])
    )
    assert torch.allclose(trajectory_weighted["macro"], torch.tensor([-1.0, -1.0, 1.0]))


def test_action_mean_local_credit_remains_an_explicit_bace_extension():
    kwargs = {
        "token_level_rewards": torch.tensor([[0.0], [1.0], [0.0]]),
        "step_rewards": torch.tensor([0.0, 4.0, 1.0]),
        "response_mask": torch.ones((3, 1)),
        "anchor_obs": np.array(["z", "z", "z"], dtype=object),
        "task_ids": np.array(["g", "g", "g"], dtype=object),
        "traj_ids": np.array(["t0", "t1", "t2"], dtype=object),
        "action_ids": np.array(["a", "a", "b"], dtype=object),
        "mode": "mean_std_norm",
    }
    _, _, occurrence = compute_bace_gigpo_advantage(**kwargs)
    _, _, action_mean = compute_bace_gigpo_advantage(
        **kwargs, local_credit_mode="action_mean"
    )

    assert torch.allclose(occurrence["macro"], action_mean["macro"])
    assert torch.isclose(action_mean["local"][0], action_mean["local"][1])
    assert not torch.isclose(occurrence["local"][0], occurrence["local"][1])


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        compute_bace_gigpo_advantage(**_inputs(), mode="invalid")
