import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_invalid_action_penalty,
    compute_advantage,
)


def _batch():
    return DataProto.from_single_dict(
        data={
            "token_level_rewards": torch.tensor([[0.0], [-0.1], [1.0], [0.9]]),
            "step_rewards": torch.tensor([0.0, -0.1, 1.0, 0.9]),
            "response_mask": torch.ones((4, 1)),
            "anchor_obs": np.array(["a", "b", "a", "b"], dtype=object),
            "uid": np.array(["g"] * 4, dtype=object),
            "traj_uid": np.array(["t0", "t0", "t1", "t1"], dtype=object),
            "action_identity": np.array(["x", "y", "x", "y"], dtype=object),
            "projected_action": np.array(["x", "y", "x", "y"], dtype=object),
        }
    )


def test_trainer_bace_branch_matches_gigpo_and_records_components():
    kwargs = {
        "step_advantage_w": 1.0,
        "gigpo_mode": "mean_std_norm",
        "gigpo_enable_similarity": False,
        "gigpo_compute_mean_std_cross_steps": False,
    }
    gigpo_batch = compute_advantage(
        _batch(), adv_estimator=AdvantageEstimator.GiGPO, **kwargs
    )
    bace_batch = compute_advantage(
        _batch(), adv_estimator=AdvantageEstimator.BACE_GiGPO, **kwargs
    )

    assert torch.allclose(bace_batch.batch["advantages"], gigpo_batch.batch["advantages"])
    assert torch.allclose(bace_batch.batch["returns"], gigpo_batch.batch["returns"])
    assert "bace_macro_advantage" in bace_batch.non_tensor_batch
    assert "bace_local_advantage" in bace_batch.non_tensor_batch
    assert "bace_occurrence_advantage" in bace_batch.non_tensor_batch


def test_pure_gigpo_action_mean_reuses_bace_math_without_enabling_bace():
    kwargs = {
        "step_advantage_w": 1.0,
        "gigpo_mode": "mean_std_norm",
        "gigpo_enable_similarity": False,
        "gigpo_compute_mean_std_cross_steps": True,
    }
    occurrence = compute_advantage(
        _batch(),
        adv_estimator=AdvantageEstimator.GiGPO,
        gigpo_local_credit_mode="occurrence",
        **kwargs,
    )
    action_mean = compute_advantage(
        _batch(),
        adv_estimator=AdvantageEstimator.GiGPO,
        gigpo_local_credit_mode="action_mean",
        **kwargs,
    )

    assert np.allclose(
        occurrence.non_tensor_batch["gigpo_macro_advantage"],
        action_mean.non_tensor_batch["gigpo_macro_advantage"],
    )
    assert np.isclose(action_mean.non_tensor_batch["gigpo_local_advantage"][0],
                      action_mean.non_tensor_batch["gigpo_local_advantage"][2])
    assert "credit_diagnostics" in action_mean.meta_info
    assert action_mean.meta_info["credit_diagnostics"][
        "selected_local_within_action_variance_mean"
    ] == 0.0


def test_pure_gigpo_step_weight_zero_is_identical_between_credit_modes():
    kwargs = {
        "step_advantage_w": 0.0,
        "gigpo_mode": "mean_std_norm",
        "gigpo_enable_similarity": False,
        "gigpo_compute_mean_std_cross_steps": True,
    }
    occurrence = compute_advantage(
        _batch(), adv_estimator=AdvantageEstimator.GiGPO,
        gigpo_local_credit_mode="occurrence", **kwargs
    )
    action_mean = compute_advantage(
        _batch(), adv_estimator=AdvantageEstimator.GiGPO,
        gigpo_local_credit_mode="action_mean", **kwargs
    )
    assert torch.allclose(occurrence.batch["advantages"], action_mean.batch["advantages"])


def test_action_mean_uses_projected_fallback_for_format_invalid_rows():
    batch = _batch()
    batch.non_tensor_batch["action_identity"] = np.array(
        [None, "valid::y", None, "valid::y"], dtype=object
    )
    result = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GiGPO,
        step_advantage_w=1.0,
        gigpo_mode="mean_std_norm",
        gigpo_local_credit_mode="action_mean",
    )
    assert "gigpo_local_advantage" in result.non_tensor_batch


def test_trainer_bace_macro_reads_invalid_penalty_from_token_rewards():
    batch = DataProto.from_single_dict(
        data={
            "prompts": torch.ones((2, 1), dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "token_level_scores": torch.zeros((2, 1)),
            "step_rewards": torch.zeros(2),
            "response_mask": torch.ones((2, 1)),
            "anchor_obs": np.array(["a", "b"], dtype=object),
            "uid": np.array(["g", "g"], dtype=object),
            "traj_uid": np.array(["t0", "t1"], dtype=object),
            "is_action_valid": np.array([True, False]),
            "projected_action": np.array(["look", "bad"], dtype=object),
        }
    )
    batch, _ = apply_invalid_action_penalty(batch, invalid_action_penalty_coef=0.1)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.BACE_GiGPO,
        step_advantage_w=0.0,
        gigpo_mode="mean_std_norm",
    )

    expected = np.array([0.7071068, -0.7071068])
    assert np.allclose(batch.non_tensor_batch["bace_macro_advantage"], expected, atol=1e-5)
