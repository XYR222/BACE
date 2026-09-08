from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gigpo import core_gigpo
from recipe.bace_gigpo.advantage import compute_credit_diagnostics
from recipe.bace_gigpo.flat_leaf import expand_tree_to_full_leaf_trajectories
from verl import DataProto


def _tree_proto():
    # root r: e0 -> e1 -> e2, branch b starts at e1 then generates bs.
    values = {
        "source_type": np.array(
            ["root", "root", "root", "branch_origin", "branch_suffix"],
            dtype=object,
        ),
        "occurrence_id": np.array(["e0", "e1", "e2", "bo", "bs"], dtype=object),
        "traj_uid": np.array(["r", "r", "r", "b", "b"], dtype=object),
        "leaf_id": np.array(["lr", "lr", "lr", "lb", "lb"], dtype=object),
        "step_index": np.array([0, 1, 2, 1, 2], dtype=np.int32),
        "tree_origin_occurrence_id": np.array(["e0", "e1", "e2", "e1", "e1"], dtype=object),
        "tree_parent_root_id": np.array(["r", "r", "r", "r", "r"], dtype=object),
        "episode_rewards": np.array([1, 1, 1, 4, 4], dtype=np.float32),
        "episode_lengths": np.array([3, 3, 3, 2, 2], dtype=np.float32),
        "success_rate": np.ones(5, dtype=np.float32),
        "rewards": np.array([1, 2, 3, 2, 7], dtype=np.float32),
        "active_masks": np.ones(5, dtype=bool),
        "anchor_obs": np.array(["z0", "z1", "z2", "z1", "zb"], dtype=object),
        "uid": np.array(["task"] * 5, dtype=object),
    }
    batch = TensorDict(
        {
            "input_ids": torch.arange(5).reshape(5, 1),
            "rollout_log_probs": torch.arange(5, dtype=torch.float32).reshape(5, 1),
        },
        batch_size=(5,),
    )
    return DataProto(batch=batch, non_tensor_batch=values)


def test_c7_expands_one_root_and_one_branch_to_full_trajectories():
    flat = expand_tree_to_full_leaf_trajectories(_tree_proto())
    assert len(np.unique(flat.non_tensor_batch["traj_uid"])) == 2
    assert len(flat) == 6  # root length 3 + branch (e0 copy, origin, suffix)
    by_uid = {
        uid: np.where(flat.non_tensor_batch["traj_uid"] == uid)[0]
        for uid in np.unique(flat.non_tensor_batch["traj_uid"])
    }
    root_rows = by_uid["flat-root:lr"]
    branch_rows = by_uid["flat-branch:lb"]
    assert len(root_rows) == 3
    assert len(branch_rows) == 3
    assert flat.non_tensor_batch["bace_flat_source_occurrence_id"][branch_rows].tolist() == [
        "e0", "bo", "bs"
    ]
    assert flat.non_tensor_batch["bace_flat_is_prefix_copy"][branch_rows].tolist() == [
        True, True, False
    ]
    np.testing.assert_array_equal(
        flat.non_tensor_batch["episode_lengths"][branch_rows], [3, 3, 3]
    )
    # Copied token/logprob are exactly those of the source natural prefix row.
    assert flat.batch["input_ids"][branch_rows[0]].item() == 0
    assert flat.batch["rollout_log_probs"][branch_rows[0]].item() == pytest.approx(0.0)


def test_c7_recomputes_branch_prefix_return_from_branch_leaf():
    flat = expand_tree_to_full_leaf_trajectories(_tree_proto())
    returns = core_gigpo.compute_step_discounted_returns(flat, gamma=0.5)
    branch_rows = np.where(
        flat.non_tensor_batch["traj_uid"] == "flat-branch:lb"
    )[0]
    # Branch rewards are copied e0=1, origin=2, suffix=7, hence [3.75, 5.5, 7].
    np.testing.assert_allclose(
        returns[branch_rows].cpu().numpy(), [3.75, 5.5, 7.0]
    )
    # The root prefix return is different, proving C7 did not freeze C0 G.
    assert returns[branch_rows[0]].item() != pytest.approx(returns[0].item())


def test_c7_credit_diagnostics_count_flat_branch_evidence():
    flat = expand_tree_to_full_leaf_trajectories(_tree_proto())
    rows = len(flat)
    mask = torch.ones((rows, 1), dtype=torch.float32)
    metrics = compute_credit_diagnostics(
        step_rewards=torch.arange(rows, dtype=torch.float32),
        response_mask=mask,
        anchor_obs=flat.non_tensor_batch["anchor_obs"],
        task_ids=flat.non_tensor_batch["uid"],
        action_ids=np.asarray([f"a{index}" for index in range(rows)], dtype=object),
        macro_scores=torch.zeros(rows),
        selected_local_scores=torch.zeros(rows),
        step_advantage_w=1.0,
        mode="mean_norm",
        enable_similarity=False,
        similarity_thresh=0.95,
        source_types=flat.non_tensor_batch["source_type"],
    )
    assert metrics["branch_created_evidence_share"] == pytest.approx(0.5)
    assert metrics["branch_origin_evidence_share"] == pytest.approx(1.0 / 6.0)
    assert metrics["branch_suffix_evidence_share"] == pytest.approx(1.0 / 6.0)
