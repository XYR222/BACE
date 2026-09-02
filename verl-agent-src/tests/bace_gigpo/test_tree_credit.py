from __future__ import annotations

import numpy as np
import pytest
import torch

from recipe.bace_gigpo.advantage import (
    build_tree_credit_index,
    compute_bace_gigpo_advantage,
    compute_bace_tree_credit_advantage,
)


def _tree_batch():
    # Four natural edges, B1 branches directly at e2 and continues for one
    # fresh edge, while B2 branches directly at e3 and terminates at origin.
    source = np.array(
        ["root", "root", "root", "root", "branch_origin", "branch_suffix", "branch_origin"],
        dtype=object,
    )
    occurrence = np.array(["e0", "e1", "e2", "e3", "b1:o", "b1:s", "b2:o"], dtype=object)
    traj = np.array(["r0", "r0", "r0", "r0", "b1", "b1", "b2"], dtype=object)
    leaf = np.array(["lr", "lr", "lr", "lr", "lb1", "lb1", "lb2"], dtype=object)
    step = np.array([0, 1, 2, 3, 2, 3, 3], dtype=np.int32)
    tree_origin = np.array(["e0", "e1", "e2", "e3", "e2", "e2", "e3"], dtype=object)
    tree_parent = np.array(["r0"] * len(source), dtype=object)
    mask = torch.ones((len(source), 1), dtype=torch.float32)
    # Three distinct leaf outcomes make the frozen stable macro nontrivial.
    token_rewards = torch.tensor([[0], [0], [0], [1], [0], [3], [-1]], dtype=torch.float32)
    # Root returns are coherent with raw_rewards when gamma=0.5:
    # [3.25, 4.5, 5, 4]. Branch outcomes deliberately differ.
    step_rewards = torch.tensor([3.25, 4.5, 5, 4, 100, 50, 200], dtype=torch.float32)
    raw_rewards = np.array([1, 2, 3, 4, 0, 5, -1], dtype=np.float32)
    episode_rewards = np.array([1, 1, 1, 1, 3, 3, -1], dtype=np.float32)
    return {
        "token_level_rewards": token_rewards,
        "step_rewards": step_rewards,
        "response_mask": mask,
        "anchor_obs": np.array(["z", "z", "loop", "loop", "loop", "fresh", "loop"], dtype=object),
        "task_ids": np.array(["task"] * len(source), dtype=object),
        "traj_ids": traj,
        "occurrence_ids": occurrence,
        "source_types": source,
        "leaf_ids": leaf,
        "step_indices": step,
        "raw_rewards": raw_rewards,
        "episode_rewards": episode_rewards,
        "tree_origin_occurrence_ids": tree_origin,
        "tree_parent_root_ids": tree_parent,
    }


def _compute(mode: str, gamma: float = 1.0):
    return compute_bace_tree_credit_advantage(
        **_tree_batch(),
        tree_credit_mode=mode,
        macro_normalization_mode="stable_occurrence",
        gamma=gamma,
        mode="mean_norm",
    )


def test_direct_and_descendant_sets_preserve_concrete_occurrences():
    values = _tree_batch()
    tree = build_tree_credit_index(
        source_types=values["source_types"],
        occurrence_ids=values["occurrence_ids"],
        traj_ids=values["traj_ids"],
        leaf_ids=values["leaf_ids"],
        step_indices=values["step_indices"],
        tree_origin_occurrence_ids=values["tree_origin_occurrence_ids"],
        tree_parent_root_ids=values["tree_parent_root_ids"],
    )
    assert tree.direct_branch_ids_by_edge["e2"] == ("b1",)
    assert tree.descendant_branch_ids_by_edge["e2"] == ("b1", "b2")
    assert tree.direct_branch_ids_by_edge["e3"] == ("b2",)
    assert tree.descendant_branch_ids_by_edge["e3"] == ("b2",)
    # e2/e3 deliberately share the same anchor/action-style identity but stay distinct.
    assert "e2" in tree.root_row_by_occurrence and "e3" in tree.root_row_by_occurrence


def test_c1_direct_backup_and_copied_origin_removal():
    _, _, components = _compute("o1_local")
    metadata = components["metadata"]
    assert components["keep_indices"].tolist() == [0, 1, 2, 3, 5]
    position = list(metadata["bace_edge_id"]).index("e2")
    assert metadata["bace_g_direct_mean"][position] == pytest.approx(52.5)
    assert metadata["bace_direct_leaf_ids"][position] == ("lr", "lb1")
    assert metadata["bace_descendant_leaf_ids"][position] == ("lr", "lb1", "lb2")
    # The origin-direct terminal B2 remains evidence despite having no suffix edge.
    e3 = list(metadata["bace_edge_id"]).index("e3")
    assert metadata["bace_g_direct_mean"][e3] == pytest.approx(102.0)


def test_two_branches_from_one_origin_still_train_one_edge():
    values = _tree_batch()
    # Move B2 to the same concrete origin as B1; this is multiplicity evidence,
    # not a second policy edge.
    values["tree_origin_occurrence_ids"] = values["tree_origin_occurrence_ids"].copy()
    values["step_indices"] = values["step_indices"].copy()
    values["tree_origin_occurrence_ids"][6] = "e2"
    values["step_indices"][6] = 2
    _, _, components = compute_bace_tree_credit_advantage(
        **values,
        tree_credit_mode="o1_local",
        macro_normalization_mode="stable_occurrence",
    )
    metadata = components["metadata"]
    assert list(metadata["bace_edge_id"]).count("e2") == 1
    e2 = list(metadata["bace_edge_id"]).index("e2")
    assert metadata["bace_direct_leaf_ids"][e2] == ("lr", "lb1", "lb2")
    assert metadata["bace_g_direct_mean"][e2] == pytest.approx((5 + 100 + 200) / 3)


def test_c3_discounted_descendant_return_reconstruction():
    _, _, components = _compute("o1_full_tree", gamma=0.5)
    metadata = components["metadata"]
    e2 = list(metadata["bace_edge_id"]).index("e2")
    # Direct B1: 100. Later B2: G(e2) + gamma * (G(B2)-G(root e3))
    # = 5 + .5 * (200 - 4) = 103, equal to r(e2) + gamma * G(B2).
    assert metadata["bace_g_descendant_mean"][e2] == pytest.approx((5 + 100 + 103) / 3)
    # Fresh suffix edges are independent leaves and keep their original G.
    suffix = list(metadata["bace_edge_id"]).index("b1:s")
    assert metadata["bace_g_descendant_mean"][suffix] == pytest.approx(50.0)


def test_tree_backup_keeps_invalid_penalty_local_to_the_target_edge():
    """A later branch-origin penalty must not be shaped into an ancestor.

    `step_rewards` and frozen macro are already post-GiGPO-reward values.  A
    branch origin is a deterministic replay of its natural origin, so their
    common local invalid-action penalty cancels in `branch - natural_origin`.
    The target's own value is then added once.  This test uses the e2 ancestor
    and b2 branch that starts later at e3; a raw branch-return backup would
    incorrectly carry e3's local term into e2.
    """
    _, _, components = _compute("o1_full_tree", gamma=0.5)
    metadata = components["metadata"]
    edge_ids = metadata["bace_edge_id"].tolist()
    e2 = edge_ids.index("e2")
    e3 = edge_ids.index("e3")

    # Local G backup: b1 is direct at e2, while b2 starts at e3.
    # Its e3-local component is removed by 200 - 4 before the delta is
    # discounted back to e2: mean(5, 100, 5 + .5 * (200 - 4)).
    assert metadata["bace_g_descendant_mean"][e2] == pytest.approx(
        (5 + 100 + (5 + 0.5 * (200 - 4))) / 3
    )

    # Macro C2/C3 applies the same translation in frozen-C0 macro space.
    # This is intentionally not mean(M_e2, M_b1, M_b2): the latter would
    # import the later e3 occurrence's local penalty into e2.
    physical_macro = components["base_macro_physical"].detach().cpu().numpy()
    expected_macro = np.mean([
        physical_macro[2],
        physical_macro[4],
        physical_macro[2] + physical_macro[6] - physical_macro[3],
    ])
    assert metadata["bace_macro_descendant_mean"][e2] == pytest.approx(expected_macro)
    assert metadata["bace_macro_descendant_mean"][e2] != pytest.approx(
        np.mean([physical_macro[2], physical_macro[4], physical_macro[6]])
    )
    assert components["tree_diagnostics"]["tree_credit_invalid_penalty_is_edge_local"] == 1.0
    assert metadata["bace_g_original"][e3] == pytest.approx(4.0)


def test_c1_c2_local_and_c2_c3_macro_are_exactly_shared():
    _, _, c1 = _compute("o1_local")
    _, _, c2 = _compute("o1_tree_macro")
    _, _, c3 = _compute("o1_full_tree")
    torch.testing.assert_close(c1["local"], c2["local"], rtol=0, atol=0)
    torch.testing.assert_close(c2["macro"], c3["macro"], rtol=0, atol=0)


def test_frozen_macro_snapshot_matches_production_c0_before_filtering():
    values = _tree_batch()
    action_ids = np.array(["a"] * len(values["source_types"]), dtype=object)
    _, _, c0 = compute_bace_gigpo_advantage(
        token_level_rewards=values["token_level_rewards"],
        step_rewards=values["step_rewards"],
        response_mask=values["response_mask"],
        anchor_obs=values["anchor_obs"],
        task_ids=values["task_ids"],
        traj_ids=values["traj_ids"],
        action_ids=action_ids,
        mode="mean_norm",
    )
    for mode in ("o1_local", "o1_tree_macro", "o1_full_tree"):
        _, _, tree = _compute(mode)
        torch.testing.assert_close(tree["base_macro_physical"], c0["macro"], rtol=0, atol=0)
        keep = tree["keep_indices"]
        np.testing.assert_array_equal(
            tree["metadata"]["bace_macro_base_stable"],
            c0["macro"][keep].detach().cpu().numpy(),
        )


def test_zero_variance_and_masks_remain_finite():
    values = _tree_batch()
    values["step_rewards"] = torch.ones_like(values["step_rewards"])
    advantage, returns, components = compute_bace_tree_credit_advantage(
        **values,
        tree_credit_mode="o1_local",
        macro_normalization_mode="stable_occurrence",
        mode="mean_norm",
    )
    assert torch.isfinite(advantage).all()
    torch.testing.assert_close(advantage, returns)
    assert advantage.shape[0] == 5
    assert components["metadata"]["bace_macro_normalization_mode"].tolist() == [
        "stable_occurrence"
    ] * 5


def test_fresh_suffix_anchor_collision_does_not_merge_edge_identity():
    values = _tree_batch()
    values["anchor_obs"] = values["anchor_obs"].copy()
    values["anchor_obs"][5] = "loop"
    _, _, components = compute_bace_tree_credit_advantage(
        **values,
        tree_credit_mode="o1_full_tree",
        macro_normalization_mode="stable_occurrence",
    )
    edge_ids = components["metadata"]["bace_edge_id"].tolist()
    assert edge_ids.count("e2") == 1
    assert edge_ids.count("b1:s") == 1


def test_another_root_same_anchor_is_not_a_descendant():
    values = _tree_batch()
    # Re-parent e0/e1 as a second natural root while keeping identical anchors.
    values["traj_ids"] = values["traj_ids"].copy()
    values["tree_parent_root_ids"] = values["tree_parent_root_ids"].copy()
    values["traj_ids"][:2] = "r_other"
    values["tree_parent_root_ids"][:2] = "r_other"
    values["leaf_ids"] = values["leaf_ids"].copy()
    values["leaf_ids"][:2] = "l_other"
    tree = build_tree_credit_index(
        source_types=values["source_types"], occurrence_ids=values["occurrence_ids"],
        traj_ids=values["traj_ids"], leaf_ids=values["leaf_ids"],
        step_indices=values["step_indices"],
        tree_origin_occurrence_ids=values["tree_origin_occurrence_ids"],
        tree_parent_root_ids=values["tree_parent_root_ids"],
    )
    assert "e0" not in tree.descendant_branch_ids_by_edge
    assert "e1" not in tree.descendant_branch_ids_by_edge


def test_invalid_lineage_and_strict_leaf_mode_fail_closed():
    values = _tree_batch()
    values["tree_origin_occurrence_ids"] = values["tree_origin_occurrence_ids"].copy()
    values["tree_origin_occurrence_ids"][4] = "missing"
    with pytest.raises(ValueError, match="missing natural origin"):
        compute_bace_tree_credit_advantage(
            **values,
            tree_credit_mode="o1_local",
            macro_normalization_mode="stable_occurrence",
        )
    with pytest.raises(ValueError, match="stable_occurrence"):
        compute_bace_tree_credit_advantage(
            **_tree_batch(),
            tree_credit_mode="o1_local",
            macro_normalization_mode="strict_leaf_uniform",
        )


def test_adjust_batch_copy_is_macro_evidence_but_not_a_duplicate_edge():
    values = _tree_batch()
    original_size = len(values["source_types"])
    duplicate = 2
    tensor_keys = {"token_level_rewards", "step_rewards", "response_mask"}
    for key, value in list(values.items()):
        if key in tensor_keys:
            values[key] = torch.cat([value, value[duplicate : duplicate + 1]], dim=0)
        elif isinstance(value, np.ndarray):
            values[key] = np.concatenate([value, value[duplicate : duplicate + 1]])
    values["adjustment_padding_mask"] = np.array(
        [False] * original_size + [True], dtype=bool
    )
    _, _, components = compute_bace_tree_credit_advantage(
        **values,
        tree_credit_mode="o1_local",
        macro_normalization_mode="stable_occurrence",
    )
    assert len(components["base_macro_physical"]) == original_size + 1
    assert components["keep_indices"].tolist() == [0, 1, 2, 3, 5]
    assert components["tree_diagnostics"]["tree_credit_copied_origins_removed"] == 2.0
    assert components["tree_diagnostics"]["tree_credit_adjustment_padding_removed"] == 1.0
