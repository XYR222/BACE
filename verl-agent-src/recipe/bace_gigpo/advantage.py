from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from gigpo import core_gigpo


class TreeCreditMode(str, Enum):
    CURRENT = "current"
    O1_LOCAL = "o1_local"
    O1_TREE_MACRO = "o1_tree_macro"
    O1_FULL_TREE = "o1_full_tree"


class MacroNormalizationMode(str, Enum):
    STABLE_OCCURRENCE = "stable_occurrence"
    STRICT_LEAF_UNIFORM = "strict_leaf_uniform"


@dataclass(frozen=True)
class TreeCreditIndex:
    """Exact rollout-tree ancestry keyed by concrete occurrence identity."""

    root_row_by_occurrence: dict[str, int]
    branch_origin_row_by_id: dict[str, int]
    branch_origin_occurrence_by_id: dict[str, str]
    branch_parent_root_by_id: dict[str, str]
    branch_origin_step_by_id: dict[str, int]
    root_leaf_by_root_id: dict[str, str]
    branch_leaf_by_id: dict[str, str]
    direct_branch_ids_by_edge: dict[str, tuple[str, ...]]
    descendant_branch_ids_by_edge: dict[str, tuple[str, ...]]


def _as_object_array(values, name: str, batch_size: int) -> np.ndarray:
    if values is None or len(values) != batch_size:
        raise ValueError(f"Tree credit requires one {name} value per physical occurrence")
    return np.asarray(values, dtype=object)


def _one_dimensional_object_array(values) -> np.ndarray:
    """Keep tuple-valued trace fields as one object per occurrence."""
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def build_tree_credit_index(
    *,
    source_types: np.ndarray,
    occurrence_ids: np.ndarray,
    traj_ids: np.ndarray,
    leaf_ids: np.ndarray,
    step_indices: np.ndarray,
    tree_origin_occurrence_ids: np.ndarray,
    tree_parent_root_ids: np.ndarray,
    adjustment_padding_mask: np.ndarray | None = None,
) -> TreeCreditIndex:
    """Build Direct/Desc relationships without collapsing equal anchors/actions."""
    batch_size = len(source_types)
    arrays = (occurrence_ids, traj_ids, leaf_ids, step_indices,
              tree_origin_occurrence_ids, tree_parent_root_ids)
    if any(len(values) != batch_size for values in arrays):
        raise ValueError("Tree-credit lineage arrays must match the physical batch")

    root_row_by_occurrence: dict[str, int] = {}
    branch_origin_row_by_id: dict[str, int] = {}
    branch_origin_occurrence_by_id: dict[str, str] = {}
    branch_parent_root_by_id: dict[str, str] = {}
    branch_origin_step_by_id: dict[str, int] = {}
    root_leaf_by_root_id: dict[str, str] = {}
    branch_leaf_by_id: dict[str, str] = {}

    if adjustment_padding_mask is None:
        adjustment_padding_mask = np.zeros(batch_size, dtype=bool)
    adjustment_padding_mask = np.asarray(adjustment_padding_mask, dtype=bool)
    if len(adjustment_padding_mask) != batch_size:
        raise ValueError("adjustment padding mask must match the physical batch")

    for row, source in enumerate(source_types):
        if adjustment_padding_mask[row]:
            continue
        source = str(source)
        occurrence_id = str(occurrence_ids[row])
        trajectory_id = str(traj_ids[row])
        leaf_id = str(leaf_ids[row])
        if source == "root":
            if occurrence_id in root_row_by_occurrence:
                raise ValueError(f"Duplicate natural occurrence identity: {occurrence_id}")
            root_row_by_occurrence[occurrence_id] = row
            parent_root_id = str(tree_parent_root_ids[row])
            if parent_root_id != trajectory_id:
                raise ValueError("Natural edge parent_root_id must equal its trajectory ID")
            previous = root_leaf_by_root_id.setdefault(parent_root_id, leaf_id)
            if previous != leaf_id:
                raise ValueError(f"Natural root {parent_root_id} has inconsistent leaf IDs")
        elif source == "branch_origin":
            branch_id = trajectory_id
            if branch_id in branch_origin_row_by_id:
                raise ValueError(f"Branch {branch_id} has multiple copied-origin rows")
            natural_origin = str(tree_origin_occurrence_ids[row])
            parent_root = str(tree_parent_root_ids[row])
            branch_origin_row_by_id[branch_id] = row
            branch_origin_occurrence_by_id[branch_id] = natural_origin
            branch_parent_root_by_id[branch_id] = parent_root
            branch_origin_step_by_id[branch_id] = int(step_indices[row])
            branch_leaf_by_id[branch_id] = leaf_id
        elif source == "branch_suffix":
            previous = branch_leaf_by_id.setdefault(trajectory_id, leaf_id)
            if previous != leaf_id:
                raise ValueError(f"Branch {trajectory_id} has inconsistent leaf IDs")
        else:
            raise ValueError(f"Unknown BACE physical occurrence source: {source}")

    for branch_id, origin_id in branch_origin_occurrence_by_id.items():
        if origin_id not in root_row_by_occurrence:
            raise ValueError(f"Branch {branch_id} references missing natural origin {origin_id}")
        root_row = root_row_by_occurrence[origin_id]
        if str(tree_parent_root_ids[root_row]) != branch_parent_root_by_id[branch_id]:
            raise ValueError(f"Branch {branch_id} parent root disagrees with its natural origin")
        if int(step_indices[root_row]) != branch_origin_step_by_id[branch_id]:
            raise ValueError(f"Branch {branch_id} origin step disagrees with its natural origin")

    direct: dict[str, list[str]] = defaultdict(list)
    descendants: dict[str, list[str]] = defaultdict(list)
    root_rows_by_parent: dict[str, list[int]] = defaultdict(list)
    for occurrence_id, row in root_row_by_occurrence.items():
        root_rows_by_parent[str(tree_parent_root_ids[row])].append(row)
    for rows in root_rows_by_parent.values():
        rows.sort(key=lambda row: int(step_indices[row]))

    for branch_id, origin_id in branch_origin_occurrence_by_id.items():
        direct[origin_id].append(branch_id)
        parent_root = branch_parent_root_by_id[branch_id]
        origin_step = branch_origin_step_by_id[branch_id]
        for row in root_rows_by_parent[parent_root]:
            if int(step_indices[row]) <= origin_step:
                descendants[str(occurrence_ids[row])].append(branch_id)

    order_key = lambda branch_id: (
        branch_origin_step_by_id[branch_id], branch_id
    )
    return TreeCreditIndex(
        root_row_by_occurrence=root_row_by_occurrence,
        branch_origin_row_by_id=branch_origin_row_by_id,
        branch_origin_occurrence_by_id=branch_origin_occurrence_by_id,
        branch_parent_root_by_id=branch_parent_root_by_id,
        branch_origin_step_by_id=branch_origin_step_by_id,
        root_leaf_by_root_id=root_leaf_by_root_id,
        branch_leaf_by_id=branch_leaf_by_id,
        direct_branch_ids_by_edge={
            edge: tuple(sorted(branches, key=order_key)) for edge, branches in direct.items()
        },
        descendant_branch_ids_by_edge={
            edge: tuple(sorted(branches, key=order_key))
            for edge, branches in descendants.items()
        },
    )


def _occurrence_values(token_values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Collapse a response-constant token tensor back to one value per occurrence."""
    token_counts = response_mask.sum(dim=-1)
    if torch.any(token_counts <= 0):
        raise ValueError("BACE advantage requires at least one response token per occurrence")
    return (token_values * response_mask).sum(dim=-1) / token_counts


def _normalization_mode(mode: str) -> bool:
    if mode == "mean_std_norm":
        return False
    if mode == "mean_norm":
        return True
    raise ValueError(f"Unknown mode: {mode}")


def _action_mean_local_advantage(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: np.ndarray,
    action_ids: np.ndarray,
    epsilon: float,
    remove_std: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BACE action-level credit using GiGPO's group and normalization conventions."""
    batch_size, response_length = response_mask.shape
    local_scores = torch.zeros(batch_size, device=step_rewards.device, dtype=step_rewards.dtype)
    rows_by_group: dict[object, list[int]] = defaultdict(list)
    for row_idx, group_id in enumerate(step_group_ids):
        rows_by_group[group_id].append(row_idx)

    for row_indices in rows_by_group.values():
        indices = torch.tensor(row_indices, device=step_rewards.device, dtype=torch.long)
        group_rewards = step_rewards[indices]
        group_mean = group_rewards.mean()
        if remove_std:
            denominator = None
        elif group_rewards.numel() < 2:
            denominator = None
        else:
            group_std = group_rewards.std(unbiased=True)
            valid_std = bool(torch.isfinite(group_std) and group_std > epsilon)
            denominator = group_std + epsilon if valid_std else None

        rows_by_action: dict[str, list[int]] = defaultdict(list)
        for row_idx in row_indices:
            rows_by_action[str(action_ids[row_idx])].append(row_idx)
        for action_rows in rows_by_action.values():
            action_indices = torch.tensor(action_rows, device=step_rewards.device, dtype=torch.long)
            centered = step_rewards[action_indices].mean() - group_mean
            local_scores[action_indices] = centered if remove_std else (
                centered / denominator if denominator is not None else 0.0
            )

    token_scores = local_scores.unsqueeze(-1).expand(batch_size, response_length) * response_mask
    return token_scores, local_scores


def resolve_action_ids(
    action_ids: np.ndarray | None,
    projected_actions: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    """Return one deterministic environment-level action identity per row.

    ALFWorld supplies ``action_identity`` for structurally valid responses.  A
    malformed response deliberately has no strict BACE identity, but the
    environment still receives a deterministic projected action.  Action-mean
    credit needs a total row mapping, so only those missing identities fall
    back to the projected action.  This does not make them branchable.
    """
    if action_ids is not None and len(action_ids) != batch_size:
        raise ValueError("action_ids must have one value per occurrence")
    if projected_actions is not None and len(projected_actions) != batch_size:
        raise ValueError("projected_actions must have one value per occurrence")
    resolved = []
    for row_idx in range(batch_size):
        identity = None if action_ids is None else action_ids[row_idx]
        if identity is not None and str(identity).strip():
            resolved.append(str(identity))
            continue
        projected = None if projected_actions is None else projected_actions[row_idx]
        if projected is None or not str(projected).strip():
            # Some legacy rollout paths omit the projected-action sidecar for
            # malformed rows.  Keep the row eligible for PPO while preventing
            # accidental cross-row action grouping; such rows remain
            # unbranchable because strict BACE identity is still absent.
            resolved.append(f"format_invalid::row::{row_idx}")
        else:
            resolved.append(f"format_invalid::{str(projected).strip()}")
    return np.asarray(resolved, dtype=object)


def compute_credit_diagnostics(
    *,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    task_ids: np.ndarray,
    action_ids: np.ndarray,
    macro_scores: torch.Tensor,
    selected_local_scores: torch.Tensor,
    step_advantage_w: float,
    mode: str,
    enable_similarity: bool,
    similarity_thresh: float,
    source_types: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Summarize the A1/A2 credit mechanism without changing optimization."""
    batch_size = int(step_rewards.shape[0])
    if len(action_ids) != batch_size:
        raise ValueError("credit diagnostics require one action ID per occurrence")
    remove_std = _normalization_mode(mode)
    step_group_ids = core_gigpo.build_step_group(
        anchor_obs,
        task_ids,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    occurrence_tokens = core_gigpo.step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=step_group_ids,
        epsilon=epsilon,
        remove_std=remove_std,
    )
    occurrence_local = _occurrence_values(occurrence_tokens, response_mask)
    sources = (
        np.asarray(source_types, dtype=object)
        if source_types is not None
        else np.asarray(["root"] * batch_size, dtype=object)
    )
    if len(sources) != batch_size:
        raise ValueError("source_types must have one value per occurrence")

    rows_by_action: dict[tuple[str, str], list[int]] = defaultdict(list)
    rows_by_state: dict[str, list[int]] = defaultdict(list)
    for row_idx, (group_id, action_id) in enumerate(zip(step_group_ids, action_ids)):
        group_key = str(group_id)
        rows_by_state[group_key].append(row_idx)
        rows_by_action[(group_key, str(action_id))].append(row_idx)

    group_sizes = []
    return_variances = []
    occurrence_variances = []
    selected_variances = []
    conflicts: dict[str, list[float]] = defaultdict(list)
    for rows in rows_by_action.values():
        indices = torch.tensor(rows, device=step_rewards.device, dtype=torch.long)
        group_sizes.append(float(len(rows)))
        return_variances.append(float(step_rewards[indices].var(unbiased=False).item()))
        occurrence_variances.append(float(occurrence_local[indices].var(unbiased=False).item()))
        selected_variances.append(float(selected_local_scores[indices].var(unbiased=False).item()))
        values = occurrence_local[indices]
        conflict = float(bool(torch.any(values > epsilon) and torch.any(values < -epsilon)))
        row_sources = {str(sources[row]) for row in rows}
        conflicts["all"].append(conflict)
        if row_sources == {"root"}:
            conflicts["natural_only"].append(conflict)
        if "branch_origin" in row_sources:
            conflicts["with_branch_origin"].append(conflict)
        if "branch_suffix" in row_sources:
            conflicts["with_branch_suffix"].append(conflict)

    imbalance_ratios = []
    for rows in rows_by_state.values():
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(action_ids[row])] += 1
        imbalance_ratios.append(float(max(counts.values()) / min(counts.values())))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    macro_abs = float(macro_scores.abs().mean().item())
    local_abs = float(selected_local_scores.abs().mean().item())
    weighted_local_abs = abs(float(step_advantage_w)) * local_abs
    denominator = macro_abs + weighted_local_abs
    result = {
        "macro_abs_mean": macro_abs,
        "macro_std": float(macro_scores.std(unbiased=False).item()),
        "local_abs_mean": local_abs,
        "local_std": float(selected_local_scores.std(unbiased=False).item()),
        "local_contribution_ratio": weighted_local_abs / denominator if denominator else 0.0,
        "local_positive_fraction": float((selected_local_scores > epsilon).float().mean().item()),
        "local_negative_fraction": float((selected_local_scores < -epsilon).float().mean().item()),
        "local_zero_fraction": float((selected_local_scores.abs() <= epsilon).float().mean().item()),
        "action_group_count": float(len(group_sizes)),
        "action_count_mean": mean(group_sizes),
        "action_count_max": max(group_sizes, default=0.0),
        "singleton_action_fraction": mean([float(size == 1.0) for size in group_sizes]),
        "action_imbalance_ratio_mean": mean(imbalance_ratios),
        "within_action_return_variance_mean": mean(return_variances),
        "occurrence_local_within_action_variance_mean": mean(occurrence_variances),
        "selected_local_within_action_variance_mean": mean(selected_variances),
        "branch_created_evidence_share": float(
            np.mean([str(source).startswith("branch_") for source in sources])
        ),
        "branch_origin_evidence_share": float(np.mean(sources == "branch_origin")),
        "branch_suffix_evidence_share": float(np.mean(sources == "branch_suffix")),
    }
    baseline_variance = result["occurrence_local_within_action_variance_mean"]
    result["within_action_variance_reduction_fraction"] = (
        1.0 - result["selected_local_within_action_variance_mean"] / baseline_variance
        if baseline_variance > epsilon
        else 0.0
    )
    for label in ("all", "natural_only", "with_branch_origin", "with_branch_suffix"):
        result[f"local_sign_conflict_rate/{label}"] = mean(conflicts[label])
        result[f"local_sign_conflict_groups/{label}"] = float(len(conflicts[label]))
    return result


def compute_bace_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    task_ids: np.ndarray,
    traj_ids: np.ndarray,
    action_ids: np.ndarray | None = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    compute_mean_std_cross_steps: bool = True,
    local_credit_mode: str = "occurrence",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute GiGPO macro/micro credit over the BACE training occurrences."""
    batch_size, response_length = response_mask.shape
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must have the same shape")
    if step_rewards.shape != (batch_size,):
        raise ValueError("step_rewards must have one value per training occurrence")
    if not all(len(values) == batch_size for values in (anchor_obs, task_ids, traj_ids)):
        raise ValueError("BACE advantage metadata must have one value per training occurrence")
    if local_credit_mode not in {"occurrence", "action_mean"}:
        raise ValueError(f"Unknown BACE local credit mode: {local_credit_mode}")
    if local_credit_mode == "action_mean" and (action_ids is None or len(action_ids) != batch_size):
        raise ValueError("action_mean local credit requires one action ID per occurrence")

    remove_std = _normalization_mode(mode)
    macro_tokens = core_gigpo.episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=task_ids,
        traj_index=traj_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
    )
    macro_scores = _occurrence_values(macro_tokens, response_mask)

    step_group_ids = core_gigpo.build_step_group(
        anchor_obs,
        task_ids,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    if local_credit_mode == "occurrence":
        local_tokens = core_gigpo.step_norm_reward(
            step_rewards=step_rewards,
            response_mask=response_mask,
            index=step_group_ids,
            epsilon=epsilon,
            remove_std=remove_std,
        )
        local_scores = _occurrence_values(local_tokens, response_mask)
    else:
        local_tokens, local_scores = _action_mean_local_advantage(
            step_rewards,
            response_mask,
            step_group_ids,
            action_ids,
            epsilon,
            remove_std,
        )

    token_scores = macro_tokens + step_advantage_w * local_tokens
    occurrence_scores = macro_scores + step_advantage_w * local_scores
    components = {
        "macro": macro_scores,
        "local": local_scores,
        "occurrence": occurrence_scores,
    }
    return token_scores, token_scores, components


def _current_stable_macro(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    task_ids: np.ndarray,
    traj_ids: np.ndarray,
    *,
    epsilon: float,
    remove_std: bool,
    compute_mean_std_cross_steps: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The production C0 macro path, intentionally shared by every variant."""
    tokens = core_gigpo.episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=task_ids,
        traj_index=traj_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
    )
    return tokens, _occurrence_values(tokens, response_mask)


def _current_local_scores(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    task_ids: np.ndarray,
    *,
    epsilon: float,
    remove_std: bool,
    enable_similarity: bool,
    similarity_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = core_gigpo.build_step_group(
        anchor_obs,
        task_ids,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    tokens = core_gigpo.step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=groups,
        epsilon=epsilon,
        remove_std=remove_std,
    )
    return tokens, _occurrence_values(tokens, response_mask)


def compute_bace_tree_credit_advantage(
    *,
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    task_ids: np.ndarray,
    traj_ids: np.ndarray,
    occurrence_ids: np.ndarray,
    source_types: np.ndarray,
    leaf_ids: np.ndarray,
    step_indices: np.ndarray,
    raw_rewards: np.ndarray,
    episode_rewards: np.ndarray,
    tree_origin_occurrence_ids: np.ndarray,
    tree_parent_root_ids: np.ndarray,
    adjustment_padding_mask: np.ndarray | None = None,
    tree_credit_mode: str,
    macro_normalization_mode: str = "stable_occurrence",
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    compute_mean_std_cross_steps: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Compute C1/C2/C3 credit and remove copied origins from PPO support.

    Macro normalization is always evaluated on the complete C0 physical batch
    before copied-origin rows are removed.  Direct/descendant relationships use
    concrete natural occurrence IDs and root-step ancestry, never anchor keys.
    """
    try:
        credit_mode = TreeCreditMode(tree_credit_mode)
    except ValueError as exc:
        raise ValueError(f"Unknown BACE tree credit mode: {tree_credit_mode}") from exc
    if credit_mode is TreeCreditMode.CURRENT:
        raise ValueError("Use compute_bace_gigpo_advantage for tree_credit_mode=current")
    try:
        normalization_mode = MacroNormalizationMode(macro_normalization_mode)
    except ValueError as exc:
        raise ValueError(
            f"Unknown BACE macro normalization mode: {macro_normalization_mode}"
        ) from exc
    if normalization_mode is not MacroNormalizationMode.STABLE_OCCURRENCE:
        raise ValueError(
            "C1/C2/C3 currently require macro_normalization_mode=stable_occurrence"
        )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    batch_size, response_length = response_mask.shape
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must have the same shape")
    if step_rewards.shape != (batch_size,):
        raise ValueError("step_rewards must have one value per physical occurrence")
    anchor_obs = _as_object_array(anchor_obs, "anchor_obs", batch_size)
    task_ids = _as_object_array(task_ids, "task_id", batch_size)
    traj_ids = _as_object_array(traj_ids, "traj_id", batch_size)
    occurrence_ids = _as_object_array(occurrence_ids, "occurrence_id", batch_size)
    source_types = _as_object_array(source_types, "source_type", batch_size)
    leaf_ids = _as_object_array(leaf_ids, "leaf_id", batch_size)
    step_indices = np.asarray(step_indices)
    raw_rewards = np.asarray(raw_rewards)
    episode_rewards = np.asarray(episode_rewards)
    tree_origin_occurrence_ids = _as_object_array(
        tree_origin_occurrence_ids, "tree_origin_occurrence_id", batch_size
    )
    tree_parent_root_ids = _as_object_array(
        tree_parent_root_ids, "tree_parent_root_id", batch_size
    )
    if adjustment_padding_mask is None:
        adjustment_padding_mask = np.zeros(batch_size, dtype=bool)
    adjustment_padding_mask = np.asarray(adjustment_padding_mask, dtype=bool)
    if len(adjustment_padding_mask) != batch_size:
        raise ValueError("adjustment padding mask must match the physical batch")
    if (
        len(step_indices) != batch_size
        or len(raw_rewards) != batch_size
        or len(episode_rewards) != batch_size
    ):
        raise ValueError(
            "Tree credit requires one step index, raw reward, and episode reward "
            "per occurrence"
        )

    remove_std = _normalization_mode(mode)
    _, physical_macro = _current_stable_macro(
        token_level_rewards,
        response_mask,
        task_ids,
        traj_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
    )
    _, physical_local = _current_local_scores(
        step_rewards,
        response_mask,
        anchor_obs,
        task_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    tree = build_tree_credit_index(
        source_types=source_types,
        occurrence_ids=occurrence_ids,
        traj_ids=traj_ids,
        leaf_ids=leaf_ids,
        step_indices=step_indices,
        tree_origin_occurrence_ids=tree_origin_occurrence_ids,
        tree_parent_root_ids=tree_parent_root_ids,
        adjustment_padding_mask=adjustment_padding_mask,
    )

    keep_indices = np.flatnonzero(
        (source_types != "branch_origin") & ~adjustment_padding_mask
    ).astype(np.int64)
    if len(keep_indices) == 0:
        raise ValueError("Tree credit produced an empty unique-edge training support")
    kept_sources = source_types[keep_indices]
    kept_occurrences = occurrence_ids[keep_indices]
    kept_traj_ids = traj_ids[keep_indices]
    kept_leaf_ids = leaf_ids[keep_indices]
    kept_mask = response_mask[keep_indices]
    kept_anchor_obs = anchor_obs[keep_indices]
    kept_task_ids = task_ids[keep_indices]

    for root_id, root_leaf_id in tree.root_leaf_by_root_id.items():
        rows = [
            row for row in tree.root_row_by_occurrence.values()
            if str(tree_parent_root_ids[row]) == root_id
        ]
        if any(str(leaf_ids[row]) != root_leaf_id for row in rows):
            raise ValueError(f"Natural root {root_id} changed leaf identity")
    branch_macro_by_id = {
        branch_id: physical_macro[row]
        for branch_id, row in tree.branch_origin_row_by_id.items()
    }

    g_original = step_rewards[keep_indices].clone()
    g_direct = g_original.clone()
    g_descendant = g_original.clone()
    macro_base = physical_macro[keep_indices].clone()
    macro_descendant = macro_base.clone()
    direct_leaf_ids: list[tuple[str, ...]] = []
    descendant_leaf_ids: list[tuple[str, ...]] = []

    for local_row, physical_row in enumerate(keep_indices):
        source = str(kept_sources[local_row])
        if source == "branch_suffix":
            branch_id = str(kept_traj_ids[local_row])
            leaf = tree.branch_leaf_by_id[branch_id]
            direct_leaf_ids.append((leaf,))
            descendant_leaf_ids.append((leaf,))
            continue
        if source != "root":
            raise ValueError(f"Unexpected unique-edge source: {source}")

        edge_id = str(kept_occurrences[local_row])
        root_id = str(tree_parent_root_ids[physical_row])
        root_leaf = tree.root_leaf_by_root_id[root_id]
        direct_branches = tree.direct_branch_ids_by_edge.get(edge_id, ())
        descendant_branches = tree.descendant_branch_ids_by_edge.get(edge_id, ())
        direct_leaves = (root_leaf,) + tuple(
            tree.branch_leaf_by_id[branch_id] for branch_id in direct_branches
        )
        descendant_leaves = (root_leaf,) + tuple(
            tree.branch_leaf_by_id[branch_id] for branch_id in descendant_branches
        )
        direct_leaf_ids.append(direct_leaves)
        descendant_leaf_ids.append(descendant_leaves)

        # A branch origin and its natural origin execute the exact same action,
        # including its invalid-action status.  Back up only their continuation
        # delta, then apply it at the current target edge.  This retains the
        # target edge's own local invalid penalty once, without propagating the
        # later origin's penalty to an earlier shared prefix.
        direct_values = [step_rewards[physical_row]]
        for branch_id in direct_branches:
            natural_origin_row = tree.root_row_by_occurrence[
                tree.branch_origin_occurrence_by_id[branch_id]
            ]
            direct_values.append(
                step_rewards[physical_row]
                + step_rewards[tree.branch_origin_row_by_id[branch_id]]
                - step_rewards[natural_origin_row]
            )
        g_direct[local_row] = torch.stack(direct_values).mean()

        descendant_values = [step_rewards[physical_row]]
        for branch_id in descendant_branches:
            natural_origin_row = tree.root_row_by_occurrence[
                tree.branch_origin_occurrence_by_id[branch_id]
            ]
            branch_step = tree.branch_origin_step_by_id[branch_id]
            edge_step = int(step_indices[physical_row])
            branch_g = step_rewards[tree.branch_origin_row_by_id[branch_id]]
            origin_g = step_rewards[natural_origin_row]
            descendant_values.append(
                step_rewards[physical_row]
                + (float(gamma) ** (branch_step - edge_step))
                * (branch_g - origin_g)
            )
        g_descendant[local_row] = torch.stack(descendant_values).mean()

        descendant_macro_values = [macro_base[local_row]]
        for branch_id in descendant_branches:
            natural_origin_row = tree.root_row_by_occurrence[
                tree.branch_origin_occurrence_by_id[branch_id]
            ]
            descendant_macro_values.append(
                macro_base[local_row]
                + branch_macro_by_id[branch_id]
                - physical_macro[natural_origin_row]
            )
        macro_descendant[local_row] = torch.stack(descendant_macro_values).mean()

    _, local_c1 = _current_local_scores(
        g_direct,
        kept_mask,
        kept_anchor_obs,
        kept_task_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    _, local_c3 = _current_local_scores(
        g_descendant,
        kept_mask,
        kept_anchor_obs,
        kept_task_ids,
        epsilon=epsilon,
        remove_std=remove_std,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )

    physical_current_final = physical_macro + step_advantage_w * physical_local
    current_final = physical_current_final[keep_indices]
    final_c1 = macro_base + step_advantage_w * local_c1
    final_c2 = macro_descendant + step_advantage_w * local_c1
    final_c3 = macro_descendant + step_advantage_w * local_c3
    if credit_mode is TreeCreditMode.O1_LOCAL:
        selected_macro, selected_local, selected_final = macro_base, local_c1, final_c1
    elif credit_mode is TreeCreditMode.O1_TREE_MACRO:
        selected_macro, selected_local, selected_final = (
            macro_descendant, local_c1, final_c2
        )
    else:
        selected_macro, selected_local, selected_final = (
            macro_descendant, local_c3, final_c3
        )

    token_scores = selected_final.unsqueeze(-1).expand(
        len(keep_indices), response_length
    ) * kept_mask
    edge_ids = np.asarray([
        str(kept_occurrences[row]) for row in range(len(keep_indices))
    ], dtype=object)
    root_ids = np.asarray([
        str(tree_parent_root_ids[physical_row]) for physical_row in keep_indices
    ], dtype=object)
    token_counts = kept_mask.sum(dim=-1).to(dtype=selected_final.dtype)
    current_mass = torch.sum(torch.abs(current_final) * token_counts)
    selected_mass = torch.sum(torch.abs(selected_final) * token_counts)
    sign_flip = (
        (torch.sign(current_final) != torch.sign(selected_final))
        & (torch.abs(current_final) > epsilon)
        & (torch.abs(selected_final) > epsilon)
    )
    diagnostics = {
        "tree_credit_physical_occurrences": float(batch_size),
        "tree_credit_unique_occurrences": float(len(keep_indices)),
        # Keep copied branch-origin evidence and synthetic divisibility padding
        # distinct.  They are both absent from PPO support, but only the former
        # represents a real rollout edge that was intentionally de-duplicated.
        "tree_credit_copied_origins_removed": float(np.sum(
            (source_types == "branch_origin") & ~adjustment_padding_mask
        )),
        "tree_credit_adjustment_padding_removed": float(
            np.sum(adjustment_padding_mask)
        ),
        "tree_credit_direct_continuations_mean": float(
            np.mean([len(values) for values in direct_leaf_ids])
        ),
        "tree_credit_descendant_leaves_mean": float(
            np.mean([len(values) for values in descendant_leaf_ids])
        ),
        "tree_credit_advantage_sign_flip_fraction": float(sign_flip.float().mean().item()),
        "tree_credit_invalid_penalty_is_edge_local": 1.0,
        "tree_credit_gradient_mass_proxy_ratio": float(
            (selected_mass / current_mass).item()
            if float(current_mass.item()) > epsilon
            else 0.0
        ),
        "tree_credit_macro_prefix_edges_affected": float(sum(
            len(values) > 1 for source, values in zip(kept_sources, descendant_leaf_ids)
            if str(source) == "root"
        )),
        "tree_credit_local_prefix_edges_affected": float(sum(
            len(values) > 1 for source, values in zip(kept_sources, (
                descendant_leaf_ids
                if credit_mode is TreeCreditMode.O1_FULL_TREE
                else direct_leaf_ids
            )) if str(source) == "root"
        )),
    }
    branch_evidence = [
        {
            "branch_id": branch_id,
            "origin_occurrence_id": tree.branch_origin_occurrence_by_id[branch_id],
            "parent_root_id": tree.branch_parent_root_by_id[branch_id],
            "origin_step_index": tree.branch_origin_step_by_id[branch_id],
            "leaf_id": tree.branch_leaf_by_id[branch_id],
            "origin_g_branch": float(step_rewards[row].item()),
            "terminal_reward": float(episode_rewards[row]),
        }
        for branch_id, row in sorted(tree.branch_origin_row_by_id.items())
    ]
    metadata = {
        "bace_edge_id": edge_ids,
        "bace_root_id": root_ids,
        "bace_direct_leaf_ids": _one_dimensional_object_array(direct_leaf_ids),
        "bace_descendant_leaf_ids": _one_dimensional_object_array(
            descendant_leaf_ids
        ),
        "bace_num_direct_continuations": np.asarray(
            [len(values) for values in direct_leaf_ids], dtype=np.int32
        ),
        "bace_num_descendant_leaves": np.asarray(
            [len(values) for values in descendant_leaf_ids], dtype=np.int32
        ),
        "bace_g_original": g_original.detach().cpu().numpy(),
        "bace_g_direct_mean": g_direct.detach().cpu().numpy(),
        "bace_g_descendant_mean": g_descendant.detach().cpu().numpy(),
        "bace_macro_base_stable": macro_base.detach().cpu().numpy(),
        "bace_macro_descendant_mean": macro_descendant.detach().cpu().numpy(),
        "bace_local_current": physical_local[keep_indices].detach().cpu().numpy(),
        "bace_local_c1": local_c1.detach().cpu().numpy(),
        "bace_local_c3": local_c3.detach().cpu().numpy(),
        "bace_final_current": current_final.detach().cpu().numpy(),
        "bace_final_c1": final_c1.detach().cpu().numpy(),
        "bace_final_c2": final_c2.detach().cpu().numpy(),
        "bace_final_c3": final_c3.detach().cpu().numpy(),
        "bace_tree_credit_mode": np.asarray(
            [credit_mode.value] * len(keep_indices), dtype=object
        ),
        "bace_macro_normalization_mode": np.asarray(
            [normalization_mode.value] * len(keep_indices), dtype=object
        ),
    }
    components: dict[str, object] = {
        "macro": selected_macro,
        "local": selected_local,
        "occurrence": selected_final,
        "base_macro_physical": physical_macro,
        "keep_indices": keep_indices,
        "metadata": metadata,
        "tree_diagnostics": diagnostics,
        "branch_evidence": branch_evidence,
    }
    return token_scores, token_scores, components
