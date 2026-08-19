from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from gigpo import core_gigpo


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
