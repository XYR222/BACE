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
