from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import RootEvent, RootEventLog


def _active_values(values, mask) -> tuple:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    active = np.asarray(mask, dtype=bool)
    return tuple(array[active].tolist())


def task_family_from_reset_key(reset_key: str) -> str:
    families = (
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    )
    family = next((family for family in families if family in reset_key), None)
    if family is not None:
        return family
    try:
        int(reset_key)
    except (TypeError, ValueError):
        return "unknown"
    return "webshop"


# Backward-compatible private name used by older traces/tests.
_task_family = task_family_from_reset_key


def resolve_action_identity(explicit_identity, parsed_action: str,
                            format_valid: bool, environment_valid: bool):
    if explicit_identity is not None:
        return str(explicit_identity)
    if not format_valid:
        return None
    kind = "valid" if environment_valid else "invalid"
    return f"{kind}::{parsed_action.strip()}"


def build_root_event_logs(batch) -> list[RootEventLog]:
    """Convert the flattened natural-root DataProto into immutable event logs."""
    required = {
        "uid",
        "traj_uid",
        "occurrence_id",
        "step_index",
        "task_batch_index",
        "anchor_obs",
        "projected_action",
        "admissible_actions",
        "post_action_observation",
        "environment_reset_key",
        "task_description",
        "raw_model_response",
        "done",
        "remaining_horizon",
        "rewards",
        "episode_rewards",
    }
    missing = required.difference(batch.non_tensor_batch)
    if missing:
        raise ValueError(f"Natural rollout is missing BACE metadata: {sorted(missing)}")

    rows_by_traj = defaultdict(list)
    for row_idx, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
        rows_by_traj[str(traj_uid)].append(row_idx)

    logs = []
    response_length = batch.batch["responses"].shape[-1]
    for root_id, row_indices in rows_by_traj.items():
        row_indices.sort(key=lambda idx: int(batch.non_tensor_batch["step_index"][idx]))
        first = row_indices[0]
        events = []
        for idx in row_indices:
            attention_mask = batch.batch["attention_mask"][idx].detach().cpu().numpy().astype(bool)
            prompt_mask = attention_mask[:-response_length]
            response_mask = attention_mask[-response_length:]
            old_log_probs = None
            if "rollout_log_probs" in batch.batch:
                old_log_probs = _active_values(batch.batch["rollout_log_probs"][idx], response_mask)
            action = str(batch.non_tensor_batch["projected_action"][idx])
            format_valid = bool(
                batch.non_tensor_batch.get("is_action_format_valid", batch.non_tensor_batch["is_action_valid"])[idx]
            )
            environment_valid = bool(
                batch.non_tensor_batch.get("is_action_environment_valid", batch.non_tensor_batch["is_action_valid"])[idx]
            )
            action_identity_values = batch.non_tensor_batch.get("action_identity")
            action_identity = (
                action_identity_values[idx] if action_identity_values is not None else None
            )
            action_identity = resolve_action_identity(
                action_identity, action, format_valid, environment_valid
            )
            identity_kind = (
                "unparsed" if not format_valid else ("valid" if environment_valid else "invalid")
            )
            events.append(
                RootEvent(
                    occurrence_id=str(batch.non_tensor_batch["occurrence_id"][idx]),
                    step_index=int(batch.non_tensor_batch["step_index"][idx]),
                    pre_action_observation=batch.non_tensor_batch["anchor_obs"][idx],
                    prompt_token_ids=_active_values(batch.batch["input_ids"][idx][:-response_length], prompt_mask),
                    response_token_ids=_active_values(batch.batch["responses"][idx], response_mask),
                    response_loss_mask=tuple(int(value) for value in response_mask[response_mask]),
                    old_log_probs=old_log_probs,
                    raw_model_response=str(batch.non_tensor_batch["raw_model_response"][idx]),
                    parsed_environment_action=action,
                    canonical_action=action_identity or "INVALID",
                    admissible_actions=tuple(batch.non_tensor_batch["admissible_actions"][idx]),
                    post_action_observation=batch.non_tensor_batch["post_action_observation"][idx],
                    reward=float(batch.non_tensor_batch["rewards"][idx]),
                    done=bool(batch.non_tensor_batch["done"][idx]),
                    remaining_horizon=int(batch.non_tensor_batch["remaining_horizon"][idx]),
                    action_identity=action_identity,
                    action_identity_kind=identity_kind,
                    action_format_valid=format_valid,
                    action_environment_valid=environment_valid,
                )
            )
        reset_key = str(batch.non_tensor_batch["environment_reset_key"][first])
        terminal_reward = float(batch.non_tensor_batch["episode_rewards"][first])
        logs.append(
            RootEventLog(
                task_id=str(batch.non_tensor_batch["uid"][first]),
                task_family=task_family_from_reset_key(reset_key),
                episode_group_id=str(batch.non_tensor_batch["uid"][first]),
                root_id=root_id,
                environment_reset_key=reset_key,
                task_description=str(batch.non_tensor_batch["task_description"][first]),
                task_batch_index=int(batch.non_tensor_batch["task_batch_index"][first]),
                events=tuple(events),
                terminal_reward=terminal_reward,
                won=terminal_reward > 0,
            )
        )
    return logs
