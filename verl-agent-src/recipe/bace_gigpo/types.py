from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RootEvent:
    occurrence_id: str
    step_index: int
    pre_action_observation: Any
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_loss_mask: tuple[int, ...]
    old_log_probs: tuple[float, ...] | None
    raw_model_response: str
    parsed_environment_action: str
    canonical_action: str
    admissible_actions: tuple[str, ...]
    post_action_observation: Any
    reward: float
    done: bool
    remaining_horizon: int
    action_identity: str | None = None
    action_identity_kind: str = "legacy"
    action_format_valid: bool = True
    action_environment_valid: bool | None = None


@dataclass(frozen=True)
class RootEventLog:
    task_id: str
    task_family: str
    episode_group_id: str
    root_id: str
    environment_reset_key: str
    task_description: str
    task_batch_index: int
    events: tuple[RootEvent, ...]
    terminal_reward: float
    won: bool


@dataclass(frozen=True)
class OriginOccurrence:
    occurrence_id: str
    task_id: str
    root_id: str
    step_index: int
    anchor_id: str
    action_id: str
    environment_reset_key: str
    remaining_horizon: int


@dataclass
class AnchorRecord:
    task_id: str
    anchor_id: str
    anchor_key: Any
    occurrence_ids: list[str] = field(default_factory=list)
    observed_action_ids: list[str] = field(default_factory=list)
    origins_by_action: dict[str, list[OriginOccurrence]] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayRequest:
    request_id: str
    task_id: str
    branch_id: str
    origin_occurrence_id: str
    environment_reset_key: str
    task_description: str
    task_batch_index: int
    target_turn: int
    parsed_action_prefix: tuple[str, ...]
    prefix_observations: tuple[Any, ...]
    expected_anchor_key: Any
    expected_observation: Any
    expected_action_set: tuple[str, ...]
    selected_canonical_action: str
    copied_response_token_ids: tuple[int, ...]
    copied_raw_model_response: str
    copied_response_loss_mask: tuple[int, ...]
    copied_old_log_probs: tuple[float, ...] | None
    original_prompt_token_ids: tuple[int, ...]
    remaining_horizon: int
    copied_parsed_environment_action: str = ""
    copied_action_identity: str | None = None
    copied_action_identity_kind: str = "legacy"
    copied_action_environment_valid: bool | None = None
    expected_post_action_observation: Any = None
    expected_immediate_reward: float = 0.0
    expected_post_action_done: bool = False


@dataclass(frozen=True)
class ReplayResult:
    request_id: str
    replay_ok: bool
    category: str
    restored_observation: Any
    restored_anchor_key: Any
    restored_action_set: tuple[str, ...]
    restored_done: bool
    error_message: str = ""
