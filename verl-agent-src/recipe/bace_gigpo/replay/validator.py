from __future__ import annotations

from enum import Enum

from gigpo.core_gigpo import to_hashable

from ..types import ReplayRequest, ReplayResult


class ReplayCategory(str, Enum):
    VALIDATED = "VALIDATED"
    ANCHOR_KEY_MISMATCH = "ANCHOR_KEY_MISMATCH"
    ACTION_SET_MISMATCH = "ACTION_SET_MISMATCH"
    SELECTED_ACTION_NOT_EXECUTABLE = "SELECTED_ACTION_NOT_EXECUTABLE"
    EARLY_TERMINATION = "EARLY_TERMINATION"
    RESET_KEY_MISMATCH = "RESET_KEY_MISMATCH"
    PROMPT_IDENTITY_MISMATCH = "PROMPT_IDENTITY_MISMATCH"
    ENV_EXCEPTION = "ENV_EXCEPTION"
    ACTION_IDENTITY_MISMATCH = "ACTION_IDENTITY_MISMATCH"
    TRANSITION_OBSERVATION_MISMATCH = "TRANSITION_OBSERVATION_MISMATCH"
    TRANSITION_REWARD_MISMATCH = "TRANSITION_REWARD_MISMATCH"
    TRANSITION_DONE_MISMATCH = "TRANSITION_DONE_MISMATCH"


class ReplayValidator:
    def __init__(self, compare_action_set: bool = True):
        self.compare_action_set = compare_action_set

    def validate(self, request: ReplayRequest, observation, action_set, done: bool) -> ReplayResult:
        restored_key = to_hashable(observation)
        expected_key = to_hashable(request.expected_anchor_key)
        restored_actions = tuple(action_set)
        if done:
            category = ReplayCategory.EARLY_TERMINATION
        elif restored_key != expected_key:
            category = ReplayCategory.ANCHOR_KEY_MISMATCH
        elif request.copied_action_environment_valid is not False and (
            request.copied_parsed_environment_action or request.selected_canonical_action
        ) not in restored_actions:
            category = ReplayCategory.SELECTED_ACTION_NOT_EXECUTABLE
        elif self.compare_action_set and set(restored_actions) != set(request.expected_action_set):
            category = ReplayCategory.ACTION_SET_MISMATCH
        else:
            category = ReplayCategory.VALIDATED
        return ReplayResult(
            request_id=request.request_id,
            replay_ok=category is ReplayCategory.VALIDATED,
            category=category.value,
            restored_observation=observation,
            restored_anchor_key=restored_key,
            restored_action_set=restored_actions,
            restored_done=done,
        )

    def validate_transition(self, request: ReplayRequest, observation, reward: float,
                            done: bool, action_identity: str | None) -> ReplayResult:
        restored_key = to_hashable(observation)
        expected_key = to_hashable(request.expected_post_action_observation)
        if action_identity != request.copied_action_identity:
            category = ReplayCategory.ACTION_IDENTITY_MISMATCH
        elif restored_key != expected_key:
            category = ReplayCategory.TRANSITION_OBSERVATION_MISMATCH
        elif float(reward) != float(request.expected_immediate_reward):
            category = ReplayCategory.TRANSITION_REWARD_MISMATCH
        elif bool(done) != bool(request.expected_post_action_done):
            category = ReplayCategory.TRANSITION_DONE_MISMATCH
        else:
            category = ReplayCategory.VALIDATED
        return ReplayResult(
            request_id=request.request_id,
            replay_ok=category is ReplayCategory.VALIDATED,
            category=category.value,
            restored_observation=observation,
            restored_anchor_key=restored_key,
            restored_action_set=(),
            restored_done=bool(done),
        )

    def from_exception(self, request: ReplayRequest, error: Exception) -> ReplayResult:
        return ReplayResult(
            request_id=request.request_id,
            replay_ok=False,
            category=ReplayCategory.ENV_EXCEPTION.value,
            restored_observation=None,
            restored_anchor_key=None,
            restored_action_set=(),
            restored_done=False,
            error_message=str(error),
        )
