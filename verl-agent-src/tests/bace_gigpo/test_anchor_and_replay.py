from dataclasses import replace

import numpy as np
import torch

from recipe.bace_gigpo.anchor_index import AnchorIndex
from recipe.bace_gigpo.coordinator import FixedTopologyCoordinator
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector
from recipe.bace_gigpo.replay.validator import ReplayCategory, ReplayValidator
from recipe.bace_gigpo.types import ReplayRequest, RootEvent, RootEventLog
from verl import DataProto


def event(occurrence_id, step, observation, action, reward=0.0):
    return RootEvent(
        occurrence_id=occurrence_id,
        step_index=step,
        pre_action_observation=observation,
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4),
        response_loss_mask=(1, 1),
        old_log_probs=(-0.1, -0.2),
        raw_model_response=f"<think>x</think><action>{action}</action>",
        parsed_environment_action=action,
        canonical_action=action,
        admissible_actions=("open fridge", "go to table"),
        post_action_observation="next",
        reward=reward,
        done=False,
        remaining_horizon=4,
    )


def root(root_id, action, task="task-1"):
    return RootEventLog(
        task_id=task,
        task_family="pick_and_place",
        episode_group_id=task,
        root_id=root_id,
        environment_reset_key="/data/pick_and_place/game.tw-pddl",
        task_description="put an apple on the table",
        task_batch_index=0,
        events=(
            event(f"{root_id}:0", 0, "initial", "look"),
            event(f"{root_id}:1", 1, {"room": "kitchen"}, action),
        ),
        terminal_reward=0.0,
        won=False,
    )


def test_anchor_index_requires_repeated_state_with_two_actions():
    roots = [root("r1", "open fridge"), root("r2", "go to table"), root("r3", "open fridge")]
    index = AnchorIndex(roots)
    anchors = index.anchors_for_task("task-1")
    assert len(anchors) == 1
    assert anchors[0].observed_action_ids == ["go to table", "open fridge"]
    assert len(anchors[0].origins_by_action["open fridge"]) == 2


def test_fixed_topology_builds_concrete_origin_request():
    roots = [root("r1", "open fridge"), root("r2", "go to table"), root("r3", "open fridge")]
    requests, skipped = FixedTopologyCoordinator(seed=4).build_requests(roots)
    assert not skipped
    assert len(requests) == 1
    assert requests[0].target_turn == 1
    assert requests[0].parsed_action_prefix == ("look",)
    assert requests[0].selected_canonical_action in {"open fridge", "go to table"}


def test_fixed_topology_reports_missing_structural_anchor():
    roots = [root("r1", "open fridge"), root("r2", "open fridge"), root("r3", "open fridge")]
    requests, skipped = FixedTopologyCoordinator(seed=4).build_requests(roots)
    assert requests == []
    assert skipped == {"task-1": "NO_STRUCTURAL_ANCHOR"}


def request():
    roots = [root("r1", "open fridge"), root("r2", "go to table"), root("r3", "open fridge")]
    return FixedTopologyCoordinator(seed=4).build_requests(roots)[0][0]


def test_replay_validator_accepts_exact_executable_restore():
    req = request()
    result = ReplayValidator().validate(
        req, req.expected_observation, req.expected_action_set, done=False
    )
    assert result.replay_ok
    assert result.category == ReplayCategory.VALIDATED.value


def test_replay_validator_rejects_mismatch_and_early_terminal():
    req = request()
    mismatch = ReplayValidator().validate(req, "wrong", req.expected_action_set, done=False)
    terminal = ReplayValidator().validate(req, req.expected_observation, req.expected_action_set, done=True)
    assert mismatch.category == ReplayCategory.ANCHOR_KEY_MISMATCH.value
    assert terminal.category == ReplayCategory.EARLY_TERMINATION.value


def test_nested_non_tensor_metadata_keeps_one_batch_dimension():
    root = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1], [2]])},
        non_tensors={
            "admissible_actions": np.array(
                [("open fridge",), ("go to table", "look")], dtype=object
            )
        },
    )
    suffix = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[3]])},
        non_tensors={
            "admissible_actions": np.array([["open fridge", "look"]], dtype=object)
        },
    )

    batches = [root, suffix]
    BaceTrajectoryCollector._normalize_non_tensor_fields(batches)
    merged = DataProto.concat(batches)

    assert merged.non_tensor_batch["admissible_actions"].shape == (3,)
    assert list(merged.non_tensor_batch["admissible_actions"][2]) == ["open fridge", "look"]


def test_wave_concat_restores_numeric_episode_metadata():
    batches = [
        DataProto.from_dict(
            tensors={"input_ids": torch.tensor([[value]])},
            non_tensors={
                "episode_rewards": np.array([float(value)], dtype=object),
                "episode_lengths": np.array([value], dtype=object),
                "tool_callings": np.array([0.0], dtype=object),
                "success_rate": np.array([float(value > 0)], dtype=object),
            },
        )
        for value in (0, 1)
    ]

    collector = BaceTrajectoryCollector.__new__(BaceTrajectoryCollector)
    merged = collector._concat_batches(batches)

    for key in ("episode_rewards", "episode_lengths", "tool_callings", "success_rate"):
        assert merged.non_tensor_batch[key].dtype == np.float32
        assert hasattr(merged.non_tensor_batch[key].max(), "item")


def test_origin_fallback_stays_in_frozen_anchor_action_support():
    roots = [root("r1", "open fridge"), root("r2", "go to table"), root("r3", "open fridge")]
    req = next(
        FixedTopologyCoordinator(seed=seed).build_requests(roots)[0][0]
        for seed in range(20)
        if FixedTopologyCoordinator(seed=seed).build_requests(roots)[0][0].selected_canonical_action
        == "open fridge"
    )
    collector = BaceTrajectoryCollector.__new__(BaceTrajectoryCollector)
    replacement = collector._fallback_request(req, roots, {req.origin_occurrence_id})

    assert replacement is not None
    assert replacement.request_id != req.request_id
    assert replacement.branch_id == req.branch_id
    assert replacement.origin_occurrence_id != req.origin_occurrence_id
    assert replacement.selected_canonical_action == req.selected_canonical_action
    assert replacement.expected_anchor_key == req.expected_anchor_key
