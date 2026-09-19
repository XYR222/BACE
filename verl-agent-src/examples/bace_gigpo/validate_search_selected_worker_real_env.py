#!/usr/bin/env python3
"""Validate Search selected-worker replay against a running fixed retriever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search import (
    build_search_envs,
    encode_search_reset_key,
    search_projection,
)
from recipe.bace_gigpo.replay.validator import ReplayValidator
from recipe.bace_gigpo.types import ReplayRequest


def request_for(reset_key, question, prefix_action, initial_obs, anchor_obs,
                origin_action, origin_response, post_obs, reward, done, identity):
    return ReplayRequest(
        request_id="search-real-replay", task_id="search-task", branch_id="search-branch",
        origin_occurrence_id="search-root:1", environment_reset_key=reset_key,
        task_description=question, task_batch_index=0, target_turn=1,
        parsed_action_prefix=(prefix_action,), prefix_observations=(initial_obs,),
        expected_anchor_key=anchor_obs, expected_observation=anchor_obs,
        expected_action_set=(), selected_canonical_action=identity,
        copied_response_token_ids=(1,), copied_raw_model_response=origin_response,
        copied_response_loss_mask=(1,), copied_old_log_probs=(-0.1,),
        original_prompt_token_ids=(2,), remaining_horizon=2,
        copied_parsed_environment_action=origin_action,
        copied_action_identity=identity, copied_action_identity_kind="valid",
        copied_action_environment_valid=True,
        expected_post_action_observation=post_obs,
        expected_immediate_reward=float(reward), expected_post_action_done=bool(done),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = OmegaConf.create({
        "data": {"train_batch_size": 2},
        "env": {
            "seed": 0, "max_steps": 4, "history_length": 4,
            "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
            "search": {"search_url": args.search_url, "topk": 3,
                       "timeout": 60, "log_requests": False},
        },
    })
    vector = build_search_envs(
        seed=0, env_num=2, group_n=1, is_train=True, env_config=config.env
    )
    manager = SearchEnvironmentManager(vector, search_projection, config)
    question = "Who directed the film The Godfather?"
    spec = {"question": question,
            "ground_truth": {"target": ["Francis Ford Coppola"]},
            "data_source": "hotpotqa"}
    reset_key = encode_search_reset_key(spec)
    prefix_response = "<think>Find the director.</think><search>The Godfather director</search>"
    origin_response = "<think>Verify the name.</think><search>Francis Ford Coppola The Godfather</search>"
    prefix_action = search_projection([prefix_response])[0][0]
    origin_action = search_projection([origin_response])[0][0]
    try:
        initial, _ = manager.reset_selected([0], [reset_key])
        anchor, _, prefix_dones, _ = manager.step_selected([0], [prefix_response])
        if bool(prefix_dones[0]):
            raise AssertionError("Search prefix terminated unexpectedly")
        natural_post, natural_rewards, natural_dones, natural_infos = manager.step_selected(
            [0], [origin_response]
        )
        identity = natural_infos[0]["action_identity"]
        request = request_for(
            reset_key, question, prefix_action, initial["anchor"][0],
            anchor["anchor"][0], origin_action, origin_response,
            natural_post["anchor"][0], natural_rewards[0], natural_dones[0], identity,
        )
        restored, restored_dones, _ = manager.replay_selected([1], [request])
        validator = ReplayValidator(
            compare_action_set=True, action_is_executable=manager.is_action_executable
        )
        replay_result = validator.validate(
            request, restored["anchor"][0], (), bool(restored_dones[0])
        )
        branch_post, branch_rewards, branch_dones, branch_infos = manager.step_selected(
            [1], [origin_response]
        )
        transition_result = validator.validate_transition(
            request, branch_post["anchor"][0], branch_rewards[0],
            branch_dones[0], branch_infos[0]["action_identity"],
        )
        report = {
            "ok": replay_result.replay_ok and transition_result.replay_ok,
            "reset_key_schema": reset_key.split(":", 1)[0],
            "prefix_action": prefix_action,
            "origin_action": origin_action,
            "action_identity": identity,
            "replay_category": replay_result.category,
            "transition_category": transition_result.category,
            "retrieval_exact_match": (
                natural_post["anchor"][0] == branch_post["anchor"][0]
            ),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
        if not report["ok"]:
            raise SystemExit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
