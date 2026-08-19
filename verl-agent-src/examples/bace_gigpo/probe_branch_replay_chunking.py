#!/usr/bin/env python3
"""Exercise capacity-sized waves against a real ALFWorld replay pool."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import ray
from omegaconf import OmegaConf

from recipe.bace_gigpo.env_factory import make_branch_env
from recipe.bace_gigpo.replay.validator import ReplayValidator
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector
from recipe.bace_gigpo.types import ReplayRequest


TUPLE_FIELDS = {
    "parsed_action_prefix",
    "prefix_observations",
    "expected_action_set",
    "copied_response_token_ids",
    "copied_response_loss_mask",
    "copied_old_log_probs",
    "original_prompt_token_ids",
}


def load_request(path: Path) -> ReplayRequest:
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    request = payload["requests"][0]
    for key in TUPLE_FIELDS:
        if request.get(key) is not None:
            request[key] = tuple(request[key])
    return ReplayRequest(**request)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition_rounds", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[17, 48])
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_request = load_request(args.acquisition_rounds)
    config = OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "seed": 0,
            "history_length": 2,
            "resources_per_worker": {"num_cpus": 0.25, "num_gpus": 0},
            "alfworld": {"eval_dataset": "eval_in_distribution"},
        },
        "data": {"train_batch_size": args.capacity},
    })

    ray.init(num_cpus=8, include_dashboard=False)
    manager = make_branch_env(config)
    validator = ReplayValidator(compare_action_set=True)
    report = {
        "ok": False,
        "capacity": manager.replay_capacity,
        "counts": [],
    }
    try:
        if manager.replay_capacity != args.capacity:
            raise AssertionError(
                f"Expected capacity {args.capacity}, got {manager.replay_capacity}"
            )
        for count in args.counts:
            requests = [
                dataclasses.replace(
                    base_request,
                    request_id=f"probe-request-{count}-{index}",
                    branch_id=f"probe-branch-{count}-{index}",
                )
                for index in range(count)
            ]

            capacity_error = None
            try:
                manager.replay(requests)
            except ValueError as error:
                capacity_error = str(error)
            if capacity_error is None:
                raise AssertionError("Oversized unchunked replay unexpectedly succeeded")

            waves = BaceTrajectoryCollector._chunk_requests(
                requests, manager.replay_capacity
            )
            replay_validated = 0
            transition_validated = 0
            for wave in waves:
                observations, dones, _ = manager.replay(wave)
                replay_results = [
                    validator.validate(
                        request,
                        observations["anchor"][index],
                        observations["admissible_actions"][index],
                        bool(dones[index]),
                    )
                    for index, request in enumerate(wave)
                ]
                failures = [result for result in replay_results if not result.replay_ok]
                if failures:
                    raise AssertionError(f"Replay validation failures: {failures}")
                replay_validated += len(replay_results)

                next_obs, rewards, next_dones, infos = manager.step(
                    [request.copied_raw_model_response for request in wave]
                )
                transition_results = validator.validate_transition
                checked = [
                    transition_results(
                        request=request,
                        observation=next_obs["anchor"][index],
                        reward=float(rewards[index]),
                        done=bool(next_dones[index]),
                        action_identity=infos[index].get("action_identity"),
                    )
                    for index, request in enumerate(wave)
                ]
                failures = [result for result in checked if not result.replay_ok]
                if failures:
                    raise AssertionError(f"Transition validation failures: {failures}")
                transition_validated += len(checked)

            report["counts"].append({
                "request_count": count,
                "wave_sizes": [len(wave) for wave in waves],
                "unchunked_capacity_error": capacity_error,
                "replay_validated": replay_validated,
                "transition_validated": transition_validated,
            })
        report["ok"] = True
        return 0
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manager.envs.close()
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
