#!/usr/bin/env python3
"""Exercise BACE WebShop selected-worker replay on the real environment.

This is a deterministic environment gate, deliberately separate from model
sampling: free-form base-model searches need not produce a structural Exact
anchor in one PPO step, whereas this verifies the selected slot machinery with
an actual legal search and a real clickable returned by WebShop.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import ray


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop.projection import webshop_projection
from recipe.bace_gigpo.replay.validator import ReplayValidator


def raw(action: str) -> str:
    return f"<think>Execute the exact admissible action.</think><action>{action}</action>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ray-num-cpus", type=int, default=2)
    # The first WebShop worker imports spaCy and starts a Lucene JVM.  On a
    # cold filesystem this can take several minutes, so the guard must be long
    # enough to distinguish slow initialization from an unbounded hang.
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    if args.ray_num_cpus <= 0:
        raise ValueError("--ray-num-cpus must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(
            f"WebShop selected-worker validation exceeded {args.timeout_seconds} seconds"
        )

    # A validator must never turn a login node into a large Ray cluster.  The
    # trainer already initializes Ray with its Slurm CPU allocation; this path
    # is only for the standalone environment gate.
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(
            num_cpus=args.ray_num_cpus,
            include_dashboard=False,
            log_to_driver=True,
        )
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout_seconds)

    data_root = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    manager = None
    report = {"session": args.session, "passed": False, "checks": {}}
    try:
        envs = build_webshop_envs(
            seed=0,
            env_num=1,
            group_n=2,
            is_train=True,
            env_kwargs={
                "observation_mode": "text",
                "num_products": None,
                "human_goals": False,
                "file_path": str(data_root / "items_shuffle_1000.json"),
                "attr_path": str(data_root / "items_ins_v2_1000.json"),
            },
            resources_per_worker={"num_cpus": 0.1},
            search_backend="ray_shared",
            search_pool_size=1,
            search_actor_num_cpus=0.1,
            worker_init_batch_size=1,
            sessions_per_actor=2,
        )
        manager = WebshopEnvironmentManager(
            envs, webshop_projection, SimpleNamespace(env=SimpleNamespace(history_length=2))
        )
        initial, _ = manager.reset_selected([0], [args.session])
        initial_anchor = initial["anchor"][0]
        search = "search[shirt]"
        searched, _, _, search_infos = manager.step_selected([0], [raw(search)])
        search_anchor = searched["anchor"][0]
        search_pool = searched["admissible_actions"][0]
        click = next(action for action in search_pool if action.startswith("click["))

        natural_next, natural_rewards, natural_dones, natural_infos = manager.step_selected(
            [0], [raw(click)]
        )
        request = SimpleNamespace(
            request_id="real-webshop-selected",
            environment_reset_key=str(args.session),
            parsed_action_prefix=(search,),
            prefix_observations=(initial_anchor,),
            task_description=manager.get_tasks_selected([0])[0],
            expected_anchor_key=search_anchor,
            expected_action_set=tuple(search_pool),
            copied_action_environment_valid=True,
            copied_parsed_environment_action=click,
            selected_canonical_action=f"valid::{click}",
            copied_action_identity=f"valid::{click}",
            expected_post_action_observation=natural_next["anchor"][0],
            expected_immediate_reward=float(natural_rewards[0]),
            expected_post_action_done=bool(natural_dones[0]),
        )
        restored, restored_dones, _ = manager.replay_selected([1], [request])
        validator = ReplayValidator(action_is_executable=manager.is_action_executable)
        replay = validator.validate(
            request, restored["anchor"][0], restored["admissible_actions"][0],
            bool(restored_dones[0]),
        )
        branch_next, branch_rewards, branch_dones, branch_infos = manager.step_selected(
            [1], [raw(click)]
        )
        transition = validator.validate_transition(
            request,
            branch_next["anchor"][0],
            float(branch_rewards[0]),
            bool(branch_dones[0]),
            branch_infos[0]["action_identity"],
        )
        report["checks"] = {
            "search_format_valid": bool(search_infos[0]["is_action_format_valid"]),
            "search_environment_valid": bool(search_infos[0]["is_action_environment_valid"]),
            "search_identity": search_infos[0]["action_identity"],
            "selected_replay": replay.category,
            "selected_transition": transition.category,
            "natural_identity": natural_infos[0]["action_identity"],
            "branch_identity": branch_infos[0]["action_identity"],
            "click": click,
        }
        report["passed"] = replay.replay_ok and transition.replay_ok
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if manager is not None:
            manager.close()
        if owns_ray and ray.is_initialized():
            ray.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
