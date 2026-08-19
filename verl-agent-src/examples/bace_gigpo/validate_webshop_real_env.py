#!/usr/bin/env python3
"""Validate WebShop replay identities against the real embedded environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBSHOP_ROOT = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop"
ALFWORLD_PACKAGE_ROOT = REPO_ROOT / "agent_system/environments/env_package/alfworld"
for path in (REPO_ROOT, ALFWORLD_PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_system.environments.env_package.webshop.envs import WebshopWorker
from agent_system.environments.env_package.webshop.projection import webshop_projection


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_raw_action(action: str, label: str) -> tuple[str, str, int]:
    raw = f"<think>Execute the deterministic {label} action.</think><action>{action}</action>"
    projected, valid = webshop_projection([raw])
    return raw, projected[0], int(valid[0])


def choose_product_action(available_actions: dict) -> str:
    for clickable in available_actions["clickables"]:
        if re.fullmatch(r"[a-z0-9]{10}", clickable):
            return f"click[{clickable}]"
    raise RuntimeError("Search results did not expose a product ASIN")


def service_status(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
            return {
                "url": url,
                "http_status": response.status,
                "body_bytes": len(body),
                "contains_webshop": b"WebShop" in body,
            }
    except Exception as error:
        return {"url": url, "error": repr(error)}


def validate_session(worker: WebshopWorker, session_id: int) -> dict:
    initial_obs, initial_info = worker.reset(session_id)
    initial_actions = initial_info["available_actions"]

    search_action = "search[shirt]"
    search_raw, search_parsed, search_format_valid = parse_raw_action(search_action, "search")
    search_obs, search_reward, search_done, search_info = worker.step(search_parsed)
    search_actions = search_info["available_actions"]

    product_action = choose_product_action(search_actions)
    product_raw, product_parsed, product_format_valid = parse_raw_action(product_action, "product")
    product_obs, product_reward, product_done, product_info = worker.step(product_parsed)
    product_actions = product_info["available_actions"]

    replay_initial_obs, replay_initial_info = worker.reset(session_id)
    replay_search_obs, replay_search_done, replay_search_info, replay_search_actions = worker.replay(
        session_id, [search_parsed]
    )
    replay_product_obs, replay_product_reward, replay_product_done, replay_product_info = worker.step(
        product_parsed
    )
    replay_product_actions = replay_product_info["available_actions"]

    replay_full_obs, replay_full_done, replay_full_info, replay_full_actions = worker.replay(
        session_id, [search_parsed, product_parsed]
    )

    checks = {
        "session_reset_identity": (
            replay_initial_obs == initial_obs
            and replay_initial_info["session_idx"] == session_id
            and replay_initial_info["available_actions"] == initial_actions
        ),
        "available_action_restoration": (
            replay_search_actions == search_actions
            and replay_search_info["available_actions"] == search_actions
            and replay_product_actions == product_actions
            and replay_full_actions == product_actions
        ),
        "raw_action_identity": (
            search_format_valid == 1
            and product_format_valid == 1
            and search_parsed == search_action
            and product_parsed == product_action
        ),
        "post_transition_consistency": (
            replay_search_obs == search_obs
            and replay_search_done == search_done
            and replay_product_obs == product_obs
            and replay_product_reward == product_reward
            and replay_product_done == product_done
            and replay_full_obs == product_obs
            and replay_full_done == product_done
            and replay_full_info["task_score"] == product_info["task_score"]
        ),
    }
    return {
        "session_id": session_id,
        "passed": all(checks.values()),
        "checks": checks,
        "actions": {
            "search": {"raw": search_raw, "parsed": search_parsed},
            "product": {"raw": product_raw, "parsed": product_parsed},
        },
        "original": {
            "initial_observation": initial_obs,
            "initial_available_actions": initial_actions,
            "search_observation": search_obs,
            "search_reward": search_reward,
            "search_done": search_done,
            "search_available_actions": search_actions,
            "product_observation": product_obs,
            "product_reward": product_reward,
            "product_done": product_done,
            "product_available_actions": product_actions,
        },
        "replay": {
            "search_observation": replay_search_obs,
            "search_done": replay_search_done,
            "search_available_actions": replay_search_actions,
            "product_observation": replay_product_obs,
            "product_reward": replay_product_reward,
            "product_done": replay_product_done,
            "product_available_actions": replay_product_actions,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, nargs="+", default=[500, 501, 502])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:3000/ABC")
    args = parser.parse_args()

    data_dir = WEBSHOP_ROOT / "data"
    product_file = data_dir / "items_shuffle_1000.json"
    attribute_file = data_dir / "items_ins_v2_1000.json"
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": str(product_file),
        "attr_path": str(attribute_file),
    }
    worker = WebshopWorker(seed=0, env_kwargs=env_kwargs)
    try:
        sessions = [validate_session(worker, session_id) for session_id in args.sessions]
    finally:
        worker.close()

    report = {
        "schema_version": "webshop-real-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "protocol": {
            "use_small": True,
            "human_goals": False,
            "validation_goal_ids": "0:500",
            "training_goal_ids": "500:len(goals)",
        },
        "data": {
            "product_file": str(product_file),
            "product_sha256": sha256(product_file),
            "attribute_file": str(attribute_file),
            "attribute_sha256": sha256(attribute_file),
        },
        "service": service_status(args.service_url),
        "sessions": sessions,
        "passed": all(session["passed"] for session in sessions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
