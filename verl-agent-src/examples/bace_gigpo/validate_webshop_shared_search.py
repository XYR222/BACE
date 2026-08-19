#!/usr/bin/env python3
"""Validate shared Lucene search without involving model inference."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ray


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBSHOP_PACKAGE = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop"
for path in (REPO_ROOT, WEBSHOP_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# PyJNIus does not reliably discover a Conda OpenJDK when the environment's
# Python is invoked by absolute path without `conda activate`.
if "JVM_PATH" not in os.environ:
    candidate = Path(sys.prefix) / "lib/server/libjvm.so"
    if candidate.is_file():
        os.environ["JAVA_HOME"] = str(Path(sys.prefix))
        os.environ["JVM_PATH"] = str(candidate)

from agent_system.environments.env_package.webshop.envs import (
    RayLuceneSearchClient,
    PackedWebshopWorker,
    WebshopMultiProcessEnv,
    WebshopWorker,
    create_webshop_search_pool,
)
from web_agent_site.engine.engine import init_search_engine


DEFAULT_QUERIES = [
    "shirt",
    "wireless headphones",
    "black running shoes",
    "stainless steel water bottle",
    "women cotton summer dress",
    "usb c charging cable",
    "nonexistent product phrase xyzzy",
    "4k monitor 27 inch",
    "organic green tea",
    "phone case",
    "noise cancelling earbuds",
    "men leather wallet",
    "kids educational toy",
    "kitchen storage container",
    "red backpack",
    "vitamin c serum",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def env_kwargs():
    data = WEBSHOP_PACKAGE / "data"
    return {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": str(data / "items_shuffle_1000.json"),
        "attr_path": str(data / "items_ins_v2_1000.json"),
    }


def stop_actors(actors):
    for actor in actors:
        try:
            ray.kill(actor)
        except Exception:
            pass


def run_parity(args):
    local = init_search_engine(num_products=None)
    actors, ready = create_webshop_search_pool(args.pool_size, num_cpus=args.search_cpus)
    client = RayLuceneSearchClient(actors[0])
    rows = []
    try:
        for query in DEFAULT_QUERIES:
            hits = local.search(query, k=args.top_k)
            local_asins = [json.loads(local.doc(hit.docid).raw())["id"] for hit in hits]
            shared_asins = client.search_asins(query, args.top_k)
            rows.append({
                "query": query,
                "local_asins": local_asins,
                "shared_asins": shared_asins,
                "identical": local_asins == shared_asins,
            })
        stats = ray.get([actor.stats.remote() for actor in actors])
    finally:
        stop_actors(actors)
    report = {
        "schema_version": "webshop-search-parity-v1",
        "created_at": utc_now(),
        "pool_size": args.pool_size,
        "ready": ready,
        "stats": stats,
        "results": rows,
        "passed": all(row["identical"] for row in rows),
    }
    write_json(args.output, report)
    return report["passed"]


def run_search_stress(args):
    actors, ready = create_webshop_search_pool(args.pool_size, num_cpus=args.search_cpus)
    rounds = []
    request_seconds = []
    failures = []
    try:
        for round_id in range(args.rounds):
            futures = []
            started = time.perf_counter()
            for request_id in range(args.concurrency):
                actor = actors[request_id % len(actors)]
                query = DEFAULT_QUERIES[(round_id * args.concurrency + request_id) % len(DEFAULT_QUERIES)]
                futures.append(actor.search_timed.remote(query, args.top_k))
            try:
                results = ray.get(futures)
                request_seconds.extend(result["seconds"] for result in results)
                result_count = len(results)
            except Exception as error:
                failures.append({"round": round_id, "error": repr(error)})
                result_count = 0
            rounds.append({
                "round": round_id,
                "batch_seconds": time.perf_counter() - started,
                "result_count": result_count,
            })
        stats = ray.get([actor.stats.remote() for actor in actors])
    finally:
        stop_actors(actors)
    report = {
        "schema_version": "webshop-search-stress-v1",
        "created_at": utc_now(),
        "pool_size": args.pool_size,
        "concurrency": args.concurrency,
        "round_count": args.rounds,
        "ready": ready,
        "rounds": rounds,
        "request_latency_seconds": {
            "mean": statistics.fmean(request_seconds) if request_seconds else None,
            "p50": percentile(request_seconds, 0.50),
            "p95": percentile(request_seconds, 0.95),
            "p99": percentile(request_seconds, 0.99),
        },
        "search_stats": stats,
        "failures": failures,
        "passed": not failures and all(row["result_count"] == args.concurrency for row in rounds),
    }
    write_json(args.output, report)
    return report["passed"]


def run_env_stress(args):
    actors, ready = create_webshop_search_pool(args.pool_size, num_cpus=args.search_cpus)
    train = validation = None
    stages = []
    started = time.perf_counter()
    try:
        train = WebshopMultiProcessEnv(
            seed=0,
            env_num=args.train_batch_size,
            group_n=args.group_size,
            resources_per_worker={"num_cpus": args.session_cpus, "num_gpus": 0},
            is_train=True,
            env_kwargs=env_kwargs(),
            search_backend="ray_shared",
            search_workers=actors,
            worker_init_batch_size=args.init_batch_size,
            diagnostics_dir=str(args.output.parent / "environment_runtime"),
            sessions_per_actor=args.sessions_per_actor,
        )
        stages.append({"stage": "train_ready", "seconds": time.perf_counter() - started})
        validation = WebshopMultiProcessEnv(
            seed=1000,
            env_num=args.val_batch_size,
            group_n=1,
            resources_per_worker={"num_cpus": args.session_cpus, "num_gpus": 0},
            is_train=False,
            env_kwargs=env_kwargs(),
            search_backend="ray_shared",
            search_workers=actors,
            worker_init_batch_size=args.init_batch_size,
            diagnostics_dir=str(args.output.parent / "environment_runtime"),
            sessions_per_actor=args.sessions_per_actor,
        )
        stages.append({"stage": "validation_ready", "seconds": time.perf_counter() - started})
        train_obs, train_info = train.reset()
        stages.append({"stage": "train_reset", "seconds": time.perf_counter() - started, "count": len(train_obs)})
        val_obs, val_info = validation.reset()
        stages.append({"stage": "validation_reset", "seconds": time.perf_counter() - started, "count": len(val_obs)})
        train_results = train.step(["search[shirt]"] * train.num_processes)
        stages.append({"stage": "train_search", "seconds": time.perf_counter() - started, "count": len(train_results[0])})
        val_results = validation.step(["search[shirt]"] * validation.num_processes)
        stages.append({"stage": "validation_search", "seconds": time.perf_counter() - started, "count": len(val_results[0])})
        search_stats = ray.get([actor.stats.remote() for actor in actors])
        passed = (
            len(train_obs) == args.train_batch_size * args.group_size
            and len(val_obs) == args.val_batch_size
            and len(train_results[0]) == args.train_batch_size * args.group_size
            and len(val_results[0]) == args.val_batch_size
        )
        error = None
    except Exception as caught:
        passed = False
        error = repr(caught)
        search_stats = []
    finally:
        if validation is not None:
            validation.close()
        if train is not None:
            train.close()
        stop_actors(actors)
    report = {
        "schema_version": "webshop-environment-stress-v1",
        "created_at": utc_now(),
        "configuration": {
            "train_batch_size": args.train_batch_size,
            "group_size": args.group_size,
            "train_workers": args.train_batch_size * args.group_size,
            "val_workers": args.val_batch_size,
            "search_pool_size": args.pool_size,
            "sessions_per_actor": args.sessions_per_actor,
        },
        "search_ready": ready,
        "stages": stages,
        "search_stats": search_stats,
        "error": error,
        "passed": passed,
    }
    write_json(args.output, report)
    return passed


def execute_session(worker, session_id):
    initial_obs, initial_info = worker.reset(session_id)
    search_obs, search_reward, search_done, search_info = worker.step("search[shirt]")
    product = next(
        value for value in search_info["available_actions"]["clickables"]
        if len(value) == 10 and value.isalnum()
    )
    product_obs, product_reward, product_done, product_info = worker.step(f"click[{product}]")
    replay_obs, replay_done, replay_info, replay_actions = worker.replay(
        session_id, ["search[shirt]", f"click[{product}]"]
    )
    return {
        "initial": [initial_obs, initial_info],
        "search": [search_obs, search_reward, search_done, search_info],
        "product": [product_obs, product_reward, product_done, product_info],
        "replay": [replay_obs, replay_done, replay_info, replay_actions],
    }


class PackedSlotAdapter:
    def __init__(self, packed, slot):
        self.packed = packed
        self.slot = slot

    def reset(self, session_id):
        return self.packed.reset_batch([(self.slot, session_id)])[0]

    def step(self, action):
        return self.packed.step_batch([(self.slot, action)])[0]

    def replay(self, session_id, prefix_actions):
        return self.packed.replay_batch([(self.slot, session_id, prefix_actions)])[0]


def run_semantic(args):
    actors, ready = create_webshop_search_pool(1, num_cpus=args.search_cpus)
    local = WebshopWorker(seed=0, env_kwargs=env_kwargs())
    packed = PackedWebshopWorker(
        [0] * len(args.sessions),
        env_kwargs(),
        [actors[0]] * len(args.sessions),
        list(range(len(args.sessions))),
    )
    rows = []
    try:
        for slot, session_id in enumerate(args.sessions):
            local_result = execute_session(local, session_id)
            shared_result = execute_session(PackedSlotAdapter(packed, slot), session_id)
            rows.append({
                "session_id": session_id,
                "local": local_result,
                "shared": shared_result,
                "identical": local_result == shared_result,
            })
    finally:
        local.close()
        packed.close()
        stop_actors(actors)
    report = {
        "schema_version": "webshop-semantic-parity-v1",
        "created_at": utc_now(),
        "search_ready": ready,
        "sessions": rows,
        "passed": all(row["identical"] for row in rows),
    }
    write_json(args.output, report)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["parity", "search-stress", "env-stress", "semantic"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--search-cpus", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--session-cpus", type=float, default=0.05)
    parser.add_argument("--init-batch-size", type=int, default=16)
    parser.add_argument("--sessions-per-actor", type=int, default=8)
    parser.add_argument("--sessions", type=int, nargs="+", default=[500, 501, 502])
    args = parser.parse_args()
    passed = {
        "parity": run_parity,
        "search-stress": run_search_stress,
        "env-stress": run_env_stress,
        "semantic": run_semantic,
    }[args.mode](args)
    print(json.dumps({"mode": args.mode, "output": str(args.output), "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
