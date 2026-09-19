import os
from functools import partial

from omegaconf import OmegaConf


def make_branch_env(config):
    """Build a dedicated one-worker-per-task replay pool."""
    env_name = str(config.env.env_name).lower()
    resources = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    if env_name == "alfworld/alfredtwenv":
        from agent_system.environments.env_manager import AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        )
        raw_envs = build_alfworld_envs(
            config_path,
            config.env.seed + 2000,
            config.data.train_batch_size,
            1,
            resources_per_worker=resources,
            is_train=True,
            env_kwargs={"eval_dataset": config.env.alfworld.eval_dataset},
        )
        return AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)

    if "webshop" in env_name:
        from agent_system.environments.env_manager import WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

        data_dir = os.path.join(
            os.path.dirname(__file__),
            "../../agent_system/environments/env_package/webshop/webshop/data",
        )
        suffix = "_1000" if config.env.webshop.use_small else ""
        raw_envs = build_webshop_envs(
            seed=config.env.seed + 2000,
            env_num=config.data.train_batch_size,
            group_n=1,
            resources_per_worker=resources,
            is_train=True,
            env_kwargs={
                "observation_mode": "text",
                "num_products": None,
                "human_goals": config.env.webshop.human_goals,
                "file_path": os.path.join(data_dir, f"items_shuffle{suffix}.json"),
                "attr_path": os.path.join(data_dir, f"items_ins_v2{suffix}.json"),
            },
            search_backend=str(getattr(config.env.webshop, "search_backend", "local")),
            search_pool_size=int(getattr(config.env.webshop, "search_pool_size", 16)),
            search_actor_num_cpus=float(getattr(config.env.webshop, "search_actor_num_cpus", 0.1)),
            worker_init_batch_size=int(getattr(config.env.webshop, "worker_init_batch_size", 16)),
            sessions_per_actor=int(getattr(config.env.webshop, "sessions_per_actor", 1)),
            diagnostics_dir=getattr(config.env.webshop, "diagnostics_dir", None),
        )
        return WebshopEnvironmentManager(raw_envs, partial(webshop_projection), config)

    if "search" in env_name:
        from agent_system.environments.env_manager import SearchEnvironmentManager
        from agent_system.environments.env_package.search import (
            build_search_envs,
            search_projection,
        )

        raw_envs = build_search_envs(
            seed=config.env.seed + 2000,
            env_num=config.data.train_batch_size,
            group_n=1,
            is_train=True,
            env_config=config.env,
        )
        return SearchEnvironmentManager(raw_envs, partial(search_projection), config)

    raise NotImplementedError(f"BACE replay does not support environment {config.env.env_name}")


make_alfworld_branch_env = make_branch_env
