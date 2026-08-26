# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)

    def _walk_env_nodes(self):
        pending = [self.env]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            for attribute in ("env", "_env", "batch_env"):
                child = getattr(node, attribute, None)
                if child is not None:
                    pending.append(child)

    def _bind_game_file(self, game_file):
        """Temporarily bind every game-pool wrapper to one concrete game.

        The returned snapshots must be restored after ``env.reset()`` has
        loaded the requested game.  Keeping the singleton iterators installed
        would pin a persistent Ray worker to its first BACE task forever and
        prevent subsequent natural-root resets from sampling new games.
        """
        updated = False
        snapshots = []
        for node in self._walk_env_nodes():
            if hasattr(node, "gamefiles"):
                snapshots.append((node, "gamefiles", node.gamefiles))
                node.gamefiles = [game_file]
                if hasattr(node, "_gamefiles_iterator"):
                    snapshots.append(
                        (node, "_gamefiles_iterator", node._gamefiles_iterator)
                    )
                    node._gamefiles_iterator = itertools.cycle([game_file])
                updated = True
            if hasattr(node, "_game_files"):
                snapshots.append((node, "_game_files", node._game_files))
                node._game_files = [game_file]
                if hasattr(node, "_next_game"):
                    snapshots.append((node, "_game_iterator", node._game_iterator))
                    node._game_iterator = node._next_game()
                updated = True
        if not updated:
            raise RuntimeError("Unable to bind ALFWorld worker to the requested game file")
        return snapshots

    @staticmethod
    def _restore_game_pool(snapshots):
        for node, attribute, value in reversed(snapshots):
            setattr(node, attribute, value)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self, game_file=None):
        """Reset the environment"""
        snapshots = None
        if game_file is not None:
            snapshots = self._bind_game_file(game_file)
        try:
            obs, infos = self.env.reset()
        finally:
            if snapshots is not None:
                self._restore_game_pool(snapshots)
        infos['observation_text'] = obs
        return obs, infos

    def replay(self, game_file, prefix_actions):
        """Reset to a concrete game and replay actions up to an anchor."""
        obs, infos = self.reset(game_file=game_file)
        done = bool(infos.get("won", [False])[0]) if isinstance(infos, dict) else False
        for turn, action in enumerate(prefix_actions):
            obs, _, dones, infos = self.step(action)
            done = bool(dones[0])
            if done and turn < len(prefix_actions) - 1:
                raise RuntimeError(f"Replay terminated early at turn {turn}")
        return obs, infos, done
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]
        self.active_processes = self.num_processes
        self.active_worker_indices = list(range(self.num_processes))

    def _validate_worker_indices(self, worker_indices):
        worker_indices = [int(index) for index in worker_indices]
        if not worker_indices or len(set(worker_indices)) != len(worker_indices):
            raise ValueError("worker_indices must be non-empty and unique")
        if min(worker_indices) < 0 or max(worker_indices) >= self.num_processes:
            raise ValueError("worker_indices must fit the ALFWorld worker pool")
        return worker_indices

    def step_selected(self, worker_indices, actions):
        """Step caller-selected workers without changing the legacy active subset."""
        worker_indices = self._validate_worker_indices(worker_indices)
        if len(actions) != len(worker_indices):
            raise ValueError("actions must match worker_indices")

        futures = [
            self.workers[worker_idx].step.remote(action)
            for worker_idx, action in zip(worker_indices, actions)
        ]

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            worker_idx = worker_indices[i]
            self.prev_admissible_commands[worker_idx] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs_selected(worker_indices)
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def step(self, actions):
        if not 0 < len(actions) <= self.num_processes:
            raise ValueError("The action batch must fit the ALFWorld worker pool")
        self.active_processes = len(actions)
        if len(self.active_worker_indices) != self.active_processes:
            self.active_worker_indices = list(range(self.active_processes))
        return self.step_selected(self.active_worker_indices, actions)

    def reset(self, game_files=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        if game_files is not None and len(game_files) != self.num_processes:
            raise ValueError("game_files must match the number of ALFWorld workers")
        self.active_processes = self.num_processes
        self.active_worker_indices = list(range(self.num_processes))
        for i, worker in enumerate(self.workers):
            game_file = None if game_files is None else game_files[i]
            future = worker.reset.remote(game_file=game_file)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def reset_selected(self, worker_indices, game_files=None):
        """Reset caller-selected workers without changing the legacy active subset."""
        worker_indices = self._validate_worker_indices(worker_indices)
        if game_files is None:
            game_files = [None] * len(worker_indices)
        if len(game_files) != len(worker_indices):
            raise ValueError("game_files must match worker_indices")

        futures = [
            self.workers[worker_idx].reset.remote(game_file=game_file)
            for worker_idx, game_file in zip(worker_indices, game_files)
        ]
        results = ray.get(futures)
        text_obs_list, info_list = [], []
        for worker_idx, (obs, info) in zip(worker_indices, results):
            for key in info.keys():
                info[key] = info[key][0]
            text_obs_list.append(obs[0])
            info_list.append(info)
            self.prev_admissible_commands[worker_idx] = info['admissible_commands']
        image_obs_list = self.getobs_selected(worker_indices) if self.multi_modal else None
        return text_obs_list, image_obs_list, info_list

    def reset_subset(self, worker_indices, game_files=None):
        """Legacy selected reset whose subset is consumed by subsequent step calls."""
        worker_indices = self._validate_worker_indices(worker_indices)
        self.active_worker_indices = worker_indices
        self.active_processes = len(worker_indices)
        return self.reset_selected(worker_indices, game_files)

    def replay(self, game_files, prefix_actions_per_env):
        """Replay one independently sized prefix on every worker."""
        if len(game_files) != len(prefix_actions_per_env) or not 0 < len(game_files) <= self.num_processes:
            raise ValueError("Replay inputs must fit the ALFWorld worker pool")
        self.active_processes = len(game_files)
        self.active_worker_indices = list(range(self.active_processes))

        futures = [
            worker.replay.remote(game_file, list(prefix_actions))
            for worker, game_file, prefix_actions in zip(self.workers[:self.active_processes], game_files, prefix_actions_per_env)
        ]
        results = ray.get(futures)

        text_obs_list = []
        info_list = []
        dones_list = []
        for i, (obs, info, done) in enumerate(results):
            for key in info.keys():
                info[key] = info[key][0]
            text_obs_list.append(obs[0])
            info_list.append(info)
            dones_list.append(done)
            self.prev_admissible_commands[i] = info['admissible_commands']

        image_obs_list = self.getobs() if self.multi_modal else None
        return text_obs_list, image_obs_list, dones_list, info_list

    def replay_selected(self, worker_indices, game_files, prefix_actions_per_env):
        """Replay independent prefixes on caller-selected persistent sibling slots."""
        worker_indices = self._validate_worker_indices(worker_indices)
        if len(game_files) != len(worker_indices) or len(prefix_actions_per_env) != len(worker_indices):
            raise ValueError("Replay inputs must match worker_indices")
        futures = [
            self.workers[worker_idx].replay.remote(game_file, list(prefix_actions))
            for worker_idx, game_file, prefix_actions in zip(
                worker_indices, game_files, prefix_actions_per_env
            )
        ]
        results = ray.get(futures)
        text_obs_list, info_list, dones_list = [], [], []
        for worker_idx, (obs, info, done) in zip(worker_indices, results):
            for key in info.keys():
                info[key] = info[key][0]
            text_obs_list.append(obs[0])
            info_list.append(info)
            dones_list.append(done)
            self.prev_admissible_commands[worker_idx] = info['admissible_commands']
        image_obs_list = self.getobs_selected(worker_indices) if self.multi_modal else None
        return text_obs_list, image_obs_list, dones_list, info_list

    def getobs_selected(self, worker_indices):
        worker_indices = self._validate_worker_indices(worker_indices)
        return ray.get([
            self.workers[worker_idx].getobs.remote() for worker_idx in worker_indices
        ])

    def get_admissible_selected(self, worker_indices):
        worker_indices = self._validate_worker_indices(worker_indices)
        return [self.prev_admissible_commands[index] for index in worker_indices]

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        return self.getobs_selected(self.active_worker_indices)

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return [self.prev_admissible_commands[index] for index in self.active_worker_indices]

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
