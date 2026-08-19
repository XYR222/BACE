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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import json
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf
from agent_system.environments.env_package.alfworld.projection import alfworld_action_identity

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        # Frontier mode keeps state by persistent raw-worker id.  The legacy
        # batch-global fields below remain unchanged for existing collectors.
        self._bace_slots = {}
        super().__init__(envs, projection_f, config)

    @staticmethod
    def _extract_one_task(text_obs):
        marker = 'Your task is to: '
        task_start = text_obs.find(marker)
        if task_start == -1:
            raise ValueError("Task description not found in text observation.")
        return text_obs[task_start + len(marker):].strip()

    @staticmethod
    def _history_context(history, history_length):
        recent = history[-history_length:]
        start_idx = len(history) - len(recent)
        lines = []
        for offset, record in enumerate(recent):
            step = start_idx + offset + 1
            lines.append(
                f"[Observation {step}: '{record['text_obs']}', "
                f"Action {step}: '{record['action']}']"
            )
        return "\n".join(lines), len(recent)

    def _build_selected_prompt(self, state, init=False):
        admissible = "\n ".join(
            f"'{action}'" for action in state['admissible_actions'] if action != 'help'
        )
        history = state['history']
        if init or self.config.env.history_length <= 0 or not history:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=state['text_obs'],
                admissible_actions=admissible,
            )
        context, valid_len = self._history_context(
            history, int(self.config.env.history_length)
        )
        return ALFWORLD_TEMPLATE.format(
            task_description=state['task'],
            step_count=len(history),
            history_length=valid_len,
            action_history=context,
            current_step=len(history) + 1,
            current_observation=state['text_obs'],
            admissible_actions=admissible,
        )

    def _selected_observations(self, worker_indices, images=None, init=False):
        states = [self._bace_slots[int(index)] for index in worker_indices]
        return {
            'text': [self._build_selected_prompt(state, init=init) for state in states],
            'image': images,
            'anchor': [state['text_obs'] for state in states],
            'admissible_actions': [state['admissible_actions'] for state in states],
        }

    def reset_selected(self, worker_indices, game_files=None):
        """Reset persistent slots and initialize independent prompt histories."""
        worker_indices = [int(index) for index in worker_indices]
        text_obs, image_obs, infos = self.envs.reset_selected(worker_indices, game_files)
        admissible = self.envs.get_admissible_selected(worker_indices)
        for index, observation, commands, info in zip(
            worker_indices, text_obs, admissible, infos
        ):
            self._bace_slots[index] = {
                'text_obs': observation,
                'task': self._extract_one_task(observation),
                'gamefile': info.get('extra.gamefile'),
                'admissible_actions': tuple(commands),
                'history': [],
                'done': False,
            }
        return self._selected_observations(worker_indices, image_obs, init=True), infos

    def get_observations_selected(self, worker_indices):
        worker_indices = [int(index) for index in worker_indices]
        missing = [index for index in worker_indices if index not in self._bace_slots]
        if missing:
            raise ValueError(f"BACE slots have not been reset: {missing}")
        images = self.envs.getobs_selected(worker_indices) if self.envs.multi_modal else None
        return self._selected_observations(worker_indices, images, init=False)

    def replay_selected(self, worker_indices, requests):
        """Restore prefixes on unused sibling slots and rebuild per-slot memory."""
        worker_indices = [int(index) for index in worker_indices]
        if len(worker_indices) != len(requests):
            raise ValueError("worker_indices must match replay requests")
        game_files = [request.environment_reset_key for request in requests]
        prefixes = [list(request.parsed_action_prefix) for request in requests]
        text_obs, image_obs, dones, infos = self.envs.replay_selected(
            worker_indices, game_files, prefixes
        )
        admissible = self.envs.get_admissible_selected(worker_indices)
        for index, request, observation, commands, done in zip(
            worker_indices, requests, text_obs, admissible, dones
        ):
            if len(request.prefix_observations) != len(request.parsed_action_prefix):
                raise ValueError("Each replay action requires its pre-action observation")
            self._bace_slots[index] = {
                'text_obs': observation,
                'task': request.task_description,
                'gamefile': request.environment_reset_key,
                'admissible_actions': tuple(commands),
                'history': [
                    {'text_obs': pre_obs, 'action': action}
                    for pre_obs, action in zip(
                        request.prefix_observations, request.parsed_action_prefix
                    )
                ],
                'done': bool(done),
            }
        return self._selected_observations(worker_indices, image_obs), np.asarray(dones, dtype=bool), infos

    def step_selected(self, worker_indices, text_actions):
        """Project and step only MODEL_READY slots, preserving all other states."""
        worker_indices = [int(index) for index in worker_indices]
        if len(worker_indices) != len(text_actions):
            raise ValueError("worker_indices must match text_actions")
        states = [self._bace_slots[index] for index in worker_indices]
        action_pools = [tuple(state['admissible_actions']) for state in states]
        actions, format_valids = self.projection_f(text_actions, action_pools)
        environment_valids = [
            bool(valid) and action in pool
            for action, valid, pool in zip(actions, format_valids, action_pools)
        ]
        text_obs, image_obs, rewards, dones, infos = self.envs.step_selected(
            worker_indices, actions
        )
        admissible = self.envs.get_admissible_selected(worker_indices)
        for index, old_state, action, observation, commands, done, info, raw, format_valid, env_valid in zip(
            worker_indices, states, actions, text_obs, admissible, dones, infos,
            text_actions, format_valids, environment_valids,
        ):
            old_state['history'].append({
                'text_obs': old_state['text_obs'],
                'action': action,
            })
            old_state['text_obs'] = observation
            old_state['admissible_actions'] = tuple(commands)
            old_state['done'] = bool(done)
            if info.get('extra.gamefile') is None:
                info['extra.gamefile'] = old_state['gamefile']
            info['is_action_valid'] = to_numpy(format_valid)
            info['is_action_format_valid'] = to_numpy(format_valid)
            info['is_action_environment_valid'] = to_numpy(env_valid)
            info['projected_action'] = action
            info['action_identity'] = alfworld_action_identity(
                raw, action, bool(format_valid), env_valid
            )
        return (
            self._selected_observations(worker_indices, image_obs),
            to_numpy(rewards),
            to_numpy(dones),
            infos,
        )
    
    def reset(self, kwargs):
        staged = kwargs if isinstance(kwargs, dict) and '_bace_worker_indices' in kwargs else None
        if staged is None:
            text_obs, image_obs, infos = self.envs.reset()
        else:
            text_obs, image_obs, infos = self.envs.reset_subset(
                staged['_bace_worker_indices'], staged.get('_bace_reset_keys')
            )
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands, init=True)
        return {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'admissible_actions': admissible_commands,
        }, infos

    def replay(self, requests):
        """Restore concrete natural prefixes in a dedicated BACE branch pool."""
        if not 0 < len(requests) <= self.envs.num_processes:
            raise ValueError("Replay requests must fit the branch environment pool")

        game_files = [request.environment_reset_key for request in requests]
        prefix_actions = [list(request.parsed_action_prefix) for request in requests]
        text_obs, image_obs, dones, infos = self.envs.replay(game_files, prefix_actions)

        self.gamefile = list(game_files)
        self.tasks = [request.task_description for request in requests]
        self.pre_text_obs = list(text_obs)
        self.memory.reset(batch_size=len(requests))
        self.memory.keys = ['text_obs', 'action']
        for env_idx, request in enumerate(requests):
            observations = list(request.prefix_observations)
            if len(observations) != len(request.parsed_action_prefix):
                raise ValueError("Each replay action requires its pre-action observation")
            self.memory._data[env_idx] = [
                {'text_obs': observation, 'action': action}
                for observation, action in zip(observations, request.parsed_action_prefix)
            ]

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands, init=False)
        observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'admissible_actions': admissible_commands,
        }
        return observations, np.asarray(dones, dtype=bool), infos
    
    def step(self, text_actions: List[str]):
        action_pools = [tuple(pool) for pool in self.envs.get_admissible_commands]
        actions, format_valids = self.projection_f(text_actions, action_pools)
        environment_valids = [
            bool(format_valid) and action in action_pool
            for action, format_valid, action_pool in zip(actions, format_valids, action_pools)
        ]
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(format_valids[i])
            info['is_action_format_valid'] = to_numpy(format_valids[i])
            info['is_action_environment_valid'] = to_numpy(environment_valids[i])
            info['projected_action'] = actions[i]
            info['action_identity'] = alfworld_action_identity(
                text_actions[i], actions[i], bool(format_valids[i]), environment_valids[i]
            )

        next_observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'admissible_actions': admissible_commands,
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        staged = kwargs if isinstance(kwargs, dict) and '_bace_worker_indices' in kwargs else None
        if staged is None:
            obs, infos = self.envs.reset()
        else:
            obs, infos = self.envs.reset_subset(
                staged['_bace_worker_indices'], staged.get('_bace_reset_keys')
            )
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        staged = kwargs if isinstance(kwargs, dict) and '_bace_worker_indices' in kwargs else None
        if staged is None:
            obs, infos = self.envs.reset()
        else:
            obs, infos = self.envs.reset_subset(
                staged['_bace_worker_indices'], staged.get('_bace_reset_keys')
            )
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        admissible_actions = [self.format_avail_actions(info['available_actions']) for info in infos]
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy(),
                        'admissible_actions': admissible_actions,
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def replay(self, requests):
        """Restore concrete WebShop sessions and rebuild their prompt histories."""
        if not 0 < len(requests) <= self.envs.num_processes:
            raise ValueError("Replay requests must fit the branch environment pool")

        session_ids = [int(request.environment_reset_key) for request in requests]
        prefix_actions = [list(request.parsed_action_prefix) for request in requests]
        raw_obs, dones, infos, raw_available_actions = self.envs.replay(session_ids, prefix_actions)

        self.tasks = [request.task_description for request in requests]
        obs = self.format_obs(raw_obs)
        self.pre_text_obs = list(obs)
        self.memory.reset(batch_size=len(requests))
        self.memory.keys = ['text_obs', 'action']
        for env_idx, request in enumerate(requests):
            observations = list(request.prefix_observations)
            if len(observations) != len(request.parsed_action_prefix):
                raise ValueError("Each replay action requires its pre-action observation")
            self.memory._data[env_idx] = [
                {'text_obs': observation, 'action': action}
                for observation, action in zip(observations, request.parsed_action_prefix)
            ]

        for info, session_id, available in zip(infos, session_ids, raw_available_actions):
            info['session_idx'] = session_id
            info['available_actions'] = available
        admissible_actions = [self.format_avail_actions(available) for available in raw_available_actions]
        observations = {
            'text': self.build_text_obs(obs, infos, init=False),
            'image': None,
            'anchor': obs.copy(),
            'admissible_actions': admissible_actions,
        }
        return observations, np.asarray(dones, dtype=bool), infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy(),
            'admissible_actions': [
                self.format_avail_actions(info['available_actions']) for info in infos
            ],
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['projected_action'] = actions[i]

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        from agent_system.environments.env_package.webshop.envs import create_webshop_search_pool
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        search_backend = str(getattr(config.env.webshop, 'search_backend', 'local'))
        search_pool_size = int(getattr(config.env.webshop, 'search_pool_size', 16))
        search_actor_num_cpus = float(getattr(config.env.webshop, 'search_actor_num_cpus', 1))
        worker_init_batch_size = int(getattr(config.env.webshop, 'worker_init_batch_size', 16))
        sessions_per_actor = int(getattr(config.env.webshop, 'sessions_per_actor', 1))
        configured_diagnostics = getattr(config.env.webshop, 'diagnostics_dir', None)
        diagnostics_dir = configured_diagnostics or os.path.join(
            config.trainer.rollout_data_dir or config.trainer.default_local_dir,
            'webshop_runtime',
        )
        search_workers = None
        if search_backend == 'ray_shared':
            search_workers, search_ready = create_webshop_search_pool(
                search_pool_size,
                env_kwargs.get('num_products'),
                search_actor_num_cpus,
            )
            os.makedirs(diagnostics_dir, exist_ok=True)
            with open(os.path.join(diagnostics_dir, 'search_pool_ready.json'), 'w', encoding='utf-8') as handle:
                json.dump(search_ready, handle, indent=2, sort_keys=True)
                handle.write('\n')

        common_webshop_kwargs = {
            'env_kwargs': env_kwargs,
            'resources_per_worker': resources_per_worker,
            'search_backend': search_backend,
            'search_pool_size': search_pool_size,
            'search_actor_num_cpus': search_actor_num_cpus,
            'worker_init_batch_size': worker_init_batch_size,
            'sessions_per_actor': sessions_per_actor,
            'diagnostics_dir': diagnostics_dir,
            'search_workers': search_workers,
        }
        _envs = build_webshop_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            owns_search_workers=search_backend == 'ray_shared',
            **common_webshop_kwargs,
        )
        _val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            owns_search_workers=False,
            **common_webshop_kwargs,
        )

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
