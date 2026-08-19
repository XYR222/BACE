import sys
from types import SimpleNamespace

import pytest

from agent_system.environments.env_manager import WebshopEnvironmentManager

if "gym" not in sys.modules:
    sys.modules["gym"] = SimpleNamespace(Env=object)

from agent_system.environments.env_package.webshop.envs import WebshopWorker
from agent_system.environments.env_package.webshop.projection import webshop_projection


class FakeWebshopEnv:
    def __init__(self):
        self.session = None
        self.turn = 0

    def reset(self, session):
        self.session = int(session)
        self.turn = 0
        return self._obs(), {}

    def step(self, action):
        self.turn += 1
        done = action == "buy"
        reward = 1.0 if done else 0.0
        return self._obs(), reward, done, {}

    def _obs(self):
        return f"site [SEP] Instruction: [SEP] task-{self.session} [SEP] page-{self.turn}"

    def get_available_actions(self):
        return {
            "has_search_bar": self.turn == 0,
            "clickables": [f"item-{self.session}", f"turn-{self.turn}"],
        }


def worker_with_fake_env():
    worker = WebshopWorker.__new__(WebshopWorker)
    worker.env = FakeWebshopEnv()
    worker._last_session_idx = None
    return worker


def test_worker_replays_exact_session_and_restores_available_actions():
    worker = worker_with_fake_env()
    obs, done, info, available = worker.replay(23, ["search[x]", "click[item]"])

    assert "task-23" in obs
    assert "page-2" in obs
    assert not done
    assert info["session_idx"] == 23
    assert info["available_actions"] == available
    assert available["clickables"] == ["item-23", "turn-2"]


def test_worker_rejects_early_terminal():
    worker = worker_with_fake_env()
    with pytest.raises(RuntimeError, match="terminated early"):
        worker.replay(7, ["buy", "click[next]"])


def test_worker_sessions_are_isolated():
    first = worker_with_fake_env()
    second = worker_with_fake_env()
    first.replay(11, ["click[a]"])
    second.replay(29, ["click[b]", "click[c]"])

    assert first._last_session_idx == 11
    assert first.env.turn == 1
    assert second._last_session_idx == 29
    assert second.env.turn == 2


class FakeVectorWebshop:
    num_processes = 2

    def replay(self, session_ids, prefixes):
        assert session_ids == [42]
        assert prefixes == [["search[red shoes]"]]
        obs = ["site [SEP] Instruction: [SEP] find red shoes [SEP] results"]
        available = [{"has_search_bar": True, "clickables": ["red shoe"]}]
        infos = [{"available_actions": available[0], "session_idx": 42}]
        return obs, [False], infos, available


class FakeStagedVectorWebshop:
    def __init__(self):
        self.calls = []

    def reset(self):
        raise AssertionError("staged reset must not reset the full worker pool")

    def reset_subset(self, worker_indices, session_ids):
        self.calls.append((worker_indices, session_ids))
        observations = [
            f"site [SEP] Instruction: [SEP] task-{session_id} [SEP] home"
            for session_id in session_ids
        ]
        infos = [
            {
                "available_actions": {"has_search_bar": True, "clickables": []},
                "session_idx": session_id,
            }
            for session_id in session_ids
        ]
        return observations, infos


def test_manager_rebuilds_memory_prompt_and_action_set():
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = WebshopEnvironmentManager(FakeVectorWebshop(), webshop_projection, config)
    request = SimpleNamespace(
        environment_reset_key="42",
        parsed_action_prefix=("search[red shoes]",),
        prefix_observations=("'home'",),
        task_description="find red shoes",
    )

    observations, dones, infos = manager.replay([request])

    assert dones.tolist() == [False]
    assert observations["anchor"] == ["'results'"]
    assert observations["admissible_actions"] == [
        ["search[<your query>]", "click[red shoe]"]
    ]
    assert manager.memory[0] == [
        {"text_obs": "'home'", "action": "search[red shoes]"}
    ]
    assert "search[red shoes]" in observations["text"][0]
    assert "'results'" in observations["text"][0]
    assert infos[0]["session_idx"] == 42


def test_manager_staged_reset_uses_only_requested_workers_and_sessions():
    envs = FakeStagedVectorWebshop()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = WebshopEnvironmentManager(envs, webshop_projection, config)

    observations, infos = manager.reset({
        "_bace_worker_indices": [2],
        "_bace_reset_keys": [503],
    })

    assert envs.calls == [([2], [503])]
    assert len(observations["text"]) == 1
    assert observations["admissible_actions"] == [["search[<your query>]"]]
    assert infos[0]["session_idx"] == 503
