import sys
from types import SimpleNamespace

import pytest

from agent_system.environments.env_manager import WebshopEnvironmentManager

if "gym" not in sys.modules:
    sys.modules["gym"] = SimpleNamespace(Env=object)

from agent_system.environments.env_package.webshop.envs import WebshopWorker
from agent_system.environments.env_package.webshop.projection import (
    webshop_action_identity,
    webshop_action_is_executable,
    webshop_projection,
)
from agent_system.environments.strict_actions import (
    parse_webshop_environment_action,
)
from recipe.bace_gigpo.replay.validator import ReplayCategory, ReplayValidator


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


class FakeSelectedVectorWebshop:
    num_processes = 4

    def __init__(self):
        self.sessions = {}
        self.turns = {}

    @staticmethod
    def _available(turn):
        return {
            "has_search_bar": turn == 0,
            "clickables": ["red shoe", f"turn-{turn}"],
        }

    def _result(self, worker_index):
        session = self.sessions[worker_index]
        turn = self.turns[worker_index]
        obs = f"site [SEP] Instruction: [SEP] task-{session} [SEP] page-{turn}"
        return obs, {
            "available_actions": self._available(turn),
            "session_idx": session,
            "won": False,
            "task_score": 0.0,
        }

    def reset_selected(self, worker_indices, session_ids):
        output = []
        for worker_index, session_id in zip(worker_indices, session_ids):
            self.sessions[worker_index] = int(session_id)
            self.turns[worker_index] = 0
            output.append(self._result(worker_index))
        observations, infos = zip(*output)
        return list(observations), list(infos)

    def replay_selected(self, worker_indices, session_ids, prefixes):
        observations, infos = self.reset_selected(worker_indices, session_ids)
        output = []
        for worker_index, prefix in zip(worker_indices, prefixes):
            self.turns[worker_index] = len(prefix)
            observation, info = self._result(worker_index)
            output.append((observation, False, info, info["available_actions"]))
        observations, dones, infos, available = zip(*output)
        return list(observations), list(dones), list(infos), list(available)

    def step_selected(self, worker_indices, actions):
        output = []
        for worker_index, action in zip(worker_indices, actions):
            self.turns[worker_index] += 1
            observation, info = self._result(worker_index)
            output.append((observation, 0.0, False, info))
        observations, rewards, dones, infos = zip(*output)
        return list(observations), list(rewards), list(dones), list(infos)


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


def test_projection_preserves_raw_responses_and_builds_strict_identity():
    raw = "<think>Find the requested item.</think><action>Search[Red Shoes]</action>"
    responses = [raw]
    projected, valid = webshop_projection(responses)

    assert responses == [raw]
    assert projected == ["search[red shoes]"]
    assert valid == [1]
    assert webshop_action_identity(raw, projected[0], True, True) == (
        "valid::search[red shoes]"
    )
    assert webshop_action_identity(raw, projected[0], True, False) == (
        "invalid::Search[Red Shoes]"
    )


def test_webshop_search_template_is_executable_but_clicks_are_exact():
    action_set = ["search[<your query>]", "click[red shoe]"]
    assert webshop_action_is_executable("search[red waterproof shoes]", action_set)
    assert webshop_action_is_executable("click[red shoe]", action_set)
    assert not webshop_action_is_executable("search[]", action_set)
    assert not webshop_action_is_executable("click[blue shoe]", action_set)


def test_webshop_dependency_light_parser_preserves_upstream_match_semantics():
    assert parse_webshop_environment_action("click[item]") == ("click", "item")
    assert parse_webshop_environment_action("click[item]trailing text") == (
        "click", "item"
    )
    assert parse_webshop_environment_action("search[red shoes]trailing text") == (
        "search", "red shoes"
    )
    assert parse_webshop_environment_action("search[]") == ("search[]", None)
    assert parse_webshop_environment_action("not-an-action") == (
        "not-an-action", None
    )


def test_webshop_executability_matches_real_dispatch_corner_cases():
    action_set = ["search[<your query>]", "click[search]", "click[item]"]

    # The upstream parser ignores text after the matched closing bracket.
    assert webshop_action_is_executable("click[item]trailing text", action_set)
    assert webshop_action_is_executable("search[query]trailing text", ())

    # WebShop can dispatch search even if the rendered page has no search-bar
    # template, while click[search] is explicitly rejected by env.step().
    assert webshop_action_is_executable("search[query]", ())
    assert not webshop_action_is_executable("click[search]", action_set)


def test_webshop_valid_identity_uses_the_environment_parsed_action():
    raw = (
        "<think>Open the item.</think>"
        "<action>click[item]trailing text</action>"
    )
    assert webshop_action_identity(
        raw, "click[item]trailing text", True, True
    ) == "valid::click[item]"


def test_selected_manager_rejects_ambiguous_slots_and_unreset_steps():
    manager = WebshopEnvironmentManager(
        FakeSelectedVectorWebshop(), webshop_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=2)),
    )

    with pytest.raises(ValueError, match="unique"):
        manager.reset_selected([1, 1], [503, 509])
    with pytest.raises(ValueError, match="have not been reset"):
        manager.step_selected([2], ["<think>x</think><action>search[x]</action>"])
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        manager.reset_selected([4], [503])


def test_replay_validator_uses_webshop_executability_callback():
    request = SimpleNamespace(
        request_id="r1",
        expected_anchor_key="page",
        copied_action_environment_valid=True,
        copied_parsed_environment_action="search[red shoes]",
        selected_canonical_action="search[red shoes]",
        expected_action_set=("search[<your query>]",),
    )
    validator = ReplayValidator(
        compare_action_set=True,
        action_is_executable=webshop_action_is_executable,
    )
    result = validator.validate(
        request, "page", ["search[<your query>]"], done=False
    )
    assert result.category == ReplayCategory.VALIDATED.value


def test_selected_slots_restore_and_step_independently_with_strict_metadata():
    envs = FakeSelectedVectorWebshop()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = WebshopEnvironmentManager(envs, webshop_projection, config)

    reset_obs, reset_infos = manager.reset_selected([3, 1], [503, 509])
    assert manager.get_tasks_selected([3, 1]) == ["task-503", "task-509"]
    assert reset_obs["admissible_actions"] == [
        ["search[<your query>]", "click[red shoe]", "click[turn-0]"],
        ["search[<your query>]", "click[red shoe]", "click[turn-0]"],
    ]

    requests = [
        SimpleNamespace(
            environment_reset_key="503",
            parsed_action_prefix=("search[red shoes]",),
            prefix_observations=("'page-0'",),
            task_description="task-503",
        ),
        SimpleNamespace(
            environment_reset_key="509",
            parsed_action_prefix=(),
            prefix_observations=(),
            task_description="task-509",
        ),
    ]
    restored, dones, _ = manager.replay_selected([3, 1], requests)
    assert not dones.any()
    assert restored["anchor"] == ["'page-1'", "'page-0'"]

    responses = [
        "<think>Open it.</think><action>click[red shoe]</action>",
        "<think>Search.</think><action>search[blue shoes]</action>",
    ]
    stepped, _, _, infos = manager.step_selected([3, 1], responses)
    assert stepped["anchor"] == ["'page-2'", "'page-1'"]
    assert [info["is_action_environment_valid"].item() for info in infos] == [True, True]
    assert [info["action_identity"] for info in infos] == [
        "valid::click[red shoe]",
        "valid::search[blue shoes]",
    ]
    assert envs.sessions == {3: 503, 1: 509}
