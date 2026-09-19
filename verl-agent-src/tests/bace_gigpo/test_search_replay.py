from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.environments.env_package.search.replay_state import (
    decode_search_reset_key,
    encode_search_reset_key,
)
import agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search as search_tool_module
from agent_system.environments.strict_actions import (
    search_action_identity,
    search_action_identity_kind,
    search_action_is_executable,
)
from recipe.bace_gigpo.anchor_index import AnchorIndex
from recipe.bace_gigpo.coordinator import ExactBatchErvCoordinator
from recipe.bace_gigpo.replay.validator import ReplayValidator
from recipe.bace_gigpo.types import ReplayRequest, RootEvent, RootEventLog


def test_search_projection_and_exact_executed_identity():
    raw = "<think>x</think><SEARCH>  The Godfather director  </SEARCH> trailing"
    projected, valid = search_projection([raw])
    assert projected == ["<search>The Godfather director</search>"]
    assert valid == [1]
    assert search_action_identity(raw, projected[0], True, True) == (
        "valid::<search>The Godfather director</search>"
    )
    assert search_action_identity_kind(projected[0], True) == "valid"
    assert search_action_is_executable(projected[0])


def test_answer_is_terminal_and_malformed_or_empty_actions_are_unparsed():
    actions = [
        "<think>x</think><answer>Paris</answer>",
        "<search></search>",
        "<search>a</search><answer>b</answer>",
        "plain text",
    ]
    projected, valid = search_projection(actions)
    assert valid == [1, 0, 0, 0]
    assert search_action_identity_kind(projected[0], True) == "terminal"
    assert not search_action_is_executable(projected[0])
    assert all(
        search_action_identity(raw, action, bool(ok), bool(ok)) is None
        for raw, action, ok in zip(actions[1:], projected[1:], valid[1:])
    )


def _event(occurrence, observation, action, kind="valid", step_index=1):
    return RootEvent(
        occurrence_id=occurrence,
        step_index=step_index,
        pre_action_observation=observation,
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        response_loss_mask=(1,),
        old_log_probs=(-0.1,),
        raw_model_response=action,
        parsed_environment_action=action,
        canonical_action=f"valid::{action}",
        admissible_actions=(),
        post_action_observation="information",
        reward=0.0,
        done=kind == "terminal",
        remaining_horizon=2,
        action_identity=f"valid::{action}",
        action_identity_kind=kind,
        action_format_valid=True,
        action_environment_valid=True,
    )


def _root(root_id, observation, action, kind="valid"):
    return RootEventLog(
        task_id="question-group",
        task_family="search",
        episode_group_id="question-group",
        root_id=root_id,
        environment_reset_key=encode_search_reset_key({
            "question": "Who directed The Godfather?",
            "ground_truth": {"target": ["Francis Ford Coppola"]},
            "data_source": "hotpotqa",
        }),
        task_description="Who directed The Godfather?",
        task_batch_index=0,
        events=(_event(f"{root_id}:1", observation, action, kind),),
        terminal_reward=0.0,
        won=False,
    )


def test_search_anchor_uses_gigpo_similarity_and_excludes_answer_candidates():
    roots = [
        _root("r1", "Doc 1: Francis Ford Coppola directed The Godfather.", "<search>The Godfather director</search>"),
        _root("r2", "Doc 1: Francis Ford Coppola directed The Godfather!", "<search>Coppola nationality</search>"),
        _root("r3", "Doc 1: Francis Ford Coppola directed The Godfather.", "<answer>Francis Ford Coppola</answer>", "terminal"),
    ]
    assert AnchorIndex(roots).anchors_for_task("question-group") == []
    index = AnchorIndex(
        roots, anchor_similarity_enabled=True, anchor_similarity_threshold=0.9
    )
    anchors = index.anchors_for_task("question-group")
    assert len(anchors) == 1
    assert anchors[0].observed_action_ids == [
        "valid::<search>Coppola nationality</search>",
        "valid::<search>The Godfather director</search>",
    ]
    assert index.anchor_for_observation(
        "question-group", "Doc 1: Francis Ford Coppola directed The Godfather!"
    ) == anchors[0]


def test_search_can_opt_in_to_initial_anchor_without_changing_default():
    roots = [
        RootEventLog(
            **{
                **_root("r1", "unused", "<search>first query</search>").__dict__,
                "events": (_event("r1:0", "Who directed it?", "<search>first query</search>", step_index=0),),
            }
        ),
        RootEventLog(
            **{
                **_root("r2", "unused", "<search>second query</search>").__dict__,
                "events": (_event("r2:0", "Who directed it?", "<search>second query</search>", step_index=0),),
            }
        ),
    ]
    assert AnchorIndex(roots).anchors_for_task("question-group") == []
    anchors = AnchorIndex(
        roots, allow_initial_search_anchor=True
    ).anchors_for_task("question-group")
    assert len(anchors) == 1
    assert anchors[0].observed_action_ids == [
        "valid::<search>first query</search>",
        "valid::<search>second query</search>",
    ]


def test_initial_anchor_opt_in_does_not_change_non_search_tasks():
    roots = [
        RootEventLog(
            **{
                **_root("r1", "unused", "<search>first query</search>").__dict__,
                "task_family": "webshop",
                "events": (_event("r1:0", "initial", "<search>first query</search>", step_index=0),),
            }
        ),
        RootEventLog(
            **{
                **_root("r2", "unused", "<search>second query</search>").__dict__,
                "task_family": "webshop",
                "events": (_event("r2:0", "initial", "<search>second query</search>", step_index=0),),
            }
        ),
    ]
    assert AnchorIndex(
        roots, allow_initial_search_anchor=True
    ).anchors_for_task("question-group") == []


def test_similarity_allocation_replays_the_concrete_origin_not_cluster_rep():
    roots = [
        _root("r1", "Doc 1: Francis Ford Coppola directed The Godfather.", "<search>The Godfather director</search>"),
        _root("r2", "Doc 1: Francis Ford Coppola directed The Godfather!", "<search>Coppola nationality</search>"),
    ]
    coordinator = ExactBatchErvCoordinator(
        max_branches_per_anchor=2,
        prior_strength=2.0,
        threshold=0.0,
        anchor_similarity_enabled=True,
        anchor_similarity_threshold=0.9,
    )
    coordinator.initialize(
        roots,
        branch_quota_by_task={"question-group": 1},
        prior_mean_by_task={"question-group": 0.5},
    )
    request = coordinator.build_round_requests()[0]
    assert request.expected_anchor_key == request.expected_observation
    assert request.expected_anchor_key in {
        "Doc 1: Francis Ford Coppola directed The Godfather.",
        "Doc 1: Francis Ford Coppola directed The Godfather!",
    }


class FakeSelectedSearchVector:
    num_processes = 4
    group_n = 4

    def __init__(self):
        self.specs = {}
        self.turns = {}

    def reset_selected(self, indices, specs):
        for index, spec in zip(indices, specs):
            self.specs[index] = spec
            self.turns[index] = 0
        return [spec["question"] for spec in specs], [
            {"data_source": spec["data_source"]} for spec in specs
        ]

    def step_selected(self, indices, actions):
        observations, rewards, dones, infos = [], [], [], []
        for index, action in zip(indices, actions):
            self.turns[index] += 1
            done = action.startswith("<answer>")
            observations.append("" if done else f"<information>{action}</information>")
            rewards.append(1.0 if done else 0.0)
            dones.append(done)
            infos.append({"won": done, "data_source": self.specs[index]["data_source"]})
        return observations, rewards, dones, infos


def _request(reset_key):
    return ReplayRequest(
        request_id="request-1", task_id="task-1", branch_id="branch-1",
        origin_occurrence_id="root:1", environment_reset_key=reset_key,
        task_description="Who directed The Godfather?", task_batch_index=0,
        target_turn=1,
        parsed_action_prefix=("<search>The Godfather director</search>",),
        prefix_observations=("Who directed The Godfather?",),
        expected_anchor_key="<information><search>The Godfather director</search></information>",
        expected_observation="<information><search>The Godfather director</search></information>",
        expected_action_set=(),
        selected_canonical_action="valid::<search>Coppola nationality</search>",
        copied_response_token_ids=(1,), copied_raw_model_response="<search>Coppola nationality</search>",
        copied_response_loss_mask=(1,), copied_old_log_probs=(-0.1,),
        original_prompt_token_ids=(2,), remaining_horizon=2,
        copied_parsed_environment_action="<search>Coppola nationality</search>",
        copied_action_identity="valid::<search>Coppola nationality</search>",
        copied_action_identity_kind="valid", copied_action_environment_valid=True,
        expected_post_action_observation="<information><search>Coppola nationality</search></information>",
    )


def test_search_selected_worker_replay_restores_task_prefix_and_identity():
    spec = {
        "question": "Who directed The Godfather?",
        "ground_truth": {"target": ["Francis Ford Coppola"]},
        "data_source": "hotpotqa",
    }
    reset_key = encode_search_reset_key(spec)
    assert decode_search_reset_key(reset_key) == spec
    manager = SearchEnvironmentManager(
        FakeSelectedSearchVector(), search_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=4)),
    )
    request = _request(reset_key)
    observations, dones, infos = manager.replay_selected([3], [request])
    assert not dones[0]
    assert observations["anchor"] == [request.expected_anchor_key]
    assert ReplayValidator(
        compare_action_set=True,
        action_is_executable=manager.is_action_executable,
    ).validate(request, observations["anchor"][0], (), False).replay_ok

    next_obs, rewards, dones, infos = manager.step_selected(
        [3], ["<think>x</think><search>Coppola nationality</search>"]
    )
    assert np.asarray(rewards).tolist() == [0.0]
    assert not dones[0]
    assert infos[0]["action_identity"] == request.copied_action_identity
    assert infos[0]["environment_reset_key"] == reset_key
    assert next_obs["anchor"] == [request.expected_post_action_observation]


def test_search_selected_worker_replay_supports_initial_anchor_empty_prefix():
    spec = {
        "question": "Who directed The Godfather?",
        "ground_truth": {"target": ["Francis Ford Coppola"]},
        "data_source": "hotpotqa",
    }
    reset_key = encode_search_reset_key(spec)
    manager = SearchEnvironmentManager(
        FakeSelectedSearchVector(), search_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=4)),
    )
    request = replace(
        _request(reset_key),
        target_turn=0,
        parsed_action_prefix=(),
        prefix_observations=(),
        expected_anchor_key=spec["question"],
        expected_observation=spec["question"],
        remaining_horizon=3,
    )
    observations, dones, _ = manager.replay_selected([2], [request])
    assert not dones[0]
    assert observations["anchor"] == [spec["question"]]
    assert ReplayValidator(
        compare_action_set=True,
        action_is_executable=manager.is_action_executable,
    ).validate(request, observations["anchor"][0], (), False).replay_ok


def test_search_staged_reset_uses_explicit_parquet_task_specs():
    spec = {
        "question": "Who directed The Godfather?",
        "ground_truth": {"target": ["Francis Ford Coppola"]},
        "data_source": "hotpotqa",
    }
    vector = FakeSelectedSearchVector()
    manager = SearchEnvironmentManager(
        vector, search_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=4)),
    )
    observations, infos = manager.reset({
        "_bace_worker_indices": [2],
        "_bace_reset_keys": [None],
        "_bace_reset_specs": [spec],
    })
    assert vector.specs[2] == spec
    assert observations["anchor"] == [spec["question"]]
    assert decode_search_reset_key(infos[0]["environment_reset_key"]) == spec


def test_search_reset_key_normalizes_parquet_numpy_ground_truth():
    reset_key = encode_search_reset_key({
        "question": "Who directed The Godfather?",
        "ground_truth": {"target": np.asarray(["Francis Ford Coppola"], dtype=object)},
        "data_source": "nq",
    })
    assert decode_search_reset_key(reset_key)["ground_truth"] == {
        "target": ["Francis Ford Coppola"]
    }


def test_search_tool_reuses_successful_retrieval_response(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["query"])
        return {"result": []}, None

    monkeypatch.setattr(search_tool_module, "call_search_api", fake_call)
    group = search_tool_module.SearchToolGroup.__new__(search_tool_module.SearchToolGroup)
    group.search_url = "http://fixed-retriever/retrieve"
    group.topk = 3
    group.timeout = 1
    group.log_requests = False
    group.session = object()
    group._response_cache.clear()
    first = group.search("same exact query")
    second = group.search("same exact query")
    assert first == second
    assert calls == ["same exact query"]
