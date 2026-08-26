from types import MethodType, SimpleNamespace

import numpy as np
import torch

from recipe.bace_gigpo import rollout_collector as collector_module
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector
from verl import DataProto


def _request(index, *, horizon=2, origin=None):
    origin = origin or f"origin-{index}"
    return SimpleNamespace(
        request_id=f"request-{index}",
        branch_id=f"branch-{index}",
        origin_occurrence_id=origin,
        parsed_action_prefix=(f"prefix-{index}",),
        copied_raw_model_response=f"origin-action-{index}",
        copied_action_identity=f"valid::origin-{index}",
        copied_action_identity_kind="valid",
        copied_response_token_ids=(index,),
        copied_response_loss_mask=(1,),
        copied_old_log_probs=(0.0,),
        selected_canonical_action=f"valid::origin-{index}",
        task_id=f"task-{index}",
        task_batch_index=index,
        environment_reset_key=f"game-{index}",
        task_description=f"task description {index}",
        target_turn=0,
        remaining_horizon=horizon,
        expected_immediate_reward=0.0,
        expected_post_action_done=False,
    )


def _result(request, ok=True):
    return SimpleNamespace(
        replay_ok=ok,
        category="VALIDATED" if ok else "OBSERVATION_MISMATCH",
        error_message="" if ok else "mismatch",
        request_id=request.request_id,
    )


class _FakeBatch:
    def __init__(self, occurrence_ids):
        self.non_tensor_batch = {
            "occurrence_id": np.asarray(occurrence_ids, dtype=object),
        }

    def select_idxs(self, indices):
        return _FakeBatch([
            self.non_tensor_batch["occurrence_id"][index] for index in indices
        ])


class _FakeGenBatch:
    def __init__(self, size):
        self.indices = list(range(size))
        self.meta_info = {}

    def select_idxs(self, indices):
        selected = _FakeGenBatch(0)
        selected.indices = [self.indices[index] for index in indices]
        return selected


def _selected_collector(requests, replay_adapter, branch_envs):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.variant = "batch_erv_exact"
    collector.branch_execution_mode = "selected_worker"
    collector.branch_envs = branch_envs
    collector.replay_adapter = replay_adapter
    collector.max_origin_retries = 1
    collector.replay_retry_metadata = {}
    collector.trace_diagnostics = {}
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.coordinator = SimpleNamespace(adopt_retry_request=lambda *_: None)
    collector._refresh_success_rate = lambda *_: None
    collector._set_lineage = lambda *_: None

    def no_suffix(self, gen_batch, actor, selected, slots, rewards, dones):
        del gen_batch, actor, slots, dones
        return None, np.asarray(rewards, dtype=np.float32), np.zeros(len(selected), dtype=bool)

    collector._collect_suffixes_selected = MethodType(no_suffix, collector)
    root_output = _FakeBatch([request.origin_occurrence_id for request in requests])
    return collector, root_output


def test_selected_happy_path_replays_once_and_steps_restored_slots_directly():
    requests = [_request(index) for index in range(3)]

    class Replay:
        def __init__(self):
            self.calls = []

        def replay_selected_and_validate(self, slots, selected):
            self.calls.append((list(slots), list(selected)))
            return None, None, [_result(request) for request in selected]

        def validate_transitions(self, selected, observations, rewards, dones, infos):
            del observations, rewards, dones, infos
            return [_result(request) for request in selected]

    class Envs:
        def __init__(self):
            self.step_calls = []

        def step_selected(self, slots, actions):
            self.step_calls.append((list(slots), list(actions)))
            size = len(slots)
            return (
                {"anchor": [f"post-{slot}" for slot in slots]},
                np.zeros(size),
                np.ones(size, dtype=bool),
                [{} for _ in slots],
            )

    replay = Replay()
    envs = Envs()
    collector, root_output = _selected_collector(requests, replay, envs)
    valid, _, _, _ = collector._execute_chunk_selected(
        root_output, _FakeGenBatch(3), None, requests
    )

    assert [request.branch_id for request in valid] == [
        request.branch_id for request in requests
    ]
    assert len(replay.calls) == 1
    assert replay.calls[0][0] == [0, 1, 2]
    assert envs.step_calls == [
        ([0, 1, 2], [request.copied_raw_model_response for request in requests])
    ]
    assert collector.orchestration_metrics["branch_execution_restore_replay_steps"] == 0
    assert collector.orchestration_metrics["branch_total_mechanical_replay_steps"] == 3
    assert collector.orchestration_metrics["branch_origin_transition_steps"] == 3


def test_selected_retry_replays_only_the_failed_worker(monkeypatch):
    requests = [_request(index) for index in range(3)]
    replacement = _request(11, origin="fallback-origin")
    replacement.branch_id = requests[1].branch_id
    replacement.task_batch_index = requests[1].task_batch_index
    monkeypatch.setattr(collector_module, "build_root_event_logs", lambda _: [])

    class Replay:
        def __init__(self):
            self.calls = []

        def replay_selected_and_validate(self, slots, selected):
            self.calls.append((list(slots), [request.request_id for request in selected]))
            if len(self.calls) == 1:
                return None, None, [
                    _result(selected[0]),
                    _result(selected[1], ok=False),
                    _result(selected[2]),
                ]
            return None, None, [_result(selected[0])]

        def validate_transitions(self, selected, observations, rewards, dones, infos):
            del observations, rewards, dones, infos
            return [_result(request) for request in selected]

    class Envs:
        def step_selected(self, slots, actions):
            size = len(slots)
            return (
                {"anchor": [f"post-{slot}" for slot in slots]},
                np.zeros(size),
                np.ones(size, dtype=bool),
                [{} for _ in slots],
            )

    replay = Replay()
    collector, root_output = _selected_collector(requests, replay, Envs())
    collector._fallback_request = lambda *_: replacement
    root_output.non_tensor_batch["occurrence_id"] = np.asarray(
        [requests[0].origin_occurrence_id, replacement.origin_occurrence_id, requests[2].origin_occurrence_id],
        dtype=object,
    )
    valid, _, _, _ = collector._execute_chunk_selected(
        root_output, _FakeGenBatch(3), None, requests
    )

    assert replay.calls == [
        ([0, 1, 2], ["request-0", "request-1", "request-2"]),
        ([1], [replacement.request_id]),
    ]
    assert [request.branch_id for request in valid] == [
        "branch-0", "branch-1", "branch-2"
    ]
    assert collector.orchestration_metrics["branch_validation_replay_steps"] == 3
    assert collector.orchestration_metrics["branch_retry_replay_steps"] == 1
    assert collector.orchestration_metrics.get(
        "branch_execution_restore_replay_steps", 0
    ) == 0


def test_selected_suffix_compacts_terminal_and_horizon_expired_slots():
    requests = [
        _request(0, horizon=3),
        _request(1, horizon=3),
        _request(2, horizon=1),
    ]

    class Envs:
        def __init__(self):
            self.step_calls = []

        def get_observations_selected(self, slots):
            return {
                "text": [f"obs-{slot}" for slot in slots],
                "image": None,
                "anchor": [f"anchor-{slot}" for slot in slots],
                "admissible_actions": [("look",) for _ in slots],
            }

        def step_selected(self, slots, actions):
            del actions
            self.step_calls.append(list(slots))
            call = len(self.step_calls)
            dones = {
                1: [True, False, False],
                2: [False],
                3: [True],
            }[call]
            return (
                {"anchor": [f"post-{slot}-{call}" for slot in slots]},
                np.zeros(len(slots), dtype=np.float32),
                np.asarray(dones, dtype=bool),
                [
                    {
                        "is_action_valid": True,
                        "is_action_format_valid": True,
                        "is_action_environment_valid": True,
                        "projected_action": "look",
                        "action_identity": "valid::look",
                    }
                    for _ in slots
                ],
            )

    class Actor:
        world_size = 1

        def generate_sequences(self, batch):
            batch.batch["responses"] = torch.ones((len(batch), 1), dtype=torch.long)
            return batch

    collector = object.__new__(BaceTrajectoryCollector)
    collector.branch_envs = Envs()
    collector.actor = Actor()
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.tokenizer = SimpleNamespace(
        batch_decode=lambda responses, skip_special_tokens: [
            "look" for _ in range(len(responses))
        ]
    )

    def preprocess(self, gen_batch, obs):
        size = len(obs["text"])
        return DataProto.from_single_dict({
            "input_ids": torch.zeros((size, 2), dtype=torch.long),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
            "position_ids": torch.zeros((size, 2), dtype=torch.long),
            "raw_prompt_ids": np.asarray([[1, 2] for _ in range(size)], dtype=object),
            "anchor_obs": np.asarray(obs["anchor"], dtype=object),
            "index": np.arange(size),
            "data_source": np.asarray(["alfworld"] * size, dtype=object),
        })

    collector.preprocess_batch = MethodType(preprocess, collector)
    gen_batch = _FakeGenBatch(3)
    output, rewards, dones = collector._collect_suffixes_selected(
        gen_batch,
        Actor(),
        requests,
        [0, 1, 2],
        np.zeros(3),
        np.zeros(3, dtype=bool),
    )

    assert collector.branch_envs.step_calls == [[0, 1, 2], [1], [1]]
    assert len(output) == 5
    assert rewards.tolist() == [0.0, 0.0, 0.0]
    assert dones.tolist() == [True, True, False]
    assert collector.orchestration_metrics["branch_suffix_active_sequences"] == 5
    assert collector.orchestration_metrics[
        "branch_suffix_dense_equivalent_sequences"
    ] == 9
    assert collector.orchestration_metrics[
        "branch_suffix_inactive_sequences_avoided"
    ] == 4
    assert collector.orchestration_metrics["branch_suffix_environment_steps"] == 5
    assert collector.orchestration_metrics["branch_suffix_active_efficiency"] == 5 / 9


def test_legacy_executor_remains_available():
    collector = object.__new__(BaceTrajectoryCollector)
    collector.branch_execution_mode = "legacy_dense"
    collector._execute_chunk_legacy = lambda *args, **kwargs: "legacy"
    collector._execute_chunk_selected = lambda *args, **kwargs: "selected"
    assert collector._execute_chunk() == "legacy"
