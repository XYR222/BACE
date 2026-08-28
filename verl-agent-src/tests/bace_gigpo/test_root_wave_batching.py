from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import uuid

from recipe.bace_gigpo import rollout_collector as collector_module
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector
from recipe.bace_gigpo.root_store import build_root_event_logs
from verl import DataProto


def test_pending_root_slots_packs_all_known_missing_slots():
    pending = BaceTrajectoryCollector._pending_root_slots(
        [0, 1, 2],
        generated_by_task={0: 2, 1: 2, 2: 3},
        target_by_task={0: 4, 1: 3, 2: 4},
    )
    assert pending == [(0, 2), (1, 2), (0, 3), (2, 3)]


def test_active_root_executor_submits_only_unfinished_logical_roots():
    class Envs:
        def __init__(self):
            self.step_calls = []
            self.tasks = {0: "task zero", 1: "task one", 2: "task two"}

        def reset_selected(self, slots, reset_keys):
            assert list(slots) == [0, 1, 2]
            assert list(reset_keys) == ["game-0", "game-1", "game-2"]
            return self.get_observations_selected(slots), [
                {"extra.gamefile": reset_key} for reset_key in reset_keys
            ]

        def get_tasks_selected(self, slots):
            return [self.tasks[slot] for slot in slots]

        def get_observations_selected(self, slots):
            return {
                "text": [f"observation-{slot}" for slot in slots],
                "image": None,
                "anchor": [f"anchor-{slot}" for slot in slots],
                "admissible_actions": [("look",) for _ in slots],
            }

        def step_selected(self, slots, actions):
            assert len(slots) == len(actions)
            self.step_calls.append(list(slots))
            dones = ([True, False, False], [False, True], [True])[
                len(self.step_calls) - 1
            ]
            infos = [
                {
                    "won": bool(done),
                    "extra.gamefile": f"game-{slot}",
                    "is_action_valid": True,
                    "is_action_format_valid": True,
                    "is_action_environment_valid": True,
                    "projected_action": "look",
                    "action_identity": "valid::look",
                }
                for slot, done in zip(slots, dones)
            ]
            return (
                {
                    "anchor": [f"post-{slot}" for slot in slots],
                    "text": [f"next-{slot}" for slot in slots],
                    "image": None,
                    "admissible_actions": [("look",) for _ in slots],
                },
                np.zeros(len(slots), dtype=np.float32),
                np.asarray(dones, dtype=bool),
                infos,
            )

        def success_evaluator(self, total_infos, total_batch_list, **_):
            return {
                "success_rate": np.asarray(
                    [float(infos[-1]["won"]) for infos in total_infos]
                )
            }

    class Actor:
        world_size = 4

        def generate_sequences(self, batch):
            batch.batch["responses"] = torch.ones(
                (len(batch), 1), dtype=torch.long
            )
            return batch

    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(env=SimpleNamespace(max_steps=3))
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.tokenizer = SimpleNamespace(
        batch_decode=lambda responses, skip_special_tokens: [
            "look" for _ in range(len(responses))
        ]
    )

    def preprocess(self, gen_batch, observations):
        size = len(observations["text"])
        return DataProto.from_single_dict({
            "input_ids": torch.zeros((size, 2), dtype=torch.long),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
            "position_ids": torch.zeros((size, 2), dtype=torch.long),
            "raw_prompt_ids": np.asarray([[1, 2] for _ in range(size)], dtype=object),
            "anchor_obs": np.asarray(observations["anchor"], dtype=object),
            "index": np.arange(size),
            "data_source": np.asarray(["alfworld"] * size, dtype=object),
        })

    collector.preprocess_batch = MethodType(preprocess, collector)
    gen_batch = DataProto.from_single_dict({
        "prompts": torch.zeros((3, 1), dtype=torch.long),
        "data_source": np.asarray(["alfworld"] * 3, dtype=object),
    })

    rollout = collector._active_root_multi_turn_loop(
        gen_batch=gen_batch,
        actor_rollout_wg=Actor(),
        envs=Envs(),
        worker_indices=[0, 1, 2],
        reset_keys=["game-0", "game-1", "game-2"],
        uid_batch=["task-0", "task-1", "task-2"],
        task_batch_indices=[0, 1, 2],
    )

    rows, rewards, lengths, success, traj_uids, _ = rollout
    assert [len(trajectory) for trajectory in rows] == [1, 3, 2]
    assert rewards.tolist() == [0.0, 0.0, 0.0]
    assert lengths.tolist() == [1.0, 3.0, 2.0]
    assert success["success_rate"].tolist() == [1.0, 1.0, 1.0]
    assert len(set(traj_uids.tolist())) == 3
    assert [row["occurrence_id"] for trajectory in rows for row in trajectory] == [
        f"{traj_uids[0]}:0",
        f"{traj_uids[1]}:0",
        f"{traj_uids[1]}:1",
        f"{traj_uids[1]}:2",
        f"{traj_uids[2]}:0",
        f"{traj_uids[2]}:1",
    ]
    assert collector.orchestration_metrics["root_active_sequences"] == 6.0
    assert collector.orchestration_metrics[
        "root_dense_equivalent_sequences"
    ] == 9.0
    assert collector.orchestration_metrics["root_rows_avoided"] == 3.0
    assert collector.orchestration_metrics["root_model_rows_submitted"] == 12.0
    assert collector.orchestration_metrics["root_padding_rows_submitted"] == 6.0
    assert collector.orchestration_metrics["root_active_efficiency"] == 2 / 3


@pytest.mark.parametrize(
    "logical_count",
    [1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127],
)
def test_active_root_padding_never_creates_logical_environment_rows(logical_count):
    class Envs:
        def __init__(self):
            self.stepped_slots = None

        def reset_selected(self, slots, reset_keys):
            del reset_keys
            return self.get_observations_selected(slots), [{} for _ in slots]

        def get_tasks_selected(self, slots):
            return [f"task-{slot}" for slot in slots]

        def get_observations_selected(self, slots):
            return {
                "text": [f"observation-{slot}" for slot in slots],
                "image": None,
                "anchor": [f"anchor-{slot}" for slot in slots],
                "admissible_actions": [("look",) for _ in slots],
            }

        def step_selected(self, slots, actions):
            assert len(slots) == len(actions) == logical_count
            self.stepped_slots = list(slots)
            return (
                {"anchor": [f"post-{slot}" for slot in slots]},
                np.ones(logical_count, dtype=np.float32),
                np.ones(logical_count, dtype=bool),
                [
                    {
                        "won": True,
                        "extra.gamefile": f"game-{slot}",
                        "is_action_valid": True,
                        "projected_action": "look",
                    }
                    for slot in slots
                ],
            )

        def success_evaluator(self, total_infos, **_):
            return {"success_rate": np.ones(len(total_infos), dtype=np.float32)}

    class Actor:
        world_size = 4

        def __init__(self):
            self.submitted = None

        def generate_sequences(self, batch):
            self.submitted = len(batch)
            batch.batch["responses"] = torch.ones(
                (len(batch), 1), dtype=torch.long
            )
            return batch

    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(env=SimpleNamespace(max_steps=1))
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.tokenizer = SimpleNamespace(
        batch_decode=lambda responses, skip_special_tokens: [
            "look" for _ in range(len(responses))
        ]
    )

    def preprocess(self, gen_batch, observations):
        size = len(observations["text"])
        return DataProto.from_single_dict({
            "input_ids": torch.zeros((size, 2), dtype=torch.long),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
            "position_ids": torch.zeros((size, 2), dtype=torch.long),
            "raw_prompt_ids": np.asarray([[1, 2]] * size, dtype=object),
            "anchor_obs": np.asarray(observations["anchor"], dtype=object),
            "index": np.arange(size),
            "data_source": np.asarray(["alfworld"] * size, dtype=object),
        })

    collector.preprocess_batch = MethodType(preprocess, collector)
    gen_batch = DataProto.from_single_dict({
        "prompts": torch.zeros((logical_count, 1), dtype=torch.long),
        "data_source": np.asarray(["alfworld"] * logical_count, dtype=object),
    })
    actor = Actor()
    envs = Envs()
    rollout = collector._active_root_multi_turn_loop(
        gen_batch=gen_batch,
        actor_rollout_wg=actor,
        envs=envs,
        worker_indices=list(range(logical_count)),
        reset_keys=[f"game-{index}" for index in range(logical_count)],
        uid_batch=[f"task-{index}" for index in range(logical_count)],
        task_batch_indices=list(range(logical_count)),
    )

    expected_submitted = ((logical_count + 3) // 4) * 4
    assert actor.submitted == expected_submitted
    assert envs.stepped_slots == list(range(logical_count))
    assert [len(rows) for rows in rollout[0]] == [1] * logical_count
    assert collector.orchestration_metrics["root_active_sequences"] == logical_count
    assert collector.orchestration_metrics[
        "root_model_rows_submitted"
    ] == expected_submitted
    assert collector.orchestration_metrics[
        "root_padding_rows_submitted"
    ] == expected_submitted - logical_count


def test_collect_root_wave_keeps_dense_executor_as_default():
    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(bace=SimpleNamespace(total_leaf_budget=4))
    )
    collector.orchestration_metrics = {}
    selected = []
    dense_calls = []
    sentinel = object()

    class GenBatch:
        def select_idxs(self, indices):
            selected.append(list(indices))
            return "selected-batch"

    class RawEnvs:
        group_n = 8

    class Envs:
        envs = RawEnvs()

    def dense_loop(**kwargs):
        dense_calls.append(kwargs)
        return ([], [], [], [], [], [])

    collector.vanilla_multi_turn_loop = dense_loop
    collector.gather_rollout_data = lambda **_: sentinel

    result = collector._collect_root_wave(
        gen_batch=GenBatch(),
        actor_rollout_wg="actor",
        envs=Envs(),
        task_indices=[1, 3],
        root_slots=[2, 7],
        task_uids=["unused", "task-1", "unused", "task-3"],
        reset_keys=["game-1", "game-3"],
    )

    assert result is sentinel
    assert selected == [[1, 3]]
    assert len(dense_calls) == 1
    assert dense_calls[0]["gen_batch"] == "selected-batch"
    assert dense_calls[0]["reset_options"] == {
        "_bace_worker_indices": [10, 31],
        "_bace_reset_keys": ["game-1", "game-3"],
    }
    assert dense_calls[0]["uid_batch"] == ["task-1", "task-3"]
    assert dense_calls[0]["task_batch_indices"] == [1, 3]
    assert collector.orchestration_metrics["root_active_executor_enabled"] == 0.0


def test_collect_root_wave_rejects_slot_outside_runtime_group():
    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(bace=SimpleNamespace(total_leaf_budget=8))
    )

    class RawEnvs:
        group_n = 4

    class Envs:
        envs = RawEnvs()

    try:
        collector._collect_root_wave(
            gen_batch=None,
            actor_rollout_wg=None,
            envs=Envs(),
            task_indices=[0],
            root_slots=[4],
            task_uids=["task-0"],
            reset_keys=["game-0"],
        )
    except ValueError as error:
        assert "runtime main-environment group size" in str(error)
    else:
        raise AssertionError("Expected runtime root-slot validation to fail")


def test_dense_and_active_root_executors_have_same_deterministic_evidence(
    monkeypatch,
):
    class Envs:
        def __init__(self):
            self.thresholds = {0: 1, 1: 3, 2: 2}
            self.turns = {slot: 0 for slot in self.thresholds}
            self.done = {slot: False for slot in self.thresholds}
            self.tasks = ["task zero", "task one", "task two"]

        def _observations(self, slots):
            return {
                "text": [f"observation-{slot}-{self.turns[slot]}" for slot in slots],
                "image": None,
                "anchor": [f"anchor-{slot}-{self.turns[slot]}" for slot in slots],
                "admissible_actions": [("look",) for _ in slots],
            }

        def reset(self, kwargs):
            slots = list(kwargs["_bace_worker_indices"])
            return self._observations(slots), [
                {"extra.gamefile": f"game-{slot}"} for slot in slots
            ]

        def reset_selected(self, slots, reset_keys):
            assert list(reset_keys) == [f"game-{slot}" for slot in slots]
            return self._observations(slots), [
                {"extra.gamefile": reset_key} for reset_key in reset_keys
            ]

        def get_observations_selected(self, slots):
            return self._observations(slots)

        def get_tasks_selected(self, slots):
            return [self.tasks[slot] for slot in slots]

        def _step(self, slots, actions):
            infos = []
            rewards = []
            dones = []
            for slot, action in zip(slots, actions):
                assert action == "look"
                reward = 0.0
                if not self.done[slot]:
                    self.turns[slot] += 1
                    self.done[slot] = self.turns[slot] >= self.thresholds[slot]
                    reward = float(self.done[slot])
                rewards.append(reward)
                dones.append(self.done[slot])
                infos.append({
                    "won": self.done[slot],
                    "extra.gamefile": f"game-{slot}",
                    "is_action_valid": True,
                    "is_action_format_valid": True,
                    "is_action_environment_valid": True,
                    "projected_action": "look",
                    "action_identity": "valid::look",
                })
            return (
                self._observations(slots),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool),
                infos,
            )

        def step(self, actions):
            return self._step([0, 1, 2], actions)

        def step_selected(self, slots, actions):
            return self._step(list(slots), actions)

        def success_evaluator(self, total_infos, **_):
            return {
                "success_rate": np.asarray(
                    [float(infos[-1]["won"]) for infos in total_infos]
                )
            }

    class Actor:
        world_size = 4

        def generate_sequences(self, batch):
            size = len(batch)
            response = torch.ones((size, 1), dtype=torch.long)
            batch.batch["responses"] = response
            batch.batch["input_ids"] = torch.cat(
                [batch.batch["input_ids"], response], dim=-1
            )
            batch.batch["attention_mask"] = torch.cat(
                [
                    batch.batch["attention_mask"],
                    torch.ones((size, 1), dtype=torch.long),
                ],
                dim=-1,
            )
            batch.batch["position_ids"] = torch.cat(
                [
                    batch.batch["position_ids"],
                    batch.batch["position_ids"][:, -1:] + 1,
                ],
                dim=-1,
            )
            return batch

    def run(active):
        ids = iter(uuid.UUID(int=index + 1) for index in range(3))
        monkeypatch.setattr(collector_module.uuid, "uuid4", lambda: next(ids))
        collector = object.__new__(BaceTrajectoryCollector)
        collector.config = SimpleNamespace(
            env=SimpleNamespace(max_steps=3, rollout=SimpleNamespace(n=8)),
            algorithm={"bace": {"enabled": True}},
        )
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
                "position_ids": torch.tensor([[0, 1]] * size, dtype=torch.long),
                "raw_prompt_ids": np.asarray(
                    [[1, 2] for _ in range(size)], dtype=object
                ),
                "anchor_obs": np.asarray(obs["anchor"], dtype=object),
                "index": np.asarray(
                    gen_batch.non_tensor_batch["logical_index"]
                ),
                "data_source": np.asarray(["alfworld"] * size, dtype=object),
            })

        collector.preprocess_batch = MethodType(preprocess, collector)
        gen_batch = DataProto.from_single_dict({
            "prompts": torch.zeros((3, 1), dtype=torch.long),
            "logical_index": np.arange(3, dtype=np.int32),
            "data_source": np.asarray(["alfworld"] * 3, dtype=object),
        })
        envs = Envs()
        if active:
            rollout = collector._active_root_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=Actor(),
                envs=envs,
                worker_indices=[0, 1, 2],
                reset_keys=["game-0", "game-1", "game-2"],
                uid_batch=["task-0", "task-1", "task-2"],
                task_batch_indices=[0, 1, 2],
            )
        else:
            rollout = collector.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=Actor(),
                envs=envs,
                reset_options={
                    "_bace_worker_indices": [0, 1, 2],
                    "_bace_reset_keys": ["game-0", "game-1", "game-2"],
                },
                uid_batch=["task-0", "task-1", "task-2"],
                task_batch_indices=[0, 1, 2],
            )
        flattened = collector.gather_rollout_data(
            total_batch_list=rollout[0],
            episode_rewards=rollout[1],
            episode_lengths=rollout[2],
            success=rollout[3],
            traj_uid=rollout[4],
            tool_callings=rollout[5],
        )
        return rollout, build_root_event_logs(flattened)

    dense, dense_logs = run(active=False)
    active, active_logs = run(active=True)
    assert dense[1].tolist() == active[1].tolist() == [1.0, 1.0, 1.0]
    assert dense[2].tolist() == active[2].tolist() == [1.0, 3.0, 2.0]
    assert dense[3]["success_rate"].tolist() == active[3]["success_rate"].tolist()
    assert dense_logs == active_logs


def test_packed_staged_roots_use_one_pilot_and_one_completion_wave(monkeypatch):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(
            bace=SimpleNamespace(total_leaf_budget=4, pilot_roots=2)
        )
    )
    collector.orchestration_metrics = {}
    waves = []

    class FakeEnvs:
        def reset(self, kwargs):
            assert kwargs["_bace_worker_indices"] == [0, 4]
            return None, [
                {"extra.gamefile": "game-0"},
                {"extra.gamefile": "game-1"},
            ]

    def collect_wave(gen_batch, actor_rollout_wg, envs, task_indices, root_slots,
                     task_uids, reset_keys):
        wave = SimpleNamespace(
            task_indices=list(task_indices),
            root_slots=list(root_slots),
            task_uids=[task_uids[index] for index in task_indices],
            reset_keys=list(reset_keys),
        )
        waves.append(wave)
        return wave

    def build_logs(wave):
        return [
            SimpleNamespace(task_id=task_id, root_id=f"{task_id}:{slot}")
            for task_id, slot in zip(wave.task_uids, wave.root_slots)
        ]

    class FakePlanner:
        def initialize_staged(self, logs):
            task_ids = list(dict.fromkeys(log.task_id for log in logs))
            return {
                task_id: SimpleNamespace(root_count=4)
                for task_id in task_ids
            }

        def correct_staged_capacity(self, logs, states):
            return set()

        def finalize_staged(self, logs, states):
            return SimpleNamespace(roots=tuple(logs))

    collector._collect_root_wave = collect_wave
    collector._concat_batches = lambda batches: tuple(batches)
    collector.topology_planner = FakePlanner()
    monkeypatch.setattr(collector_module, "build_root_event_logs", build_logs)

    root_output, root_logs, _, generated = collector._collect_staged_dynamic_roots_packed(
        gen_batch=[object(), object()], actor_rollout_wg=None, envs=FakeEnvs()
    )

    assert len(root_output) == 2
    assert len(root_logs) == 8
    assert list(generated.values()) == [4, 4]
    assert [wave.root_slots for wave in waves] == [[0, 0, 1, 1], [2, 2, 3, 3]]
    assert waves[0].reset_keys == ["game-0", "game-1", "game-0", "game-1"]
    assert collector.orchestration_metrics["root_generation_waves"] == 2.0
    assert collector.orchestration_metrics["pilot_root_trajectories"] == 4.0
    assert collector.orchestration_metrics["completion_root_trajectories"] == 4.0


def test_exact_batch_roots_have_no_pilot_wave_and_pack_all_planned_roots(monkeypatch):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(bace=SimpleNamespace(total_leaf_budget=4))
    )
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.current_step = 3
    waves = []

    class FakeEnvs:
        def reset(self, kwargs):
            return None, [
                {"extra.gamefile": "/data/pick_and_place/game-0"},
                {"extra.gamefile": "/data/pick_and_place/game-1"},
            ]

    def collect_wave(gen_batch, actor_rollout_wg, envs, task_indices, root_slots,
                     task_uids, reset_keys):
        wave = SimpleNamespace(
            task_indices=list(task_indices),
            root_slots=list(root_slots),
            task_uids=[task_uids[index] for index in task_indices],
            reset_keys=list(reset_keys),
        )
        waves.append(wave)
        return wave

    def build_logs(wave):
        return [
            SimpleNamespace(task_id=task_id, root_id=f"{task_id}:{slot}")
            for task_id, slot in zip(wave.task_uids, wave.root_slots)
        ]

    class FakePlanner:
        def initialize(self, families):
            task_ids = list(families)
            return {
                task_ids[0]: SimpleNamespace(root_count=3),
                task_ids[1]: SimpleNamespace(root_count=2),
            }

        def correct_capacity(self, logs, states):
            return set()

        def finalize(self, logs, states):
            return SimpleNamespace(roots=tuple(logs))

    collector._collect_root_wave = collect_wave
    collector._concat_batches = lambda batches: tuple(batches)
    collector.topology_planner = FakePlanner()
    monkeypatch.setattr(collector_module, "build_root_event_logs", build_logs)

    _, root_logs, _, generated = collector._collect_exact_batch_dynamic_roots_packed(
        gen_batch=[object(), object()], actor_rollout_wg=None, envs=FakeEnvs()
    )

    assert len(waves) == 1
    assert waves[0].root_slots == [0, 0, 1, 1, 2]
    assert len(root_logs) == 5
    assert sorted(generated.values()) == [2, 3]


def test_exact_capacity_correction_uses_the_shared_root_wave_executor(monkeypatch):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(bace=SimpleNamespace(total_leaf_budget=4))
    )
    collector.root_active_executor = True
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.current_step = 7
    waves = []

    class FakeEnvs:
        def reset(self, kwargs):
            assert kwargs["_bace_worker_indices"] == [0, 4]
            return None, [
                {"extra.gamefile": "/data/pick_and_place/game-0"},
                {"extra.gamefile": "/data/pick_and_place/game-1"},
            ]

    def collect_wave(
        gen_batch,
        actor_rollout_wg,
        envs,
        task_indices,
        root_slots,
        task_uids,
        reset_keys,
    ):
        del gen_batch, actor_rollout_wg, envs, reset_keys
        wave = SimpleNamespace(
            task_indices=list(task_indices),
            root_slots=list(root_slots),
            task_uids=[task_uids[index] for index in task_indices],
            active_executor=collector.root_active_executor,
        )
        waves.append(wave)
        return wave

    def build_logs(wave):
        return [
            SimpleNamespace(task_id=task_id, root_id=f"{task_id}:{slot}")
            for task_id, slot in zip(wave.task_uids, wave.root_slots)
        ]

    class FakePlanner:
        def __init__(self):
            self.calls = 0
            self.first_task = None

        def initialize(self, families):
            task_ids = list(families)
            self.first_task = task_ids[0]
            return {
                task_id: SimpleNamespace(root_count=2) for task_id in task_ids
            }

        def correct_capacity(self, logs, states):
            del logs
            self.calls += 1
            if self.calls == 1:
                states[self.first_task].root_count = 3
                return {self.first_task}
            return set()

        def finalize(self, logs, states):
            del states
            return SimpleNamespace(roots=tuple(logs))

    collector._collect_root_wave = collect_wave
    collector._concat_batches = lambda batches: tuple(batches)
    collector.topology_planner = FakePlanner()
    monkeypatch.setattr(collector_module, "build_root_event_logs", build_logs)

    _, root_logs, _, generated = collector._collect_exact_batch_dynamic_roots_packed(
        gen_batch=[object(), object()], actor_rollout_wg=None, envs=FakeEnvs()
    )

    assert len(waves) == 2
    assert waves[0].root_slots == [0, 0, 1, 1]
    assert waves[1].root_slots == [2]
    assert all(wave.active_executor for wave in waves)
    assert len(root_logs) == 5
    assert sorted(generated.values()) == [2, 3]
    assert collector.orchestration_metrics[
        "capacity_correction_root_trajectories"
    ] == 1.0
    assert collector.orchestration_metrics["planned_root_waves"] == 1.0
    assert "pilot_root_waves" not in collector.orchestration_metrics
