from types import SimpleNamespace

import numpy as np
import pytest

from recipe.bace_gigpo.frontier import (
    BACEFrontierOrchestrator,
    FrontierInvariantError,
    FrontierJob,
    FrontierProfile,
    FrontierTaskController,
)
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector


class _Planner:
    def __init__(self, root_count=3, branch_count=1):
        self.root_count = root_count
        self.branch_count = branch_count

    def initialize_staged(self, roots):
        return {roots[0].task_id: SimpleNamespace(
            root_count=self.root_count,
            branch_count=self.branch_count,
        )}

    def correct_staged_capacity(self, roots, states):
        return set()

    def finalize_staged(self, roots, states):
        task_id = roots[0].task_id
        return SimpleNamespace(tasks={task_id: object()})


def _root(task_id="task"):
    return SimpleNamespace(task_id=task_id)


def test_frontier_task_consumes_each_sibling_once_and_freezes_roots():
    controller = FrontierTaskController(
        task_index=0,
        task_id="task",
        slots=(0, 1, 2, 3),
        total_budget=4,
        pilot_target=2,
    )
    assert [controller.allocate_slot(), controller.allocate_slot()] == [0, 1]
    controller.add_root(_root())
    controller.add_root(_root())
    assert controller.plan_after_pilots(_Planner())
    assert controller.allocate_slot() == 2
    controller.add_root(_root())
    assert controller.capacity_check(_Planner())
    assert controller.root_backbone_frozen
    assert controller.root_target + controller.branch_quota == 4
    assert controller.allocate_slot() == 3
    controller.branch_completed = 1
    assert controller.complete
    controller.assert_budget()


def test_frontier_rejects_root_after_freeze():
    controller = FrontierTaskController(0, "task", (0, 1), 2, 1)
    controller.root_backbone_frozen = True
    with pytest.raises(FrontierInvariantError):
        controller.add_root(_root())


def test_frontier_rejects_slot_overconsumption():
    controller = FrontierTaskController(0, "task", (0,), 1, 1)
    assert controller.allocate_slot() == 0
    with pytest.raises(FrontierInvariantError):
        controller.allocate_slot()


def test_optional_coalescing_caps_only_ready_jobs_and_mixes_phases():
    orchestrator = object.__new__(BACEFrontierOrchestrator)
    orchestrator.batch_coalescing_enabled = True
    orchestrator.batch_coalescing_max_size = 2
    orchestrator.batch_coalescing_min_size = 2
    orchestrator.profile = FrontierProfile()
    orchestrator.jobs = {
        0: FrontierJob("root", 0, "task-a", 0, "root-a"),
        1: FrontierJob("root", 0, "task-a", 1, "root-b"),
        2: FrontierJob("branch", 1, "task-b", 2, "branch-a"),
        3: FrontierJob("branch", 1, "task-b", 3, "branch-b"),
    }
    selected = orchestrator._select_generation_slots()
    assert len(selected) == 2
    assert {orchestrator.jobs[slot].kind for slot in selected} == {"root", "branch"}
    assert len(orchestrator.jobs) == 4


def test_initial_reset_rebinds_mismatched_siblings_in_structured_observations():
    orchestrator = object.__new__(BACEFrontierOrchestrator)
    orchestrator.gen_batch = [object()]
    orchestrator.budget = 2
    orchestrator.pilots = 1
    orchestrator.tasks = {}

    class FakeEnvs:
        def __init__(self):
            self.calls = 0
            self._bace_slots = {}

        def reset_selected(self, worker_indices, game_files=None):
            self.calls += 1
            if self.calls == 1:
                keys = ["game-a", "game-b"]
                texts = ["old-a", "old-b"]
                tasks = ["task-a", "task-b"]
            else:
                assert worker_indices == [0, 1]
                assert game_files == ["game-a", "game-a"]
                keys = ["game-a", "game-a"]
                texts = ["new-a", "new-b"]
                tasks = ["task-a", "task-a"]
            for index, task in zip(worker_indices, tasks):
                self._bace_slots[index] = {"task": task}
            observations = {
                "text": list(texts),
                "image": None,
                "anchor": [f"anchor-{text}" for text in texts],
                "admissible_actions": [("look",)] * len(texts),
            }
            infos = [{"extra.gamefile": key} for key in keys]
            return observations, infos

    orchestrator.envs = FakeEnvs()
    observations = orchestrator._initial_reset()

    assert observations["text"] == ["new-a", "new-b"]
    assert observations["anchor"] == ["anchor-new-a", "anchor-new-b"]
    assert observations["image"] is None
    assert next(iter(orchestrator.tasks.values())).reset_key == "game-a"
    assert orchestrator.envs.calls == 2


def test_collect_exits_when_last_replay_completes_all_tasks():
    orchestrator = object.__new__(BACEFrontierOrchestrator)
    controller = SimpleNamespace(
        task_id="task-a",
        pilot_target=0,
        complete=False,
        assert_budget=lambda: None,
    )
    orchestrator.tasks = {controller.task_id: controller}
    orchestrator.jobs = {}
    orchestrator._pending_replay = [object()]
    orchestrator.profile = FrontierProfile()
    orchestrator._initial_reset = lambda: None
    orchestrator._enqueue_root = lambda task: None
    orchestrator._generate_frontier = lambda: None
    orchestrator._finalize = lambda: "complete"

    replay_calls = 0

    def finish_replay():
        nonlocal replay_calls
        replay_calls += 1
        orchestrator._pending_replay.clear()
        controller.complete = True

    orchestrator._run_replay = finish_replay

    assert orchestrator.collect() == "complete"
    assert replay_calls == 2


def test_success_metrics_separate_root_branch_and_refresh_stale_origin():
    class FakeBatch:
        def __init__(self):
            self.non_tensor_batch = {
                "traj_uid": np.asarray(["root-a", "root-a", "branch-b", "branch-b"], dtype=object),
                "source_type": np.asarray(
                    ["root", "root", "branch_origin", "branch_suffix"], dtype=object
                ),
                "episode_rewards": np.asarray([10.0, 10.0, 0.0, 0.0], dtype=np.float32),
                "success_rate": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            }

        def __len__(self):
            return len(self.non_tensor_batch["traj_uid"])

    batch = FakeBatch()
    BaceTrajectoryCollector._refresh_success_rate(batch)
    assert batch.non_tensor_batch["success_rate"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert BaceTrajectoryCollector._episode_success_metrics(batch) == {
        "root_success_count": 1,
        "root_success_episodes": 1,
        "root_success_rate": 1.0,
        "branch_success_count": 0,
        "branch_success_episodes": 1,
        "branch_success_rate": 0.0,
        "mixed_success_count": 1,
        "mixed_success_episodes": 2,
        "mixed_success_rate": 0.5,
    }
