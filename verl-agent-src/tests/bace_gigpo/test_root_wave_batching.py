from types import SimpleNamespace

from recipe.bace_gigpo import rollout_collector as collector_module
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector


def test_pending_root_slots_packs_all_known_missing_slots():
    pending = BaceTrajectoryCollector._pending_root_slots(
        [0, 1, 2],
        generated_by_task={0: 2, 1: 2, 2: 3},
        target_by_task={0: 4, 1: 3, 2: 4},
    )
    assert pending == [(0, 2), (1, 2), (0, 3), (2, 3)]


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
    assert collector.orchestration_metrics["planned_root_waves"] == 1.0
    assert "pilot_root_waves" not in collector.orchestration_metrics
