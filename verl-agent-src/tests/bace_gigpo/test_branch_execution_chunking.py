from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from agent_system.environments.base import EnvironmentManagerBase
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector


def make_request(index):
    return SimpleNamespace(
        request_id=f"request-{index}",
        branch_id=f"branch-{index}",
        origin_occurrence_id=f"origin-{index}",
        task_batch_index=index % 16,
    )


def make_collector(capacity=16, variant="batch_erv_exact"):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.branch_envs = SimpleNamespace(replay_capacity=capacity)
    collector.variant = variant
    collector.orchestration_metrics = {}
    collector.artifact_store = None
    collector.replay_retry_metadata = {}
    return collector


def install_successful_chunk_executor(collector, calls):
    def execute_chunk(
        self,
        root_output,
        gen_batch,
        actor_rollout_wg,
        requests,
        used_origins=None,
        execution_wave=0,
    ):
        calls.append((execution_wave, list(requests), used_origins))
        rewards = np.arange(len(requests), dtype=np.float32)
        return list(requests), None, None, rewards

    collector._execute_chunk = MethodType(execute_chunk, collector)


def test_environment_manager_exposes_replay_capacity():
    manager = EnvironmentManagerBase(
        envs=SimpleNamespace(num_processes=16),
        projection_f=None,
        config=None,
    )
    assert manager.replay_capacity == 16


def test_environment_manager_rejects_empty_replay_pool():
    manager = EnvironmentManagerBase(
        envs=SimpleNamespace(num_processes=0),
        projection_f=None,
        config=None,
    )
    with pytest.raises(RuntimeError, match="no workers"):
        _ = manager.replay_capacity


@pytest.mark.parametrize(
    ("request_count", "expected_wave_sizes"),
    [
        (17, [16, 1]),
        (48, [16, 16, 16]),
        (96, [16, 16, 16, 16, 16, 16]),
    ],
)
def test_exact_round_chunks_frozen_requests_without_reordering(
    request_count, expected_wave_sizes
):
    collector = make_collector()
    calls = []
    install_successful_chunk_executor(collector, calls)
    requests = [make_request(index) for index in range(request_count)]

    valid, origin, suffix, rewards = collector._execute_round(
        None, None, None, requests
    )

    assert [len(call[1]) for call in calls] == expected_wave_sizes
    assert all(len(call[1]) <= collector.branch_envs.replay_capacity for call in calls)
    assert [request.branch_id for request in valid] == [
        request.branch_id for request in requests
    ]
    assert origin is None
    assert suffix is None
    assert len(rewards) == request_count
    assert collector.orchestration_metrics["branch_replay_capacity"] == 16.0
    assert collector.orchestration_metrics["branch_execution_waves"] == float(
        len(expected_wave_sizes)
    )


def test_all_waves_share_origins_reserved_by_the_full_frozen_round():
    collector = make_collector()
    requests = [make_request(index) for index in range(17)]
    calls = []

    def execute_chunk(
        self,
        root_output,
        gen_batch,
        actor_rollout_wg,
        chunk,
        used_origins=None,
        execution_wave=0,
    ):
        calls.append((execution_wave, used_origins, set(used_origins)))
        if execution_wave == 0:
            used_origins.add("fallback-origin")
        return list(chunk), None, None, np.zeros(len(chunk), dtype=np.float32)

    collector._execute_chunk = MethodType(execute_chunk, collector)
    collector._execute_round(None, None, None, requests)

    expected_origins = {request.origin_occurrence_id for request in requests}
    assert calls[0][2] == expected_origins
    assert calls[0][1] is calls[1][1]
    assert "fallback-origin" in calls[1][2]


def test_exact_round_fails_instead_of_dropping_a_chunk_request():
    collector = make_collector()
    requests = [make_request(index) for index in range(17)]

    def execute_chunk(
        self,
        root_output,
        gen_batch,
        actor_rollout_wg,
        chunk,
        used_origins=None,
        execution_wave=0,
    ):
        valid = list(chunk) if execution_wave == 0 else []
        return valid, None, None, np.zeros(len(valid), dtype=np.float32)

    collector._execute_chunk = MethodType(execute_chunk, collector)

    with pytest.raises(RuntimeError, match="execution wave 1.*validated 0 of 1"):
        collector._execute_round(None, None, None, requests)


def test_chunk_helper_rejects_nonpositive_capacity():
    with pytest.raises(ValueError, match="positive"):
        BaceTrajectoryCollector._chunk_requests([make_request(0)], 0)


def make_main_reuse_collector(capacity=128, group_size=8):
    collector = make_collector(capacity=capacity)
    collector.branch_pool_mode = "main_reuse"
    collector.main_group_size = group_size
    return collector


def test_main_reuse_maps_current_maximum_96_requests_to_one_cohort():
    collector = make_main_reuse_collector()
    requests = []
    for task_index in range(16):
        for local_index in range(6):
            request = make_request(len(requests))
            request.task_batch_index = task_index
            request.branch_id = f"task-{task_index}:branch-{local_index}"
            requests.append(request)

    cohorts = collector._branch_capacity_cohorts(requests, 128)
    slots = collector._branch_worker_slots(cohorts[0])

    assert len(cohorts) == 1
    assert len(slots) == 96
    assert len(set(slots)) == 96
    assert slots[:6] == [0, 1, 2, 3, 4, 5]
    assert slots[-6:] == [120, 121, 122, 123, 124, 125]


@pytest.mark.parametrize(("request_count", "expected_sizes"), [
    (129, [128, 1]),
    (200, [128, 72]),
])
def test_main_reuse_capacity_overflow_has_safe_fallback(
    request_count, expected_sizes
):
    collector = make_main_reuse_collector()
    requests = [make_request(index) for index in range(request_count)]

    cohorts = collector._branch_capacity_cohorts(requests, 128)

    assert [len(cohort) for cohort in cohorts] == expected_sizes
    assert [request.request_id for cohort in cohorts for request in cohort] == [
        request.request_id for request in requests
    ]
    for cohort in cohorts:
        slots = collector._branch_worker_slots(cohort)
        assert len(slots) == len(set(slots))


def test_main_reuse_splits_a_task_that_exceeds_its_physical_group():
    collector = make_main_reuse_collector()
    requests = [make_request(index) for index in range(17)]
    for request in requests:
        request.task_batch_index = 3

    cohorts = collector._branch_capacity_cohorts(requests, 128)

    assert [len(cohort) for cohort in cohorts] == [8, 8, 1]
    assert [collector._branch_worker_slots(cohort) for cohort in cohorts] == [
        list(range(24, 32)),
        list(range(24, 32)),
        [24],
    ]
