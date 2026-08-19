import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "bace_gigpo"
    / "analyze_host_resources.py"
)
SPEC = importlib.util.spec_from_file_location("bace_host_resources", SCRIPT)
host_resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host_resources)


def test_accepts_cgroup_threads_fallback_samples():
    report = host_resources.analyze(
        [
            {
                "pids_current": "410",
                "pids_limit": "unavailable",
                "pids_source": "cgroup.threads",
            },
            {
                "pids_current": "875",
                "pids_limit": "unavailable",
                "pids_source": "cgroup.threads",
            },
        ],
        max_pids=12000,
    )

    assert report["ok"] is True
    assert report["peak_pids_current"] == 875
    assert report["pids_sources"] == ["cgroup.threads"]


def test_legacy_samples_remain_supported_and_limit_is_enforced():
    report = host_resources.analyze(
        [{"pids_current": "12001", "pids_limit": "16384"}],
        max_pids=12000,
    )

    assert report["ok"] is False
    assert report["pids_sources"] == ["legacy"]


def test_rejects_run_without_numeric_task_samples():
    with pytest.raises(ValueError, match="no numeric cgroup PID samples"):
        host_resources.analyze(
            [{"pids_current": "unavailable", "pids_limit": "unavailable"}],
            max_pids=12000,
        )
