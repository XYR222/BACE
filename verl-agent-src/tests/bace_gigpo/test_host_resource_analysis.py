import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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


def test_warn_only_cli_records_violation_without_failing(tmp_path):
    samples = tmp_path / "samples.csv"
    samples.write_text(
        "timestamp,pids_current,pids_limit,pids_source\n"
        "now,27529,unavailable,cgroup.threads\n"
    )
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(samples),
            "--max-pids",
            "12000",
            "--warn-only",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "continuing because monitoring is warn-only" in result.stderr
    report = json.loads(output.read_text())
    assert report["ok"] is False
    assert report["enforcement"] == "warning_only"
    assert report["peak_pids_current"] == 27529
