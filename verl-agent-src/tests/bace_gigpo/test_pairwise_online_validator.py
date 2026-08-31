import json
import sys
from pathlib import Path

import pytest

from examples.bace_gigpo import validate_pairwise_online_smoke


@pytest.mark.parametrize(
    ("arm", "fixed", "stopping"),
    (("c1", 1.0, 0.0), ("c2", 0.0, 1.0)),
)
def test_pairwise_validator_accepts_legitimate_zero_branch_cold_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, fixed: float, stopping: float
):
    artifact_root = tmp_path / "artifacts"
    step_dir = artifact_root / "step_00000001"
    step_dir.mkdir(parents=True)
    (step_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "step": 1,
                "bace_metrics": {
                    "batch_erv_exact": 1.0,
                    "staged_root_batching_packed": 1.0,
                    "branch_selected_worker_execution": 1.0,
                    "final_roots_mean": 8.0,
                    "final_branches_mean": 0.0,
                    "requested": 0.0,
                    "pairwise_fixed": fixed,
                    "pairwise_stopping": stopping,
                    "pairwise_branch_rounds": 0.0,
                    "fallback_root_count": 0.0,
                },
            }
        )
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_pairwise_online_smoke.py",
            str(artifact_root),
            "--arm",
            arm,
            "--output",
            str(output),
        ],
    )

    validate_pairwise_online_smoke.main()

    report = json.loads(output.read_text())
    assert report["ok"] is True
    assert report["steps"][0]["requested"] == 0.0
