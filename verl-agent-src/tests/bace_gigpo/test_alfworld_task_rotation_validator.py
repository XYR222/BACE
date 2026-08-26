import json
from pathlib import Path

import pytest

from examples.bace_gigpo.validate_alfworld_task_rotation import validate_rotation


def _write_roots(root: Path, step: int, reset_keys: list[str]) -> None:
    step_dir = root / f"step_{step:08d}"
    step_dir.mkdir(parents=True)
    rows = [
        {
            "task_batch_index": task_index,
            "environment_reset_key": reset_key,
        }
        for task_index, reset_key in enumerate(reset_keys)
    ]
    (step_dir / "roots.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_rotation_validator_accepts_new_natural_games(tmp_path):
    _write_roots(tmp_path, 1, ["game-a", "game-b"])
    _write_roots(tmp_path, 2, ["game-c", "game-d"])

    result = validate_rotation(tmp_path, 1, 2)

    assert result["ok"] is True
    assert result["changed_task_count"] == 2


def test_rotation_validator_rejects_pinned_worker(tmp_path):
    _write_roots(tmp_path, 1, ["game-a", "game-b"])
    _write_roots(tmp_path, 2, ["game-a", "game-c"])

    with pytest.raises(RuntimeError, match="remained pinned"):
        validate_rotation(tmp_path, 1, 2)
