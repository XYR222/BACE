import json
import shutil
from pathlib import Path

import pytest

from verl.utils.checkpoint.milestone import preserve_checkpoint


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "model_world_size_4_rank_0.pt").write_bytes(b"model")
    (actor / "optim_world_size_4_rank_0.pt").write_bytes(b"optim")
    (actor / "extra_state_world_size_4_rank_0.pt").write_bytes(b"extra")
    (checkpoint / "data.pt").write_bytes(b"data")
    (checkpoint / "bace_collector_state.json").write_text("{}")
    return checkpoint


def test_preserve_checkpoint_is_atomic_hardlink_and_idempotent(tmp_path: Path):
    source = _checkpoint(tmp_path / "active", 10)
    archive_root = tmp_path / "preserved"

    first = preserve_checkpoint(source, archive_root, 10)
    destination = archive_root / "global_step_10"
    assert (destination / ".complete").is_file()
    assert first == json.loads((destination / "preservation_manifest.json").read_text())
    assert first["method"] == "hardlink"
    assert (source / "data.pt").stat().st_ino == (destination / "data.pt").stat().st_ino
    assert preserve_checkpoint(source, archive_root, 10) == first

    # Normal active-root rotation may remove the entire source checkpoint;
    # the independent directory entries must keep the milestone readable.
    shutil.rmtree(source)
    assert (destination / "data.pt").read_bytes() == b"data"

    assert json.loads((destination / "preservation_manifest.json").read_text()) == first


def test_preserve_checkpoint_rejects_incomplete_source(tmp_path: Path):
    source = tmp_path / "active" / "global_step_75"
    source.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        preserve_checkpoint(source, tmp_path / "preserved", 75)
