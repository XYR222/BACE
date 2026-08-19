import importlib.util
from pathlib import Path
from types import SimpleNamespace

from verl.utils.checkpoint.checkpoint_manager import (
    find_existing_component_checkpoints,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "bace_gigpo"
    / "prune_checkpoints.py"
)
SPEC = importlib.util.spec_from_file_location("bace_prune_checkpoints", SCRIPT)
prune_checkpoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prune_checkpoints)


def make_checkpoint(root: Path, step: int, complete: bool = True) -> Path:
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "weights.bin").write_bytes(b"actor")
    if complete:
        (checkpoint / "data.pt").write_bytes(b"data")
        (checkpoint / "bace_collector_state.json").write_text("{}")
    return checkpoint


def test_cross_process_checkpoint_rotation_keeps_latest_two(tmp_path):
    for step in (5, 10, 15):
        make_checkpoint(tmp_path, step)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("15")

    result = prune_checkpoints.prune(tmp_path, keep=2)

    assert result["retained"] == [10, 15]
    assert not (tmp_path / "global_step_5").exists()
    assert (tmp_path / "global_step_10").is_dir()
    assert (tmp_path / "global_step_15").is_dir()


def test_rotation_removes_uncommitted_newer_directory(tmp_path):
    make_checkpoint(tmp_path, 5)
    make_checkpoint(tmp_path, 10, complete=False)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5")

    result = prune_checkpoints.prune(tmp_path, keep=1)

    assert result["retained"] == [5]
    assert not (tmp_path / "global_step_10").exists()


def test_recommendation_requires_space_for_checkpoint_plus_reserve(
    tmp_path, monkeypatch
):
    make_checkpoint(tmp_path, 5)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5")
    checkpoint_bytes = prune_checkpoints.directory_size(tmp_path / "global_step_5")

    monkeypatch.setattr(
        prune_checkpoints.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            free=checkpoint_bytes + 9 * prune_checkpoints.GIB
        ),
    )
    assert prune_checkpoints.recommend_keep(tmp_path)["recommended_keep"] == 1

    monkeypatch.setattr(
        prune_checkpoints.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            free=checkpoint_bytes + 10 * prune_checkpoints.GIB
        ),
    )
    assert prune_checkpoints.recommend_keep(tmp_path)["recommended_keep"] == 2


def test_resumed_manager_discovers_prior_actor_checkpoints(tmp_path):
    actor_5 = make_checkpoint(tmp_path, 5) / "actor"
    actor_10 = make_checkpoint(tmp_path, 10) / "actor"
    actor_15 = make_checkpoint(tmp_path, 15) / "actor"

    # Step 15 may be a partially written newer save. Loading step 10 must seed
    # only paths at or before the checkpoint selected by the atomic tracker.
    assert find_existing_component_checkpoints(str(actor_10)) == [
        str(actor_5.resolve()),
        str(actor_10.resolve()),
    ]
    assert str(actor_15.resolve()) not in find_existing_component_checkpoints(
        str(actor_10)
    )
