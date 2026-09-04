from pathlib import Path

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.trainer.ppo.ray_trainer import _rotate_global_checkpoint_directories


def test_cross_root_restore_never_rotates_read_only_source(tmp_path: Path):
    source = tmp_path / "source" / "global_step_145" / "actor"
    destination = tmp_path / "destination"
    source.mkdir(parents=True)
    destination.mkdir()

    retained = FSDPCheckpointManager._destination_owned_previous_paths(
        [str(source)], str(destination / "global_step_150" / "actor")
    )

    assert retained == []
    assert source.is_dir()


def test_same_root_restore_still_rotates_old_destination(tmp_path: Path):
    root = tmp_path / "destination"
    old = root / "global_step_145" / "actor"
    new = root / "global_step_150" / "actor"
    old.mkdir(parents=True)
    new.parent.mkdir(parents=True)

    retained = FSDPCheckpointManager._destination_owned_previous_paths(
        [str(old)], str(new)
    )

    assert retained == [str(old)]


def test_successful_save_rotation_preserves_cross_experiment_source(tmp_path: Path):
    source = tmp_path / "source" / "global_step_145" / "actor"
    destination_root = tmp_path / "destination"
    old = destination_root / "global_step_145" / "actor"
    new = destination_root / "global_step_150" / "actor"
    source.mkdir(parents=True)
    old.mkdir(parents=True)
    new.parent.mkdir(parents=True)

    manager = object.__new__(FSDPCheckpointManager)
    manager.previous_saved_paths = [str(source), str(old)]
    manager._record_successful_save_and_rotate(str(new), 1)

    assert source.is_dir()
    assert not old.exists()
    assert manager.previous_saved_paths == [str(new)]


def test_global_rotation_removes_only_old_committed_step_trees(tmp_path: Path):
    for step in (5, 10, 15):
        (tmp_path / f"global_step_{step}" / "actor").mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    removed = _rotate_global_checkpoint_directories(str(tmp_path), 15, keep=2)

    assert removed == [str(tmp_path / "global_step_5")]
    assert not (tmp_path / "global_step_5").exists()
    assert (tmp_path / "global_step_10").is_dir()
    assert (tmp_path / "global_step_15").is_dir()
    assert (tmp_path / "notes").is_dir()
