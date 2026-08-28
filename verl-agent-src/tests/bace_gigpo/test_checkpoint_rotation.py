from pathlib import Path

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


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
