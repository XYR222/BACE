import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from recipe.bace_gigpo.competence import CompetenceHistory
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class FakeActorWorker:
    def __init__(self):
        self.loaded_path = None

    def save_checkpoint(self, path, _remote, _step, max_ckpt_to_keep=None):
        del max_ckpt_to_keep
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "marker").write_text("actor")

    def load_checkpoint(self, path, del_local_after_load=False):
        del del_local_after_load
        assert Path(path, "marker").read_text() == "actor"
        self.loaded_path = path


class FakeDataLoader:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"cursor": 7}

    def load_state_dict(self, state):
        self.loaded = state


class FakeCollector:
    requires_checkpoint_state = True

    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"version": 1, "history": {"heat": [1.0, 2.0]}}

    def load_state_dict(self, state):
        self.loaded = state


class FakeMigrationCollector(FakeCollector):
    def checkpoint_migration_metadata(self, source_checkpoint, source_global_step):
        return {
            "migration_mode": "diagnostic_only",
            "source_checkpoint": source_checkpoint,
            "source_global_step": source_global_step,
            "allowed_diff": {
                "min_natural_roots": {"saved": 2, "current": 4}
            },
        }


def make_trainer(checkpoint_dir):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "trainer": {
            "default_local_dir": str(checkpoint_dir),
            "default_hdfs_dir": None,
            "remove_previous_ckpt_in_save": False,
            "max_actor_ckpt_to_keep": 1,
            "max_critic_ckpt_to_keep": 1,
            "resume_mode": "auto",
            "resume_from_path": None,
            "del_local_ckpt_after_load": False,
        }
    })
    trainer.global_steps = 3
    trainer.actor_rollout_wg = FakeActorWorker()
    trainer.use_critic = False
    trainer.train_dataloader = FakeDataLoader()
    trainer.traj_collector = FakeCollector()
    return trainer


def test_bace_checkpoint_state_is_atomic_and_restored(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_checkpoint()

    step_dir = tmp_path / "global_step_3"
    payload = json.loads((step_dir / "bace_collector_state.json").read_text())
    assert payload["global_step"] == 3
    assert payload["collector"] == trainer.traj_collector.state_dict()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"

    restored = make_trainer(tmp_path)
    restored.global_steps = 0
    restored._load_checkpoint()
    assert restored.global_steps == 3
    assert restored.train_dataloader.loaded == {"cursor": 7}
    assert restored.traj_collector.loaded == payload["collector"]


def test_dynamic_bace_resume_rejects_missing_or_mismatched_state(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_checkpoint()
    state_path = tmp_path / "global_step_3" / "bace_collector_state.json"
    state_path.unlink()

    with pytest.raises(FileNotFoundError, match="missing collector state"):
        make_trainer(tmp_path)._load_checkpoint()

    trainer._save_checkpoint()
    payload = json.loads(state_path.read_text())
    payload["global_step"] = 2
    state_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="global_step"):
        make_trainer(tmp_path)._load_checkpoint()

    state_path.write_text("{corrupt json")
    with pytest.raises(json.JSONDecodeError):
        make_trainer(tmp_path)._load_checkpoint()


def test_trainer_writes_migration_metadata_only_to_distinct_output(tmp_path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source = make_trainer(source_root)
    source._save_checkpoint()
    source_step = source_root / "global_step_3"

    restored = make_trainer(destination_root)
    restored.config.trainer.resume_mode = "resume_path"
    restored.config.trainer.resume_from_path = str(source_step)
    restored.traj_collector = FakeMigrationCollector()
    restored.global_steps = 0
    restored._load_checkpoint()

    metadata_path = destination_root / "checkpoint_migration.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["migration_mode"] == "diagnostic_only"
    assert metadata["source_global_step"] == 3
    assert restored.actor_rollout_wg.loaded_path == str(source_step / "actor")
    assert not (source_root / "checkpoint_migration.json").exists()

    unsafe = make_trainer(source_root)
    unsafe.config.trainer.resume_mode = "resume_path"
    unsafe.config.trainer.resume_from_path = str(source_step)
    unsafe.traj_collector = FakeMigrationCollector()
    unsafe.global_steps = 0
    with pytest.raises(ValueError, match="distinct output directory"):
        unsafe._load_checkpoint()
    assert unsafe.actor_rollout_wg.loaded_path is None


def bare_collector(
    threshold=0.005,
    tie_break_identity_mode="legacy_uuid",
    min_natural_roots=2,
    migration_enabled=False,
):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.variant = "batch_erv_exact"
    collector.topology = "dynamic"
    collector.acquisition = "batch_erv_exact"
    collector.current_step = 7
    collector.competence_history = CompetenceHistory(
        base_alpha=0.2,
        base_beta=1.8,
        forgetting=0.8,
        transfer_fraction=0.1,
        min_strength=2.0,
        max_strength=8.0,
    )
    collector.parameter_signature = {
        "batch_erv_threshold": threshold,
        "min_natural_roots": min_natural_roots,
    }
    collector.tie_break_identity_mode = tie_break_identity_mode
    collector.checkpoint_migration_enabled = migration_enabled
    collector.checkpoint_migration_mode = (
        "diagnostic_rmin2_to4" if migration_enabled else "strict"
    )
    collector.last_checkpoint_migration = None
    if tie_break_identity_mode != "legacy_uuid":
        collector.parameter_signature["tie_break_identity_mode"] = tie_break_identity_mode
    return collector


def test_collector_state_round_trip_and_parameter_signature_rejection():
    collector = bare_collector()
    collector.competence_history.update({"heat": [True, False]})
    payload = json.loads(json.dumps(collector.state_dict()))

    restored = bare_collector()
    restored.current_step = 0
    restored.load_state_dict(payload)
    assert restored.current_step == 7
    assert restored.competence_history.snapshot() == {"heat": (1.0, 1.0)}

    incompatible = bare_collector(threshold=0.01)
    with pytest.raises(ValueError, match="parameter signature"):
        incompatible.load_state_dict(payload)


def test_collector_checkpoint_rejects_tie_break_identity_mode_change():
    legacy_payload = json.loads(json.dumps(bare_collector().state_dict()))
    stable_payload = json.loads(json.dumps(
        bare_collector(tie_break_identity_mode="stable_v1").state_dict()
    ))

    with pytest.raises(ValueError, match="parameter signature"):
        bare_collector(tie_break_identity_mode="stable_v1").load_state_dict(
            legacy_payload
        )
    with pytest.raises(ValueError, match="parameter signature"):
        bare_collector().load_state_dict(stable_payload)


def test_diagnostic_checkpoint_migration_allows_only_rmin2_to4():
    source = bare_collector(min_natural_roots=2)
    source.competence_history.update({"heat": [True, False, True]})
    payload = json.loads(json.dumps(source.state_dict()))

    migrated = bare_collector(
        min_natural_roots=4, migration_enabled=True
    )
    migrated.load_state_dict(payload)
    assert migrated.current_step == source.current_step
    assert migrated.competence_history.snapshot() == source.competence_history.snapshot()
    metadata = migrated.checkpoint_migration_metadata(
        "/source/global_step_7", 7
    )
    assert metadata["migration_mode"] == "diagnostic_only"
    assert metadata["allowed_diff"] == {
        "min_natural_roots": {"saved": 2, "current": 4}
    }
    assert len(metadata["source_competence_history_sha256"]) == 64

    strict = bare_collector(min_natural_roots=4)
    with pytest.raises(ValueError, match="parameter signature"):
        strict.load_state_dict(payload)

    extra_difference = bare_collector(
        threshold=0.01,
        min_natural_roots=4,
        migration_enabled=True,
    )
    with pytest.raises(ValueError, match="parameter signature"):
        extra_difference.load_state_dict(payload)

    reverse_payload = json.loads(json.dumps(
        bare_collector(min_natural_roots=4).state_dict()
    ))
    reverse = bare_collector(
        min_natural_roots=2, migration_enabled=True
    )
    with pytest.raises(ValueError, match="parameter signature"):
        reverse.load_state_dict(reverse_payload)
