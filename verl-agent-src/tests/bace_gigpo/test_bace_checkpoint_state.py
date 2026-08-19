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


def bare_collector(threshold=0.005):
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
    collector.parameter_signature = {"batch_erv_threshold": threshold}
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
