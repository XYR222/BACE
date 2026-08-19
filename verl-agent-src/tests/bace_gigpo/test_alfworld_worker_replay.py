import pytest

from agent_system.environments.env_package.alfworld import envs as alfworld_envs
from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs, AlfworldWorker


class FakeBatchEnv:
    def __init__(self):
        self.gamefiles = ["old-game"]
        self._gamefiles_iterator = iter(self.gamefiles)


class FakeEnv:
    def __init__(self):
        self.batch_env = FakeBatchEnv()
        self.turn = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.turn = 0
        return ["turn-0"], {
            "admissible_commands": [["a", "stop"]],
            "won": [False],
        }

    def step(self, actions):
        self.turn += 1
        done = actions[0] == "stop"
        return [f"turn-{self.turn}"], [0.0], [done], {
            "admissible_commands": [["a", "stop"]],
            "won": [False],
        }


class FakeBaseEnv:
    def init_env(self, batch_size):
        assert batch_size == 1
        return FakeEnv()


def test_worker_binds_game_and_replays_prefix():
    worker = AlfworldWorker(config={}, seed=7, base_env=FakeBaseEnv())
    obs, info, done = worker.replay("new-game", ["a", "a"])
    assert obs == ["turn-2"]
    assert not done
    assert worker.env.batch_env.gamefiles == ["new-game"]
    assert info["admissible_commands"] == [["a", "stop"]]


def test_worker_rejects_early_terminal():
    worker = AlfworldWorker(config={}, seed=7, base_env=FakeBaseEnv())
    with pytest.raises(RuntimeError, match="terminated early"):
        worker.replay("new-game", ["stop", "a"])


class FakeRemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class FakeRemoteAlfworldWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.calls = []
        self.reset = FakeRemoteMethod(self._reset)
        self.step = FakeRemoteMethod(self._step)
        self.replay = FakeRemoteMethod(self._replay)

    def _reset(self, game_file=None):
        self.calls.append(("reset", game_file))
        return [f"worker-{self.worker_id}-reset"], {
            "admissible_commands": [[f"action-{self.worker_id}"]],
            "won": [False],
            "extra.gamefile": [game_file],
        }

    def _step(self, action):
        self.calls.append(("step", action))
        return [f"worker-{self.worker_id}-step"], [0.0], [False], {
            "admissible_commands": [[f"action-{self.worker_id}"]],
            "won": [False],
            "extra.gamefile": [f"game-{self.worker_id}"],
        }

    def _replay(self, game_file, prefix_actions):
        self.calls.append(("replay", game_file, tuple(prefix_actions)))
        return [f"worker-{self.worker_id}-replay-{len(prefix_actions)}"], {
            "admissible_commands": [[f"action-{self.worker_id}"]],
            "won": [False],
            "extra.gamefile": [game_file],
        }, False


def test_vector_env_subset_preserves_requested_worker_order(monkeypatch):
    monkeypatch.setattr(alfworld_envs.ray, "get", lambda values: values)
    vector = AlfworldEnvs.__new__(AlfworldEnvs)
    vector.num_processes = 4
    vector.multi_modal = False
    vector.workers = [FakeRemoteAlfworldWorker(index) for index in range(4)]
    vector.prev_admissible_commands = [None] * 4
    vector.active_processes = 4
    vector.active_worker_indices = list(range(4))

    obs, _, infos = vector.reset_subset([2, 0], ["game-2", "game-0"])
    assert obs == ["worker-2-reset", "worker-0-reset"]
    assert [info["extra.gamefile"] for info in infos] == ["game-2", "game-0"]

    next_obs, _, _, _, _ = vector.step(["a2", "a0"])
    assert next_obs == ["worker-2-step", "worker-0-step"]
    assert vector.workers[1].calls == []
    assert vector.workers[3].calls == []
    assert vector.get_admissible_commands == [["action-2"], ["action-0"]]


def test_selected_step_and_replay_do_not_change_legacy_active_subset(monkeypatch):
    monkeypatch.setattr(alfworld_envs.ray, "get", lambda values: values)
    vector = AlfworldEnvs.__new__(AlfworldEnvs)
    vector.num_processes = 4
    vector.multi_modal = False
    vector.workers = [FakeRemoteAlfworldWorker(index) for index in range(4)]
    vector.prev_admissible_commands = [None] * 4
    vector.active_processes = 2
    vector.active_worker_indices = [0, 1]

    obs, _, _, _, _ = vector.step_selected([3, 1], ["a3", "a1"])
    assert obs == ["worker-3-step", "worker-1-step"]
    assert vector.active_worker_indices == [0, 1]

    obs, _, dones, infos = vector.replay_selected(
        [2, 0], ["game-2", "game-0"], [["a", "b"], []]
    )
    assert obs == ["worker-2-replay-2", "worker-0-replay-0"]
    assert dones == [False, False]
    assert [info["extra.gamefile"] for info in infos] == ["game-2", "game-0"]
    assert vector.active_worker_indices == [0, 1]
