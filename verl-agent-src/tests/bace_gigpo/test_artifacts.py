import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from recipe.bace_gigpo.artifacts import BaceArtifactStore, SCHEMA_VERSION, jsonable
from recipe.bace_gigpo.validate_trace import validate_step
from verl import DataProto
from recipe.bace_gigpo.rollout_collector import BaceTrajectoryCollector


@dataclass
class Example:
    value: int


def test_jsonable_handles_numpy_and_dataclass():
    value = jsonable({"array": np.array([1, 2]), "scalar": np.float32(1.5), "item": Example(3)})
    assert value == {"array": [1, 2], "scalar": 1.5, "item": {"value": 3}}


def test_store_flushes_streams_and_writes_manifest_summary(tmp_path):
    store = BaceArtifactStore(tmp_path, 7, {"experiment": "unit"})
    store.append("replay_attempts", {"category": "OBSERVATION_MISMATCH"})
    stream = store.step_dir / "replay_attempts.jsonl"
    assert stream.exists()
    assert json.loads(stream.read_text().splitlines()[0])["schema_version"] == SCHEMA_VERSION
    store.finalize({"status": "complete"})
    summary = json.loads((store.step_dir / "summary.json").read_text())
    manifest = json.loads((store.step_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert summary["record_counts"]["replay_attempts"] == 1


def test_store_uses_attempt_suffix_for_repeated_step(tmp_path):
    first = BaceArtifactStore(tmp_path, 1, {})
    second = BaceArtifactStore(tmp_path, 1, {})
    assert first.step_dir.name == "step_00000001"
    assert second.step_dir.name == "step_00000001_attempt_01"


def test_disabled_mode_is_configurable_at_caller(tmp_path):
    # The collector owns the switch; the store itself remains deterministic when used directly.
    store = BaceArtifactStore(tmp_path, 0, {"enabled": False})
    store.close()
    assert (store.step_dir / "summary.json").exists()


def build_valid_trace(tmp_path):
    store = BaceArtifactStore(tmp_path, 3, {
        "total_leaf_budget": 2,
        "max_branches_per_anchor": 1,
        "include_token_arrays": True,
    })
    store.append("roots", {"root_id": "root-1", "selected_for_training": True})
    store.append("leaves", {
        "root_id": "root-1",
        "occurrence_id": "root-1:1",
        "action_format_valid": True,
        "action_environment_valid": False,
        "action_identity": "invalid::bad action",
        "response_token_ids": [3, 4],
        "response_loss_mask": [1, 1],
        "old_log_probs": [-0.1, -0.2],
    })
    store.append("topology", {
        "root_ids": ["root-1"],
        "plan": {"tasks": {"task-1": {
            "final_root_count": 1,
            "final_branch_count": 1,
            "effective_anchor_count": 1,
        }}},
    })
    store.append("acquisition_rounds", {
        "round": 1,
        "diagnostics": [{
            "task_id": "task-1",
            "selected_allocation": {"anchor-1": 1},
            "global_allocation": {
                "solver": "quota_aware_exact_dp",
                "num_anchors": 1,
                "branch_quota": 1,
                "total_information_capacity": 1,
                "reachable_state_count": 3,
                "optimal_value": 0.1,
                "optimal_tie_count": 1,
                "selected_allocation": {"anchor-1": 1},
                "solver_wall_time_ms": 0.1,
            },
        }],
        "posterior_snapshot": {"task-1": {"anchor-1": {
            "invalid::bad action": {"alpha": 1, "beta": 1},
            "valid::look": {"alpha": 1, "beta": 1},
        }}},
    })
    store.append("replay_attempts", {
        "phase": "origin_transition_validation",
        "request": {"request_id": "request-1"},
        "result": {"replay_ok": True, "category": "VALIDATED"},
    })
    store.append("branches", {
        "branch_id": "branch-1",
        "request_id": "request-1",
        "origin_occurrence_id": "root-1:1",
        "terminal_reward": 0.0,
        "suffix_occurrence_count": 0,
        "suffix_occurrence_ids": [],
    })
    store.append("trainable_occurrences", {
        "source_type": "branch_origin",
        "traj_uid": "branch-1",
        "leaf_id": "branch-1",
        "leaf_advantage": 0.5,
        "action_identity": "invalid::bad action",
        "response_token_ids": [3, 4],
        "response_loss_mask": [1, 1],
        "rollout_old_log_probs": [-0.1, -0.2],
    })
    store.finalize({
        "status": "complete",
        "diagnostics": {"old_log_prob_by_source": {
            "branch_origin": {"max_abs_diff": 0.2, "probability_max_abs_diff": 0.05}
        }},
    })
    return store.step_dir


def test_trace_validator_accepts_complete_trace_and_reports_logprob_stats(tmp_path):
    step_dir = build_valid_trace(tmp_path)
    result = validate_step(step_dir)
    assert result["ok"], result["errors"]
    assert result["checks"]["copied_branch_origins_audited"] == 1
    assert result["checks"]["old_log_prob_by_source"]["branch_origin"]["max_abs_diff"] == 0.2


def test_trace_validator_rejects_copied_origin_and_threshold_violations(tmp_path):
    step_dir = build_valid_trace(tmp_path)
    stream = step_dir / "trainable_occurrences.jsonl"
    record = json.loads(stream.read_text())
    record["response_token_ids"] = [9, 4]
    stream.write_text(json.dumps(record) + "\n")

    result = validate_step(step_dir, max_recomputed_logprob_diff=0.1)
    assert not result["ok"]
    assert any("copied response token ids" in error for error in result["errors"])
    assert any("exceeds threshold" in error for error in result["errors"])


def test_trace_validator_rejects_wrong_global_allocation_quota(tmp_path):
    step_dir = build_valid_trace(tmp_path)
    stream = step_dir / "acquisition_rounds.jsonl"
    record = json.loads(stream.read_text())
    record["diagnostics"][0]["global_allocation"]["branch_quota"] = 2
    stream.write_text(json.dumps(record) + "\n")

    result = validate_step(step_dir)
    assert not result["ok"]
    assert any("uses 1 branches, expected 2" in error for error in result["errors"])


def test_training_diagnostics_tolerates_missing_optional_advantages(tmp_path):
    collector = object.__new__(BaceTrajectoryCollector)
    collector.artifact_store = BaceArtifactStore(tmp_path, 0, {})
    collector.config = SimpleNamespace(
        algorithm=SimpleNamespace(
            bace=SimpleNamespace(
                artifacts={"include_token_arrays": False},
            )
        )
    )
    collector.trace_diagnostics = {}
    collector.last_bace_metrics = {}
    batch = DataProto.from_single_dict(
        data={"responses": torch.zeros((2, 1), dtype=torch.long)}
    )
    batch.non_tensor_batch["source_type"] = np.asarray(["root", "branch"], dtype=object)
    collector.save_training_diagnostics(batch)
    records = (collector.artifact_store.step_dir / "trainable_occurrences.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert '"macro_advantage": null' in records[0]


def test_trace_validator_accepts_occurrence_level_gigpo_macro_values(tmp_path):
    store = BaceArtifactStore(tmp_path, 4, {"advantage_semantics": "gigpo_macro"})
    store.append("roots", {"root_id": "root-1"})
    store.append("leaves", {
        "root_id": "root-1",
        "occurrence_id": "root-1:0",
        "action_format_valid": True,
        "action_environment_valid": True,
        "action_identity": "valid::look",
    })
    store.append("topology", {"root_ids": ["root-1"], "plan": {"tasks": {}}})
    for index, macro in enumerate((-0.5, 0.5)):
        store.append("trainable_occurrences", {
            "source_type": "root",
            "occurrence_id": f"root-1:{index}",
            "traj_uid": "root-1",
            "leaf_id": "root-1",
            "macro_advantage": macro,
            "local_advantage": 0.0,
            "occurrence_advantage": macro,
        })
    store.finalize({"status": "complete"})

    result = validate_step(store.step_dir)
    assert result["ok"], result["errors"]
