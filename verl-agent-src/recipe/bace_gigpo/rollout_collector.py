from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid

import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy

from .anchor_index import AnchorIndex
from .artifacts import BaceArtifactStore
from .coordinator import ExactBatchErvCoordinator, ExpectedErvCoordinator, FixedTopologyCoordinator
from .competence import CompetenceHistory
from .replay.alfworld import ReplayAdapter
from .root_store import build_root_event_logs, task_family_from_reset_key
from .topology import DynamicTopologyPlanner, ExactBatchTopologyPlanner
from .frontier import BACEFrontierOrchestrator


class BaceTrajectoryCollector(TrajectoryCollector):
    """Natural roots plus validated, optionally sequential ERV branches."""

    STATE_VERSION = 1

    def close_branch_pool(self):
        """Release only a dedicated branch pool; a reused main pool is borrowed."""
        if not getattr(self, "branch_pool_owns_resources", False):
            return
        close = getattr(self.branch_envs, "close", None)
        if callable(close):
            close()

    def __init__(self, config, tokenizer, processor, branch_envs):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor)
        self.branch_envs = branch_envs
        bace_config = config.algorithm.bace
        self.variant = str(bace_config.get("variant", "legacy"))
        if self.variant not in {"legacy", "batch_erv_exact"}:
            raise ValueError("algorithm.bace.variant must be legacy or batch_erv_exact")
        self.topology = str(bace_config.get("topology", "fixed"))
        self.acquisition = str(bace_config.get("acquisition", "erv"))
        self.dynamic_root_generation = str(
            bace_config.get("dynamic_root_generation", "preallocated")
        )
        self.staged_root_batching = str(
            bace_config.get("staged_root_batching", "sequential")
        )
        self.branch_execution_mode = str(
            bace_config.get("branch_execution_mode", "legacy_dense")
        )
        self.branch_pool_mode = str(
            bace_config.get("branch_pool_mode", "dedicated")
        )
        self.tree_credit_mode = str(
            bace_config.get("tree_credit_mode", "current")
        )
        self.macro_normalization_mode = str(
            bace_config.get("macro_normalization_mode", "stable_occurrence")
        )
        self.tree_ppo_padding_mode = str(
            bace_config.get("tree_ppo_padding_mode", "copy_trainable")
        )
        if self.tree_credit_mode not in {
            "current", "o1_local", "o1_tree_macro", "o1_full_tree"
        }:
            raise ValueError(
                "algorithm.bace.tree_credit_mode must be current, o1_local, "
                "o1_tree_macro, or o1_full_tree"
            )
        if self.macro_normalization_mode != "stable_occurrence":
            raise ValueError(
                "C0/C1/C2/C3 currently require "
                "algorithm.bace.macro_normalization_mode=stable_occurrence"
            )
        if self.tree_ppo_padding_mode not in {"copy_trainable", "zero_loss"}:
            raise ValueError(
                "algorithm.bace.tree_ppo_padding_mode must be "
                "copy_trainable or zero_loss"
            )
        self.branch_pool_owns_resources = self.branch_pool_mode == "dedicated"
        pairwise_config = bace_config.get("pairwise", {})
        self.pairwise_mode = str(pairwise_config.get("mode", "full"))
        self.pairwise_batch_size = int(pairwise_config.get("batch_size", 2))
        self.capacity_correction_batch_size = int(
            bace_config.get("capacity_correction_batch_size", 1)
        )
        if self.capacity_correction_batch_size < 1:
            raise ValueError(
                "algorithm.bace.capacity_correction_batch_size must be positive"
            )
        self.root_active_executor = bool(
            bace_config.get("root_active_executor", False)
        )
        configured_quota_policy = bace_config.get("quota_policy", None)
        min_natural_roots = int(bace_config.get("min_natural_roots", 2))
        if configured_quota_policy is None:
            self.quota_policy = (
                "current"
                if min_natural_roots == 2
                else "conservative_rmin4"
                if min_natural_roots == 4
                else f"custom_rmin{min_natural_roots}"
            )
        else:
            self.quota_policy = str(configured_quota_policy)
        migration_config = bace_config.get("checkpoint_migration", {})
        self.checkpoint_migration_enabled = bool(
            migration_config.get("enabled", False)
        )
        self.checkpoint_migration_mode = str(
            migration_config.get("mode", "strict")
        )
        if self.checkpoint_migration_mode not in {
            "strict",
            "diagnostic_rmin2_to4",
            "pairwise_from_full",
        }:
            raise ValueError(
                "algorithm.bace.checkpoint_migration.mode must be strict or "
                "diagnostic_rmin2_to4 or pairwise_from_full"
            )
        if (
            self.checkpoint_migration_enabled
            and self.checkpoint_migration_mode not in {
                "diagnostic_rmin2_to4", "pairwise_from_full"
            }
        ):
            raise ValueError(
                "checkpoint_migration.enabled=true requires "
                "mode=diagnostic_rmin2_to4 or pairwise_from_full"
            )
        self.last_checkpoint_migration = None
        self.invalid_action_mode = str(
            bace_config.get("invalid_action_mode", "strict_identity")
        )
        self.tie_break_identity_mode = str(
            bace_config.get("tie_break_identity_mode", "legacy_uuid")
        )
        if self.tie_break_identity_mode not in AnchorIndex.IDENTITY_MODES:
            raise ValueError(
                "algorithm.bace.tie_break_identity_mode must be "
                "legacy_uuid or stable_v1"
            )
        if self.dynamic_root_generation not in {"preallocated", "staged"}:
            raise ValueError(
                "algorithm.bace.dynamic_root_generation must be preallocated or staged"
            )
        if self.staged_root_batching not in {"sequential", "packed", "frontier"}:
            raise ValueError(
                "algorithm.bace.staged_root_batching must be sequential, packed, or frontier"
            )
        if self.branch_execution_mode not in {"legacy_dense", "selected_worker"}:
            raise ValueError(
                "algorithm.bace.branch_execution_mode must be legacy_dense or selected_worker"
            )
        if self.branch_pool_mode not in {"dedicated", "main_reuse"}:
            raise ValueError(
                "algorithm.bace.branch_pool_mode must be dedicated or main_reuse"
            )
        if (
            self.branch_pool_mode == "main_reuse"
            and self.branch_execution_mode != "selected_worker"
        ):
            raise ValueError(
                "algorithm.bace.branch_pool_mode=main_reuse requires "
                "branch_execution_mode=selected_worker"
            )
        if self.branch_execution_mode == "selected_worker":
            required = (
                "get_observations_selected",
                "replay_selected",
                "step_selected",
            )
            missing = [name for name in required if not callable(getattr(branch_envs, name, None))]
            if missing:
                raise ValueError(
                    "selected_worker branch execution requires environment methods: "
                    + ", ".join(missing)
                )
        if self.staged_root_batching == "frontier" and (
            self.topology != "dynamic" or self.dynamic_root_generation != "staged"
        ):
            raise ValueError(
                "staged_root_batching=frontier requires topology=dynamic and "
                "dynamic_root_generation=staged"
            )
        if self.tree_credit_mode != "current" and self.staged_root_batching == "frontier":
            raise ValueError("C1/C2/C3 tree credit currently require packed rollout batching")
        if self.variant == "batch_erv_exact" and (
            self.topology != "dynamic"
            or self.dynamic_root_generation != "staged"
            or self.staged_root_batching != "packed"
        ):
            raise ValueError(
                "variant=batch_erv_exact requires topology=dynamic, "
                "dynamic_root_generation=staged, and staged_root_batching=packed"
            )
        if self.variant == "batch_erv_exact" and self.acquisition != "batch_erv_exact":
            raise ValueError(
                "variant=batch_erv_exact requires acquisition=batch_erv_exact"
            )
        if self.pairwise_mode not in {"full", "fixed", "stopping"}:
            raise ValueError("algorithm.bace.pairwise.mode must be full, fixed, or stopping")
        if self.pairwise_batch_size != 2:
            raise ValueError("algorithm.bace.pairwise.batch_size currently must equal 2")
        if self.pairwise_mode != "full" and self.variant != "batch_erv_exact":
            raise ValueError("Pairwise modes require variant=batch_erv_exact")
        self.competence_history = None
        self.topology_planner = None
        if self.topology == "dynamic":
            if self.acquisition not in {"erv", "batch_erv_exact"}:
                raise ValueError("Dynamic BACE topology requires ERV acquisition")
            initial_mean = bace_config.get("history_initial_mean")
            initial_strength = bace_config.get("history_initial_strength")
            if (initial_mean is None) != (initial_strength is None):
                raise ValueError(
                    "history_initial_mean and history_initial_strength must be set together"
                )
            if initial_mean is None:
                base_alpha = float(bace_config.history_base_alpha)
                base_beta = float(bace_config.history_base_beta)
            else:
                initial_mean = float(initial_mean)
                initial_strength = float(initial_strength)
                if not 0 < initial_mean < 1 or initial_strength <= 0:
                    raise ValueError(
                        "history initial mean must lie in (0, 1) and strength must be positive"
                    )
                base_alpha = initial_mean * initial_strength
                base_beta = (1.0 - initial_mean) * initial_strength
            self.competence_history = CompetenceHistory(
                base_alpha=base_alpha,
                base_beta=base_beta,
                forgetting=float(bace_config.history_forgetting),
                transfer_fraction=float(bace_config.history_transfer_fraction),
                min_strength=float(bace_config.history_min_strength),
                max_strength=float(bace_config.history_max_strength),
            )
            if self.variant == "batch_erv_exact":
                self.topology_planner = ExactBatchTopologyPlanner(
                    history=self.competence_history,
                    total_budget=int(bace_config.total_leaf_budget),
                    min_natural_roots=int(bace_config.min_natural_roots),
                    competence_threshold=float(bace_config.competence_threshold),
                    max_branches_per_anchor=int(bace_config.max_branches_per_anchor),
                    local_prior_strength=float(bace_config.local_prior_strength),
                    batch_erv_threshold=float(bace_config.batch_erv_threshold),
                    tie_abs_tolerance=float(bace_config.batch_erv_tie_abs_tolerance),
                    tie_rel_tolerance=float(bace_config.batch_erv_tie_rel_tolerance),
                    seed=int(config.env.seed),
                    invalid_action_mode=self.invalid_action_mode,
                    tie_break_identity_mode=self.tie_break_identity_mode,
                    capacity_correction_batch_size=self.capacity_correction_batch_size,
                )
            else:
                self.topology_planner = DynamicTopologyPlanner(
                    history=self.competence_history,
                    total_budget=int(bace_config.total_leaf_budget),
                    pilot_roots=int(bace_config.pilot_roots),
                    competence_threshold=float(bace_config.competence_threshold),
                    max_branches_per_anchor=int(bace_config.max_branches_per_anchor),
                    local_prior_strength=float(bace_config.local_prior_strength),
                    erv_threshold=float(bace_config.erv_threshold),
                    erv_temperature=float(bace_config.erv_temperature),
                    erv_mc_samples=int(bace_config.erv_mc_samples),
                    seed=int(config.env.seed),
                    invalid_action_mode=self.invalid_action_mode,
                )
        if self.acquisition == "random":
            self.coordinator = FixedTopologyCoordinator(
                root_count=int(bace_config.fixed_root_count),
                branch_count=int(bace_config.fixed_branch_count),
                seed=int(config.env.seed),
                invalid_action_mode=self.invalid_action_mode,
            )
        elif self.acquisition == "erv":
            self.coordinator = ExpectedErvCoordinator(
                root_count=int(bace_config.fixed_root_count),
                branch_count=int(bace_config.fixed_branch_count),
                max_branches_per_anchor=int(bace_config.max_branches_per_anchor),
                prior_strength=float(bace_config.local_prior_strength),
                erv_threshold=float(bace_config.erv_threshold),
                temperature=float(bace_config.erv_temperature),
                mc_samples=int(bace_config.erv_mc_samples),
                seed=int(config.env.seed),
                invalid_action_mode=self.invalid_action_mode,
            )
        elif self.acquisition == "batch_erv_exact":
            if self.variant != "batch_erv_exact":
                raise ValueError("acquisition=batch_erv_exact requires variant=batch_erv_exact")
            self.coordinator = ExactBatchErvCoordinator(
                max_branches_per_anchor=int(bace_config.max_branches_per_anchor),
                prior_strength=float(bace_config.local_prior_strength),
                threshold=float(bace_config.batch_erv_threshold),
                tie_abs_tolerance=float(bace_config.batch_erv_tie_abs_tolerance),
                tie_rel_tolerance=float(bace_config.batch_erv_tie_rel_tolerance),
                seed=int(config.env.seed),
                invalid_action_mode=self.invalid_action_mode,
                tie_break_identity_mode=self.tie_break_identity_mode,
                pairwise_mode=self.pairwise_mode,
                pairwise_batch_size=self.pairwise_batch_size,
            )
        else:
            raise ValueError(f"Unknown BACE acquisition mode: {self.acquisition}")
        self.replay_adapter = ReplayAdapter(
            manager=branch_envs,
            tokenizer=tokenizer,
            config=config,
            compare_action_set=bool(config.algorithm.bace.replay.compare_action_set),
        )
        self.max_origin_retries = int(config.algorithm.bace.replay.get("max_origin_retries", 0))
        self.main_group_size = None
        if self.branch_pool_mode == "main_reuse":
            raw_envs = getattr(branch_envs, "envs", None)
            runtime_group_size = getattr(raw_envs, "group_n", None)
            if runtime_group_size is None:
                raise ValueError(
                    "main_reuse requires the main environment to expose its "
                    "physical rollout group size"
                )
            self.main_group_size = int(runtime_group_size)
            if self.main_group_size <= 0:
                raise ValueError("main rollout group size must be positive")
            configured_group_size = int(config.env.rollout.n)
            if configured_group_size != self.main_group_size:
                raise ValueError(
                    "main environment group size does not match env.rollout.n: "
                    f"{self.main_group_size} != {configured_group_size}"
                )
            if int(bace_config.total_leaf_budget) != self.main_group_size:
                raise ValueError(
                    "Current packed Exact main-reuse requires total_leaf_budget "
                    "to equal the physical main rollout group size"
                )
        self.last_bace_metrics = {}
        self.current_step = 0
        self.artifact_store = None
        self.trace_diagnostics = {}
        self.replay_retry_metadata = {}
        self.orchestration_metrics = {}
        self.parameter_signature = {
            "variant": self.variant,
            "topology": self.topology,
            "acquisition": self.acquisition,
            "dynamic_root_generation": self.dynamic_root_generation,
            "staged_root_batching": self.staged_root_batching,
            "branch_execution_mode": self.branch_execution_mode,
            "invalid_action_mode": self.invalid_action_mode,
            "local_credit_mode": str(bace_config.local_credit_mode),
            "total_leaf_budget": int(bace_config.total_leaf_budget),
            "pilot_roots": int(bace_config.pilot_roots),
            "min_natural_roots": int(bace_config.get("min_natural_roots", 2)),
            "competence_threshold": float(bace_config.competence_threshold),
            "max_branches_per_anchor": int(bace_config.max_branches_per_anchor),
            "local_prior_strength": float(bace_config.local_prior_strength),
            "erv_threshold": float(bace_config.erv_threshold),
            "erv_temperature": float(bace_config.erv_temperature),
            "erv_mc_samples": int(bace_config.erv_mc_samples),
            "batch_erv_threshold": float(bace_config.batch_erv_threshold),
            "batch_erv_tie_abs_tolerance": float(
                bace_config.batch_erv_tie_abs_tolerance
            ),
            "batch_erv_tie_rel_tolerance": float(
                bace_config.batch_erv_tie_rel_tolerance
            ),
            "replay_compare_action_set": bool(bace_config.replay.compare_action_set),
            "replay_max_origin_retries": self.max_origin_retries,
        }
        # ``full`` is the historical one-round protocol.  Keep its persisted
        # signature byte-for-byte compatible with checkpoints produced before
        # the optional Pairwise feature existed; non-full modes are deliberately
        # distinct experiment semantics and must never resume each other.
        if self.pairwise_mode != "full":
            self.parameter_signature.update({
                "pairwise_mode": self.pairwise_mode,
                "pairwise_batch_size": self.pairwise_batch_size,
            })
        # ``1`` is the historical exact-capacity protocol.  Preserve existing
        # checkpoints byte-for-byte in that case; any accelerated correction
        # policy is a distinct sampling schedule and must resume strictly.
        if self.capacity_correction_batch_size != 1:
            self.parameter_signature["capacity_correction_batch_size"] = (
                self.capacity_correction_batch_size
            )
        # Preserve byte-for-byte compatibility with checkpoints written before
        # stable_v1 existed.  Stable runs add the field, so cross-mode resume is
        # rejected by the existing strict signature comparison.
        if self.tie_break_identity_mode != "legacy_uuid":
            self.parameter_signature["tie_break_identity_mode"] = (
                self.tie_break_identity_mode
            )
        # Tree credit only changes the learner-side use of a completed rollout.
        # It deliberately stays outside the controller/history signature so the
        # four variants can fork from one identical BACE checkpoint, as required
        # by the C0/C1/C2/C3 controlled-comparison protocol.

    def set_step(self, step):
        self.current_step = int(step)

    @property
    def requires_checkpoint_state(self) -> bool:
        return self.topology == "dynamic" and self.competence_history is not None

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "variant": self.variant,
            "topology": self.topology,
            "acquisition": self.acquisition,
            "parameters": self.parameter_signature,
            "current_step": self.current_step,
            "competence_history": (
                self.competence_history.state_dict()
                if self.competence_history is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported or invalid BACE collector state")
        for name, expected in (
            ("variant", self.variant),
            ("topology", self.topology),
            ("acquisition", self.acquisition),
        ):
            if state.get(name) != expected:
                raise ValueError(
                    f"BACE collector state mismatch for {name}: "
                    f"checkpoint={state.get(name)!r}, current={expected!r}"
                )
        saved_parameters = state.get("parameters")
        if not isinstance(saved_parameters, dict):
            raise ValueError("BACE collector checkpoint has invalid parameters")
        if saved_parameters != self.parameter_signature:
            all_keys = set(saved_parameters) | set(self.parameter_signature)
            differences = {
                key: {
                    "saved": saved_parameters.get(key),
                    "current": self.parameter_signature.get(key),
                }
                for key in sorted(all_keys)
                if saved_parameters.get(key) != self.parameter_signature.get(key)
            }
            rmin2_to4_allowed = (
                getattr(self, "checkpoint_migration_enabled", False)
                and getattr(self, "checkpoint_migration_mode", "strict")
                == "diagnostic_rmin2_to4"
                and differences
                == {
                    "min_natural_roots": {"saved": 2, "current": 4}
                }
            )
            # A continuation from Full Batch to a Pairwise arm intentionally
            # changes only the acquisition *schedule*.  Model/optimizer/RNG,
            # topology, prior, threshold, action identity, and competence
            # history must remain identical.  Historical Full checkpoints
            # predate the optional pairwise signature fields, so accept only
            # their absence and exactly the new mode/K=2 declaration.
            pairwise_from_full_allowed = (
                getattr(self, "checkpoint_migration_enabled", False)
                and getattr(self, "checkpoint_migration_mode", "strict")
                == "pairwise_from_full"
                and differences
                == {
                    "pairwise_batch_size": {"saved": None, "current": 2},
                    "pairwise_mode": {
                        "saved": None,
                        "current": self.pairwise_mode,
                    },
                }
                and self.pairwise_mode in {"fixed", "stopping"}
            )
            allowed = rmin2_to4_allowed or pairwise_from_full_allowed
            if not allowed:
                raise ValueError(
                    "BACE collector parameter signature does not match the checkpoint"
                )
            history_state_for_hash = state.get("competence_history")
            history_payload = json.dumps(
                history_state_for_hash,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.last_checkpoint_migration = {
                "migration_mode": (
                    "diagnostic_only"
                    if self.checkpoint_migration_mode == "diagnostic_rmin2_to4"
                    else "controlled_semantic_fork"
                ),
                "migration_policy": self.checkpoint_migration_mode,
                "allowed_diff": differences,
                "source_parameter_signature": saved_parameters,
                "current_parameter_signature": self.parameter_signature,
                "source_competence_history_sha256": hashlib.sha256(
                    history_payload
                ).hexdigest(),
            }
        else:
            self.last_checkpoint_migration = None
        history_state = state.get("competence_history")
        if self.requires_checkpoint_state:
            if history_state is None:
                raise ValueError("dynamic BACE checkpoint is missing competence history")
            self.competence_history.load_state_dict(history_state)
        elif history_state is not None:
            raise ValueError("checkpoint contains competence history for non-dynamic BACE")
        try:
            current_step = int(state["current_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("BACE collector state has an invalid current_step") from exc
        if current_step < 0:
            raise ValueError("BACE collector current_step must be non-negative")
        self.current_step = current_step

    def checkpoint_migration_metadata(
        self, source_checkpoint: str, source_global_step: int
    ) -> dict | None:
        if self.last_checkpoint_migration is None:
            return None
        metadata = dict(self.last_checkpoint_migration)
        metadata.update({
            "source_checkpoint": os.path.abspath(source_checkpoint),
            "source_global_step": int(source_global_step),
            "restored_collector_step": int(self.current_step),
            "source_competence_history_summary": (
                self.competence_history.snapshot()
                if self.competence_history is not None
                else None
            ),
        })
        return metadata

    def _start_artifact_store(self):
        artifact_config = self.config.algorithm.bace.get("artifacts", {})
        if not bool(artifact_config.get("enabled", True)):
            return None
        directory = artifact_config.get("directory")
        if not directory:
            directory = self.config.trainer.get("rollout_data_dir")
            if directory:
                directory = os.path.join(str(directory), "bace_trace")
            else:
                directory = os.path.join(str(self.config.trainer.default_local_dir), "bace_trace")
        return BaceArtifactStore(directory, self.current_step, {
            "variant": self.variant,
            "topology": self.topology,
            "acquisition": self.acquisition,
            "dynamic_root_generation": self.dynamic_root_generation,
            "staged_root_batching": self.staged_root_batching,
            "branch_execution_mode": self.branch_execution_mode,
            "frontier_batch_coalescing": {
                "enabled": bool(self.config.algorithm.bace.get("frontier_batch_coalescing", {}).get("enabled", False)),
                "max_batch_size": int(self.config.algorithm.bace.get("frontier_batch_coalescing", {}).get("max_batch_size", 0)),
                "min_batch_size": int(self.config.algorithm.bace.get("frontier_batch_coalescing", {}).get("min_batch_size", 1)),
            },
            "invalid_action_mode": self.invalid_action_mode,
            "tie_break_identity_mode": self.tie_break_identity_mode,
            "local_credit_mode": str(self.config.algorithm.bace.local_credit_mode),
            "tree_credit_mode": self.tree_credit_mode,
            "macro_normalization_mode": self.macro_normalization_mode,
            "tree_ppo_padding_mode": self.tree_ppo_padding_mode,
            "advantage_semantics": "gigpo_macro",
            "gigpo_mode": str(self.config.algorithm.gigpo.mode),
            "gigpo_compute_mean_std_cross_steps": bool(
                self.config.algorithm.gigpo.get("compute_mean_std_cross_steps", True)
            ),
            "gigpo_step_advantage_w": float(self.config.algorithm.gigpo.step_advantage_w),
            "total_leaf_budget": int(self.config.algorithm.bace.total_leaf_budget),
            "max_branches_per_anchor": int(self.config.algorithm.bace.max_branches_per_anchor),
            "min_natural_roots": int(self.config.algorithm.bace.get("min_natural_roots", 2)),
            "quota_policy": self.quota_policy,
            "batch_erv_threshold": float(self.config.algorithm.bace.get("batch_erv_threshold", 0.0)),
            "batch_erv_tie_abs_tolerance": float(
                self.config.algorithm.bace.get("batch_erv_tie_abs_tolerance", 1e-12)
            ),
            "batch_erv_tie_rel_tolerance": float(
                self.config.algorithm.bace.get("batch_erv_tie_rel_tolerance", 1e-10)
            ),
            "history_initial_mean": self.config.algorithm.bace.get("history_initial_mean"),
            "history_initial_strength": self.config.algorithm.bace.get(
                "history_initial_strength"
            ),
            "history_resolved_alpha": (
                self.competence_history.base_alpha
                if self.competence_history is not None
                else None
            ),
            "history_resolved_beta": (
                self.competence_history.base_beta
                if self.competence_history is not None
                else None
            ),
            "include_token_arrays": bool(artifact_config.get("include_token_arrays", True)),
        }, fsync=bool(artifact_config.get("fsync", False)))

    def _trace_roots(self, logs, selected_ids=None):
        if self.artifact_store is None:
            return
        selected_ids = set(selected_ids or ())
        parsed_count = 0
        invalid_count = 0
        for root in logs:
            payload = dataclasses.asdict(root)
            payload["selected_for_training"] = not selected_ids or root.root_id in selected_ids
            if not bool(self.config.algorithm.bace.artifacts.get("include_token_arrays", True)):
                for event in payload["events"]:
                    event.pop("prompt_token_ids", None)
                    event.pop("response_token_ids", None)
                    event.pop("response_loss_mask", None)
                    event.pop("old_log_probs", None)
            self.artifact_store.append("roots", payload)
            for event in root.events:
                if event.action_format_valid:
                    parsed_count += 1
                    invalid_count += int(event.action_environment_valid is False)
                event_payload = dataclasses.asdict(event)
                if not bool(self.config.algorithm.bace.artifacts.get("include_token_arrays", True)):
                    for key in ("prompt_token_ids", "response_token_ids", "response_loss_mask", "old_log_probs"):
                        event_payload.pop(key, None)
                event_payload.update({"root_id": root.root_id, "task_id": root.task_id})
                self.artifact_store.append("leaves", event_payload)
        self.trace_diagnostics["parsed_action_occurrences"] = parsed_count
        self.trace_diagnostics["environment_invalid_occurrences"] = invalid_count
        self.trace_diagnostics["invalid_occurrence_ratio"] = (
            invalid_count / parsed_count if parsed_count else 0.0
        )

    def _trace_anchors(self, root_logs):
        if self.artifact_store is None:
            return
        index = AnchorIndex(
            root_logs,
            invalid_action_mode=getattr(self, "invalid_action_mode", "strict_identity"),
            tie_break_identity_mode=getattr(
                self, "tie_break_identity_mode", "legacy_uuid"
            ),
        )
        candidate_action_counts = []
        invalid_fragmentation = {}
        for anchor in index._anchors.values():
            self.artifact_store.append("anchors", anchor)
            candidate_action_counts.append(len(anchor.observed_action_ids))
            invalid_fragmentation[anchor.anchor_id] = sum(
                str(action).startswith("invalid::") for action in anchor.observed_action_ids
            )
        self.trace_diagnostics["candidate_action_counts"] = candidate_action_counts
        self.trace_diagnostics["invalid_fragmentation_by_anchor"] = invalid_fragmentation

    def _trace_topology(self, topology_plan, root_logs):
        if self.artifact_store is None:
            return
        self.artifact_store.append("topology", {
            "plan": topology_plan,
            "root_count": len(root_logs),
            "root_ids": [root.root_id for root in root_logs],
        })

    def _record_branch_execution_cost(self, kind, steps, elapsed, jobs):
        """Record physical branch work, including operations with no validator result."""
        metric_names = {
            "validation_replay": (
                "branch_validation_replay_steps",
                "branch_validation_replay_seconds",
                "branch_validation_replay_jobs",
            ),
            "retry_replay": (
                "branch_retry_replay_steps",
                "branch_retry_replay_seconds",
                "branch_retry_replay_jobs",
            ),
            "execution_restore_replay": (
                "branch_execution_restore_replay_steps",
                "branch_execution_restore_replay_seconds",
                "branch_execution_restore_replay_jobs",
            ),
            "origin_transition": (
                "branch_origin_transition_steps",
                "branch_origin_transition_seconds",
                "branch_origin_transition_jobs",
            ),
        }
        if kind not in metric_names:
            raise ValueError(f"Unknown branch execution cost kind: {kind}")
        step_key, seconds_key, jobs_key = metric_names[kind]
        steps = int(steps)
        jobs = int(jobs)
        elapsed = float(elapsed)
        for target in (self.trace_diagnostics, self.orchestration_metrics):
            target[step_key] = target.get(step_key, 0) + steps
            target[seconds_key] = target.get(seconds_key, 0.0) + elapsed
            target[jobs_key] = target.get(jobs_key, 0) + jobs
            target["branch_total_environment_steps"] = (
                target.get("branch_total_environment_steps", 0) + steps
            )
            target["branch_total_environment_seconds"] = (
                target.get("branch_total_environment_seconds", 0.0) + elapsed
            )
            if kind != "origin_transition":
                target["branch_total_mechanical_replay_steps"] = (
                    target.get("branch_total_mechanical_replay_steps", 0) + steps
                )
                target["branch_total_mechanical_replay_seconds"] = (
                    target.get("branch_total_mechanical_replay_seconds", 0.0) + elapsed
                )
        # Backward-compatible diagnostic, now counting every physical branch
        # environment action rather than omitting execution restores.
        self.trace_diagnostics["replay_environment_steps"] = self.trace_diagnostics[
            "branch_total_environment_steps"
        ]
        if self.artifact_store is not None:
            self.artifact_store.append("branch_execution_costs", {
                "kind": kind,
                "jobs": jobs,
                "environment_steps": steps,
                "elapsed_seconds": elapsed,
            })

    def _trace_replay_results(
        self,
        requests,
        results,
        phase,
        elapsed,
        prefix_lengths=None,
        cost_kind="validation_replay",
    ):
        prefix_lengths = prefix_lengths or [len(request.parsed_action_prefix) for request in requests]
        categories = self.trace_diagnostics.setdefault("replay_category_counts", {})
        timings = self.trace_diagnostics.setdefault("phase_timing_seconds", {})
        timings[phase] = timings.get(phase, 0.0) + float(elapsed)
        self._record_branch_execution_cost(
            cost_kind,
            steps=sum(prefix_lengths),
            elapsed=elapsed,
            jobs=len(requests),
        )
        for request, result, prefix_length in zip(requests, results, prefix_lengths):
            categories[result.category] = categories.get(result.category, 0) + 1
            if self.artifact_store is not None:
                payload = {"phase": phase, "elapsed_seconds": elapsed, "prefix_length": prefix_length,
                           "cost_kind": cost_kind, "request": request, "result": result,
                           **self.replay_retry_metadata.get(request.request_id, {})}
                self.artifact_store.append("replay_attempts", payload)

    def _trace_branch_selections(self, requests):
        if not requests:
            return
        total = self.trace_diagnostics.get("selected_branch_count", 0) + len(requests)
        invalid = self.trace_diagnostics.get("invalid_selected_branch_count", 0) + sum(
            request.copied_action_identity_kind == "invalid" for request in requests
        )
        self.trace_diagnostics["selected_branch_count"] = total
        self.trace_diagnostics["invalid_selected_branch_count"] = invalid
        self.trace_diagnostics["invalid_branch_ratio"] = invalid / total

    def _fallback_request(self, request, root_logs, used_occurrences):
        index = AnchorIndex(
            root_logs,
            invalid_action_mode=getattr(self, "invalid_action_mode", "strict_identity"),
            tie_break_identity_mode=getattr(
                self, "tie_break_identity_mode", "legacy_uuid"
            ),
        )
        candidates = []
        for anchor in index.anchors_for_task(request.task_id):
            if repr(anchor.anchor_key) != repr(request.expected_anchor_key):
                continue
            candidates.extend(anchor.origins_by_action.get(request.selected_canonical_action, []))
        candidates = index.ordered_origins(candidates)
        origin = next((item for item in candidates if item.occurrence_id not in used_occurrences), None)
        if origin is None:
            return None
        root = index.root_for(origin.root_id)
        event = index.event_for(origin.occurrence_id)
        prefix = root.events[: event.step_index]
        return dataclasses.replace(
            request,
            request_id=str(uuid.uuid4()),
            origin_occurrence_id=origin.occurrence_id,
            environment_reset_key=root.environment_reset_key,
            target_turn=event.step_index,
            parsed_action_prefix=tuple(item.parsed_environment_action for item in prefix),
            prefix_observations=tuple(item.pre_action_observation for item in prefix),
            expected_observation=event.pre_action_observation,
            expected_action_set=event.admissible_actions,
            copied_response_token_ids=event.response_token_ids,
            copied_raw_model_response=event.raw_model_response,
            copied_response_loss_mask=event.response_loss_mask,
            copied_old_log_probs=event.old_log_probs,
            original_prompt_token_ids=event.prompt_token_ids,
            remaining_horizon=event.remaining_horizon,
            copied_parsed_environment_action=event.parsed_environment_action,
            copied_action_identity=event.action_identity or event.canonical_action,
            copied_action_identity_kind=event.action_identity_kind,
            copied_action_environment_valid=event.action_environment_valid,
            expected_post_action_observation=event.post_action_observation,
            expected_immediate_reward=event.reward,
            expected_post_action_done=event.done,
        )

    def save_training_diagnostics(self, batch):
        if self.artifact_store is None:
            return
        artifact_started = time.monotonic()
        response_mask = batch.batch.get("response_mask")
        rollout_lp = batch.batch.get("rollout_log_probs")
        recomputed_lp = batch.batch.get("old_log_probs")
        advantages = batch.batch.get("advantages")
        responses = batch.batch.get("responses")
        include_token_arrays = bool(
            self.config.algorithm.bace.artifacts.get("include_token_arrays", True)
        )
        token_counts = {"root": 0, "branch": 0}
        source_counts = {}
        old_log_prob_diffs = []
        source_log_prob_stats = {}
        for branch_record in batch.meta_info.get("tree_credit_branch_evidence", []):
            self.artifact_store.append("tree_credit_branches", branch_record)

        def metadata_value(key, index, default=None):
            values = batch.non_tensor_batch.get(key)
            if values is None:
                return default
            try:
                return values[index]
            except (IndexError, KeyError, TypeError):
                return default

        for index in range(len(batch)):
            mask = response_mask[index].detach().cpu().numpy().astype(bool) if response_mask is not None else None
            rollout = rollout_lp[index].detach().cpu().numpy() if rollout_lp is not None else None
            recomputed = recomputed_lp[index].detach().cpu().numpy() if recomputed_lp is not None else None
            diff = None
            mean_diff = None
            probability_max_diff = None
            active_rollout = None
            active_recomputed = None
            if rollout is not None and recomputed is not None and mask is not None:
                active_rollout = rollout[-len(mask):][mask]
                active_recomputed = recomputed[-len(mask):][mask]
                absolute_diff = np.abs(active_rollout - active_recomputed)
                diff = float(np.max(absolute_diff)) if mask.any() else 0.0
                mean_diff = float(np.mean(absolute_diff)) if mask.any() else 0.0
                probability_max_diff = (
                    float(np.max(np.abs(np.exp(active_rollout) - np.exp(active_recomputed))))
                    if mask.any() else 0.0
                )
                old_log_prob_diffs.append(diff)
            source_type = str(metadata_value("source_type", index, "unknown"))
            if source_type == "ppo_padding":
                self.trace_diagnostics["tree_credit_ppo_padding_rows"] = (
                    self.trace_diagnostics.get("tree_credit_ppo_padding_rows", 0) + 1
                )
                continue
            if bool(metadata_value("bace_ppo_is_padding_copy", index, False)):
                token_count = int(mask.sum()) if mask is not None else 0
                masked_advantage_mean = None
                if advantages is not None and mask is not None and mask.any():
                    values = advantages[index].detach().cpu().numpy()
                    masked_advantage_mean = float(values[-len(mask):][mask].mean())
                self.artifact_store.append("ppo_training_copies", {
                    "physical_batch_index": index,
                    "copy_source_row": int(metadata_value(
                        "bace_ppo_copy_source_row", index, -1
                    )),
                    "copy_of_edge_id": metadata_value(
                        "bace_ppo_copy_of_edge_id", index
                    ),
                    "source_type": source_type,
                    "task_id": metadata_value("uid", index),
                    "traj_uid": metadata_value("traj_uid", index),
                    "occurrence_id": metadata_value("occurrence_id", index),
                    "response_token_count": token_count,
                    "masked_token_advantage_mean": masked_advantage_mean,
                })
                self.trace_diagnostics["tree_credit_ppo_trainable_copy_rows"] = (
                    self.trace_diagnostics.get(
                        "tree_credit_ppo_trainable_copy_rows", 0
                    ) + 1
                )
                continue
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            if diff is not None:
                stats = source_log_prob_stats.setdefault(
                    source_type, {"max": 0.0, "sum": 0.0, "count": 0, "probability_max": 0.0}
                )
                stats["max"] = max(stats["max"], diff)
                stats["sum"] += float(np.abs(active_rollout - active_recomputed).sum())
                stats["count"] += int(len(active_rollout))
                stats["probability_max"] = max(stats["probability_max"], probability_max_diff)
            token_count = int(mask.sum()) if mask is not None else 0
            token_counts["root" if source_type == "root" else "branch"] += token_count
            masked_advantage_mean = None
            if advantages is not None and mask is not None and mask.any():
                values = advantages[index].detach().cpu().numpy()
                masked_advantage_mean = float(values[-len(mask):][mask].mean())
            payload = {"index": index, "task_id": metadata_value("uid", index),
                       "leaf_id": metadata_value("leaf_id", index),
                       "traj_uid": metadata_value("traj_uid", index),
                       "occurrence_id": metadata_value("occurrence_id", index),
                       "parsed_environment_action": metadata_value("projected_action", index),
                       "action_identity": metadata_value("action_identity", index),
                       "action_format_valid": metadata_value("is_action_format_valid", index),
                       "action_environment_valid": metadata_value("is_action_environment_valid", index),
                       "anchor": metadata_value("anchor_obs", index),
                       "terminal_reward": metadata_value("episode_rewards", index),
                       "episode_success": bool(float(metadata_value("episode_rewards", index, 0.0)) > 0),
                       "success_scope": (
                           "root" if source_type == "root"
                           else "branch" if source_type.startswith("branch_")
                           else "unknown"
                       ),
                       "macro_advantage": metadata_value("bace_macro_advantage", index),
                       "local_advantage": metadata_value("bace_local_advantage", index),
                       "occurrence_advantage": metadata_value("bace_occurrence_advantage", index),
                       "source_type": source_type,
                       "masked_token_advantage_mean": masked_advantage_mean,
                       "rollout_vs_recomputed_old_log_prob_max_abs_diff": diff,
                       "rollout_vs_recomputed_old_log_prob_mean_abs_diff": mean_diff,
                       "rollout_vs_recomputed_probability_max_abs_diff": probability_max_diff,
                       "response_token_count": token_count}
            tree_trace_fields = {
                "edge_id": "bace_edge_id",
                "root_id": "bace_root_id",
                "direct_leaf_ids": "bace_direct_leaf_ids",
                "descendant_leaf_ids": "bace_descendant_leaf_ids",
                "num_direct_continuations": "bace_num_direct_continuations",
                "num_descendant_leaves": "bace_num_descendant_leaves",
                "g_original": "bace_g_original",
                "g_direct_mean": "bace_g_direct_mean",
                "g_descendant_mean": "bace_g_descendant_mean",
                "macro_base_stable": "bace_macro_base_stable",
                "macro_descendant_mean": "bace_macro_descendant_mean",
                "local_current": "bace_local_current",
                "local_c1": "bace_local_c1",
                "local_c3": "bace_local_c3",
                "final_current": "bace_final_current",
                "final_c1": "bace_final_c1",
                "final_c2": "bace_final_c2",
                "final_c3": "bace_final_c3",
                "tree_credit_mode": "bace_tree_credit_mode",
                "macro_normalization_mode": "bace_macro_normalization_mode",
            }
            for trace_name, metadata_name in tree_trace_fields.items():
                value = metadata_value(metadata_name, index)
                if value is not None:
                    payload[trace_name] = value
            if metadata_value("tree_origin_occurrence_id", index) is not None:
                payload["origin_occurrence_id"] = metadata_value(
                    "tree_origin_occurrence_id", index
                )
                payload["parent_root_id"] = metadata_value(
                    "tree_parent_root_id", index
                )
                payload["step_index"] = metadata_value("step_index", index)
            if include_token_arrays and mask is not None:
                response_values = responses[index].detach().cpu().numpy() if responses is not None else None
                payload.update({
                    "response_token_ids": response_values[-len(mask):][mask] if response_values is not None else None,
                    "response_loss_mask": mask[mask].astype(np.int8),
                    "rollout_old_log_probs": active_rollout,
                    "recomputed_old_log_probs": active_recomputed,
                })
            self.artifact_store.append("trainable_occurrences", payload)
        total_tokens = token_counts["root"] + token_counts["branch"]
        self.trace_diagnostics.update({
            "tree_credit_ppo_physical_rows": int(len(batch)),
            "tree_credit_unique_trainable_rows": int(sum(source_counts.values())),
            "source_occurrence_counts": source_counts,
            "root_response_tokens": token_counts["root"],
            "branch_response_tokens": token_counts["branch"],
            "root_token_fraction": token_counts["root"] / total_tokens if total_tokens else 0.0,
            "branch_token_fraction": token_counts["branch"] / total_tokens if total_tokens else 0.0,
            "old_log_prob_max_abs_diff": max(old_log_prob_diffs, default=None),
            "old_log_prob_by_source": {
                source: {
                    "max_abs_diff": stats["max"],
                    "mean_abs_diff": stats["sum"] / stats["count"] if stats["count"] else None,
                    "probability_max_abs_diff": stats["probability_max"],
                    "token_count": stats["count"],
                }
                for source, stats in source_log_prob_stats.items()
            },
        })
        branch_occurrences = sum(
            count
            for source, count in source_counts.items()
            if str(source).startswith("branch_")
        )
        self.last_bace_metrics.update({
            "root_trainable_occurrences": int(source_counts.get("root", 0)),
            "branch_trainable_occurrences": int(branch_occurrences),
            "root_trainable_tokens": int(token_counts["root"]),
            "branch_trainable_tokens": int(token_counts["branch"]),
        })
        artifact_elapsed = float(time.monotonic() - artifact_started)
        self.trace_diagnostics["artifact_write_seconds"] = artifact_elapsed
        self.last_bace_metrics["artifact_write_seconds"] = artifact_elapsed
        self.artifact_store.finalize({"status": "complete", "bace_metrics": self.last_bace_metrics,
                                      "diagnostics": self.trace_diagnostics})

    @staticmethod
    def _set_lineage(batch, source_type, leaf_ids):
        size = len(batch)
        batch.non_tensor_batch["source_type"] = np.array([source_type] * size, dtype=object)
        batch.non_tensor_batch["leaf_id"] = np.asarray(leaf_ids, dtype=object)

    @staticmethod
    def _annotate_tree_lineage(batch, branch_origin_by_id):
        """Attach exact root/branch ancestry after all physical rows are merged."""
        sources = np.asarray(batch.non_tensor_batch["source_type"], dtype=object)
        occurrences = np.asarray(batch.non_tensor_batch["occurrence_id"], dtype=object)
        trajectories = np.asarray(batch.non_tensor_batch["traj_uid"], dtype=object)
        natural_parent_by_occurrence = {
            str(occurrence): str(trajectory)
            for source, occurrence, trajectory in zip(sources, occurrences, trajectories)
            if str(source) == "root"
        }
        origin_ids = []
        parent_root_ids = []
        for source, occurrence, trajectory in zip(sources, occurrences, trajectories):
            source = str(source)
            trajectory = str(trajectory)
            if source == "root":
                natural_origin = str(occurrence)
                parent_root = trajectory
            elif source in {"branch_origin", "branch_suffix"}:
                if trajectory not in branch_origin_by_id:
                    raise ValueError(
                        f"Missing tree-credit origin metadata for branch {trajectory}"
                    )
                natural_origin = str(branch_origin_by_id[trajectory])
                if natural_origin not in natural_parent_by_occurrence:
                    raise ValueError(
                        f"Branch {trajectory} references unavailable natural origin "
                        f"{natural_origin}"
                    )
                parent_root = natural_parent_by_occurrence[natural_origin]
            else:
                raise ValueError(f"Unknown BACE occurrence source: {source}")
            origin_ids.append(natural_origin)
            parent_root_ids.append(parent_root)
        batch.non_tensor_batch["tree_origin_occurrence_id"] = np.asarray(
            origin_ids, dtype=object
        )
        batch.non_tensor_batch["tree_parent_root_id"] = np.asarray(
            parent_root_ids, dtype=object
        )
        return batch

    @staticmethod
    def _rollout_generation_metrics(batch):
        """Count newly generated response tokens without counting copied origins."""
        responses = batch.batch.get("responses")
        attention_mask = batch.batch.get("attention_mask")
        if responses is None or attention_mask is None:
            return {
                "root_generated_tokens": 0,
                "branch_generated_tokens": 0,
                "root_generated_occurrences": 0,
                "branch_generated_occurrences": 0,
            }
        response_length = int(responses.shape[-1])
        response_attention = attention_mask[:, -response_length:]
        source_types = np.asarray(
            batch.non_tensor_batch.get("source_type", ["unknown"] * len(batch)),
            dtype=object,
        )
        result = {
            "root_generated_tokens": 0,
            "branch_generated_tokens": 0,
            "root_generated_occurrences": 0,
            "branch_generated_occurrences": 0,
        }
        for index, source_type in enumerate(source_types):
            source_type = str(source_type)
            if source_type == "root":
                prefix = "root"
            elif source_type == "branch_suffix":
                prefix = "branch"
            else:
                # branch_origin is copied from a root occurrence, not generated
                # again during branch execution.
                continue
            result[f"{prefix}_generated_tokens"] += int(
                response_attention[index].sum().item()
            )
            result[f"{prefix}_generated_occurrences"] += 1
        return result

    @staticmethod
    def _refresh_success_rate(batch):
        """Keep the legacy per-row metric aligned with the canonical terminal reward."""
        if batch is None or "episode_rewards" not in batch.non_tensor_batch:
            return
        rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float32)
        batch.non_tensor_batch["success_rate"] = (rewards > 0).astype(np.float32)

    @staticmethod
    def _episode_success_metrics(batch):
        """Summarize root, branch, and mixed outcomes by unique trajectory."""
        if batch is None or "episode_rewards" not in batch.non_tensor_batch:
            return {}
        traj_uids = np.asarray(batch.non_tensor_batch.get("traj_uid", np.arange(len(batch))), dtype=object)
        rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float32)
        source_types = np.asarray(
            batch.non_tensor_batch.get("source_type", ["unknown"] * len(batch)), dtype=object
        )
        episodes = {}
        for traj_uid, reward, source_type in zip(traj_uids, rewards, source_types):
            key = str(traj_uid)
            success = bool(float(reward) > 0)
            scope = "root" if str(source_type) == "root" else (
                "branch" if str(source_type).startswith("branch_") else "mixed"
            )
            record = episodes.setdefault(key, {"success": success, "scopes": set()})
            if record["success"] != success:
                raise ValueError(f"Trajectory {key} has inconsistent terminal rewards")
            record["scopes"].add(scope)

        grouped = {"root": [], "branch": [], "mixed": []}
        for record in episodes.values():
            scopes = record["scopes"]
            scope = "root" if scopes == {"root"} else "branch" if scopes == {"branch"} else "mixed"
            grouped[scope].append(float(record["success"]))
            if scope != "mixed":
                grouped["mixed"].append(float(record["success"]))

        metrics = {}
        for scope, values in grouped.items():
            if not values:
                continue
            prefix = f"{scope}_success"
            metrics[f"{prefix}_count"] = int(sum(values))
            metrics[f"{prefix}_episodes"] = int(len(values))
            metrics[f"{prefix}_rate"] = float(np.mean(values))
        return metrics

    @staticmethod
    def _normalize_non_tensor_fields(batches):
        """Keep nested metadata values behind a single batch dimension."""
        for batch in batches:
            for key, value in batch.non_tensor_batch.items():
                value = np.asarray(value)
                if value.ndim <= 1:
                    continue
                normalized = np.empty(len(batch), dtype=object)
                normalized[:] = [value[index] for index in range(len(batch))]
                batch.non_tensor_batch[key] = normalized

    @staticmethod
    def _align_non_tensor_fields(batches):
        all_keys = set().union(*(batch.non_tensor_batch.keys() for batch in batches))
        for key in all_keys:
            exemplar = next((batch.non_tensor_batch[key] for batch in batches if key in batch.non_tensor_batch), None)
            for batch in batches:
                if key in batch.non_tensor_batch:
                    continue
                if exemplar.dtype == object:
                    batch.non_tensor_batch[key] = np.array([None] * len(batch), dtype=object)
                else:
                    batch.non_tensor_batch[key] = np.zeros(len(batch), dtype=exemplar.dtype)

    def _concat_batches(self, batches):
        self._normalize_non_tensor_fields(batches)
        self._align_non_tensor_fields(batches)
        merged = DataProto.concat(batches)
        numeric_metadata = {
            "episode_rewards": np.float32,
            "episode_lengths": np.float32,
            "tool_callings": np.float32,
        }
        for key in merged.non_tensor_batch:
            if "success_rate" in key:
                numeric_metadata[key] = np.float32
        for key, dtype in numeric_metadata.items():
            if key in merged.non_tensor_batch:
                merged.non_tensor_batch[key] = np.asarray(
                    merged.non_tensor_batch[key], dtype=dtype
                )
        return merged

    def _runtime_main_group_size(self, envs):
        """Return the physical main-pool stride, independent of BACE budget."""
        fallback = int(self.config.algorithm.bace.total_leaf_budget)
        raw_envs = getattr(envs, "envs", None)
        runtime_group_size = getattr(raw_envs, "group_n", None)
        group_size = (
            int(runtime_group_size) if runtime_group_size is not None else fallback
        )
        if group_size <= 0:
            raise ValueError("Root environment group size must be positive")
        return group_size

    def _collect_root_wave(
        self,
        gen_batch,
        actor_rollout_wg,
        envs,
        task_indices,
        root_slots,
        task_uids,
        reset_keys,
    ):
        physical_stride = self._runtime_main_group_size(envs)
        if any(int(slot) < 0 or int(slot) >= physical_stride for slot in root_slots):
            raise ValueError(
                "Root slot exceeds the runtime main-environment group size"
            )
        worker_indices = [
            int(task_idx) * physical_stride + int(slot)
            for task_idx, slot in zip(task_indices, root_slots)
        ]
        wave_batch = gen_batch.select_idxs(task_indices)
        wave_uids = [task_uids[index] for index in task_indices]
        if getattr(self, "root_active_executor", False):
            rollout = self._active_root_multi_turn_loop(
                gen_batch=wave_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                worker_indices=worker_indices,
                reset_keys=reset_keys,
                uid_batch=wave_uids,
                task_batch_indices=task_indices,
            )
        else:
            self.orchestration_metrics["root_active_executor_enabled"] = 0.0
            rollout = self.vanilla_multi_turn_loop(
                gen_batch=wave_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                reset_options={
                    "_bace_worker_indices": worker_indices,
                    "_bace_reset_keys": reset_keys,
                },
                uid_batch=wave_uids,
                task_batch_indices=task_indices,
            )
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = rollout
        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )

    def _active_root_multi_turn_loop(
        self,
        gen_batch,
        actor_rollout_wg,
        envs,
        worker_indices,
        reset_keys,
        uid_batch,
        task_batch_indices,
    ):
        """Run one packed root wave while submitting only unfinished roots."""
        required = (
            "reset_selected",
            "get_observations_selected",
            "get_tasks_selected",
            "step_selected",
        )
        missing = [name for name in required if not callable(getattr(envs, name, None))]
        if missing:
            raise ValueError(
                "root_active_executor requires selected-slot environment methods: "
                + ", ".join(missing)
            )
        batch_size = len(gen_batch)
        if not (
            len(worker_indices)
            == len(reset_keys)
            == len(uid_batch)
            == len(task_batch_indices)
            == batch_size
        ):
            raise ValueError("Active root logical and physical mappings must match")
        if len(set(int(slot) for slot in worker_indices)) != batch_size:
            raise ValueError("Active root worker slots must be unique within a wave")

        envs.reset_selected(worker_indices, reset_keys)
        task_descriptions = envs.get_tasks_selected(worker_indices)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.asarray(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )
        uid_batch = np.asarray(uid_batch, dtype=object)
        task_batch_indices = np.asarray(task_batch_indices, dtype=np.int32)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        self.orchestration_metrics["root_active_executor_enabled"] = 1.0
        self.orchestration_metrics["root_logical_trajectories"] = (
            self.orchestration_metrics.get("root_logical_trajectories", 0.0)
            + float(batch_size)
        )
        for step_index in range(int(self.config.env.max_steps)):
            active_indices = np.flatnonzero(~is_done).tolist()
            if not active_indices:
                break
            active_slots = [worker_indices[index] for index in active_indices]
            observations = envs.get_observations_selected(active_slots)
            active_batch = gen_batch.select_idxs(active_indices)
            batch = self.preprocess_batch(active_batch, observations)
            active_size = len(active_indices)
            batch.non_tensor_batch["step_index"] = np.full(
                active_size, step_index, dtype=np.int32
            )
            batch.non_tensor_batch["admissible_actions"] = np.asarray(
                observations.get(
                    "admissible_actions", [tuple() for _ in range(active_size)]
                ),
                dtype=object,
            )
            batch_input = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    key
                    for key in (
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "raw_prompt",
                        "tools_kwargs",
                    )
                    if key in batch.non_tensor_batch
                ],
            )
            batch_input.meta_info = active_batch.meta_info
            padded, pad_size = pad_dataproto_to_divisor(
                batch_input, actor_rollout_wg.world_size
            )
            started = time.monotonic()
            output = unpad_dataproto(
                actor_rollout_wg.generate_sequences(padded), pad_size=pad_size
            )
            generation_elapsed = time.monotonic() - started
            batch = batch.union(output)
            text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )
            env_started = time.monotonic()
            next_obs, rewards, dones, infos = envs.step_selected(
                active_slots, text_actions
            )
            environment_elapsed = time.monotonic() - env_started
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones, dtype=bool).reshape(-1)

            logical_indices = np.asarray(active_indices, dtype=np.int32)
            batch.non_tensor_batch["uid"] = uid_batch[logical_indices]
            batch.non_tensor_batch["traj_uid"] = traj_uid[logical_indices]
            batch.non_tensor_batch["occurrence_id"] = np.asarray(
                [
                    f"{traj_uid[logical_index]}:{step_index}"
                    for logical_index in active_indices
                ],
                dtype=object,
            )
            batch.non_tensor_batch["task_batch_index"] = task_batch_indices[
                logical_indices
            ]
            batch.non_tensor_batch["is_action_valid"] = np.asarray(
                [info.get("is_action_valid", True) for info in infos], dtype=bool
            )
            batch.non_tensor_batch["raw_model_response"] = np.asarray(
                text_actions, dtype=object
            )
            batch.non_tensor_batch["projected_action"] = np.asarray(
                [info.get("projected_action", "INVALID") for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch["is_action_format_valid"] = np.asarray(
                [
                    info.get(
                        "is_action_format_valid", info.get("is_action_valid", False)
                    )
                    for info in infos
                ],
                dtype=bool,
            )
            batch.non_tensor_batch["is_action_environment_valid"] = np.asarray(
                [
                    info.get(
                        "is_action_environment_valid",
                        info.get("is_action_valid", False),
                    )
                    for info in infos
                ],
                dtype=bool,
            )
            batch.non_tensor_batch["action_identity"] = np.asarray(
                [info.get("action_identity") for info in infos], dtype=object
            )
            batch.non_tensor_batch["post_action_observation"] = np.asarray(
                next_obs.get("anchor", [None for _ in range(active_size)]),
                dtype=object,
            )
            batch.non_tensor_batch["environment_reset_key"] = np.asarray(
                [
                    info.get("extra.gamefile", info.get("session_idx", ""))
                    for info in infos
                ],
                dtype=object,
            )
            batch.non_tensor_batch["task_description"] = np.asarray(
                [task_descriptions[index] for index in active_indices], dtype=object
            )
            batch.non_tensor_batch["done"] = dones
            batch.non_tensor_batch["remaining_horizon"] = np.full(
                active_size,
                int(self.config.env.max_steps) - step_index - 1,
                dtype=np.int32,
            )
            batch.non_tensor_batch["rewards"] = rewards.astype(object)
            batch.non_tensor_batch["active_masks"] = np.ones(
                active_size, dtype=object
            )

            rows = to_list_of_dict(batch)
            for row_index, logical_index in enumerate(active_indices):
                total_batch_list[logical_index].append(rows[row_index])
                total_infos[logical_index].append(infos[row_index])
                episode_rewards[logical_index] += float(rewards[row_index])
                episode_lengths[logical_index] += 1
                if "tool_calling" in infos[row_index]:
                    tool_callings[logical_index] += float(
                        infos[row_index]["tool_calling"]
                    )
                is_done[logical_index] = bool(dones[row_index])

            submitted_size = active_size + int(pad_size)
            dense_size = batch_size
            for key, value in (
                ("root_generation_turns", 1.0),
                ("root_active_sequences", float(active_size)),
                ("root_dense_equivalent_sequences", float(dense_size)),
                ("root_rows_avoided", float(dense_size - active_size)),
                ("root_model_rows_submitted", float(submitted_size)),
                ("root_padding_rows_submitted", float(pad_size)),
                ("root_environment_steps", float(active_size)),
                ("root_generate_wallclock", float(generation_elapsed)),
                ("root_env_wallclock", float(environment_elapsed)),
            ):
                self.orchestration_metrics[key] = (
                    self.orchestration_metrics.get(key, 0.0) + value
                )
            if self.artifact_store is not None:
                self.artifact_store.append("root_active_execution_waves", {
                    "step_index": step_index,
                    "logical_batch_size": batch_size,
                    "active_logical_indices": active_indices,
                    "active_worker_slots": active_slots,
                    "active_sequences": active_size,
                    "dense_equivalent_sequences": dense_size,
                    "submitted_sequences": submitted_size,
                    "padding_sequences": int(pad_size),
                    "generation_seconds": generation_elapsed,
                    "environment_seconds": environment_elapsed,
                })

        dense_total = float(
            self.orchestration_metrics.get("root_dense_equivalent_sequences", 0.0)
        )
        active_total = float(
            self.orchestration_metrics.get("root_active_sequences", 0.0)
        )
        self.orchestration_metrics["root_active_efficiency"] = (
            active_total / dense_total if dense_total else 1.0
        )
        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        )

    @staticmethod
    def _pending_root_slots(task_indices, generated_by_task, target_by_task):
        """Return all currently known missing task/slot pairs in slot-major order."""
        pending = []
        max_target = max((int(target_by_task[index]) for index in task_indices), default=0)
        for slot in range(max_target):
            for index in task_indices:
                if int(generated_by_task[index]) <= slot < int(target_by_task[index]):
                    pending.append((index, slot))
        return pending

    @staticmethod
    def _reset_key_from_info(info):
        reset_key = info.get("extra.gamefile", info.get("session_idx"))
        if reset_key is None or str(reset_key) == "":
            raise ValueError("Staged packed root generation could not resolve an environment reset key")
        return str(reset_key)

    def _record_root_wave(self, phase, batch_size, elapsed):
        prefix = f"{phase}_root"
        self.orchestration_metrics[f"{prefix}_waves"] = (
            self.orchestration_metrics.get(f"{prefix}_waves", 0.0) + 1.0
        )
        self.orchestration_metrics[f"{prefix}_trajectories"] = (
            self.orchestration_metrics.get(f"{prefix}_trajectories", 0.0)
            + float(batch_size)
        )
        self.orchestration_metrics[f"{prefix}_generation_seconds"] = (
            self.orchestration_metrics.get(f"{prefix}_generation_seconds", 0.0)
            + float(elapsed)
        )

    def _collect_staged_dynamic_roots_packed(self, gen_batch, actor_rollout_wg, envs):
        """Generate staged roots in dependency-respecting, accelerator-sized waves."""
        task_count = len(gen_batch)
        budget = int(self.config.algorithm.bace.total_leaf_budget)
        physical_stride = self._runtime_main_group_size(envs)
        pilot_roots = int(self.config.algorithm.bace.pilot_roots)
        task_indices = list(range(task_count))
        task_uids = [str(uuid.uuid4()) for _ in task_indices]
        generated_by_task = {task_idx: 0 for task_idx in task_indices}
        root_batches = []
        root_logs = []

        probe_started = time.monotonic()
        _, probe_infos = envs.reset(kwargs={
            "_bace_worker_indices": [
                task_idx * physical_stride for task_idx in task_indices
            ],
            "_bace_reset_keys": [None] * task_count,
        })
        reset_key_by_task = {
            task_idx: self._reset_key_from_info(info)
            for task_idx, info in zip(task_indices, probe_infos)
        }
        self.orchestration_metrics["root_reset_key_probe_seconds"] = float(
            time.monotonic() - probe_started
        )

        def collect(pairs, phase):
            if not pairs:
                return
            indices = [task_idx for task_idx, _ in pairs]
            slots = [slot for _, slot in pairs]
            keys = [reset_key_by_task[task_idx] for task_idx in indices]
            started = time.monotonic()
            output = self._collect_root_wave(
                gen_batch,
                actor_rollout_wg,
                envs,
                indices,
                slots,
                task_uids,
                keys,
            )
            elapsed = time.monotonic() - started
            self._record_root_wave(phase, len(pairs), elapsed)
            root_batches.append(output)
            log_started = time.monotonic()
            root_logs.extend(build_root_event_logs(output))
            self.orchestration_metrics["root_event_logging_seconds"] = (
                self.orchestration_metrics.get("root_event_logging_seconds", 0.0)
                + float(time.monotonic() - log_started)
            )
            for task_idx, slot in pairs:
                if slot != generated_by_task[task_idx]:
                    raise AssertionError("Packed root slots must be collected contiguously per task")
                generated_by_task[task_idx] += 1

        pilot_targets = {task_idx: pilot_roots for task_idx in task_indices}
        collect(
            self._pending_root_slots(task_indices, generated_by_task, pilot_targets),
            "pilot",
        )
        states = self.topology_planner.initialize_staged(root_logs)
        uid_to_task_index = {uid: index for index, uid in enumerate(task_uids)}

        completion_targets = {
            uid_to_task_index[task_id]: state.root_count
            for task_id, state in states.items()
        }
        collect(
            self._pending_root_slots(task_indices, generated_by_task, completion_targets),
            "completion",
        )

        while True:
            correction_started = time.monotonic()
            deficient = self.topology_planner.correct_staged_capacity(root_logs, states)
            self.orchestration_metrics["capacity_planning_seconds"] = (
                self.orchestration_metrics.get("capacity_planning_seconds", 0.0)
                + float(time.monotonic() - correction_started)
            )
            if not deficient:
                break
            pairs = [
                (uid_to_task_index[task_id], generated_by_task[uid_to_task_index[task_id]])
                for task_id in deficient
            ]
            pairs.sort(key=lambda item: (item[1], item[0]))
            collect(pairs, "capacity_correction")

        topology_plan = self.topology_planner.finalize_staged(root_logs, states)
        if any(generated_by_task[index] > budget for index in task_indices):
            raise AssertionError("Staged root generation exceeded the per-task budget")
        concat_started = time.monotonic()
        root_output = self._concat_batches(root_batches)
        self.orchestration_metrics["root_batch_concat_seconds"] = float(
            time.monotonic() - concat_started
        )
        self.orchestration_metrics["root_generation_waves"] = float(len(root_batches))
        self.orchestration_metrics["staged_root_batching_packed"] = 1.0
        generated_by_uid = {
            task_uids[index]: generated_by_task[index] for index in task_indices
        }
        return root_output, list(topology_plan.roots), topology_plan, generated_by_uid

    def _collect_staged_dynamic_roots_sequential(self, gen_batch, actor_rollout_wg, envs):
        task_count = len(gen_batch)
        budget = int(self.config.algorithm.bace.total_leaf_budget)
        pilot_roots = int(self.config.algorithm.bace.pilot_roots)
        task_indices = list(range(task_count))
        task_uids = [str(uuid.uuid4()) for _ in task_indices]
        reset_key_by_task = {task_idx: None for task_idx in task_indices}
        generated_by_task = {task_idx: 0 for task_idx in task_indices}
        root_batches = []

        def collect(indices, phase):
            started = time.monotonic()
            slots = [generated_by_task[index] for index in indices]
            keys = [reset_key_by_task[index] for index in indices]
            output = self._collect_root_wave(
                gen_batch,
                actor_rollout_wg,
                envs,
                indices,
                slots,
                task_uids,
                keys,
            )
            self._record_root_wave(phase, len(indices), time.monotonic() - started)
            root_batches.append(output)
            for index in indices:
                generated_by_task[index] += 1
            for row_idx, task_idx in enumerate(output.non_tensor_batch["task_batch_index"]):
                task_idx = int(task_idx)
                if reset_key_by_task[task_idx] is None:
                    reset_key_by_task[task_idx] = str(
                        output.non_tensor_batch["environment_reset_key"][row_idx]
                    )

        for _ in range(pilot_roots):
            collect(task_indices, "pilot")

        concat_started = time.monotonic()
        root_output = self._concat_batches(root_batches)
        self.orchestration_metrics["root_batch_concat_seconds"] = (
            self.orchestration_metrics.get("root_batch_concat_seconds", 0.0)
            + time.monotonic() - concat_started
        )
        log_started = time.monotonic()
        pilot_logs = build_root_event_logs(root_output)
        self.orchestration_metrics["root_event_logging_seconds"] = (
            self.orchestration_metrics.get("root_event_logging_seconds", 0.0)
            + time.monotonic() - log_started
        )
        states = self.topology_planner.initialize_staged(pilot_logs)
        uid_to_task_index = {uid: index for index, uid in enumerate(task_uids)}

        while True:
            needed = [
                uid_to_task_index[task_id]
                for task_id, state in states.items()
                if generated_by_task[uid_to_task_index[task_id]] < state.root_count
            ]
            if needed:
                collect(needed, "completion")
                concat_started = time.monotonic()
                root_output = self._concat_batches(root_batches)
                self.orchestration_metrics["root_batch_concat_seconds"] += (
                    time.monotonic() - concat_started
                )
                continue

            log_started = time.monotonic()
            root_logs = build_root_event_logs(root_output)
            self.orchestration_metrics["root_event_logging_seconds"] += (
                time.monotonic() - log_started
            )
            deficient = self.topology_planner.correct_staged_capacity(root_logs, states)
            if not deficient:
                break
            needed = [uid_to_task_index[task_id] for task_id in deficient]
            collect(needed, "capacity_correction")
            concat_started = time.monotonic()
            root_output = self._concat_batches(root_batches)
            self.orchestration_metrics["root_batch_concat_seconds"] += (
                time.monotonic() - concat_started
            )

        log_started = time.monotonic()
        root_logs = build_root_event_logs(root_output)
        self.orchestration_metrics["root_event_logging_seconds"] += (
            time.monotonic() - log_started
        )
        topology_plan = self.topology_planner.finalize_staged(root_logs, states)
        if any(generated_by_task[index] > budget for index in task_indices):
            raise AssertionError("Staged root generation exceeded the per-task budget")
        generated_by_uid = {
            task_uids[index]: generated_by_task[index] for index in task_indices
        }
        self.orchestration_metrics["root_generation_waves"] = float(len(root_batches))
        return root_output, list(topology_plan.roots), topology_plan, generated_by_uid

    def _collect_exact_batch_dynamic_roots_packed(self, gen_batch, actor_rollout_wg, envs):
        """Generate no-pilot roots from lagged family plans, then correct capacity."""
        # Keep this low-level helper usable by the lightweight root-wave tests
        # and external diagnostics that construct a collector with ``__new__``.
        # A real collector always sets the field in ``__init__``.
        pairwise_mode = getattr(self, "pairwise_mode", "full")
        task_count = len(gen_batch)
        budget = int(self.config.algorithm.bace.total_leaf_budget)
        physical_stride = self._runtime_main_group_size(envs)
        task_indices = list(range(task_count))
        task_uids = [str(uuid.uuid4()) for _ in task_indices]
        generated_by_task = {task_idx: 0 for task_idx in task_indices}
        root_batches = []
        root_logs = []

        probe_started = time.monotonic()
        _, probe_infos = envs.reset(kwargs={
            "_bace_worker_indices": [
                task_idx * physical_stride for task_idx in task_indices
            ],
            "_bace_reset_keys": [None] * task_count,
        })
        reset_key_by_task = {
            task_idx: self._reset_key_from_info(info)
            for task_idx, info in zip(task_indices, probe_infos)
        }
        self.orchestration_metrics["root_reset_key_probe_seconds"] = float(
            time.monotonic() - probe_started
        )
        states = self.topology_planner.initialize({
            task_uids[task_idx]: task_family_from_reset_key(reset_key_by_task[task_idx])
            for task_idx in task_indices
        })
        uid_to_task_index = {uid: index for index, uid in enumerate(task_uids)}

        if self.artifact_store is not None:
            self.artifact_store.append("family_topology_plans", {
                "policy_update_id": self.current_step,
                "history_before_update": self.competence_history.snapshot(),
                "tasks": states,
            })

        def collect(pairs, phase):
            if not pairs:
                return
            indices = [task_idx for task_idx, _ in pairs]
            slots = [slot for _, slot in pairs]
            started = time.monotonic()
            output = self._collect_root_wave(
                gen_batch,
                actor_rollout_wg,
                envs,
                indices,
                slots,
                task_uids,
                [reset_key_by_task[task_idx] for task_idx in indices],
            )
            self._record_root_wave(phase, len(pairs), time.monotonic() - started)
            root_batches.append(output)
            log_started = time.monotonic()
            root_logs.extend(build_root_event_logs(output))
            self.orchestration_metrics["root_event_logging_seconds"] = (
                self.orchestration_metrics.get("root_event_logging_seconds", 0.0)
                + float(time.monotonic() - log_started)
            )
            for task_idx, slot in pairs:
                if slot != generated_by_task[task_idx]:
                    raise AssertionError("Exact root slots must be contiguous per task")
                generated_by_task[task_idx] += 1

        planned_targets = {
            uid_to_task_index[task_id]: state.root_count
            for task_id, state in states.items()
        }
        collect(
            self._pending_root_slots(task_indices, generated_by_task, planned_targets),
            "planned",
        )

        if pairwise_mode == "stopping":
            # P2 deliberately starts as soon as its initially planned natural
            # roots are available.  Threshold capacity is checked per pair by
            # the coordinator; C=0 later becomes one-way packed fallback roots.
            correction_started = time.monotonic()
            for state in states.values():
                task_roots = [root for root in root_logs if root.task_id == state.task_id]
                self.topology_planner._assess_task(task_roots, state)
            self.orchestration_metrics["capacity_planning_seconds"] = float(
                time.monotonic() - correction_started
            )
            if self.artifact_store is not None:
                self.artifact_store.append("capacity_checks", {
                    "policy_update_id": self.current_step,
                    "deficient_task_ids": [],
                    "lazy_pairwise_capacity": True,
                    "tasks": states,
                })
            topology_plan = self.topology_planner.finalize(
                root_logs, states, allow_capacity_shortfall=True
            )
        else:
            while True:
                correction_started = time.monotonic()
                deficient = self.topology_planner.correct_capacity(root_logs, states)
                self.orchestration_metrics["capacity_planning_seconds"] = (
                    self.orchestration_metrics.get("capacity_planning_seconds", 0.0)
                    + float(time.monotonic() - correction_started)
                )
                if self.artifact_store is not None:
                    self.artifact_store.append("capacity_checks", {
                        "policy_update_id": self.current_step,
                        "deficient_task_ids": sorted(deficient),
                        "tasks": states,
                    })
                if not deficient:
                    break
                # A capacity correction may convert more than one branch slot
                # into root slots.  Generate every newly required contiguous
                # slot in this packed wave; emitting only the first missing
                # slot would leave ``generated_by_task < state.root_count``
                # before the next assessment.
                correction_indices = sorted(
                    uid_to_task_index[task_id] for task_id in deficient
                )
                correction_targets = {
                    index: states[task_uids[index]].root_count
                    for index in correction_indices
                }
                pairs = self._pending_root_slots(
                    correction_indices, generated_by_task, correction_targets
                )
                collect(pairs, "capacity_correction")
            topology_plan = self.topology_planner.finalize(root_logs, states)
        if any(generated_by_task[index] > budget for index in task_indices):
            raise AssertionError("Exact root generation exceeded the per-task budget")
        concat_started = time.monotonic()
        root_output = self._concat_batches(root_batches)
        self.orchestration_metrics["root_batch_concat_seconds"] = float(
            time.monotonic() - concat_started
        )
        self.orchestration_metrics["root_generation_waves"] = float(len(root_batches))
        self.orchestration_metrics["staged_root_batching_packed"] = 1.0
        self.orchestration_metrics["batch_erv_exact"] = 1.0
        generated_by_uid = {
            task_uids[index]: generated_by_task[index] for index in task_indices
        }
        if pairwise_mode == "stopping":
            self._pairwise_root_context = {
                "task_uids": task_uids,
                "reset_key_by_task": reset_key_by_task,
                "generated_by_task": generated_by_task,
                "uid_to_task_index": uid_to_task_index,
            }
        return root_output, list(topology_plan.roots), topology_plan, generated_by_uid

    def _collect_pairwise_fallback_roots(
        self, gen_batch, actor_rollout_wg, envs, fallback_by_task: dict[str, int]
    ):
        """Generate P2's one-way fallback roots in one global packed wave."""
        context = getattr(self, "_pairwise_root_context", None)
        if context is None:
            raise RuntimeError("Pairwise fallback roots require lazy root context")
        pairs = []
        for task_id in sorted(fallback_by_task):
            count = int(fallback_by_task[task_id])
            if count <= 0:
                continue
            task_index = context["uid_to_task_index"][task_id]
            for _ in range(count):
                slot = context["generated_by_task"][task_index]
                pairs.append((task_index, slot))
                context["generated_by_task"][task_index] += 1
        if not pairs:
            return None, []
        budget = int(self.config.algorithm.bace.total_leaf_budget)
        if any(slot >= budget for _, slot in pairs):
            raise AssertionError("Pairwise fallback root exceeded the leaf budget")
        indices = [index for index, _ in pairs]
        slots = [slot for _, slot in pairs]
        started = time.monotonic()
        output = self._collect_root_wave(
            gen_batch,
            actor_rollout_wg,
            envs,
            indices,
            slots,
            context["task_uids"],
            [context["reset_key_by_task"][index] for index in indices],
        )
        self._record_root_wave("fallback", len(pairs), time.monotonic() - started)
        logs = build_root_event_logs(output)
        self.orchestration_metrics["fallback_root_count"] = float(len(pairs))
        return output, logs

    def _collect_staged_dynamic_roots(self, gen_batch, actor_rollout_wg, envs):
        if self.variant == "batch_erv_exact":
            return self._collect_exact_batch_dynamic_roots_packed(
                gen_batch, actor_rollout_wg, envs
            )
        if self.staged_root_batching == "packed":
            return self._collect_staged_dynamic_roots_packed(
                gen_batch, actor_rollout_wg, envs
            )
        started = time.monotonic()
        result = self._collect_staged_dynamic_roots_sequential(
            gen_batch, actor_rollout_wg, envs
        )
        self.orchestration_metrics.update({
            "staged_root_batching_packed": 0.0,
            "sequential_root_generation_seconds": float(time.monotonic() - started),
        })
        return result

    def _collect_frontier_dynamic(self, gen_batch, actor_rollout_wg, envs):
        """Collect via the optional mixed MODEL_READY frontier scheduler."""
        orchestrator = BACEFrontierOrchestrator(
            collector=self,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
        )
        started = time.monotonic()
        output, root_logs, controllers, logical_branches, profile = orchestrator.collect()
        elapsed = time.monotonic() - started
        success_metrics = self._episode_success_metrics(output)
        self.orchestration_metrics.update(profile)
        self.orchestration_metrics.update(success_metrics)
        self.orchestration_metrics["frontier/total_seconds"] = float(elapsed)
        self.orchestration_metrics["frontier/task_count"] = float(len(controllers))
        self.orchestration_metrics["frontier/leaf_budget"] = float(
            self.config.algorithm.bace.total_leaf_budget
        )
        self._frontier_branch_count = len(logical_branches)
        task_plans = {
            task_id: controller.topology_plan.tasks[task_id]
            for task_id, controller in controllers.items()
            if controller.topology_plan is not None
        }
        topology_plan = None
        if task_plans:
            from .topology import TopologyPlan
            topology_plan = TopologyPlan(roots=tuple(root_logs), tasks=task_plans)
        selected_ids = {root.root_id for root in root_logs}
        self._trace_roots(root_logs, selected_ids=selected_ids)
        self._trace_anchors(root_logs)
        self._trace_topology(topology_plan, root_logs)
        if self.artifact_store is not None:
            self.artifact_store.append("frontier_events", {
                "status": "complete",
                "logical_branch_count": len(logical_branches),
                "profile": profile,
                "success_metrics": success_metrics,
                "generation_wave_stream": "generation_waves.jsonl",
                "tasks": {
                    task_id: {
                        "slots": list(controller.slots),
                        "root_count": len(controller.root_logs),
                        "branch_count": controller.branch_completed,
                        "root_backbone_frozen": controller.root_backbone_frozen,
                        "slot_consumed": controller.slot_cursor,
                    }
                    for task_id, controller in controllers.items()
                },
                "logical_branches": logical_branches,
            })
        if topology_plan is not None:
            self.topology_planner.update_history(topology_plan)
        return output, root_logs, topology_plan

    def _collect_suffixes_legacy(self, gen_batch, actor_rollout_wg, requests, initial_obs, origin_rewards, origin_dones):
        """Original dense suffix executor retained for compatibility and A/B checks."""
        batch_size = len(requests)
        is_done = np.asarray(origin_dones, dtype=bool).copy()
        episode_rewards = np.asarray(origin_rewards, dtype=np.float32).copy()
        episode_lengths = np.ones(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        total_batch_list = [[] for _ in range(batch_size)]
        obs = initial_obs

        for suffix_step in range(max(request.remaining_horizon for request in requests)):
            horizon_active = np.array([suffix_step < request.remaining_horizon for request in requests])
            active_masks = np.logical_and(~is_done, horizon_active)
            if not active_masks.any():
                break

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch.non_tensor_batch["step_index"] = np.array(
                [request.target_turn + 1 + suffix_step for request in requests], dtype=np.int32
            )
            batch.non_tensor_batch["admissible_actions"] = np.array(obs["admissible_actions"], dtype=object)
            batch_keys = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs"):
                if key in batch.non_tensor_batch:
                    non_tensor_keys.append(key)
            batch_input = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
            batch_input.meta_info = gen_batch.meta_info
            padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            output = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)

            branch_ids = np.array([request.branch_id for request in requests], dtype=object)
            batch.non_tensor_batch["uid"] = np.array([request.task_id for request in requests], dtype=object)
            batch.non_tensor_batch["traj_uid"] = branch_ids
            batch.non_tensor_batch["occurrence_id"] = np.array(
                [f"{request.branch_id}:suffix:{suffix_step}" for request in requests], dtype=object
            )
            batch.non_tensor_batch["task_batch_index"] = np.array(
                [request.task_batch_index for request in requests], dtype=np.int32
            )
            batch = batch.union(output)
            text_actions = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            next_obs, rewards, dones, infos = self.branch_envs.step(text_actions)
            rewards = np.asarray(rewards).squeeze()
            dones = np.asarray(dones).squeeze()
            if rewards.ndim == 0:
                rewards = rewards.reshape(1)
                dones = dones.reshape(1)

            batch.non_tensor_batch["is_action_valid"] = np.array(
                [info.get("is_action_valid", True) for info in infos], dtype=bool
            )
            batch.non_tensor_batch["raw_model_response"] = np.array(text_actions, dtype=object)
            batch.non_tensor_batch["projected_action"] = np.array(
                [info.get("projected_action", "INVALID") for info in infos], dtype=object
            )
            batch.non_tensor_batch["is_action_format_valid"] = np.array(
                [info.get("is_action_format_valid", info.get("is_action_valid", False)) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["is_action_environment_valid"] = np.array(
                [info.get("is_action_environment_valid", info.get("is_action_valid", False)) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_identity"] = np.array(
                [info.get("action_identity") for info in infos], dtype=object
            )
            batch.non_tensor_batch["post_action_observation"] = np.array(next_obs["anchor"], dtype=object)
            batch.non_tensor_batch["environment_reset_key"] = np.array(
                [request.environment_reset_key for request in requests], dtype=object
            )
            batch.non_tensor_batch["task_description"] = np.array(
                [request.task_description for request in requests], dtype=object
            )
            batch.non_tensor_batch["done"] = dones.astype(bool)
            batch.non_tensor_batch["remaining_horizon"] = np.array(
                [max(0, request.remaining_horizon - suffix_step - 1) for request in requests], dtype=np.int32
            )
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

            rows = to_list_of_dict(batch)
            for idx, row in enumerate(rows):
                total_batch_list[idx].append(row)
            episode_rewards[active_masks] += rewards[active_masks]
            episode_lengths[active_masks] += 1
            is_done = np.logical_or(is_done, dones)
            obs = next_obs

        effective_rows = []
        for env_idx, rows in enumerate(total_batch_list):
            for row in rows:
                if row["active_masks"]:
                    row["episode_rewards"] = episode_rewards[env_idx]
                    row["episode_lengths"] = episode_lengths[env_idx]
                    row["tool_callings"] = tool_callings[env_idx]
                    row["success_rate"] = float(episode_rewards[env_idx] > 0)
                    effective_rows.append(row)
        output = DataProto.from_single_dict(collate_fn(effective_rows)) if effective_rows else None
        return output, episode_rewards, is_done

    @staticmethod
    def _chunk_requests(requests, capacity):
        """Split frozen requests into physical waves without replanning them."""
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        requests = list(requests)
        return [
            requests[start : start + capacity]
            for start in range(0, len(requests), capacity)
        ]

    def _execute_chunk_legacy(
        self,
        root_output,
        gen_batch,
        actor_rollout_wg,
        requests,
        used_origins=None,
        execution_wave=0,
    ):
        """Execute one capacity-bounded replay/origin/suffix transaction.

        This is the original round execution path.  The outer ``_execute_round``
        now invokes it in physical waves while keeping the Exact Batch-ERV
        allocation frozen.
        """
        active_requests = list(requests)
        all_results = []
        if used_origins is None:
            used_origins = set()
        used_origins.update(
            request.origin_occurrence_id for request in active_requests
        )
        for request in active_requests:
            self.replay_retry_metadata.setdefault(request.request_id, {
                "attempt": 0,
                "fallback_level": "selected_origin",
                "parent_request_id": None,
                "execution_wave": int(execution_wave),
            })
        for attempt in range(self.max_origin_retries + 1):
            started = time.monotonic()
            replay_results = self.replay_adapter.replay_and_validate(active_requests)
            self._trace_replay_results(
                active_requests,
                replay_results,
                f"initial_validation_{attempt}",
                time.monotonic() - started,
                cost_kind="validation_replay" if attempt == 0 else "retry_replay",
            )
            all_results.extend(zip(active_requests, replay_results))
            failed = []
            retry_requests = []
            for request, result in zip(active_requests, replay_results):
                if result.replay_ok:
                    continue
                if attempt >= self.max_origin_retries:
                    failed.append(request)
                    continue
                replacement = self._fallback_request(request, build_root_event_logs(root_output), used_origins)
                if replacement is not None:
                    used_origins.add(replacement.origin_occurrence_id)
                    self.replay_retry_metadata[replacement.request_id] = {
                        "attempt": attempt + 1,
                        "fallback_level": "same_anchor_action_alternate_origin",
                        "parent_request_id": request.request_id,
                        "execution_wave": int(execution_wave),
                    }
                    retry_requests.append(replacement)
                    if hasattr(self.coordinator, "adopt_retry_request"):
                        self.coordinator.adopt_retry_request(request, replacement)
                else:
                    failed.append(request)
            if not retry_requests:
                break
            active_requests = retry_requests
        # The successful request set is the latest attempt for each branch.
        result_by_branch = {request.branch_id: (request, result) for request, result in all_results}
        active_pairs = list(result_by_branch.values())
        valid_requests = [request for request, result in active_pairs if result.replay_ok]
        if self.artifact_store is not None:
            for request, result in active_pairs:
                if result.replay_ok:
                    continue
                self.artifact_store.append("branches", {
                    "request_id": request.request_id,
                    "branch_id": request.branch_id,
                    "task_id": request.task_id,
                    "origin_occurrence_id": request.origin_occurrence_id,
                    "selected_action": request.selected_canonical_action,
                    "status": "replay_budget_exhausted",
                    "failure_category": result.category,
                    "failure_message": result.error_message,
                    **self.replay_retry_metadata.get(request.request_id, {}),
                })
        if not valid_requests:
            return [], None, None, None
        if len(valid_requests) != len(active_pairs):
            started = time.monotonic()
            replay_results = self.replay_adapter.replay_and_validate(valid_requests)
            self._trace_replay_results(valid_requests, replay_results, "aligned_batch_validation", time.monotonic() - started)
            valid_requests = [
                request for request, result in zip(valid_requests, replay_results) if result.replay_ok
            ]
        if not valid_requests:
            return [], None, None, None

        def execute_origins(origin_requests, phase):
            restore_started = time.monotonic()
            self.branch_envs.replay(origin_requests)
            restore_elapsed = time.monotonic() - restore_started
            self._record_branch_execution_cost(
                "execution_restore_replay",
                steps=sum(len(request.parsed_action_prefix) for request in origin_requests),
                elapsed=restore_elapsed,
                jobs=len(origin_requests),
            )
            started = time.monotonic()
            observations, rewards, dones, origin_infos = self.branch_envs.step(
                [request.copied_raw_model_response for request in origin_requests]
            )
            results = self.replay_adapter.validate_transitions(
                origin_requests, observations, rewards, dones, origin_infos
            )
            self._trace_replay_results(
                origin_requests, results, phase, time.monotonic() - started,
                prefix_lengths=[1] * len(origin_requests),
                cost_kind="origin_transition",
            )
            return observations, np.asarray(rewards), np.asarray(dones), origin_infos, results

        next_obs, origin_rewards, origin_dones, infos, transition_results = execute_origins(
            valid_requests, "origin_transition_validation"
        )
        transition_valid = [index for index, result in enumerate(transition_results) if result.replay_ok]
        for request, result in zip(valid_requests, transition_results):
            if result.replay_ok or self.artifact_store is None:
                continue
            self.artifact_store.append("branches", {
                "request_id": request.request_id,
                "branch_id": request.branch_id,
                "task_id": request.task_id,
                "origin_occurrence_id": request.origin_occurrence_id,
                "selected_action": request.selected_canonical_action,
                "copied_action_identity": request.copied_action_identity,
                "status": "transition_replay_mismatch",
                "failure_category": result.category,
            })
        if not transition_valid:
            return [], None, None, None
        if len(transition_valid) != len(valid_requests):
            valid_requests = [valid_requests[index] for index in transition_valid]
            next_obs, origin_rewards, origin_dones, infos, transition_results = execute_origins(
                valid_requests, "aligned_origin_transition_validation"
            )
            final_valid = [index for index, result in enumerate(transition_results) if result.replay_ok]
            valid_requests = [valid_requests[index] for index in final_valid]
            if not valid_requests:
                return [], None, None, None
            next_obs = {key: [values[index] for index in final_valid] for key, values in next_obs.items()}
            origin_rewards = origin_rewards[final_valid]
            origin_dones = origin_dones[final_valid]

        branch_gen_batch = gen_batch.select_idxs([request.task_batch_index for request in valid_requests])
        suffix_started = time.monotonic()
        suffix_output, terminal_rewards, terminal_dones = self._collect_suffixes_legacy(
            branch_gen_batch,
            actor_rollout_wg,
            valid_requests,
            next_obs,
            origin_rewards,
            origin_dones,
        )
        self.orchestration_metrics["branch_suffix_generation_seconds"] = (
            self.orchestration_metrics.get("branch_suffix_generation_seconds", 0.0)
            + float(time.monotonic() - suffix_started)
        )
        self.orchestration_metrics["branch_suffix_waves"] = (
            self.orchestration_metrics.get("branch_suffix_waves", 0.0) + 1.0
        )
        self.orchestration_metrics["branch_suffix_trajectories"] = (
            self.orchestration_metrics.get("branch_suffix_trajectories", 0.0)
            + float(len(valid_requests))
        )

        occurrence_to_idx = {
            str(occurrence_id): idx
            for idx, occurrence_id in enumerate(root_output.non_tensor_batch["occurrence_id"])
        }
        origin_output = root_output.select_idxs(
            [occurrence_to_idx[request.origin_occurrence_id] for request in valid_requests]
        )
        origin_output.non_tensor_batch["traj_uid"] = np.array(
            [request.branch_id for request in valid_requests], dtype=object
        )
        origin_output.non_tensor_batch["occurrence_id"] = np.array(
            [f"{request.branch_id}:origin" for request in valid_requests], dtype=object
        )
        origin_output.non_tensor_batch["episode_rewards"] = terminal_rewards.astype(np.float32)
        origin_output.non_tensor_batch["rewards"] = np.asarray(origin_rewards, dtype=object)
        origin_output.non_tensor_batch["done"] = np.asarray(origin_dones, dtype=bool)
        self._refresh_success_rate(origin_output)
        self._set_lineage(
            origin_output, "branch_origin", [request.branch_id for request in valid_requests]
        )
        if suffix_output is not None:
            self._set_lineage(
                suffix_output,
                "branch_suffix",
                suffix_output.non_tensor_batch["traj_uid"],
            )
        return valid_requests, origin_output, suffix_output, terminal_rewards

    def _collect_suffixes_selected(
        self,
        gen_batch,
        actor_rollout_wg,
        requests,
        worker_slots,
        origin_rewards,
        origin_dones,
    ):
        """Roll out only active persistent slots while preserving branch-major rows."""
        batch_size = len(requests)
        is_done = np.asarray(origin_dones, dtype=bool).reshape(-1).copy()
        episode_rewards = np.asarray(origin_rewards, dtype=np.float32).reshape(-1).copy()
        episode_lengths = np.ones(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        total_batch_list = [[] for _ in range(batch_size)]
        max_horizon = max((request.remaining_horizon for request in requests), default=0)

        for suffix_step in range(max_horizon):
            active_indices = [
                index
                for index, request in enumerate(requests)
                if not is_done[index] and suffix_step < request.remaining_horizon
            ]
            if not active_indices:
                break
            active_requests = [requests[index] for index in active_indices]
            active_slots = [worker_slots[index] for index in active_indices]
            observations = self.branch_envs.get_observations_selected(active_slots)
            active_gen_batch = gen_batch.select_idxs(active_indices)

            started = time.monotonic()
            batch = self.preprocess_batch(gen_batch=active_gen_batch, obs=observations)
            batch.non_tensor_batch["step_index"] = np.asarray(
                [request.target_turn + 1 + suffix_step for request in active_requests],
                dtype=np.int32,
            )
            batch.non_tensor_batch["admissible_actions"] = np.asarray(
                observations["admissible_actions"], dtype=object
            )
            batch_keys = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs"):
                if key in batch.non_tensor_batch:
                    non_tensor_keys.append(key)
            batch_input = batch.pop(
                batch_keys=batch_keys,
                non_tensor_batch_keys=non_tensor_keys,
            )
            batch_input.meta_info = active_gen_batch.meta_info
            padded, pad_size = pad_dataproto_to_divisor(
                batch_input, actor_rollout_wg.world_size
            )
            submitted_size = len(padded)
            output = unpad_dataproto(
                actor_rollout_wg.generate_sequences(padded), pad_size=pad_size
            )

            batch.non_tensor_batch["uid"] = np.asarray(
                [request.task_id for request in active_requests], dtype=object
            )
            batch.non_tensor_batch["traj_uid"] = np.asarray(
                [request.branch_id for request in active_requests], dtype=object
            )
            batch.non_tensor_batch["occurrence_id"] = np.asarray(
                [
                    f"{request.branch_id}:suffix:{suffix_step}"
                    for request in active_requests
                ],
                dtype=object,
            )
            batch.non_tensor_batch["task_batch_index"] = np.asarray(
                [request.task_batch_index for request in active_requests], dtype=np.int32
            )
            batch = batch.union(output)
            text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )
            next_obs, rewards, dones, infos = self.branch_envs.step_selected(
                active_slots, text_actions
            )
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones).reshape(-1)

            batch.non_tensor_batch["is_action_valid"] = np.asarray(
                [info.get("is_action_valid", True) for info in infos], dtype=bool
            )
            batch.non_tensor_batch["raw_model_response"] = np.asarray(
                text_actions, dtype=object
            )
            batch.non_tensor_batch["projected_action"] = np.asarray(
                [info.get("projected_action", "INVALID") for info in infos], dtype=object
            )
            batch.non_tensor_batch["is_action_format_valid"] = np.asarray(
                [
                    info.get(
                        "is_action_format_valid", info.get("is_action_valid", False)
                    )
                    for info in infos
                ],
                dtype=bool,
            )
            batch.non_tensor_batch["is_action_environment_valid"] = np.asarray(
                [
                    info.get(
                        "is_action_environment_valid",
                        info.get("is_action_valid", False),
                    )
                    for info in infos
                ],
                dtype=bool,
            )
            batch.non_tensor_batch["action_identity"] = np.asarray(
                [info.get("action_identity") for info in infos], dtype=object
            )
            batch.non_tensor_batch["post_action_observation"] = np.asarray(
                next_obs["anchor"], dtype=object
            )
            batch.non_tensor_batch["environment_reset_key"] = np.asarray(
                [request.environment_reset_key for request in active_requests], dtype=object
            )
            batch.non_tensor_batch["task_description"] = np.asarray(
                [request.task_description for request in active_requests], dtype=object
            )
            batch.non_tensor_batch["done"] = dones.astype(bool)
            batch.non_tensor_batch["remaining_horizon"] = np.asarray(
                [
                    max(0, request.remaining_horizon - suffix_step - 1)
                    for request in active_requests
                ],
                dtype=np.int32,
            )
            batch.non_tensor_batch["rewards"] = torch_to_numpy(
                rewards, is_object=True
            )
            batch.non_tensor_batch["active_masks"] = np.ones(
                len(active_indices), dtype=object
            )

            for logical_index, row in zip(active_indices, to_list_of_dict(batch)):
                total_batch_list[logical_index].append(row)
            for local_index, logical_index in enumerate(active_indices):
                episode_rewards[logical_index] += float(rewards[local_index])
                episode_lengths[logical_index] += 1
                is_done[logical_index] = bool(is_done[logical_index] or dones[local_index])

            active_size = len(active_indices)
            dense_size = batch_size
            elapsed = time.monotonic() - started
            increments = {
                "branch_suffix_generation_waves": 1,
                "branch_suffix_active_sequences": active_size,
                "branch_suffix_dense_equivalent_sequences": dense_size,
                "branch_suffix_inactive_sequences_avoided": dense_size - active_size,
                "branch_suffix_submitted_sequences": submitted_size,
                "branch_suffix_padding_sequences": submitted_size - active_size,
                "branch_suffix_environment_steps": active_size,
                "branch_suffix_selected_seconds": elapsed,
            }
            for key, value in increments.items():
                self.orchestration_metrics[key] = (
                    self.orchestration_metrics.get(key, 0) + value
                )
            dense_total = self.orchestration_metrics[
                "branch_suffix_dense_equivalent_sequences"
            ]
            self.orchestration_metrics["branch_suffix_active_efficiency"] = (
                self.orchestration_metrics["branch_suffix_active_sequences"]
                / dense_total
                if dense_total
                else 1.0
            )
            if self.artifact_store is not None:
                self.artifact_store.append("branch_suffix_execution_waves", {
                    "execution_wave": suffix_step,
                    "active_worker_slots": active_slots,
                    "active_branch_ids": [
                        request.branch_id for request in active_requests
                    ],
                    "active_sequences": active_size,
                    "dense_equivalent_sequences": dense_size,
                    "submitted_sequences": submitted_size,
                    "padding_sequences": submitted_size - active_size,
                    "elapsed_seconds": elapsed,
                })

        effective_rows = []
        for env_index, rows in enumerate(total_batch_list):
            for row in rows:
                row["episode_rewards"] = episode_rewards[env_index]
                row["episode_lengths"] = episode_lengths[env_index]
                row["tool_callings"] = tool_callings[env_index]
                row["success_rate"] = float(episode_rewards[env_index] > 0)
                effective_rows.append(row)
        output = (
            DataProto.from_single_dict(collate_fn(effective_rows))
            if effective_rows
            else None
        )
        return output, episode_rewards, is_done

    def _execute_chunk_selected(
        self,
        root_output,
        gen_batch,
        actor_rollout_wg,
        requests,
        used_origins=None,
        execution_wave=0,
    ):
        """Execute a frozen chunk on persistent slots without replaying successes."""
        for target in (self.trace_diagnostics, self.orchestration_metrics):
            target.setdefault("branch_execution_restore_replay_steps", 0)
            target.setdefault("branch_execution_restore_replay_seconds", 0.0)
            target.setdefault("branch_execution_restore_replay_jobs", 0)
        for key, value in (
            ("branch_suffix_generation_waves", 0),
            ("branch_suffix_active_sequences", 0),
            ("branch_suffix_dense_equivalent_sequences", 0),
            ("branch_suffix_inactive_sequences_avoided", 0),
            ("branch_suffix_submitted_sequences", 0),
            ("branch_suffix_padding_sequences", 0),
            ("branch_suffix_environment_steps", 0),
            ("branch_suffix_selected_seconds", 0.0),
            ("branch_suffix_active_efficiency", 1.0),
        ):
            self.orchestration_metrics.setdefault(key, value)
        self.orchestration_metrics["branch_selected_worker_execution"] = 1.0
        worker_slots = self._branch_worker_slots(requests)
        entries = [
            {"slot": slot, "request": request, "result": None}
            for slot, request in zip(worker_slots, requests)
        ]
        if used_origins is None:
            used_origins = set()
        used_origins.update(
            entry["request"].origin_occurrence_id for entry in entries
        )
        for entry in entries:
            request = entry["request"]
            self.replay_retry_metadata.setdefault(request.request_id, {
                "attempt": 0,
                "fallback_level": "selected_origin",
                "parent_request_id": None,
                "execution_wave": int(execution_wave),
                "worker_slot": int(entry["slot"]),
            })

        pending = list(entries)
        root_logs = None
        for attempt in range(self.max_origin_retries + 1):
            if not pending:
                break
            slots = [entry["slot"] for entry in pending]
            active_requests = [entry["request"] for entry in pending]
            started = time.monotonic()
            _, _, results = self.replay_adapter.replay_selected_and_validate(
                slots, active_requests
            )
            self._trace_replay_results(
                active_requests,
                results,
                f"initial_validation_{attempt}",
                time.monotonic() - started,
                cost_kind="validation_replay" if attempt == 0 else "retry_replay",
            )
            retry_entries = []
            for entry, result in zip(pending, results):
                entry["result"] = result
                if result.replay_ok:
                    continue
                if attempt >= self.max_origin_retries:
                    continue
                if root_logs is None:
                    root_logs = build_root_event_logs(root_output)
                request = entry["request"]
                replacement = self._fallback_request(
                    request, root_logs, used_origins
                )
                if replacement is None:
                    continue
                used_origins.add(replacement.origin_occurrence_id)
                self.replay_retry_metadata[replacement.request_id] = {
                    "attempt": attempt + 1,
                    "fallback_level": "same_anchor_action_alternate_origin",
                    "parent_request_id": request.request_id,
                    "execution_wave": int(execution_wave),
                    "worker_slot": int(entry["slot"]),
                }
                if hasattr(self.coordinator, "adopt_retry_request"):
                    self.coordinator.adopt_retry_request(request, replacement)
                entry["request"] = replacement
                entry["result"] = None
                retry_entries.append(entry)
            pending = retry_entries

        valid_entries = [
            entry
            for entry in entries
            if entry["result"] is not None and entry["result"].replay_ok
        ]
        failed_entries = [entry for entry in entries if entry not in valid_entries]
        if self.artifact_store is not None:
            for entry in failed_entries:
                request = entry["request"]
                result = entry["result"]
                self.artifact_store.append("branches", {
                    "request_id": request.request_id,
                    "branch_id": request.branch_id,
                    "task_id": request.task_id,
                    "origin_occurrence_id": request.origin_occurrence_id,
                    "selected_action": request.selected_canonical_action,
                    "status": "replay_budget_exhausted",
                    "failure_category": getattr(result, "category", "REPLAY_FAILED"),
                    "failure_message": getattr(result, "error_message", ""),
                    "worker_slot": int(entry["slot"]),
                    **self.replay_retry_metadata.get(request.request_id, {}),
                })
        if failed_entries and self.variant == "batch_erv_exact":
            raise RuntimeError(
                "Exact Batch-ERV selected-worker replay did not realize every "
                f"frozen branch in execution wave {execution_wave}; validated "
                f"{len(valid_entries)} of {len(entries)} requests"
            )
        if not valid_entries:
            return [], None, None, None

        valid_requests = [entry["request"] for entry in valid_entries]
        worker_slots = [entry["slot"] for entry in valid_entries]
        started = time.monotonic()
        next_obs, origin_rewards, origin_dones, infos = self.branch_envs.step_selected(
            worker_slots,
            [request.copied_raw_model_response for request in valid_requests],
        )
        transition_results = self.replay_adapter.validate_transitions(
            valid_requests, next_obs, origin_rewards, origin_dones, infos
        )
        self._trace_replay_results(
            valid_requests,
            transition_results,
            "origin_transition_validation",
            time.monotonic() - started,
            prefix_lengths=[1] * len(valid_requests),
            cost_kind="origin_transition",
        )
        transition_valid = [
            index
            for index, result in enumerate(transition_results)
            if result.replay_ok
        ]
        for request, slot, result in zip(
            valid_requests, worker_slots, transition_results
        ):
            if result.replay_ok or self.artifact_store is None:
                continue
            self.artifact_store.append("branches", {
                "request_id": request.request_id,
                "branch_id": request.branch_id,
                "task_id": request.task_id,
                "origin_occurrence_id": request.origin_occurrence_id,
                "selected_action": request.selected_canonical_action,
                "copied_action_identity": request.copied_action_identity,
                "status": "transition_replay_mismatch",
                "failure_category": result.category,
                "worker_slot": int(slot),
            })
        if len(transition_valid) != len(valid_requests):
            if self.variant == "batch_erv_exact":
                raise RuntimeError(
                    "Exact Batch-ERV selected-worker origin transition validation "
                    f"failed in execution wave {execution_wave}; validated "
                    f"{len(transition_valid)} of {len(valid_requests)} requests"
                )
            valid_requests = [valid_requests[index] for index in transition_valid]
            worker_slots = [worker_slots[index] for index in transition_valid]
            next_obs = {
                key: None if values is None else [values[index] for index in transition_valid]
                for key, values in next_obs.items()
            }
            origin_rewards = np.asarray(origin_rewards).reshape(-1)[transition_valid]
            origin_dones = np.asarray(origin_dones).reshape(-1)[transition_valid]
        if not valid_requests:
            return [], None, None, None

        origin_rewards = np.asarray(origin_rewards).reshape(-1)
        origin_dones = np.asarray(origin_dones).reshape(-1)
        branch_gen_batch = gen_batch.select_idxs(
            [request.task_batch_index for request in valid_requests]
        )
        suffix_started = time.monotonic()
        suffix_output, terminal_rewards, _ = self._collect_suffixes_selected(
            branch_gen_batch,
            actor_rollout_wg,
            valid_requests,
            worker_slots,
            origin_rewards,
            origin_dones,
        )
        self.orchestration_metrics["branch_suffix_generation_seconds"] = (
            self.orchestration_metrics.get("branch_suffix_generation_seconds", 0.0)
            + float(time.monotonic() - suffix_started)
        )
        self.orchestration_metrics["branch_suffix_waves"] = (
            self.orchestration_metrics.get("branch_suffix_waves", 0.0) + 1.0
        )
        self.orchestration_metrics["branch_suffix_trajectories"] = (
            self.orchestration_metrics.get("branch_suffix_trajectories", 0.0)
            + float(len(valid_requests))
        )
        self.orchestration_metrics["branch_selected_worker_execution"] = 1.0

        occurrence_to_index = {
            str(occurrence_id): index
            for index, occurrence_id in enumerate(
                root_output.non_tensor_batch["occurrence_id"]
            )
        }
        origin_output = root_output.select_idxs([
            occurrence_to_index[request.origin_occurrence_id]
            for request in valid_requests
        ])
        origin_output.non_tensor_batch["traj_uid"] = np.asarray(
            [request.branch_id for request in valid_requests], dtype=object
        )
        origin_output.non_tensor_batch["occurrence_id"] = np.asarray(
            [f"{request.branch_id}:origin" for request in valid_requests], dtype=object
        )
        origin_output.non_tensor_batch["episode_rewards"] = terminal_rewards.astype(
            np.float32
        )
        origin_output.non_tensor_batch["rewards"] = np.asarray(
            origin_rewards, dtype=object
        )
        origin_output.non_tensor_batch["done"] = origin_dones.astype(bool)
        self._refresh_success_rate(origin_output)
        self._set_lineage(
            origin_output,
            "branch_origin",
            [request.branch_id for request in valid_requests],
        )
        if suffix_output is not None:
            self._set_lineage(
                suffix_output,
                "branch_suffix",
                suffix_output.non_tensor_batch["traj_uid"],
            )
        return valid_requests, origin_output, suffix_output, terminal_rewards

    def _execute_chunk(self, *args, **kwargs):
        if self.branch_execution_mode == "selected_worker":
            return self._execute_chunk_selected(*args, **kwargs)
        return self._execute_chunk_legacy(*args, **kwargs)

    def _branch_worker_slots(self, requests):
        """Map one replay-capacity cohort to unique physical worker slots."""
        requests = list(requests)
        if getattr(self, "branch_pool_mode", "dedicated") == "dedicated":
            return list(range(len(requests)))

        capacity = int(self.branch_envs.replay_capacity)
        if len(requests) > capacity:
            raise ValueError("A branch capacity cohort exceeds replay capacity")
        group_size = int(self.main_group_size)
        if capacity % group_size:
            raise ValueError(
                "Main replay capacity must be divisible by its rollout group size"
            )
        task_capacity = capacity // group_size
        local_counts = {}
        slots = []
        for request in requests:
            task_index = int(request.task_batch_index)
            if not 0 <= task_index < task_capacity:
                raise ValueError(
                    f"Invalid task_batch_index for main replay pool: {task_index}"
                )
            local_slot = local_counts.get(task_index, 0)
            if local_slot >= group_size:
                raise ValueError(
                    "A task has more branch requests in one cohort than its "
                    "physical main-pool group can hold"
                )
            slots.append(task_index * group_size + local_slot)
            local_counts[task_index] = local_slot + 1
        if len(slots) != len(set(slots)):
            raise AssertionError("Branch physical slot collision")
        if slots and (min(slots) < 0 or max(slots) >= capacity):
            raise AssertionError("Branch physical slot falls outside replay capacity")
        return slots

    def _branch_capacity_cohorts(self, requests, capacity):
        """Chunk requests without overbooking any task-local main-pool group."""
        requests = list(requests)
        if getattr(self, "branch_pool_mode", "dedicated") == "dedicated":
            return self._chunk_requests(requests, capacity)
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        group_size = int(self.main_group_size)
        cohorts = []
        current = []
        counts = {}
        for request in requests:
            task_index = int(request.task_batch_index)
            if current and (
                len(current) >= capacity
                or counts.get(task_index, 0) >= group_size
            ):
                cohorts.append(current)
                current = []
                counts = {}
            current.append(request)
            counts[task_index] = counts.get(task_index, 0) + 1
        if current:
            cohorts.append(current)
        return cohorts

    def _execute_round(self, root_output, gen_batch, actor_rollout_wg, requests):
        """Execute one frozen acquisition round in replay-capacity-sized waves."""
        requests = list(requests)
        capacity = int(self.branch_envs.replay_capacity)
        chunks = self._branch_capacity_cohorts(requests, capacity)
        if not chunks:
            return [], None, None, None

        # Reserve every originally selected occurrence before the first wave.
        # A retry in an early wave must not steal an origin scheduled later.
        used_origins = {
            request.origin_occurrence_id for request in requests
        }
        all_valid = []
        origin_batches = []
        suffix_batches = []
        reward_batches = []

        self.orchestration_metrics["branch_replay_capacity"] = float(capacity)
        self.orchestration_metrics["branch_requests_total"] = float(len(requests))
        self.orchestration_metrics["branch_capacity_cohorts"] = (
            self.orchestration_metrics.get("branch_capacity_cohorts", 0.0)
            + float(len(chunks))
        )
        self.orchestration_metrics["branch_main_pool_reuse"] = float(
            getattr(self, "branch_pool_mode", "dedicated") == "main_reuse"
        )
        self.orchestration_metrics["branch_slot_collisions"] = 0.0
        self.orchestration_metrics["branch_execution_waves"] = (
            self.orchestration_metrics.get("branch_execution_waves", 0.0)
            + float(len(chunks))
        )
        self.orchestration_metrics["branch_execution_max_wave_size"] = max(
            self.orchestration_metrics.get("branch_execution_max_wave_size", 0.0),
            float(max(len(chunk) for chunk in chunks)),
        )

        for execution_wave, chunk in enumerate(chunks):
            worker_slots = self._branch_worker_slots(chunk)
            if self.artifact_store is not None:
                self.artifact_store.append("branch_execution_waves", {
                    "execution_wave": execution_wave,
                    "capacity_cohort": execution_wave,
                    "request_count": len(chunk),
                    "replay_capacity": capacity,
                    "branch_pool_mode": getattr(
                        self, "branch_pool_mode", "dedicated"
                    ),
                    "worker_slots": worker_slots,
                    "request_ids": [request.request_id for request in chunk],
                    "branch_ids": [request.branch_id for request in chunk],
                })
            valid, origin_output, suffix_output, terminal_rewards = self._execute_chunk(
                root_output,
                gen_batch,
                actor_rollout_wg,
                chunk,
                used_origins=used_origins,
                execution_wave=execution_wave,
            )
            if self.variant == "batch_erv_exact" and len(valid) != len(chunk):
                raise RuntimeError(
                    "Exact Batch-ERV replay did not realize every frozen branch in "
                    f"execution wave {execution_wave}; validated {len(valid)} of "
                    f"{len(chunk)} requests"
                )
            all_valid.extend(valid)
            if origin_output is not None:
                origin_batches.append(origin_output)
            if suffix_output is not None:
                suffix_batches.append(suffix_output)
            if terminal_rewards is not None:
                reward_batches.append(np.asarray(terminal_rewards))

        if self.variant == "batch_erv_exact":
            requested_branch_ids = [request.branch_id for request in requests]
            realized_branch_ids = [request.branch_id for request in all_valid]
            if realized_branch_ids != requested_branch_ids:
                raise RuntimeError(
                    "Exact Batch-ERV branch identity/order changed during chunked execution"
                )

        def merge_optional(batches):
            if not batches:
                return None
            if len(batches) == 1:
                return batches[0]
            return self._concat_batches(batches)

        terminal_rewards = (
            np.concatenate(reward_batches) if reward_batches else None
        )
        return (
            all_valid,
            merge_optional(origin_batches),
            merge_optional(suffix_batches),
            terminal_rewards,
        )

    @staticmethod
    def _suffix_pairs(suffix_output, branch_id):
        if suffix_output is None:
            return []
        pairs = []
        for idx, traj_uid in enumerate(suffix_output.non_tensor_batch["traj_uid"]):
            if str(traj_uid) != branch_id:
                continue
            if not bool(suffix_output.non_tensor_batch["is_action_valid"][idx]):
                continue
            pairs.append(
                (
                    suffix_output.non_tensor_batch["anchor_obs"][idx],
                    suffix_output.non_tensor_batch.get(
                        "action_identity", suffix_output.non_tensor_batch["projected_action"]
                    )[idx],
                )
            )
        return pairs

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if not is_train:
            return super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=False)

        self.artifact_store = self._start_artifact_store()
        self.trace_diagnostics = {}
        self.replay_retry_metadata = {}
        self.orchestration_metrics = {}
        self._pairwise_root_context = None
        rollout_started = time.monotonic()

        if (
            is_train
            and self.topology == "dynamic"
            and self.dynamic_root_generation == "staged"
            and self.staged_root_batching == "frontier"
        ):
            root_output, root_logs, topology_plan = self._collect_frontier_dynamic(
                gen_batch, actor_rollout_wg, envs
            )
            self.last_bace_metrics = {
                "requested": int(getattr(self, "_frontier_branch_count", 0)),
                "validated": int(self.orchestration_metrics.get("frontier/branch_terminal_events", 0)),
                "skipped": 0,
                "erv_rounds": self.orchestration_metrics.get("frontier/branch_terminal_events", 0),
                **self._episode_success_metrics(root_output),
                **self.orchestration_metrics,
            }
            return root_output

        topology_plan = None
        generated_by_task = None
        if self.topology == "dynamic" and self.dynamic_root_generation == "staged":
            root_output, root_logs, topology_plan, generated_by_task = (
                self._collect_staged_dynamic_roots(gen_batch, actor_rollout_wg, envs)
            )
            candidate_root_logs = root_logs
        else:
            root_output = super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=True)
            candidate_root_logs = build_root_event_logs(root_output)

        if self.topology == "dynamic" and self.dynamic_root_generation == "preallocated":
            topology_plan = self.topology_planner.plan(candidate_root_logs)
            selected_root_ids = {root.root_id for root in topology_plan.roots}
            selected_rows = [
                idx
                for idx, traj_uid in enumerate(root_output.non_tensor_batch["traj_uid"])
                if str(traj_uid) in selected_root_ids
            ]
            root_output = root_output.select_idxs(selected_rows)
            root_logs = list(topology_plan.roots)
        elif self.topology == "fixed":
            root_logs = candidate_root_logs
        elif self.topology == "dynamic" and self.dynamic_root_generation == "staged":
            pass
        else:
            raise ValueError(f"Unknown BACE topology mode: {self.topology}")
        # These aliases are intentionally stable across Full/P1/P2 so online
        # comparisons do not have to know the internal root-wave names.
        self.orchestration_metrics["time/initial_root"] = float(
            self.orchestration_metrics.get("planned_root_generation_seconds", 0.0)
            + self.orchestration_metrics.get("pilot_root_generation_seconds", 0.0)
        )
        self.orchestration_metrics["time/capacity_correction"] = float(
            self.orchestration_metrics.get(
                "capacity_correction_root_generation_seconds", 0.0
            )
        )
        self.orchestration_metrics["count/root_generation_waves"] = float(
            self.orchestration_metrics.get("root_generation_waves", 0.0)
        )
        self.orchestration_metrics["count/capacity_correction_waves"] = float(
            self.orchestration_metrics.get("capacity_correction_root_waves", 0.0)
        )
        self._set_lineage(
            root_output,
            "root",
            [f"root:{traj_uid}" for traj_uid in root_output.non_tensor_batch["traj_uid"]],
        )
        selected_ids = {root.root_id for root in root_logs}
        self._trace_roots(candidate_root_logs, selected_ids=selected_ids)
        self._trace_anchors(root_logs)
        self._trace_topology(topology_plan, root_logs)

        batches = [root_output]
        branch_origin_by_id = {}
        requested = 0
        validated = 0
        erv_rounds = 0
        if self.acquisition in {"erv", "batch_erv_exact"}:
            if hasattr(self.coordinator, "set_policy_update_id"):
                self.coordinator.set_policy_update_id(self.current_step)
            skipped = self.coordinator.initialize(
                root_logs,
                branch_quota_by_task=(topology_plan.branch_quota_by_task if topology_plan else None),
                prior_mean_by_task=(topology_plan.posterior_mean_by_task if topology_plan else None),
            )
            while True:
                acquisition_started = time.monotonic()
                requests = self.coordinator.build_round_requests()
                self.orchestration_metrics["time/acquisition_compute"] = (
                    self.orchestration_metrics.get("time/acquisition_compute", 0.0)
                    + float(time.monotonic() - acquisition_started)
                )
                if not requests:
                    break
                erv_rounds += 1
                requested += len(requests)
                self._trace_branch_selections(requests)
                if self.artifact_store is not None:
                    self.artifact_store.append("acquisition_rounds", {
                        "round": erv_rounds,
                        "requests": requests,
                        "diagnostics": self.coordinator.last_round_diagnostics,
                        "posterior_snapshot": self.coordinator.posterior_snapshot(),
                    })
                branch_started = time.monotonic()
                valid, origin_output, suffix_output, terminal_rewards = self._execute_round(
                    root_output, gen_batch, actor_rollout_wg, requests
                )
                self.orchestration_metrics["time/branch_generation"] = (
                    self.orchestration_metrics.get("time/branch_generation", 0.0)
                    + float(time.monotonic() - branch_started)
                )
                if self.variant == "batch_erv_exact" and len(valid) != len(requests):
                    raise RuntimeError(
                        "Exact Batch-ERV replay did not realize every frozen branch; "
                        f"validated {len(valid)} of {len(requests)} requests"
                    )
                validated += len(valid)
                if not valid:
                    continue
                branch_origin_by_id.update({
                    request.branch_id: request.origin_occurrence_id
                    for request in valid
                })
                if self.artifact_store is not None:
                    for request, reward in zip(valid, terminal_rewards):
                        suffix_ids = [] if suffix_output is None else [
                            str(value)
                            for value, traj_uid in zip(
                                suffix_output.non_tensor_batch["occurrence_id"],
                                suffix_output.non_tensor_batch["traj_uid"],
                            )
                            if str(traj_uid) == request.branch_id
                        ]
                        self.artifact_store.append("branches", {
                            "request_id": request.request_id,
                            "branch_id": request.branch_id,
                            "task_id": request.task_id,
                            "origin_occurrence_id": request.origin_occurrence_id,
                            "selected_action": request.selected_canonical_action,
                            "terminal_reward": reward,
                            "suffix_occurrence_ids": suffix_ids,
                            "suffix_occurrence_count": len(suffix_ids),
                        })
                batches.append(origin_output)
                if suffix_output is not None:
                    batches.append(suffix_output)
                for request, reward in zip(valid, terminal_rewards):
                    self.coordinator.update_from_branch(
                        request,
                        success=float(reward) > 0,
                        suffix_pairs=self._suffix_pairs(suffix_output, request.branch_id),
                    )
                if self.artifact_store is not None:
                    self.artifact_store.append("posterior_snapshots", {
                        "round": erv_rounds,
                        "posterior_snapshot": self.coordinator.posterior_snapshot(),
                    })
        else:
            requests, skipped = self.coordinator.build_requests(root_logs)
            requested = len(requests)
            self._trace_branch_selections(requests)
            if requests:
                valid, origin_output, suffix_output, _ = self._execute_round(
                    root_output, gen_batch, actor_rollout_wg, requests
                )
                validated = len(valid)
                if valid:
                    branch_origin_by_id.update({
                        request.branch_id: request.origin_occurrence_id
                        for request in valid
                    })
                    if self.artifact_store is not None:
                        for request in valid:
                            self.artifact_store.append("branches", {
                                "request_id": request.request_id,
                                "branch_id": request.branch_id,
                                "task_id": request.task_id,
                                "origin_occurrence_id": request.origin_occurrence_id,
                                "selected_action": request.selected_canonical_action,
                            })
                    batches.append(origin_output)
                    if suffix_output is not None:
                        batches.append(suffix_output)

        if (
            self.variant == "batch_erv_exact"
            and self.pairwise_mode == "stopping"
            and topology_plan is not None
        ):
            fallback_by_task = self.coordinator.consume_fallback_roots()
            final_counts = self.coordinator.pairwise_final_counts()
            fallback_output, fallback_logs = self._collect_pairwise_fallback_roots(
                gen_batch, actor_rollout_wg, envs, fallback_by_task
            )
            if fallback_output is not None:
                batches.append(fallback_output)
                root_logs.extend(fallback_logs)
                self._trace_roots(fallback_logs, selected_ids={root.root_id for root in fallback_logs})
            final_tasks = {}
            for task_id, task in topology_plan.tasks.items():
                counts = final_counts.get(task_id, {
                    "executed_branches": task.final_branch_count,
                    "fallback_roots": 0,
                    "planned_branches": task.planned_branch_count,
                })
                final_root_count = task.final_root_count + int(counts["fallback_roots"])
                final_branch_count = int(counts["executed_branches"])
                if final_root_count + final_branch_count != int(
                    self.config.algorithm.bace.total_leaf_budget
                ):
                    raise AssertionError("Pairwise stopping violated the leaf budget")
                final_tasks[task_id] = dataclasses.replace(
                    task,
                    final_root_count=final_root_count,
                    final_branch_count=final_branch_count,
                    correction_count=task.correction_count + int(counts["fallback_roots"]),
                )
            topology_plan = dataclasses.replace(
                topology_plan, roots=tuple(root_logs), tasks=final_tasks
            )
            # Preserve both the pre-branch plan and the actual P2 final leaf
            # topology in the trace.  Consumers can distinguish this entry by
            # its explicit phase rather than infer it from root counts.
            if self.artifact_store is not None:
                self.artifact_store.append("topology", {
                    "phase": "pairwise_stopping_final",
                    "plan": topology_plan,
                    "root_count": len(root_logs),
                    "root_ids": [root.root_id for root in root_logs],
                })
            context = self._pairwise_root_context
            generated_by_task = {
                context["task_uids"][index]: count
                for index, count in context["generated_by_task"].items()
            }
            if self.artifact_store is not None:
                self.artifact_store.append("pairwise_fallback_roots", {
                    "policy_update_id": self.current_step,
                    "mode": "stopping",
                    "fallback_by_task": fallback_by_task,
                    "final_counts": final_counts,
                    "one_way_fallback": True,
                })

        if self.variant == "batch_erv_exact":
            self.orchestration_metrics["pairwise_fixed"] = float(
                self.pairwise_mode == "fixed"
            )
            self.orchestration_metrics["pairwise_stopping"] = float(
                self.pairwise_mode == "stopping"
            )
            self.orchestration_metrics["pairwise_branch_rounds"] = float(erv_rounds)
            self.orchestration_metrics["count/pairwise_branch_rounds"] = float(erv_rounds)
            self.orchestration_metrics["time/fallback_root"] = float(
                self.orchestration_metrics.get("fallback_root_generation_seconds", 0.0)
            )
            self.orchestration_metrics["count/fallback_root_waves"] = float(
                self.orchestration_metrics.get("fallback_root_waves", 0.0)
            )
            self.orchestration_metrics["time/rollout_total"] = float(
                time.monotonic() - rollout_started
            )

        global_allocation_diagnostics = [
            diagnostic["global_allocation"]
            for diagnostic in getattr(self.coordinator, "last_round_diagnostics", [])
            if "global_allocation" in diagnostic
        ]
        if global_allocation_diagnostics:
            solver_times = np.asarray([
                item["solver_wall_time_ms"]
                for item in global_allocation_diagnostics
            ], dtype=float)
            self.orchestration_metrics.update({
                "global_alloc/num_anchors_mean": float(np.mean([
                    item["num_anchors"] for item in global_allocation_diagnostics
                ])),
                "global_alloc/quota_mean": float(np.mean([
                    item["branch_quota"] for item in global_allocation_diagnostics
                ])),
                "global_alloc/total_capacity_mean": float(np.mean([
                    item["total_information_capacity"]
                    for item in global_allocation_diagnostics
                ])),
                "global_alloc/reachable_states_mean": float(np.mean([
                    item["reachable_state_count"]
                    for item in global_allocation_diagnostics
                ])),
                "global_alloc/optimal_tie_count_mean": float(np.mean([
                    item["optimal_tie_count"]
                    for item in global_allocation_diagnostics
                ])),
                "global_alloc/solver_time_ms_mean": float(np.mean(solver_times)),
                "global_alloc/solver_time_ms_p50": float(np.percentile(solver_times, 50)),
                "global_alloc/solver_time_ms_p95": float(np.percentile(solver_times, 95)),
                "global_alloc/solver_time_ms_max": float(np.max(solver_times)),
            })

        topology_metrics = {}
        if topology_plan is not None:
            tasks = list(topology_plan.tasks.values())
            if generated_by_task is None:
                generated_counts = {
                    task.task_id: int(self.config.algorithm.bace.total_leaf_budget)
                    for task in tasks
                }
            else:
                generated_counts = {
                    task_id: count for task_id, count in generated_by_task.items()
                }
            topology_metrics = {
                "capacity_correction_batch_size": float(
                    self.capacity_correction_batch_size
                ),
                "quota_policy_current": float(self.quota_policy == "current"),
                "quota_policy_conservative_rmin4": float(
                    self.quota_policy == "conservative_rmin4"
                ),
                "root_generation_mode": float(self.dynamic_root_generation == "staged"),
                "generated_roots_mean": float(np.mean(list(generated_counts.values()))),
                "discarded_roots_mean": float(
                    np.mean([
                        generated_counts[task.task_id] - task.final_root_count
                        for task in tasks
                    ])
                ),
                "planned_branches_mean": float(np.mean([task.planned_branch_count for task in tasks])),
                "planned_branches_total": float(
                    sum(task.planned_branch_count for task in tasks)
                ),
                "planned_roots_mean": float(
                    np.mean([
                        int(self.config.algorithm.bace.total_leaf_budget)
                        - task.planned_branch_count
                        for task in tasks
                    ])
                ),
                "final_roots_mean": float(np.mean([task.final_root_count for task in tasks])),
                "final_branches_mean": float(np.mean([task.final_branch_count for task in tasks])),
                "final_branches_total": float(
                    sum(task.final_branch_count for task in tasks)
                ),
                "effective_anchors_mean": float(np.mean([task.effective_anchor_count for task in tasks])),
                "competence_readiness_mean": float(np.mean([task.readiness for task in tasks])),
                "capacity_corrections_mean": float(
                    np.mean([task.correction_count for task in tasks])
                ),
                "corrections_total": float(
                    sum(task.correction_count for task in tasks)
                ),
                "correction_rate_normalized": float(
                    sum(task.correction_count for task in tasks)
                    / max(sum(task.planned_branch_count for task in tasks), 1)
                ),
                "correction_depth_p50": float(np.percentile(
                    [task.correction_count for task in tasks], 50
                )),
                "correction_depth_p90": float(np.percentile(
                    [task.correction_count for task in tasks], 90
                )),
                "correction_depth_max": float(max(
                    task.correction_count for task in tasks
                )),
                "information_capacity_mean": float(
                    np.mean([
                        task.information_capacity
                        for task in tasks
                        if task.information_capacity is not None
                    ])
                ) if any(task.information_capacity is not None for task in tasks) else 0.0,
                "initial_information_capacity_mean": float(np.mean([
                    task.initial_information_capacity
                    for task in tasks
                    if task.initial_information_capacity is not None
                ])) if any(
                    task.initial_information_capacity is not None for task in tasks
                ) else 0.0,
                "initial_quota_deficit_mean": float(np.mean([
                    task.initial_quota_deficit
                    for task in tasks
                    if task.initial_quota_deficit is not None
                ])) if any(
                    task.initial_quota_deficit is not None for task in tasks
                ) else 0.0,
                "normalized_initial_quota_deficit_mean": float(np.mean([
                    task.initial_normalized_quota_deficit
                    for task in tasks
                    if task.initial_normalized_quota_deficit is not None
                    and task.planned_branch_count > 0
                ])) if any(
                    task.initial_normalized_quota_deficit is not None
                    and task.planned_branch_count > 0
                    for task in tasks
                ) else 0.0,
                "zero_planned_branch_task_fraction": float(np.mean([
                    task.planned_branch_count == 0 for task in tasks
                ])),
                "root_generation_seconds_total": float(
                    self.orchestration_metrics.get(
                        "planned_root_generation_seconds", 0.0
                    )
                    + self.orchestration_metrics.get(
                        "capacity_correction_root_generation_seconds", 0.0
                    )
                ),
                "family_prior_mean": float(
                    np.mean([
                        task.family_prior_mean
                        for task in tasks
                        if task.family_prior_mean is not None
                    ])
                ) if any(task.family_prior_mean is not None for task in tasks) else 0.0,
                **self.orchestration_metrics,
            }
            history_before = self.competence_history.snapshot()
            self.topology_planner.update_history(topology_plan)
            if self.artifact_store is not None:
                outcomes_by_family = {}
                for root in topology_plan.roots:
                    counts = outcomes_by_family.setdefault(
                        root.task_family, {"successes": 0, "failures": 0, "root_ids": []}
                    )
                    counts["successes" if root.won else "failures"] += 1
                    counts["root_ids"].append(root.root_id)
                self.artifact_store.append("family_history_updates", {
                    "policy_update_id": self.current_step,
                    "history_before": history_before,
                    "natural_root_evidence": outcomes_by_family,
                    "history_after": self.competence_history.snapshot(),
                    "branch_outcomes_included": False,
                })

        if len(batches) == 1:
            self._annotate_tree_lineage(root_output, branch_origin_by_id)
            replay_seconds = sum(
                self.trace_diagnostics.get("phase_timing_seconds", {}).values()
            )
            generation_metrics = self._rollout_generation_metrics(root_output)
            self.last_bace_metrics = {
                "requested": requested,
                "validated": validated,
                "skipped": len(skipped),
                "erv_rounds": erv_rounds,
                "replay_validation_seconds": float(replay_seconds),
                **self._episode_success_metrics(root_output),
                **topology_metrics,
                **generation_metrics,
                "generated_tokens/root": generation_metrics["root_generated_tokens"],
                "generated_tokens/branch": generation_metrics["branch_generated_tokens"],
                "generated_tokens/total": (
                    generation_metrics["root_generated_tokens"]
                    + generation_metrics["branch_generated_tokens"]
                ),
            }
            return root_output
        concat_started = time.monotonic()
        merged_output = self._concat_batches(batches)
        self._annotate_tree_lineage(merged_output, branch_origin_by_id)
        self.orchestration_metrics["training_batch_concat_seconds"] = float(
            time.monotonic() - concat_started
        )
        replay_seconds = sum(
            self.trace_diagnostics.get("phase_timing_seconds", {}).values()
        )
        generation_metrics = self._rollout_generation_metrics(merged_output)
        self.last_bace_metrics = {
            "requested": requested,
            "validated": validated,
            "skipped": len(skipped),
            "erv_rounds": erv_rounds,
            "replay_validation_seconds": float(replay_seconds),
            **self._episode_success_metrics(merged_output),
            **topology_metrics,
            **generation_metrics,
            "generated_tokens/root": generation_metrics["root_generated_tokens"],
            "generated_tokens/branch": generation_metrics["branch_generated_tokens"],
            "generated_tokens/total": (
                generation_metrics["root_generated_tokens"]
                + generation_metrics["branch_generated_tokens"]
            ),
            "training_batch_concat_seconds": self.orchestration_metrics[
                "training_batch_concat_seconds"
            ],
        }
        return merged_output
