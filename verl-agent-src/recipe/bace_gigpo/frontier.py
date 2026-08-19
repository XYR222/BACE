"""Optional task-local BACE frontier scheduler.

The legacy collector deliberately remains in ``rollout_collector.py``.  This
module owns the slot/state-machine path described by the 2026-08-10 pipeline
specification and is only called when ``staged_root_batching=frontier``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn

from .coordinator import ExpectedErvCoordinator
from .root_store import build_root_event_logs
from .replay.alfworld import ReplayAdapter
from .types import ReplayRequest


@dataclass
class FrontierJob:
    kind: str  # root or branch
    task_index: int
    task_id: str
    slot_id: int
    traj_uid: str
    step_index: int = 0
    remaining_horizon: int | None = None
    rows: list[dict] = field(default_factory=list)
    episode_reward: float = 0.0
    episode_length: int = 0
    tool_callings: float = 0.0
    request: ReplayRequest | None = None


class FrontierInvariantError(RuntimeError):
    pass


@dataclass
class FrontierTaskController:
    """Task-local controller; no state is shared between task groups."""

    task_index: int
    task_id: str
    slots: tuple[int, ...]
    total_budget: int
    pilot_target: int
    root_logs: list = field(default_factory=list)
    root_target: int | None = None
    branch_quota: int = 0
    branch_completed: int = 0
    root_backbone_frozen: bool = False
    branch_coordinator: object | None = None
    topology_plan: object | None = None
    slot_cursor: int = 0
    pilot_planned: bool = False
    reset_key: str = ""
    _staged_state: object | None = field(default=None, repr=False)

    def allocate_slot(self):
        if self.slot_cursor >= self.total_budget:
            raise FrontierInvariantError(
                f"task {self.task_id} consumed more than {self.total_budget} slots"
            )
        slot = self.slots[self.slot_cursor]
        self.slot_cursor += 1
        return slot

    def add_root(self, root):
        if self.root_backbone_frozen:
            raise FrontierInvariantError(f"task {self.task_id} generated a root after freeze")
        self.root_logs.append(root)

    def plan_after_pilots(self, planner):
        if self.pilot_planned or len(self.root_logs) != self.pilot_target:
            return False
        states = planner.initialize_staged(list(self.root_logs))
        state = states[self.task_id]
        self.root_target = int(state.root_count)
        self.branch_quota = int(state.branch_count)
        self._staged_state = state
        self.pilot_planned = True
        return True

    def capacity_check(self, planner):
        if not self.pilot_planned or self.root_backbone_frozen:
            return False
        states = {self.task_id: self._staged_state}
        deficient = planner.correct_staged_capacity(list(self.root_logs), states)
        self.root_target = int(self._staged_state.root_count)
        self.branch_quota = int(self._staged_state.branch_count)
        if deficient:
            return True
        if len(self.root_logs) != self.root_target:
            return False
        self.topology_plan = planner.finalize_staged(list(self.root_logs), states)
        self.root_backbone_frozen = True
        return True

    @property
    def roots_ready(self):
        return self.pilot_planned and len(self.root_logs) >= int(self.root_target or 0)

    @property
    def complete(self):
        return self.root_backbone_frozen and self.branch_completed >= self.branch_quota

    def assert_budget(self):
        roots = len(self.root_logs)
        branches = self.branch_completed
        if roots + branches > self.total_budget:
            raise FrontierInvariantError(f"task {self.task_id} exceeded leaf budget")
        if self.complete and roots + branches != self.total_budget:
            raise FrontierInvariantError(
                f"task {self.task_id}: roots + branches != {self.total_budget}"
            )
        if self.complete and self.slot_cursor != self.total_budget:
            raise FrontierInvariantError(
                f"task {self.task_id}: consumed {self.slot_cursor} sibling slots, "
                f"expected {self.total_budget}"
            )


@dataclass
class FrontierProfile:
    generation_waves: int = 0
    generated_decisions: int = 0
    generation_seconds: float = 0.0
    selected_environment_steps: int = 0
    replay_jobs: int = 0
    replay_steps: int = 0
    replay_seconds: float = 0.0
    root_terminal_events: int = 0
    branch_terminal_events: int = 0
    model_ready_sizes: list[int] = field(default_factory=list)
    root_queue_sizes: list[int] = field(default_factory=list)
    replay_queue_sizes: list[int] = field(default_factory=list)
    wave_records: list[dict] = field(default_factory=list)
    coalescing_fallback_waves: int = 0
    coalesced_ready_jobs: int = 0
    last_replay_batch_jobs: int = 0

    def snapshot(self):
        def percentile(values, q):
            return float(np.percentile(values, q)) if values else 0.0

        return {
            "frontier/generation_waves": self.generation_waves,
            "frontier/generated_decisions": self.generated_decisions,
            "frontier/generation_seconds": self.generation_seconds,
            "frontier/selected_environment_steps": self.selected_environment_steps,
            "frontier/replay_jobs": self.replay_jobs,
            "frontier/replay_steps": self.replay_steps,
            "frontier/replay_seconds": self.replay_seconds,
            "frontier/root_terminal_events": self.root_terminal_events,
            "frontier/branch_terminal_events": self.branch_terminal_events,
            "frontier/model_ready_batch_size_mean": float(np.mean(self.model_ready_sizes)) if self.model_ready_sizes else 0.0,
            "frontier/model_ready_batch_size_p10": percentile(self.model_ready_sizes, 10),
            "frontier/model_ready_batch_size_p50": percentile(self.model_ready_sizes, 50),
            "frontier/model_ready_batch_size_p90": percentile(self.model_ready_sizes, 90),
            "frontier/root_queue_length": self.root_queue_sizes[-1] if self.root_queue_sizes else 0,
            "frontier/replay_queue_length": self.replay_queue_sizes[-1] if self.replay_queue_sizes else 0,
            "frontier/coalescing_fallback_waves": self.coalescing_fallback_waves,
            "frontier/coalesced_ready_jobs": self.coalesced_ready_jobs,
        }


class BACEFrontierOrchestrator:
    """Mixed MODEL_READY scheduler for dynamic BACE acquisition.

    Replay is deliberately synchronous in this first production-safe version,
    but it is selected-slot based, measured separately, and never invokes the
    actor.  The queue boundaries make asynchronous Ray replay a later optional
    optimization without changing algorithm semantics.
    """

    def __init__(self, collector, gen_batch, actor_rollout_wg, envs):
        self.collector = collector
        self.gen_batch = gen_batch
        self.actor = actor_rollout_wg
        self.envs = envs
        self.replay_adapter = ReplayAdapter(
            manager=envs,
            tokenizer=collector.tokenizer,
            config=collector.config,
            compare_action_set=bool(collector.config.algorithm.bace.replay.compare_action_set),
        )
        cfg = collector.config.algorithm.bace
        self.budget = int(cfg.total_leaf_budget)
        self.pilots = int(cfg.pilot_roots)
        self.profile = FrontierProfile()
        self.tasks: dict[str, FrontierTaskController] = {}
        self.jobs: dict[int, FrontierJob] = {}
        self.root_batches = []
        self.branch_batches = []
        self.branch_records = []
        self.root_logs = []
        self.logical_branches = []
        self._pending_replay: list[tuple[FrontierTaskController, object]] = []
        self._used_origins = set()
        coalescing = cfg.get("frontier_batch_coalescing", {})
        self.batch_coalescing_enabled = bool(coalescing.get("enabled", False))
        self.batch_coalescing_max_size = int(coalescing.get("max_batch_size", 0))
        self.batch_coalescing_min_size = max(1, int(coalescing.get("min_batch_size", 1)))

    def _new_controller(self, task_index, task_id, slots):
        controller = FrontierTaskController(
            task_index=task_index,
            task_id=task_id,
            slots=tuple(slots),
            total_budget=self.budget,
            pilot_target=self.pilots,
        )
        self.tasks[task_id] = controller
        return controller

    def _make_coordinator(self, controller):
        cfg = self.collector.config.algorithm.bace
        coordinator = ExpectedErvCoordinator(
            root_count=len(controller.root_logs),
            branch_count=controller.branch_quota,
            max_branches_per_anchor=int(cfg.max_branches_per_anchor),
            prior_strength=float(cfg.local_prior_strength),
            erv_threshold=float(cfg.erv_threshold),
            temperature=float(cfg.erv_temperature),
            mc_samples=int(cfg.erv_mc_samples),
            seed=int(self.collector.config.env.seed) + controller.task_index,
            invalid_action_mode=str(cfg.invalid_action_mode),
        )
        coordinator.initialize(
            list(controller.root_logs),
            branch_quota_by_task={controller.task_id: controller.branch_quota},
            prior_mean_by_task=controller.topology_plan.posterior_mean_by_task,
        )
        controller.branch_coordinator = coordinator

    @staticmethod
    def _replace_observation_group(observations, group_observations, start, end):
        """Replace a selected reset group in the manager's structured output."""
        if not isinstance(observations, dict) or not isinstance(group_observations, dict):
            observations[start:end] = group_observations
            return
        if observations.keys() != group_observations.keys():
            raise FrontierInvariantError("Selected reset changed observation fields")
        for key, values in group_observations.items():
            target = observations[key]
            if target is None and values is None:
                continue
            if target is None or values is None:
                raise FrontierInvariantError(
                    f"Selected reset changed observation availability for {key}"
                )
            if len(values) != end - start:
                raise FrontierInvariantError(
                    f"Selected reset returned {len(values)} values for {key}, "
                    f"expected {end - start}"
                )
            target[start:end] = values

    def _initial_reset(self):
        task_count = len(self.gen_batch)
        all_slots = list(range(task_count * self.budget))
        observations, infos = self.envs.reset_selected(all_slots)
        reset_keys = [str(info.get("extra.gamefile", "")) for info in infos]
        if any(not key for key in reset_keys):
            raise FrontierInvariantError("Frontier reset did not provide ALFWorld game keys")
        for task_index in range(task_count):
            group = list(range(task_index * self.budget, (task_index + 1) * self.budget))
            group_keys = reset_keys[task_index * self.budget:(task_index + 1) * self.budget]
            # The baseline seeds siblings identically.  If a backend advances a
            # shared iterator differently, explicitly bind all sibling workers
            # to the first concrete game before any model decision.
            if len(set(group_keys)) != 1:
                observations_group, infos_group = self.envs.reset_selected(
                    group, [group_keys[0]] * len(group)
                )
                self._replace_observation_group(
                    observations,
                    observations_group,
                    task_index * self.budget,
                    (task_index + 1) * self.budget,
                )
                infos[task_index * self.budget:(task_index + 1) * self.budget] = infos_group
                group_keys = [group_keys[0]] * len(group)
            task_id = str(uuid.uuid4())
            self._new_controller(task_index, task_id, group)
            for index in group:
                if self.envs._bace_slots[index]['task'] != self.envs._bace_slots[group[0]]['task']:
                    raise FrontierInvariantError("Sibling slots do not share task identity")
            self.tasks[task_id].reset_key = group_keys[0]
        return observations

    def _enqueue_root(self, controller):
        slot = controller.allocate_slot()
        job = FrontierJob(
            kind="root", task_index=controller.task_index, task_id=controller.task_id,
            slot_id=slot, traj_uid=f"root:{controller.task_id}:{slot}",
        )
        self.jobs[slot] = job

    def _enqueue_roots_if_ready(self):
        for controller in self.tasks.values():
            if not controller.pilot_planned:
                continue
            target = int(controller.root_target or 0)
            while len(controller.root_logs) + sum(
                job.kind == "root" and job.task_id == controller.task_id for job in self.jobs.values()
            ) < target:
                self._enqueue_root(controller)

    def _finish_root(self, job):
        for row in job.rows:
            row["episode_rewards"] = float(job.episode_reward)
            row["episode_lengths"] = float(job.episode_length)
            row["success_rate"] = float(job.episode_reward > 0)
        output = self._rows_to_batch(job.rows)
        logs = build_root_event_logs(output)
        if len(logs) != 1:
            raise FrontierInvariantError("A root slot must produce exactly one root event log")
        controller = self.tasks[job.task_id]
        controller.add_root(logs[0])
        self.root_logs.extend(logs)
        self.root_batches.append(output)
        self.profile.root_terminal_events += 1
        if controller.plan_after_pilots(self.collector.topology_planner):
            self._enqueue_roots_if_ready()
        if controller.roots_ready and not controller.root_backbone_frozen:
            changed = controller.capacity_check(self.collector.topology_planner)
            if changed:
                self._enqueue_roots_if_ready()
            if controller.root_backbone_frozen:
                self._make_coordinator(controller)
                self._enqueue_next_branch(controller)
        controller.assert_budget()

    def _enqueue_next_branch(self, controller):
        if not controller.root_backbone_frozen or controller.complete:
            return
        if controller.branch_quota == 0:
            controller.assert_budget()
            return
        requests = controller.branch_coordinator.build_round_requests()
        if not requests:
            raise FrontierInvariantError(
                f"Task {controller.task_id} has branch budget but ERV produced no request"
            )
        # ExpectedErvCoordinator emits at most one request for this task, which
        # enforces exact same-task sequential acquisition.
        self._pending_replay.append((controller, requests[0]))

    def _run_replay(self):
        if not self._pending_replay:
            self.profile.last_replay_batch_jobs = 0
            return
        active = [
            {
                "controller": controller,
                "slot": controller.allocate_slot(),
                "request": request,
                "used": set(self._used_origins),
            }
            for controller, request in self._pending_replay
        ]
        self._pending_replay.clear()
        self.profile.last_replay_batch_jobs = len(active)
        restored = []
        for attempt in range(self.collector.max_origin_retries + 1):
            if not active:
                break
            slots = [entry["slot"] for entry in active]
            requests = [entry["request"] for entry in active]
            started = time.monotonic()
            _, _, results = self.replay_adapter.replay_selected_and_validate(slots, requests)
            self.profile.replay_seconds += time.monotonic() - started
            self.profile.replay_jobs += len(requests)
            self.profile.replay_steps += sum(len(request.parsed_action_prefix) for request in requests)
            retry = []
            for entry, result in zip(active, results):
                request = entry["request"]
                if result.replay_ok:
                    restored.append(entry)
                    continue
                replacement = self.collector._fallback_request(
                    request, self.root_logs, entry["used"]
                ) if attempt < self.collector.max_origin_retries else None
                if replacement is None:
                    raise FrontierInvariantError(
                        f"Replay failed for task={entry['controller'].task_id}, "
                        f"branch={request.branch_id}, category={result.category}"
                    )
                entry["used"].add(replacement.origin_occurrence_id)
                self.collector.replay_retry_metadata[replacement.request_id] = {
                    "attempt": attempt + 1,
                    "fallback_level": "same_anchor_action_alternate_origin",
                    "parent_request_id": request.request_id,
                }
                entry["controller"].branch_coordinator.adopt_retry_request(
                    request, replacement
                )
                entry["request"] = replacement
                retry.append(entry)
            active = retry
        if active:
            raise FrontierInvariantError("Replay retry loop ended with unresolved requests")

        slots = [entry["slot"] for entry in restored]
        requests = [entry["request"] for entry in restored]
        for entry in restored:
            controller = entry["controller"]
            active_request = entry["request"]
            self._used_origins.add(active_request.origin_occurrence_id)
            if self.collector.artifact_store is not None:
                self.collector.artifact_store.append("acquisition_rounds", {
                    "task_id": controller.task_id,
                    "request": active_request,
                    "diagnostics": controller.branch_coordinator.last_round_diagnostics,
                    "posterior_snapshot": controller.branch_coordinator.posterior_snapshot(),
                })
        observations, rewards, dones, infos = self.envs.step_selected(
            slots, [request.copied_raw_model_response for request in requests]
        )
        transitions = self.replay_adapter.validate_transitions(
            requests, observations, rewards, dones, infos
        )
        for entry, transition in zip(restored, transitions):
            if not transition.replay_ok:
                raise FrontierInvariantError(
                    f"Branch origin transition mismatch for "
                    f"{entry['request'].branch_id}: {transition.category}"
                )
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1)
        for index, entry in enumerate(restored):
            controller = entry["controller"]
            slot = entry["slot"]
            active_request = entry["request"]
            reward = float(rewards[index])
            done = bool(dones[index])
            job = FrontierJob(
                kind="branch", task_index=controller.task_index,
                task_id=controller.task_id, slot_id=slot,
                traj_uid=active_request.branch_id,
                step_index=active_request.target_turn + 1,
                remaining_horizon=active_request.remaining_horizon,
                episode_reward=reward, episode_length=1,
                request=active_request,
            )
            if done or not active_request.remaining_horizon:
                single_observation = {
                    key: None if values is None else [values[index]]
                    for key, values in observations.items()
                }
                self._finish_branch(job, single_observation, reward, done)
            else:
                self.jobs[slot] = job

    def _finish_branch(self, job, observations, terminal_reward, terminal_done):
        # Every suffix row receives the branch terminal outcome.  The replayed
        # prefix is intentionally absent from ``job.rows``.
        for row in job.rows:
            row["episode_rewards"] = float(terminal_reward)
            row["episode_lengths"] = float(job.episode_length)
            row["success_rate"] = float(float(terminal_reward) > 0)
        output = self._rows_to_batch(job.rows) if job.rows else None
        if output is not None:
            self.collector._set_lineage(
                output, "branch_suffix", output.non_tensor_batch["traj_uid"]
            )
            self.branch_batches.append(output)
        controller = self.tasks[job.task_id]
        request = job.request
        self.logical_branches.append({
            "branch_id": request.branch_id,
            "task_id": request.task_id,
            "origin_occurrence_id": request.origin_occurrence_id,
            "selected_action": request.selected_canonical_action,
            "terminal_reward": float(terminal_reward),
            "terminal_done": bool(terminal_done),
            "suffix_steps": len(job.rows),
        })
        self.branch_records.append((request, float(terminal_reward)))
        suffix_pairs = []
        if output is not None:
            for index in range(len(output)):
                if bool(output.non_tensor_batch.get("is_action_valid", [True])[index]):
                    suffix_pairs.append((
                        output.non_tensor_batch["anchor_obs"][index],
                        output.non_tensor_batch.get("action_identity", output.non_tensor_batch["projected_action"])[index],
                    ))
        controller.branch_coordinator.update_from_branch(
            request, success=float(terminal_reward) > 0, suffix_pairs=suffix_pairs
        )
        controller.branch_completed += 1
        self.profile.branch_terminal_events += 1
        controller.assert_budget()
        if self.collector.artifact_store is not None:
            self.collector.artifact_store.append("branches", self.logical_branches[-1])
            self.collector.artifact_store.append("posterior_snapshots", {
                "task_id": controller.task_id,
                "completed_branches": controller.branch_completed,
                "posterior_snapshot": controller.branch_coordinator.posterior_snapshot(),
            })
        self._enqueue_next_branch(controller)

    def _rows_to_batch(self, rows):
        if not rows:
            return None
        return DataProto.from_single_dict(data=collate_fn(rows))

    def _generate_frontier(self):
        if not self.jobs:
            return
        ready_queue_before = len(self.jobs)
        slots = self._select_generation_slots()
        jobs = [self.jobs[slot] for slot in slots]
        observations = self.envs.get_observations_selected(slots)
        task_indices = [job.task_index for job in jobs]
        source_batch = self.gen_batch.select_idxs(task_indices)
        started = time.monotonic()
        batch = self.collector.preprocess_batch(source_batch, observations)
        size = len(jobs)
        batch.non_tensor_batch["step_index"] = np.asarray([job.step_index for job in jobs], dtype=np.int32)
        batch.non_tensor_batch["admissible_actions"] = np.asarray(observations["admissible_actions"], dtype=object)
        batch_input = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[key for key in ("raw_prompt_ids", "multi_modal_data", "raw_prompt", "tools_kwargs") if key in batch.non_tensor_batch],
        )
        batch_input.meta_info = source_batch.meta_info
        padded, pad_size = pad_dataproto_to_divisor(batch_input, self.actor.world_size)
        output = unpad_dataproto(self.actor.generate_sequences(padded), pad_size=pad_size)
        batch = batch.union(output)
        actions = self.collector.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        next_obs, rewards, dones, infos = self.envs.step_selected(slots, actions)
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1)
        elapsed = time.monotonic() - started
        self.profile.generation_waves += 1
        self.profile.generated_decisions += size
        self.profile.generation_seconds += elapsed
        self.profile.selected_environment_steps += size
        self.profile.model_ready_sizes.append(size)
        wave_record = {
            "wave_index": self.profile.generation_waves - 1,
            "batch_size": size,
            "root_jobs": sum(job.kind == "root" for job in jobs),
            "branch_jobs": sum(job.kind == "branch" for job in jobs),
            "task_count": len(set(job.task_id for job in jobs)),
            "replay_jobs_ready": self.profile.last_replay_batch_jobs,
            "ready_queue_before": ready_queue_before,
            "ready_queue_after": len(self.jobs) - size,
            "elapsed_seconds": elapsed,
            "coalescing_enabled": self.batch_coalescing_enabled,
        }
        self.profile.wave_records.append(wave_record)
        if self.collector.artifact_store is not None:
            self.collector.artifact_store.append("generation_waves", wave_record)
        if self.batch_coalescing_enabled:
            self.profile.coalesced_ready_jobs += size
        batch.non_tensor_batch["uid"] = np.asarray([job.task_id for job in jobs], dtype=object)
        batch.non_tensor_batch["traj_uid"] = np.asarray([job.traj_uid for job in jobs], dtype=object)
        batch.non_tensor_batch["occurrence_id"] = np.asarray(
            [f"{job.traj_uid}:{job.step_index}" for job in jobs], dtype=object
        )
        batch.non_tensor_batch["task_batch_index"] = np.asarray(task_indices, dtype=np.int32)
        batch.non_tensor_batch["is_action_valid"] = np.asarray(
            [info.get("is_action_valid", True) for info in infos], dtype=bool
        )
        batch.non_tensor_batch["raw_model_response"] = np.asarray(actions, dtype=object)
        batch.non_tensor_batch["projected_action"] = np.asarray(
            [info.get("projected_action", "INVALID") for info in infos], dtype=object
        )
        batch.non_tensor_batch["is_action_format_valid"] = np.asarray(
            [info.get("is_action_format_valid", info.get("is_action_valid", False)) for info in infos], dtype=bool
        )
        batch.non_tensor_batch["is_action_environment_valid"] = np.asarray(
            [info.get("is_action_environment_valid", info.get("is_action_valid", False)) for info in infos], dtype=bool
        )
        batch.non_tensor_batch["action_identity"] = np.asarray(
            [info.get("action_identity") for info in infos], dtype=object
        )
        batch.non_tensor_batch["post_action_observation"] = np.asarray(next_obs["anchor"], dtype=object)
        batch.non_tensor_batch["environment_reset_key"] = np.asarray(
            [info.get("extra.gamefile", "") for info in infos], dtype=object
        )
        batch.non_tensor_batch["task_description"] = np.asarray(
            [self.envs._bace_slots[slot]["task"] for slot in slots], dtype=object
        )
        batch.non_tensor_batch["done"] = dones.astype(bool)
        batch.non_tensor_batch["remaining_horizon"] = np.asarray(
            [max(0, int(self.collector.config.env.max_steps) - job.step_index - 1) for job in jobs], dtype=np.int32
        )
        batch.non_tensor_batch["rewards"] = rewards.astype(object)
        batch.non_tensor_batch["active_masks"] = np.ones(size, dtype=object)
        for index, job in enumerate(jobs):
            row = {key: value[index] for key, value in batch.non_tensor_batch.items()}
            row.update({key: value[index] for key, value in batch.batch.items()})
            row["episode_rewards"] = float(job.episode_reward + rewards[index])
            row["episode_lengths"] = float(job.episode_length + 1)
            row["tool_callings"] = float(job.tool_callings)
            row["success_rate"] = float(row["episode_rewards"] > 0)
            job.rows.append(row)
            job.episode_reward += float(rewards[index])
            job.episode_length += 1
            job.step_index += 1
        finished = []
        for slot, job in zip(slots, jobs):
            terminal = bool(dones[slots.index(slot)]) or job.step_index >= int(self.collector.config.env.max_steps)
            if terminal:
                finished.append((slot, job, next_obs))
            else:
                self.jobs[slot] = job
        for slot, job, next_obs_all in finished:
            self.jobs.pop(slot, None)
            index = slots.index(slot)
            if job.kind == "root":
                self._finish_root(job)
            else:
                self._finish_branch(job, next_obs_all, job.episode_reward, bool(dones[index]))

    def _select_generation_slots(self):
        """Select a mixed MODEL_READY wave without synthetic candidates.

        With coalescing disabled this is exactly the legacy frontier behavior.
        The optional cap is a physical dispatch control for A/B experiments;
        deferred jobs stay in ``self.jobs`` and retain their task-local state.
        No extra root is created merely to fill a small tail batch.
        """
        slots = list(self.jobs)
        if not self.batch_coalescing_enabled:
            return slots
        if len(slots) < self.batch_coalescing_min_size:
            self.profile.coalescing_fallback_waves += 1
        cap = self.batch_coalescing_max_size
        if cap <= 0 or len(slots) <= cap:
            return slots

        roots = [slot for slot in slots if self.jobs[slot].kind == "root"]
        branches = [slot for slot in slots if self.jobs[slot].kind == "branch"]
        mixed = []
        while roots or branches:
            if roots:
                mixed.append(roots.pop(0))
            if branches:
                mixed.append(branches.pop(0))
        return mixed[:cap]

    def collect(self):
        self._initial_reset()
        for controller in self.tasks.values():
            for _ in range(controller.pilot_target):
                self._enqueue_root(controller)
        while not all(controller.complete for controller in self.tasks.values()):
            self.profile.root_queue_sizes.append(sum(job.kind == "root" for job in self.jobs.values()))
            self.profile.replay_queue_sizes.append(len(self._pending_replay))
            self._run_replay()
            if self.jobs:
                self._generate_frontier()
                continue
            self._run_replay()
            if not self.jobs and not self._pending_replay:
                incomplete = [task.task_id for task in self.tasks.values() if not task.complete]
                if incomplete:
                    raise FrontierInvariantError(
                        f"Frontier stalled with incomplete tasks: {incomplete}"
                    )
                break
        for controller in self.tasks.values():
            controller.assert_budget()
        return self._finalize()

    def _finalize(self):
        if not self.root_batches:
            raise FrontierInvariantError("Frontier collected no natural roots")
        root_output = self.collector._concat_batches(self.root_batches)
        self.collector._set_lineage(
            root_output, "root", [f"root:{value}" for value in root_output.non_tensor_batch["traj_uid"]]
        )
        batches = [root_output]
        occurrence_to_idx = {
            str(value): idx for idx, value in enumerate(root_output.non_tensor_batch["occurrence_id"])
        }
        for request, terminal_reward in self.branch_records:
            if request.origin_occurrence_id not in occurrence_to_idx:
                raise FrontierInvariantError(
                    f"Branch origin {request.origin_occurrence_id} is missing from natural roots"
                )
            origin_output = root_output.select_idxs([occurrence_to_idx[request.origin_occurrence_id]])
            origin_output.non_tensor_batch["traj_uid"] = np.asarray([request.branch_id], dtype=object)
            origin_output.non_tensor_batch["occurrence_id"] = np.asarray(
                [f"{request.branch_id}:origin"], dtype=object
            )
            origin_output.non_tensor_batch["episode_rewards"] = np.asarray(
                [terminal_reward], dtype=np.float32
            )
            origin_output.non_tensor_batch["rewards"] = np.asarray(
                [request.expected_immediate_reward], dtype=object
            )
            origin_output.non_tensor_batch["done"] = np.asarray(
                [request.expected_post_action_done], dtype=bool
            )
            self.collector._refresh_success_rate(origin_output)
            self.collector._set_lineage(origin_output, "branch_origin", [request.branch_id])
            batches.append(origin_output)
        batches.extend(self.branch_batches)
        merged = self.collector._concat_batches(batches)
        return merged, self.root_logs, self.tasks, self.logical_branches, self.profile.snapshot()
