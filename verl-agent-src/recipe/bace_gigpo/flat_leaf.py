"""C7 leaf expansion for BACE-acquired rollout trees."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from verl import DataProto


def _ordered_rows(rows, step_indices):
    return sorted(rows, key=lambda row: (int(step_indices[row]), row))


def expand_tree_to_full_leaf_trajectories(data: DataProto) -> DataProto:
    """Expand every root/branch terminal leaf into one complete trajectory.

    Root leaves retain one natural path.  A branch leaf receives copied strict
    natural prefix rows, its copied origin row, and its freshly generated
    suffix.  The returned data therefore satisfies the ordinary production
    GiGPO batch contract and can have returns recomputed without tree logic.
    """
    required = {
        "source_type",
        "occurrence_id",
        "traj_uid",
        "leaf_id",
        "step_index",
        "tree_origin_occurrence_id",
        "tree_parent_root_id",
        "episode_rewards",
    }
    missing = sorted(required - set(data.non_tensor_batch))
    if missing:
        raise ValueError(f"C7 flat-leaf expansion is missing metadata: {missing}")

    sources = np.asarray(data.non_tensor_batch["source_type"], dtype=object)
    occurrences = np.asarray(data.non_tensor_batch["occurrence_id"], dtype=object)
    trajectories = np.asarray(data.non_tensor_batch["traj_uid"], dtype=object)
    steps = np.asarray(data.non_tensor_batch["step_index"])
    origins = np.asarray(
        data.non_tensor_batch["tree_origin_occurrence_id"], dtype=object
    )
    parents = np.asarray(data.non_tensor_batch["tree_parent_root_id"], dtype=object)
    episode_rewards = np.asarray(data.non_tensor_batch["episode_rewards"])

    root_rows: dict[str, list[int]] = defaultdict(list)
    branch_origin_row: dict[str, int] = {}
    branch_suffix_rows: dict[str, list[int]] = defaultdict(list)
    natural_row_by_occurrence: dict[str, int] = {}
    for row, source_value in enumerate(sources):
        source = str(source_value)
        trajectory = str(trajectories[row])
        if source == "root":
            root_rows[trajectory].append(row)
            occurrence = str(occurrences[row])
            if occurrence in natural_row_by_occurrence:
                raise ValueError(f"Duplicate natural occurrence in C7 input: {occurrence}")
            natural_row_by_occurrence[occurrence] = row
        elif source == "branch_origin":
            if trajectory in branch_origin_row:
                raise ValueError(f"Branch {trajectory} has multiple copied origins")
            branch_origin_row[trajectory] = row
        elif source == "branch_suffix":
            branch_suffix_rows[trajectory].append(row)
        else:
            raise ValueError(f"C7 cannot flatten source_type={source}")

    orphan_suffixes = sorted(set(branch_suffix_rows) - set(branch_origin_row))
    if orphan_suffixes:
        raise ValueError(
            f"C7 branch suffixes have no copied origin: {orphan_suffixes}"
        )

    pieces: list[DataProto] = []
    flat_uids: set[str] = set()

    def append_piece(
        row_ids: list[int], *, flat_uid: str, root_id: str,
        branch_id: str, prefix_length: int, terminal_reward: float,
        flat_sources: list[str], prefix_flags: list[bool],
    ) -> None:
        piece = data.select_idxs(np.asarray(row_ids, dtype=np.int64))
        size = len(piece)
        if size == 0:
            raise ValueError(f"C7 produced an empty flat trajectory {flat_uid}")
        if flat_uid in flat_uids:
            raise ValueError(f"C7 produced duplicate flat trajectory ID {flat_uid}")
        flat_uids.add(flat_uid)
        source_occurrences = np.asarray(
            piece.non_tensor_batch["occurrence_id"], dtype=object
        ).copy()
        piece.non_tensor_batch["traj_uid"] = np.asarray(
            [flat_uid] * size, dtype=object
        )
        piece.non_tensor_batch["occurrence_id"] = np.asarray(
            [f"{flat_uid}:step:{index}" for index in range(size)], dtype=object
        )
        piece.non_tensor_batch["leaf_id"] = np.asarray(
            [flat_uid] * size, dtype=object
        )
        piece.non_tensor_batch["episode_rewards"] = np.asarray(
            [terminal_reward] * size, dtype=np.float32
        )
        if "episode_lengths" in piece.non_tensor_batch:
            piece.non_tensor_batch["episode_lengths"] = np.asarray(
                [size] * size, dtype=np.float32
            )
        if "success_rate" in piece.non_tensor_batch:
            piece.non_tensor_batch["success_rate"] = np.asarray(
                [float(terminal_reward > 0)] * size, dtype=np.float32
            )
        piece.non_tensor_batch["source_type"] = np.asarray(
            flat_sources, dtype=object
        )
        piece.non_tensor_batch["tree_parent_root_id"] = np.asarray(
            [root_id] * size, dtype=object
        )
        piece.non_tensor_batch["bace_credit_mode"] = np.asarray(
            ["c7_flat_leaf_gigpo"] * size, dtype=object
        )
        piece.non_tensor_batch["bace_flat_leaf_id"] = np.asarray(
            [flat_uid] * size, dtype=object
        )
        piece.non_tensor_batch["bace_flat_traj_uid"] = np.asarray(
            [flat_uid] * size, dtype=object
        )
        piece.non_tensor_batch["bace_flat_source_root_id"] = np.asarray(
            [root_id] * size, dtype=object
        )
        piece.non_tensor_batch["bace_flat_source_branch_id"] = np.asarray(
            [branch_id] * size, dtype=object
        )
        piece.non_tensor_batch["bace_flat_copied_prefix_length"] = np.asarray(
            [prefix_length] * size, dtype=np.int32
        )
        piece.non_tensor_batch["bace_flat_is_prefix_copy"] = np.asarray(
            prefix_flags, dtype=bool
        )
        piece.non_tensor_batch["bace_flat_source_occurrence_id"] = source_occurrences
        piece.non_tensor_batch["bace_flat_full_leaf_terminal_reward"] = np.asarray(
            [terminal_reward] * size, dtype=np.float32
        )
        pieces.append(piece)

    for root_id in sorted(root_rows):
        rows = _ordered_rows(root_rows[root_id], steps)
        if [int(steps[row]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Natural root {root_id} has a non-contiguous step sequence")
        root_leaf = str(data.non_tensor_batch["leaf_id"][rows[0]])
        if any(str(data.non_tensor_batch["leaf_id"][row]) != root_leaf for row in rows):
            raise ValueError(f"Natural root {root_id} has inconsistent terminal leaf")
        terminal_reward = float(episode_rewards[rows[-1]])
        if any(float(episode_rewards[row]) != terminal_reward for row in rows):
            raise ValueError(f"Natural root {root_id} has inconsistent terminal reward")
        append_piece(
            rows,
            flat_uid=f"flat-root:{root_leaf}",
            root_id=root_id,
            branch_id="",
            prefix_length=0,
            terminal_reward=terminal_reward,
            flat_sources=["flat_root"] * len(rows),
            prefix_flags=[False] * len(rows),
        )

    for branch_id in sorted(branch_origin_row):
        origin_row = branch_origin_row[branch_id]
        natural_origin_id = str(origins[origin_row])
        if natural_origin_id not in natural_row_by_occurrence:
            raise ValueError(
                f"C7 branch {branch_id} references missing origin {natural_origin_id}"
            )
        natural_origin_row = natural_row_by_occurrence[natural_origin_id]
        root_id = str(parents[origin_row])
        if root_id not in root_rows:
            raise ValueError(f"C7 branch {branch_id} references missing root {root_id}")
        if str(trajectories[natural_origin_row]) != root_id:
            raise ValueError(
                f"C7 branch {branch_id} origin does not belong to parent root {root_id}"
            )
        origin_step = int(steps[natural_origin_row])
        if int(steps[origin_row]) != origin_step:
            raise ValueError(f"C7 branch {branch_id} copied-origin step mismatch")
        prefix = [
            row for row in _ordered_rows(root_rows[root_id], steps)
            if int(steps[row]) < origin_step
        ]
        suffix = _ordered_rows(branch_suffix_rows.get(branch_id, []), steps)
        rows = prefix + [origin_row] + suffix
        if [int(steps[row]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"C7 branch {branch_id} has a non-contiguous full path")
        branch_leaf = str(data.non_tensor_batch["leaf_id"][origin_row])
        if any(str(data.non_tensor_batch["leaf_id"][row]) != branch_leaf for row in [origin_row] + suffix):
            raise ValueError(f"Branch {branch_id} has inconsistent terminal leaf")
        terminal_reward = float(episode_rewards[origin_row])
        if any(float(episode_rewards[row]) != terminal_reward for row in [origin_row] + suffix):
            raise ValueError(f"Branch {branch_id} has inconsistent terminal reward")
        append_piece(
            rows,
            flat_uid=f"flat-branch:{branch_leaf}",
            root_id=root_id,
            branch_id=branch_id,
            prefix_length=len(prefix) + 1,
            terminal_reward=terminal_reward,
            flat_sources=(
                ["flat_branch_prefix"] * len(prefix)
                + ["flat_branch_origin"]
                + ["flat_branch_suffix"] * len(suffix)
            ),
            prefix_flags=[True] * (len(prefix) + 1) + [False] * len(suffix),
        )

    if not pieces:
        raise ValueError("C7 flat-leaf expansion produced no terminal trajectories")
    result = DataProto.concat(pieces)
    result.meta_info["flat_leaf_diagnostics"] = {
        "flat_leaf_input_rows": float(len(data)),
        "flat_leaf_output_rows": float(len(result)),
        "flat_leaf_root_trajectories": float(len(root_rows)),
        "flat_leaf_branch_trajectories": float(len(branch_origin_row)),
        "flat_leaf_trajectories": float(len(pieces)),
        "flat_leaf_copied_prefix_rows": float(np.sum(
            result.non_tensor_batch["bace_flat_is_prefix_copy"]
        )),
    }
    return result
