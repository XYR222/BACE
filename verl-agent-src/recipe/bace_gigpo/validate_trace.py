from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import SCHEMA_VERSION


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _close_sequence(left, right, tolerance=1e-7):
    if left is None or right is None or len(left) != len(right):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right))


def _float32_advantage_sum(macro: float, local: float, weight: float) -> tuple[float, float]:
    """Reconstruct the two float32 tensor ops used by advantage.py.

    Artifact components are serialized after each tensor has already rounded to
    float32.  Re-adding those values in Python float64 can differ from the
    recorded float32 combined value by one or two ULPs.  Compare against the
    actual training dtype and allow two ULPs, while retaining a small absolute
    floor around zero.
    """
    macro32 = np.float32(macro)
    local32 = np.float32(local)
    weight32 = np.float32(weight)
    expected32 = np.float32(macro32 + np.float32(weight32 * local32))
    ulp = abs(float(np.spacing(expected32)))
    return float(expected32), max(1e-6, 2.0 * ulp)


def validate_step(step_dir: Path, max_recomputed_logprob_diff=None,
                  max_recomputed_probability_diff=None):
    step_dir = Path(step_dir)
    errors = []
    warnings = []
    checks = {}

    required = ["manifest.json", "roots.jsonl", "leaves.jsonl", "topology.jsonl", "summary.json"]
    missing = [name for name in required if not (step_dir / name).exists()]
    if missing:
        return {"step_dir": str(step_dir), "ok": False,
                "errors": [f"missing required files: {missing}"], "warnings": [], "checks": {}}

    manifest = _json(step_dir / "manifest.json")
    summary = _json(step_dir / "summary.json")
    streams = {
        path.stem: _jsonl(path)
        for path in step_dir.glob("*.jsonl")
    }
    step = int(summary.get("step", -1))
    if manifest.get("schema_version") != SCHEMA_VERSION or summary.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest/summary schema version mismatch")
    if int(manifest.get("step", -2)) != step:
        errors.append("manifest and summary step mismatch")

    for stream, records in streams.items():
        indices = [record.get("record_index") for record in records]
        if indices != list(range(len(records))):
            errors.append(f"{stream}: non-contiguous record_index")
        if any(record.get("schema_version") != SCHEMA_VERSION for record in records):
            errors.append(f"{stream}: schema version mismatch")
        if any(int(record.get("step", -1)) != step for record in records):
            errors.append(f"{stream}: step mismatch")
        expected_count = summary.get("record_counts", {}).get(stream)
        if expected_count is not None and int(expected_count) != len(records):
            errors.append(f"{stream}: summary count {expected_count} != {len(records)}")
    checks["record_counts"] = {name: len(records) for name, records in streams.items()}

    roots = streams.get("roots", [])
    leaves = streams.get("leaves", [])
    leaf_by_id = {}
    root_ids = set()
    for root in roots:
        root_id = str(root["root_id"])
        if root_id in root_ids:
            errors.append(f"duplicate root_id: {root_id}")
        root_ids.add(root_id)
    for leaf in leaves:
        occurrence_id = str(leaf["occurrence_id"])
        if occurrence_id in leaf_by_id:
            errors.append(f"duplicate natural occurrence_id: {occurrence_id}")
        leaf_by_id[occurrence_id] = leaf
        identity = leaf.get("action_identity")
        format_valid = leaf.get("action_format_valid")
        environment_valid = leaf.get("action_environment_valid")
        if not format_valid and identity is not None:
            errors.append(f"unparsed occurrence has action identity: {occurrence_id}")
        if format_valid and environment_valid is True and not str(identity).startswith("valid::"):
            errors.append(f"environment-valid occurrence lacks valid identity: {occurrence_id}")
        if format_valid and environment_valid is False and not str(identity).startswith("invalid::"):
            errors.append(f"environment-invalid occurrence lacks invalid identity: {occurrence_id}")
    checks["natural_roots"] = len(root_ids)
    checks["natural_occurrences"] = len(leaf_by_id)

    topology_records = streams.get("topology", [])
    for topology in topology_records:
        plan = topology.get("plan") or {}
        tasks = plan.get("tasks", {})
        budget = manifest.get("total_leaf_budget")
        for task_id, task in tasks.items():
            roots_count = int(task["final_root_count"])
            branches_count = int(task["final_branch_count"])
            if budget is not None and roots_count + branches_count != int(budget):
                errors.append(f"task {task_id}: R + Q != B")
            max_per_anchor = manifest.get("max_branches_per_anchor")
            if max_per_anchor is not None and branches_count > int(
                task.get("effective_anchor_count", 0)
            ) * int(max_per_anchor):
                errors.append(f"task {task_id}: branch count exceeds anchor capacity")
        selected_root_ids = set(topology.get("root_ids", []))
        if not selected_root_ids.issubset(root_ids):
            errors.append("topology references unknown root ids")

    replay = streams.get("replay_attempts", [])
    request_ids = {record["request"]["request_id"] for record in replay}
    validated_transition_requests = {
        record["request"]["request_id"]
        for record in replay
        if "transition_validation" in record["phase"] and record["result"]["replay_ok"]
    }
    if manifest.get("branch_execution_mode") == "selected_worker":
        metrics = summary.get("bace_metrics", {})
        diagnostics = summary.get("diagnostics", {})
        requested = int(metrics.get("requested", 0))
        if requested:
            restore_steps = int(metrics.get("branch_execution_restore_replay_steps", -1))
            if restore_steps != 0:
                errors.append(
                    "selected-worker execution performed an execution-restore replay"
                )
            active = int(metrics.get("branch_suffix_active_sequences", -1))
            dense = int(metrics.get("branch_suffix_dense_equivalent_sequences", -1))
            avoided = int(metrics.get("branch_suffix_inactive_sequences_avoided", -1))
            suffix_steps = int(metrics.get("branch_suffix_environment_steps", -1))
            if min(active, dense, avoided, suffix_steps) < 0:
                errors.append("selected-worker suffix accounting is incomplete")
            elif active + avoided != dense:
                errors.append("selected-worker suffix accounting does not conserve sequences")
            elif suffix_steps != active:
                errors.append("selected-worker suffix environment steps differ from active sequences")
            mechanical = int(diagnostics.get("branch_total_mechanical_replay_steps", -1))
            origin = int(diagnostics.get("branch_origin_transition_steps", -1))
            total = int(diagnostics.get("branch_total_environment_steps", -1))
            compatibility_total = int(diagnostics.get("replay_environment_steps", -1))
            if min(mechanical, origin, total, compatibility_total) < 0:
                errors.append("selected-worker replay accounting is incomplete")
            elif mechanical + origin != total or compatibility_total != total:
                errors.append("selected-worker replay accounting does not conserve environment steps")
            checks["selected_worker_execution"] = {
                "execution_restore_replay_steps": restore_steps,
                "active_suffix_sequences": active,
                "dense_equivalent_suffix_sequences": dense,
                "inactive_sequences_avoided": avoided,
                "mechanical_replay_steps": mechanical,
                "origin_transition_steps": origin,
                "total_branch_environment_steps": total,
            }
    branches = streams.get("branches", [])
    successful_branches = {}
    failed_branch_ids = set()
    for branch in branches:
        branch_id = str(branch["branch_id"])
        if "terminal_reward" in branch:
            successful_branches[branch_id] = branch
            if branch.get("request_id") not in request_ids:
                errors.append(f"branch {branch_id}: unknown request_id")
            if branch.get("request_id") not in validated_transition_requests:
                errors.append(f"branch {branch_id}: missing validated origin transition")
            if branch.get("origin_occurrence_id") not in leaf_by_id:
                errors.append(f"branch {branch_id}: unknown natural origin")
            suffix_ids = branch.get("suffix_occurrence_ids", [])
            if int(branch.get("suffix_occurrence_count", 0)) != len(suffix_ids):
                errors.append(f"branch {branch_id}: suffix count mismatch")
        else:
            failed_branch_ids.add(branch_id)
    checks["successful_branches"] = len(successful_branches)
    checks["failed_branches"] = len(failed_branch_ids)

    occurrences = streams.get("trainable_occurrences", [])
    by_source = defaultdict(list)
    by_leaf = defaultdict(list)
    for occurrence in occurrences:
        by_source[str(occurrence.get("source_type"))].append(occurrence)
        by_leaf[str(occurrence.get("leaf_id"))].append(occurrence)
    trained_branch_ids = {
        str(record.get("traj_uid")) for record in occurrences
        if record.get("source_type") in {"branch_origin", "branch_suffix"}
    }
    unknown_trained = trained_branch_ids.difference(successful_branches)
    if unknown_trained:
        errors.append(f"training batch contains failed/unknown branches: {sorted(unknown_trained)}")

    advantage_semantics = manifest.get("advantage_semantics", "leaf_uniform")
    if advantage_semantics == "leaf_uniform":
        for leaf_id, records in by_leaf.items():
            values = [float(record["leaf_advantage"]) for record in records if record.get("leaf_advantage") is not None]
            if values and any(not math.isclose(value, values[0], abs_tol=1e-6) for value in values[1:]):
                errors.append(f"leaf {leaf_id}: non-uniform leaf advantage")
    elif advantage_semantics == "gigpo_macro":
        step_advantage_w = float(manifest.get("gigpo_step_advantage_w", 1.0))
        for occurrence in occurrences:
            values = (
                occurrence.get("macro_advantage"),
                occurrence.get("local_advantage"),
                occurrence.get("occurrence_advantage"),
            )
            if any(value is None or not math.isfinite(float(value)) for value in values):
                errors.append(
                    f"occurrence {occurrence.get('occurrence_id')}: missing or non-finite advantage component"
                )
                continue
            macro, local, combined = map(float, values)
            expected, tolerance = _float32_advantage_sum(macro, local, step_advantage_w)
            if not math.isclose(combined, expected, rel_tol=0.0, abs_tol=tolerance):
                errors.append(
                    f"occurrence {occurrence.get('occurrence_id')}: inconsistent advantage components"
                )
    else:
        errors.append(f"unknown advantage semantics: {advantage_semantics}")

    copied_origin_checks = 0
    for occurrence in by_source.get("branch_origin", []):
        branch_id = str(occurrence.get("traj_uid"))
        branch = successful_branches.get(branch_id)
        if branch is None:
            continue
        natural = leaf_by_id.get(branch["origin_occurrence_id"])
        if natural is None:
            continue
        arrays = ("response_token_ids", "response_loss_mask", "rollout_old_log_probs")
        if any(occurrence.get(name) is None for name in arrays):
            warnings.append(f"branch {branch_id}: token arrays absent; copied-origin audit skipped")
            continue
        copied_origin_checks += 1
        if occurrence["response_token_ids"] != natural.get("response_token_ids"):
            errors.append(f"branch {branch_id}: copied response token ids differ from natural origin")
        if occurrence["response_loss_mask"] != natural.get("response_loss_mask"):
            errors.append(f"branch {branch_id}: copied response loss mask differs from natural origin")
        if not _close_sequence(occurrence["rollout_old_log_probs"], natural.get("old_log_probs")):
            errors.append(f"branch {branch_id}: copied rollout old-log-probs differ from natural origin")
        if occurrence.get("action_identity") != natural.get("action_identity"):
            errors.append(f"branch {branch_id}: copied action identity differs from natural origin")
    checks["copied_branch_origins_audited"] = copied_origin_checks

    acquisition = streams.get("acquisition_rounds", [])
    support = None
    for round_record in acquisition:
        for diagnostic in round_record.get("diagnostics", []):
            global_allocation = diagnostic.get("global_allocation")
            if global_allocation is None:
                continue
            task_id = diagnostic.get("task_id", "<unknown>")
            solver = global_allocation.get("solver")
            if solver != "quota_aware_exact_dp":
                errors.append(
                    f"task {task_id}: unexpected global allocation solver {solver!r}"
                )
            selected = global_allocation.get("selected_allocation")
            quota = global_allocation.get("branch_quota")
            if not isinstance(selected, dict) or quota is None:
                errors.append(
                    f"task {task_id}: incomplete global allocation diagnostics"
                )
                continue
            try:
                selected_total = sum(int(value) for value in selected.values())
                expected_quota = int(quota)
            except (TypeError, ValueError):
                errors.append(f"task {task_id}: invalid global allocation quota values")
                continue
            if selected_total != expected_quota:
                errors.append(
                    f"task {task_id}: selected global allocation uses "
                    f"{selected_total} branches, expected {expected_quota}"
                )
            if selected != diagnostic.get("selected_allocation"):
                errors.append(
                    f"task {task_id}: selected global allocation diagnostics disagree"
                )
            tie_count = global_allocation.get("optimal_tie_count")
            if not isinstance(tie_count, int) or tie_count < 1:
                errors.append(f"task {task_id}: invalid global optimal tie count")
            reachable = global_allocation.get("reachable_state_count")
            if not isinstance(reachable, int) or reachable < 1:
                errors.append(f"task {task_id}: invalid reachable DP state count")
            wall_time = global_allocation.get("solver_wall_time_ms")
            if not isinstance(wall_time, (int, float)) or not math.isfinite(wall_time):
                errors.append(f"task {task_id}: invalid global allocator wall time")
        snapshot = round_record.get("posterior_snapshot", {})
        current_support = {
            (task_id, anchor_id, action_id)
            for task_id, anchors in snapshot.items()
            for anchor_id, actions in anchors.items()
            for action_id in actions
        }
        if support is None:
            support = current_support
        elif current_support != support:
            errors.append(f"round {round_record.get('round')}: posterior support changed")
    checks["frozen_support_size"] = len(support or ())

    diagnostics = summary.get("diagnostics", {})
    source_stats = diagnostics.get("old_log_prob_by_source", {})
    checks["old_log_prob_by_source"] = source_stats
    for source, stats in source_stats.items():
        if max_recomputed_logprob_diff is not None and stats.get("max_abs_diff", 0) > max_recomputed_logprob_diff:
            errors.append(f"{source}: recomputed old-log-prob diff exceeds threshold")
        if max_recomputed_probability_diff is not None and stats.get("probability_max_abs_diff", 0) > max_recomputed_probability_diff:
            errors.append(f"{source}: recomputed probability diff exceeds threshold")

    return {
        "step_dir": str(step_dir.resolve()),
        "step": step,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def find_step_dirs(path: Path):
    path = Path(path)
    if (path / "summary.json").exists():
        return [path]
    return sorted({summary.parent for summary in path.glob("step_*/summary.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate BACE per-step experiment traces")
    parser.add_argument("path", type=Path, help="A step directory or bace_trace directory")
    parser.add_argument("--max-recomputed-logprob-diff", type=float)
    parser.add_argument("--max-recomputed-probability-diff", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    step_dirs = find_step_dirs(args.path)
    if not step_dirs:
        parser.error(f"no complete step traces found under {args.path}")
    results = [
        validate_step(
            step_dir,
            max_recomputed_logprob_diff=args.max_recomputed_logprob_diff,
            max_recomputed_probability_diff=args.max_recomputed_probability_diff,
        )
        for step_dir in step_dirs
    ]
    report = {"ok": all(result["ok"] for result in results), "steps": results}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
