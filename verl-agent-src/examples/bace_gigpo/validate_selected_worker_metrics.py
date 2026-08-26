#!/usr/bin/env python3
"""Require a completed run to exercise the selected-worker branch executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_run(artifact_root: Path) -> dict:
    candidates = []
    for summary_path in sorted(artifact_root.glob("step_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = summary_path.with_name("manifest.json")
        if not manifest_path.exists() or summary.get("status") != "complete":
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = summary.get("bace_metrics", {})
        if int(metrics.get("requested", 0)) <= 0:
            continue
        candidates.append((summary_path, manifest, summary))

    if not candidates:
        raise RuntimeError("no complete step exercised a branch request")

    checked = []
    for summary_path, manifest, summary in candidates:
        metrics = summary.get("bace_metrics", {})
        diagnostics = summary.get("diagnostics", {})
        if manifest.get("branch_execution_mode") != "selected_worker":
            raise RuntimeError(f"{summary_path}: selected_worker was not enabled")
        if float(metrics.get("branch_selected_worker_execution", 0)) != 1.0:
            raise RuntimeError(f"{summary_path}: selected-worker execution flag is absent")
        if int(metrics.get("branch_execution_restore_replay_steps", -1)) != 0:
            raise RuntimeError(f"{summary_path}: execution restore replay was not eliminated")
        active = int(metrics.get("branch_suffix_active_sequences", -1))
        dense = int(metrics.get("branch_suffix_dense_equivalent_sequences", -1))
        avoided = int(metrics.get("branch_suffix_inactive_sequences_avoided", -1))
        if min(active, dense, avoided) < 0 or active + avoided != dense:
            raise RuntimeError(f"{summary_path}: suffix accounting is inconsistent")
        mechanical = int(diagnostics.get("branch_total_mechanical_replay_steps", -1))
        origin = int(diagnostics.get("branch_origin_transition_steps", -1))
        total = int(diagnostics.get("branch_total_environment_steps", -1))
        if min(mechanical, origin, total) < 0 or mechanical + origin != total:
            raise RuntimeError(f"{summary_path}: replay accounting is inconsistent")
        checked.append({
            "step_dir": str(summary_path.parent),
            "requested": int(metrics["requested"]),
            "validated": int(metrics.get("validated", 0)),
            "mechanical_replay_steps": mechanical,
            "origin_transition_steps": origin,
            "active_suffix_sequences": active,
            "inactive_sequences_avoided": avoided,
        })
    return {"ok": True, "checked_steps": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_run(args.artifact_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
