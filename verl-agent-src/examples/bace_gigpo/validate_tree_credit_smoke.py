#!/usr/bin/env python3
"""Fail closed unless a one-step C1/C2/C3 smoke exercised tree credit."""

import argparse
import json
from pathlib import Path

from recipe.bace_gigpo.validate_trace import validate_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    step_dirs = sorted(args.artifact_root.glob("step_*"))
    errors = []
    reports = []
    if not step_dirs:
        errors.append("no trace step directory found")
    for step_dir in step_dirs:
        report = validate_step(step_dir)
        reports.append(report)
        if not report["ok"]:
            errors.extend(report["errors"])
        summary_path = step_dir / "summary.json"
        manifest_path = step_dir / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tree_credit_mode") != args.mode:
            errors.append(f"{step_dir.name}: resolved tree-credit mode mismatch")
        metrics = summary.get("bace_metrics", {})
        diagnostics = summary.get("diagnostics", {})
        requested = int(metrics.get("requested", 0))
        if requested < 1:
            errors.append(f"{step_dir.name}: smoke did not request a branch")
        counts = diagnostics.get("source_occurrence_counts", {})
        if int(counts.get("branch_origin", 0)) != 0:
            errors.append(f"{step_dir.name}: copied branch origin reached PPO")
        if int(counts.get("branch_suffix", 0)) < 0:
            errors.append(f"{step_dir.name}: invalid branch-suffix count")
        tree = report.get("checks", {}).get("tree_credit", {})
        if tree.get("mode") != args.mode:
            errors.append(f"{step_dir.name}: trace tree-credit check is missing")

    payload = {
        "ok": not errors,
        "mode": args.mode,
        "step_count": len(step_dirs),
        "errors": errors,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
