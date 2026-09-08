#!/usr/bin/env python3
"""Fail closed unless a one-step C0.5/C4/C7/C8 smoke exercised its path."""

import argparse
import json
from pathlib import Path

from recipe.bace_gigpo.validate_trace import validate_step


PHYSICAL_MODES = {
    "c0_5_origin_family_local_mean",
    "c4_macro_strict_ancestor",
    "c8_macro_local_strict_ancestor",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors = []
    reports = []
    step_dirs = sorted(args.artifact_root.glob("step_*"))
    if not step_dirs:
        errors.append("no trace step directory found")
    for step_dir in step_dirs:
        report = validate_step(step_dir)
        reports.append(report)
        errors.extend(report.get("errors", []))
        summary_path = step_dir / "summary.json"
        manifest_path = step_dir / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved = manifest.get("credit_mode") or manifest.get("tree_credit_mode")
        if resolved != args.mode:
            errors.append(f"{step_dir.name}: resolved credit mode {resolved!r} != {args.mode!r}")
        requested = int(summary.get("bace_metrics", {}).get("requested", 0))
        if requested < 1:
            errors.append(f"{step_dir.name}: smoke did not request a branch")
        counts = summary.get("diagnostics", {}).get("source_occurrence_counts", {})
        if args.mode in PHYSICAL_MODES:
            if int(counts.get("branch_origin", 0)) < 1:
                errors.append(f"{step_dir.name}: physical mode did not train copied origin")
            mode_check = report.get("checks", {}).get("physical_tree_credit", {})
            if mode_check.get("mode") != args.mode:
                errors.append(f"{step_dir.name}: physical-credit trace check missing")
        elif args.mode == "c7_flat_leaf_gigpo":
            flat_check = report.get("checks", {}).get("flat_leaf_credit", {})
            if int(flat_check.get("flat_trajectories", 0)) < 2:
                errors.append(f"{step_dir.name}: C7 did not produce root and branch leaves")
            if not any(str(source).startswith("flat_branch_") for source in counts):
                errors.append(f"{step_dir.name}: C7 flat branch rows missing")
        else:
            errors.append(f"unsupported smoke mode: {args.mode}")

    payload = {
        "ok": not errors,
        "mode": args.mode,
        "step_count": len(step_dirs),
        "errors": errors,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
