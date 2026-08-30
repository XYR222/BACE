#!/usr/bin/env python3
"""Validate the immutable two-rank BACE checkpoint used for a GiGPO fork."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, default=100)
    parser.add_argument("--deep", action="store_true", help="stream all PT files and verify ZIP CRCs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    expected_name = f"global_step_{args.expected_step}"
    errors: list[str] = []
    if checkpoint.name != expected_name or not checkpoint.is_dir():
        errors.append(f"checkpoint must be an existing {expected_name} directory")
    if not (checkpoint / ".complete").is_file():
        errors.append("missing immutable checkpoint completion marker")

    actor = checkpoint / "actor"
    required = [checkpoint / "data.pt", checkpoint / "bace_collector_state.json"]
    for rank in range(2):
        required.extend(
            actor / f"{component}_world_size_2_rank_{rank}.pt"
            for component in ("model", "optim", "extra_state")
        )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        errors.append(f"missing or empty required files: {missing}")

    collector_step = None
    dataloader_readable = False
    extra_states_readable = False
    crc_verified = False
    if not errors:
        try:
            collector = json.loads((checkpoint / "bace_collector_state.json").read_text())
            collector_step = int(collector["global_step"])
            nested_step = int(collector["collector"]["current_step"])
            if collector_step != args.expected_step or nested_step != args.expected_step:
                errors.append(
                    f"collector step mismatch: wrapper={collector_step}, collector={nested_step}"
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid BACE collector state: {exc}")

        try:
            state = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
            dataloader_readable = isinstance(state, dict)
            if not dataloader_readable:
                errors.append("dataloader state is not a dictionary")
        except Exception as exc:  # torch emits several format-specific exception classes
            errors.append(f"unreadable dataloader state: {exc}")

        try:
            for rank in range(2):
                state = torch.load(
                    actor / f"extra_state_world_size_2_rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if "lr_scheduler" not in state or "rng" not in state:
                    raise ValueError(f"rank {rank} extra state lacks scheduler or RNG")
            extra_states_readable = True
        except Exception as exc:
            errors.append(f"unreadable actor extra state: {exc}")

        if args.deep:
            try:
                for path in required:
                    if path.suffix != ".pt":
                        continue
                    if not zipfile.is_zipfile(path):
                        raise ValueError(f"not a PyTorch ZIP archive: {path}")
                    with zipfile.ZipFile(path) as archive:
                        bad_member = archive.testzip()
                    if bad_member is not None:
                        raise ValueError(f"CRC failure in {path}: {bad_member}")
                crc_verified = True
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                errors.append(f"deep checkpoint validation failed: {exc}")

    file_sizes = {
        str(path.relative_to(checkpoint)): path.stat().st_size
        for path in required
        if path.is_file()
    }
    report = {
        "checkpoint": str(checkpoint),
        "collector_step": collector_step,
        "crc_verified": crc_verified,
        "dataloader_readable": dataloader_readable,
        "errors": errors,
        "expected_step": args.expected_step,
        "extra_states_readable": extra_states_readable,
        "file_count": len(file_sizes),
        "files": file_sizes,
        "ok": not errors,
        "total_bytes": sum(file_sizes.values()),
        "world_size": 2,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        temporary.write_text(rendered)
        os.replace(temporary, args.output)
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
