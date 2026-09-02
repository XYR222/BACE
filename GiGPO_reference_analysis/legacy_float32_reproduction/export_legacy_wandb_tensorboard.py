#!/usr/bin/env python3
"""Export an offline W&B Reference Run into TensorBoard event files.

The legacy float32 reproductions intentionally run the frozen seed-0 source.
That runner records W&B in offline mode, so this utility reads the local
``run-*.wandb`` journal and writes the trainer metrics without contacting W&B.
It does not alter the source run or its raw archive.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


METRIC_PREFIXES = (
    "actor/",
    "critic/",
    "episode/",
    "global_seqlen/",
    "perf/",
    "prompt_length/",
    "response_length/",
    "timing/",
    "training/",
    "val/",
)


def _number(value: str) -> float | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
        return None
    parsed = float(parsed)
    return parsed if math.isfinite(parsed) else None


def _history_records(journal: Path):
    store = DataStore()
    store.open_for_scan(str(journal))
    while True:
        encoded = store.scan_data()
        if encoded is None:
            return
        record = wandb_internal_pb2.Record()
        record.ParseFromString(encoded)
        if record.WhichOneof("record_type") != "history":
            continue
        fields: dict[str, float] = {}
        for item in record.history.item:
            key = ".".join(item.nested_key)
            value = _number(item.value_json)
            if value is not None:
                fields[key] = value
        yield fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    journals = sorted(args.run_dir.glob("wandb/wandb/offline-run-*/run-*.wandb"))
    if len(journals) != 1:
        raise SystemExit(f"expected exactly one offline W&B journal in {args.run_dir}, got {journals}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    writer = SummaryWriter(str(args.output_dir))
    record_count = scalar_count = 0
    for fields in _history_records(journals[0]):
        # W&B's bookkeeping step is a fallback.  Trainer metrics always use
        # training/global_step, which is what the original TensorBoard used.
        step = int(fields.get("training/global_step", fields.get("_step", 0)))
        for name, value in fields.items():
            if name.startswith(METRIC_PREFIXES):
                writer.add_scalar(name, value, step)
                scalar_count += 1
        record_count += 1
    writer.flush()
    writer.close()
    (args.output_dir / "export_metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(args.run_dir),
                "source_wandb_journal": str(journals[0]),
                "record_count": record_count,
                "scalar_count": scalar_count,
                "step_policy": "training/global_step, falling back to W&B _step",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
