#!/usr/bin/env python3
"""Export scalar metrics from a VeRL trainer log to a TensorBoard event file.

This is intentionally a one-way, provenance-preserving adapter for old runs
that wrote console metrics but did not retain their TensorBoard event files.
It only extracts ``step:<int> - key:value`` records; it never changes the
source log or claims that the result is a native event stream.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)$")
METRIC_RE = re.compile(r"\s+-\s+([^:]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_records(path: Path, wanted: set[str]) -> list[tuple[int, dict[str, float]]]:
    records: list[tuple[int, dict[str, float]]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        payload = " - " + match.group(2)
        values = {
            key.strip(): float(value)
            for key, value in METRIC_RE.findall(payload)
            if key.strip() in wanted
        }
        if values:
            records.append((step, values))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "val/success_rate",
            "val/webshop_task_score (not success_rate)",
            "episode/success_rate",
            "episode/webshop_task_score (not success_rate)",
            "episode/reward/mean",
            "timing_s/step",
        ],
    )
    args = parser.parse_args()

    records = parse_records(args.source_log, set(args.metrics))
    if not records:
        raise SystemExit(f"No requested step metrics found in {args.source_log}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = args.output_dir / "DERIVED_FROM_TRAINER_LOG.md"
    provenance.write_text(
        "# TensorBoard provenance\n\n"
        "These event scalars were extracted from the immutable VeRL console log. "
        "They are not native TensorBoard events.\n\n"
        f"- Source: `{args.source_log}`\n"
        f"- Extracted records: {len(records)}\n"
        f"- Metrics: {', '.join(args.metrics)}\n",
        encoding="utf-8",
    )

    writer = EventFileWriter(str(args.output_dir))
    wall_time = time.time()
    for index, (step, values) in enumerate(records):
        writer.add_event(
            event_pb2.Event(
                wall_time=wall_time + index,
                step=step,
                summary=summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag=key, simple_value=value) for key, value in values.items()]
                ),
            )
        )
    writer.flush()
    writer.close()
    print(f"exported_records={len(records)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
