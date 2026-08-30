#!/usr/bin/env python3
"""Convert the completed 2-GPU BACE console metrics into a TensorBoard run.

The 2-GPU reference-aligned invocation used the console/W&B logger, so it did
not produce a local event file.  This small, deterministic exporter preserves
the scalar names and global-step values emitted in the training log; it does
not rerun or modify training.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE = re.compile(r"step:(\d+) - (.*)")
METRIC = re.compile(
    r"(?:^| - )([A-Za-z0-9_./-]+):"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|-inf)"
)


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = STEP_LINE.search(ANSI.sub("", raw))
            if not match:
                continue
            step = int(match.group(1))
            metrics = values.setdefault(step, {})
            for name, raw_value in METRIC.findall(match.group(2)):
                value = float(raw_value)
                if math.isfinite(value):
                    metrics[name] = value
    return values


def export(values: dict[int, dict[str, float]], output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = EventFileWriter(str(output))
    try:
        for step in sorted(values):
            summary = summary_pb2.Summary()
            for name, value in sorted(values[step].items()):
                summary.value.add(tag=name, simple_value=value)
                count += 1
            writer.add_event(
                event_pb2.Event(wall_time=0.0, step=step, summary=summary)
            )
        writer.flush()
    finally:
        writer.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = parse_log(args.log)
    if not values:
        raise SystemExit(f"no step metrics found in {args.log}")
    count = export(values, args.output)
    print(f"exported {count} scalar values across {len(values)} steps to {args.output}")


if __name__ == "__main__":
    main()
