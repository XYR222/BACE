from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "bace-trace-v1"


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _git_commit(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


class BaceArtifactStore:
    """Line-flushed, per-step CPU trace store for postmortem analysis."""

    def __init__(self, base_dir, step: int, manifest: dict[str, Any], fsync: bool = False):
        base_dir = Path(base_dir).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        step_dir = base_dir / f"step_{step:08d}"
        attempt = 0
        while step_dir.exists():
            attempt += 1
            step_dir = base_dir / f"step_{step:08d}_attempt_{attempt:02d}"
        step_dir.mkdir(parents=False)

        self.step = int(step)
        self.step_dir = step_dir
        self.fsync = bool(fsync)
        self._handles = {}
        self._counts = {}
        self._closed = False
        self.write_json(
            "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "step": self.step,
                "pid": os.getpid(),
                "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
                **manifest,
            },
        )

    def write_json(self, filename: str, payload: Any) -> None:
        target = self.step_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, target)

    def append(self, stream: str, payload: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed BACE artifact store")
        handle = self._handles.get(stream)
        if handle is None:
            handle = (self.step_dir / f"{stream}.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
            self._handles[stream] = handle
        record = {
            "schema_version": SCHEMA_VERSION,
            "step": self.step,
            "record_index": self._counts.get(stream, 0),
            **jsonable(payload),
        }
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        self._counts[stream] = record["record_index"] + 1

    def finalize(self, summary: dict[str, Any]) -> None:
        if self._closed:
            return
        self.write_json(
            "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "step": self.step,
                "record_counts": self._counts,
                **summary,
            },
        )
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._closed = True

    def close(self) -> None:
        self.finalize({"status": "closed_without_training_diagnostics"})

