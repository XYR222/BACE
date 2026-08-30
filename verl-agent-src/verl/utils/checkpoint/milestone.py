"""Atomic, space-efficient preservation of selected local checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path


def _inventory(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def preserve_checkpoint(source: str | Path, archive_root: str | Path, step: int) -> dict:
    """Preserve a completed checkpoint with hard links and an atomic rename.

    The active checkpoint root can continue rotating normally: removing its
    directory entries does not remove the inode data referenced by the
    milestone directory.  Source and archive must be on the same filesystem;
    refusing a cross-filesystem copy prevents an unexpected multi-GiB write.
    """
    source = Path(source).resolve()
    archive_root = Path(archive_root).resolve()
    expected_name = f"global_step_{step}"
    if source.name != expected_name or not source.is_dir():
        raise ValueError(f"invalid milestone checkpoint source: {source}")

    required = (source / "actor", source / "data.pt", source / "bace_collector_state.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"milestone checkpoint is incomplete: {missing}")

    archive_root.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != archive_root.stat().st_dev:
        raise OSError("milestone archive must be on the same filesystem as the active checkpoint")

    destination = archive_root / expected_name
    if destination.exists():
        marker = destination / ".complete"
        if not marker.is_file():
            raise FileExistsError(f"incomplete milestone destination exists: {destination}")
        return json.loads((destination / "preservation_manifest.json").read_text())

    source_inventory = _inventory(source)
    temporary = archive_root / f".{expected_name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=os.link, symlinks=True)
        destination_inventory = _inventory(temporary)
        if destination_inventory != source_inventory:
            raise RuntimeError("milestone hard-link inventory does not match source checkpoint")

        manifest = {
            "archive": str(destination),
            "bytes_referenced": sum(source_inventory.values()),
            "file_count": len(source_inventory),
            "method": "hardlink",
            "source": str(source),
            "step": step,
        }
        manifest_path = temporary / "preservation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        marker = temporary / ".complete"
        marker.touch()
        with marker.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
