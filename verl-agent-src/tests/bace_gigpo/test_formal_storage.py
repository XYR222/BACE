import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "bace_gigpo"
    / "check_formal_storage.py"
)
SPEC = importlib.util.spec_from_file_location("bace_formal_storage", SCRIPT)
formal_storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal_storage)


def test_accepts_symlink_into_expected_storage(tmp_path, monkeypatch):
    project = tmp_path / "project"
    target = project / "experiment"
    target.mkdir(parents=True)
    link = tmp_path / "visible-experiment"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        formal_storage.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=200, used=100, free=100),
    )

    report = formal_storage.check_storage(link, project, 64)

    assert report["ok"] is True
    assert report["resolved_output_root"] == str(target.resolve())


def test_rejects_output_outside_expected_storage(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "personal" / "experiment"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside"):
        formal_storage.check_storage(outside, project, 1)


def test_rejects_insufficient_free_space(tmp_path, monkeypatch):
    project = tmp_path / "project"
    target = project / "experiment"
    target.mkdir(parents=True)
    monkeypatch.setattr(
        formal_storage.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=200, used=150, free=50),
    )

    with pytest.raises(ValueError, match="at least 64 required"):
        formal_storage.check_storage(target, project, 64)
