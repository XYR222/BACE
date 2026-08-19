#!/usr/bin/env python3
"""Fail-fast CUDA/H100 and bundled-asset preflight for BACE."""

import argparse
import importlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = (
    "torch",
    "torchvision",
    "vllm",
    "flash-attn",
    "ray",
    "transformers",
    "tensordict",
    "datasets",
    "alfworld",
    "textworld",
    "gymnasium",
    "stable-baselines3",
    "numpy",
    "pyarrow",
)


def inspect_model_weights(model_path: Path):
    """Return the detected safetensors layout and any missing/invalid files."""
    single_file = model_path / "model.safetensors"
    if single_file.is_file() and single_file.stat().st_size > 0:
        return "single", [single_file.name], [], []

    index_file = model_path / "model.safetensors.index.json"
    if not index_file.is_file():
        return None, [], ["model.safetensors or model.safetensors.index.json"], []

    errors = []
    try:
        index = json.loads(index_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "sharded", [], [], [f"cannot read {index_file.name}: {exc}"]

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return "sharded", [], [], [f"{index_file.name} has no non-empty weight_map"]

    shard_values = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in shard_values):
        return "sharded", [], [], [f"{index_file.name} contains an invalid shard name"]
    shard_names = sorted(set(shard_values))

    missing = []
    for name in shard_names:
        shard_path = model_path / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append(f"unsafe shard path in {index_file.name}: {name}")
        elif not shard_path.is_file() or shard_path.stat().st_size == 0:
            missing.append(name)
    return "sharded", [index_file.name, *shard_names], missing, errors


def package_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    errors = []
    warnings = []
    report = {
        "ok": False,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": package_versions(),
    }

    for module in ("torch", "vllm", "flash_attn", "ray", "transformers", "pyarrow"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # Imports can fail on binary ABI mismatches.
            errors.append(f"cannot import {module}: {exc!r}")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        nccl_version = torch.cuda.nccl.version() if cuda_available else None
        if isinstance(nccl_version, tuple):
            nccl_version = list(nccl_version)
        report["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "cudnn_version": torch.backends.cudnn.version(),
            "nccl_version": nccl_version,
            "gpus": [],
        }
        for index in range(gpu_count):
            props = torch.cuda.get_device_properties(index)
            report["torch"]["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": props.total_memory,
                }
            )
            if "H100" not in props.name:
                warnings.append(f"GPU {index} is {props.name}, not an H100")
        if not cuda_available:
            errors.append("torch.cuda.is_available() is false")
        if args.expected_gpus is not None and gpu_count != args.expected_gpus:
            errors.append(f"expected {args.expected_gpus} visible GPUs, found {gpu_count}")
    except Exception as exc:
        errors.append(f"CUDA inspection failed: {exc!r}")

    required_model_files = ("config.json", "tokenizer_config.json")
    missing_model_files = [name for name in required_model_files if not (args.model_path / name).is_file()]
    weight_layout, weight_files, missing_weight_files, weight_errors = inspect_model_weights(args.model_path)
    missing_model_files.extend(missing_weight_files)
    report["model_path"] = str(args.model_path.resolve())
    report["model_weight_layout"] = weight_layout
    report["model_weight_files"] = weight_files
    report["missing_model_files"] = missing_model_files
    if missing_model_files:
        errors.append(f"model is incomplete: missing {missing_model_files}")
    errors.extend(weight_errors)

    required_alfworld_paths = (
        "logic/alfred.pddl",
        "logic/alfred.twl2",
        "json_2.1.1/train",
        "json_2.1.1/valid_seen",
        "json_2.1.1/valid_unseen",
    )
    missing_alfworld_paths = [name for name in required_alfworld_paths if not (args.alfworld_data / name).exists()]
    report["alfworld_data"] = str(args.alfworld_data.resolve())
    report["missing_alfworld_paths"] = missing_alfworld_paths
    if missing_alfworld_paths:
        errors.append(f"ALFWorld data is incomplete: missing {missing_alfworld_paths}")

    report["warnings"] = warnings
    report["errors"] = errors
    report["ok"] = not errors
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
