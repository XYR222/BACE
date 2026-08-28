#!/usr/bin/env python3
"""Export complete BACE training curves from a TensorBoard event file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASKS = {
    "pick_and_place": "Pick & place",
    "pick_clean_then_place_in_recep": "Clean & place",
    "pick_cool_then_place_in_recep": "Cool & place",
    "look_at_obj_in_light": "Look in light",
    "pick_heat_then_place_in_recep": "Heat & place",
    "pick_two_obj_and_place": "Pick two & place",
}


def load_scalars(event_file: Path) -> pd.DataFrame:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    columns: dict[str, pd.Series] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        columns[tag] = pd.Series(
            {int(event.step): float(event.value) for event in events}, dtype=float
        )
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "training_step"
    return frame


def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_raw_and_smooth(ax, frame, tags, labels=None, ylim=None, window=10):
    labels = labels or tags
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (tag, label) in enumerate(zip(tags, labels)):
        if tag not in frame:
            continue
        series = frame[tag].dropna()
        color = colors[i % len(colors)]
        ax.plot(series.index, series.values, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(
            series.index,
            smooth(series, window),
            color=color,
            linewidth=2.0,
            label=f"{label} ({window}-step mean)",
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def save_figure(fig, output: Path):
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def branch_position_table(artifact_root: Path, max_step: int):
    rows = []
    for path in sorted(artifact_root.glob("step_*/branches.jsonl")):
        training_step = int(path.parent.name.rsplit("_", 1)[1])
        if training_step > max_step:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                env_step = int(record["origin_occurrence_id"].rsplit(":", 1)[1]) + 1
                rows.append((training_step, env_step))
    return pd.DataFrame(rows, columns=["training_step", "branch_env_step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_scalars(args.event_file)
    frame.to_csv(args.output_dir / "all_training_scalars_1_150.csv")
    max_step = int(frame.index.max())

    validation_tags = ["val/success_rate", "val/text/test_score"] + [
        f"val/{task}_success_rate" for task in TASKS
    ]
    frame[[tag for tag in validation_tags if tag in frame]].dropna(
        how="all"
    ).to_csv(args.output_dir / "validation_metrics.csv")

    fig, left = plt.subplots(figsize=(11, 5.5))
    val = frame["val/success_rate"].dropna()
    left.plot(val.index, val.values, marker="o", markersize=3, linewidth=2,
              label="Validation success rate")
    left.set_ylim(0, 1.02)
    left.set_ylabel("Success rate")
    left.set_xlabel("Training step")
    left.grid(alpha=0.25)
    right = left.twinx()
    score = frame["val/text/test_score"].dropna()
    right.plot(score.index, score.values, color="tab:orange", marker="s",
               markersize=3, linewidth=1.8, label="Validation test score")
    right.set_ylabel("Test score")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="lower right")
    left.set_title("Validation performance (every 5 training steps)")
    save_figure(fig, args.output_dir / "01_validation_overall.png")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for task, label in TASKS.items():
        tag = f"val/{task}_success_rate"
        series = frame[tag].dropna()
        ax.plot(series.index, series.values, marker="o", markersize=2.5,
                linewidth=1.7, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation success rate")
    ax.set_title("Validation success rate by ALFWorld task family")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    save_figure(fig, args.output_dir / "02_validation_by_task.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_raw_and_smooth(
        ax,
        frame,
        ["bace/root_success_rate", "bace/branch_success_rate", "bace/mixed_success_rate"],
        ["Root", "Branch", "Mixed"],
        ylim=(0, 1.02),
    )
    ax.set_ylabel("Training-batch success rate")
    ax.set_title("BACE rollout success rates")
    save_figure(fig, args.output_dir / "03_training_success_rates.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_raw_and_smooth(axes[0, 0], frame, ["episode/reward/mean"], ["Reward"])
    axes[0, 0].set_ylabel("Mean reward")
    plot_raw_and_smooth(axes[0, 1], frame, ["episode/length/mean"], ["Episode length"])
    axes[0, 1].set_ylabel("Mean env steps")
    plot_raw_and_smooth(axes[1, 0], frame, ["episode/valid_action_ratio"], ["Valid action ratio"], ylim=(0, 1.02))
    axes[1, 0].set_ylabel("Ratio")
    plot_raw_and_smooth(axes[1, 1], frame, ["response_length/mean", "prompt_length/mean"], ["Response tokens", "Prompt tokens"])
    axes[1, 1].set_ylabel("Tokens")
    fig.suptitle("Episode, reward, and sequence dynamics")
    save_figure(fig, args.output_dir / "04_episode_dynamics.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_raw_and_smooth(axes[0, 0], frame, ["actor/pg_loss"], ["Policy-gradient loss"])
    plot_raw_and_smooth(axes[0, 1], frame, ["actor/entropy_loss"], ["Entropy"])
    plot_raw_and_smooth(axes[1, 0], frame, ["actor/kl_loss", "actor/ppo_kl"], ["Reference KL loss", "PPO approximate KL"])
    plot_raw_and_smooth(axes[1, 1], frame, ["actor/grad_norm", "actor/pg_clipfrac"], ["Gradient norm", "Clip fraction"])
    fig.suptitle("PPO optimization diagnostics")
    save_figure(fig, args.output_dir / "05_ppo_optimization.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_raw_and_smooth(axes[0, 0], frame,
                        ["bace/generated_roots_mean", "bace/final_roots_mean", "bace/planned_branches_mean", "bace/final_branches_mean"],
                        ["Generated roots", "Final roots", "Planned branches", "Final branches"])
    plot_raw_and_smooth(axes[0, 1], frame,
                        ["bace/effective_anchors_mean", "bace/information_capacity_mean"],
                        ["Effective anchors", "Information capacity"])
    plot_raw_and_smooth(axes[1, 0], frame,
                        ["bace/competence_readiness_mean", "bace/family_prior_mean"],
                        ["Competence readiness", "Family prior"], ylim=(0, 1.02))
    plot_raw_and_smooth(axes[1, 1], frame,
                        ["bace/requested", "bace/validated", "bace/skipped"],
                        ["Requested branches", "Validated branches", "Skipped branches"])
    fig.suptitle("BACE allocation and posterior dynamics")
    save_figure(fig, args.output_dir / "06_bace_allocation.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_raw_and_smooth(axes[0, 0], frame,
                        ["bace/replay_validation_seconds", "bace/branch_suffix_generation_seconds"],
                        ["Replay validation", "Branch suffix generation"])
    axes[0, 0].set_ylabel("Seconds")
    plot_raw_and_smooth(axes[0, 1], frame,
                        ["bace/branch_validation_replay_steps", "bace/branch_suffix_environment_steps"],
                        ["Replay env steps", "Suffix env steps"])
    plot_raw_and_smooth(axes[1, 0], frame,
                        ["bace/branch_execution_waves", "bace/branch_execution_max_wave_size"],
                        ["Execution waves", "Maximum wave size"])
    plot_raw_and_smooth(axes[1, 1], frame,
                        ["bace/branch_suffix_active_efficiency"], ["Active compaction efficiency"], ylim=(0, 1.02))
    fig.suptitle("Branch replay and selected-worker execution cost")
    save_figure(fig, args.output_dir / "07_branch_replay_cost.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_raw_and_smooth(axes[0, 0], frame, ["timing_s/step", "timing_s/gen"], ["Total step", "Generation"])
    axes[0, 0].set_ylabel("Seconds")
    plot_raw_and_smooth(axes[0, 1], frame, ["timing_s/update_actor", "timing_s/testing"], ["Actor update", "Validation"])
    axes[0, 1].set_ylabel("Seconds")
    plot_raw_and_smooth(axes[1, 0], frame, ["perf/throughput"], ["Throughput"])
    plot_raw_and_smooth(axes[1, 1], frame,
                        ["perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb", "perf/cpu_memory_used_gb"],
                        ["GPU allocated", "GPU reserved", "CPU used"])
    axes[1, 1].set_ylabel("GiB")
    fig.suptitle("Runtime and resource curves")
    save_figure(fig, args.output_dir / "08_performance_resources.png")

    branches = branch_position_table(args.artifact_root, max_step)
    branches.to_csv(args.output_dir / "branch_origin_env_steps.csv", index=False)
    train_bins = list(range(1, max_step + 1, 10))
    matrix = np.zeros((len(train_bins), 10), dtype=float)
    counts = np.zeros(len(train_bins), dtype=int)
    for i, start in enumerate(train_bins):
        subset = branches.loc[
            branches.training_step.between(start, min(start + 9, max_step)),
            "branch_env_step",
        ]
        counts[i] = len(subset)
        hist, _ = np.histogram(subset, bins=np.arange(1, 56, 5))
        if len(subset):
            matrix[i] = hist / len(subset)
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(train_bins)))
    ax.set_yticklabels([f"{s}-{min(s + 9, max_step)} (n={n})" for s, n in zip(train_bins, counts)])
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{s}-{s+4}" for s in range(1, 50, 5)])
    ax.set_xlabel("Branch origin env step")
    ax.set_ylabel("Training-step window")
    ax.set_title("Distribution of branch origin positions over training")
    fig.colorbar(image, ax=ax, label="Fraction within training window")
    save_figure(fig, args.output_dir / "09_branch_origin_heatmap.png")

    summary = {
        "event_file": str(args.event_file.resolve()),
        "training_steps": max_step,
        "scalar_tags": len(frame.columns),
        "validation_points": int(frame["val/success_rate"].count()),
        "final_validation_success_rate": float(frame["val/success_rate"].dropna().iloc[-1]),
        "best_validation_success_rate": float(frame["val/success_rate"].max()),
        "best_validation_step": int(frame["val/success_rate"].idxmax()),
        "final_reward_mean": float(frame["episode/reward/mean"].iloc[-1]),
        "final_episode_length_mean": float(frame["episode/length/mean"].iloc[-1]),
        "branch_records": int(len(branches)),
    }
    (args.output_dir / "curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
