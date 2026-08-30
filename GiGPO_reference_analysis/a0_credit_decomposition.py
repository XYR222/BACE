#!/usr/bin/env python3
"""Offline A0 credit decomposition for the completed 2-GPU runs.

This intentionally reads immutable run artifacts only.  GiGPO advantages are
reconstructed from physical occurrences; BACE uses the per-occurrence
macro/local components emitted by its trainer and checks their sum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EPS = 1e-8


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _token_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    tab = pq.read_table(path, columns=["occurrence_id", "response_mask"])
    ids = tab.column("occurrence_id").to_pylist()
    masks = tab.column("response_mask").to_pylist()
    return {str(i): int(sum(int(x) for x in m)) for i, m in zip(ids, masks)}


def _std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def _norm(values: np.ndarray) -> np.ndarray:
    sd = _std(values)
    if sd <= 1e-12 or not np.isfinite(sd):
        return np.zeros(values.size, dtype=float)
    return (values - values.mean()) / (sd + 1e-6)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # Average ties, matching the usual Spearman convention sufficiently for
    # this diagnostic (the exact ties are mostly zero-variance groups).
    for val in np.unique(values):
        idx = np.flatnonzero(values == val)
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def _obs_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _corr(macro: np.ndarray, local: np.ndarray) -> tuple[float, float]:
    mask = (np.abs(macro) > EPS) & (np.abs(local) > EPS)
    if int(mask.sum()) < 2:
        return 0.0, 0.0
    x, y = macro[mask], local[mask]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rank(x), _rank(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0
    return pearson, spearman


def _metrics(df: pd.DataFrame, run: str, step: int, source_filter: str | None = None) -> dict[str, Any]:
    if source_filter is not None:
        df = df[df.source_type == source_filter]
    n = len(df)
    if n == 0:
        return {"run": run, "step": step, "source_filter": source_filter or "all", "n_occurrences": 0}
    macro = df.macro_advantage.to_numpy(float)
    local = df.local_advantage.to_numpy(float)
    tokens = df.response_token_count.to_numpy(float)
    nonzero = np.abs(local) > EPS
    macro_mass = float(np.abs(macro).sum())
    local_mass = float(np.abs(local).sum())
    token_macro = float((tokens * np.abs(macro)).sum())
    token_local = float((tokens * np.abs(local)).sum())
    simultaneous = (np.abs(macro) > EPS) & nonzero
    agree = float(np.mean(np.sign(macro[simultaneous]) == np.sign(local[simultaneous]))) if simultaneous.any() else 0.0
    conflict = float(np.mean((macro[simultaneous] * local[simultaneous]) < 0)) if simultaneous.any() else 0.0
    pearson, spearman = _corr(macro, local)
    groups = df.groupby("local_group", sort=False, dropna=False).size().to_numpy(float)
    action_counts = df.groupby("local_group", sort=False, dropna=False).action_identity.nunique().to_numpy(float)
    out: dict[str, Any] = {
        "run": run,
        "step": step,
        "source_filter": source_filter or "all",
        "n_occurrences": int(n),
        "n_trajectories": int(df.traj_uid.nunique()),
        "n_local_groups": int(len(groups)),
        "mean_local_group_size": float(groups.mean()),
        "p50_local_group_size": float(np.percentile(groups, 50)),
        "p90_local_group_size": float(np.percentile(groups, 90)),
        "local_group_size_ge2_ratio": float(np.mean(groups >= 2)),
        "local_coverage": float(nonzero.mean()),
        "mean_abs_macro": float(np.abs(macro).mean()),
        "mean_abs_local": float(np.abs(local).mean()),
        "median_abs_macro": float(np.median(np.abs(macro))),
        "median_abs_local": float(np.median(np.abs(local))),
        "macro_abs_mass": macro_mass,
        "local_abs_mass": local_mass,
        "local_magnitude_share": local_mass / (macro_mass + local_mass + 1e-12),
        "token_macro_abs_mass": token_macro,
        "token_local_abs_mass": token_local,
        "token_local_magnitude_share": token_local / (token_macro + token_local + 1e-12),
        "macro_local_sign_agreement": agree,
        "macro_local_sign_conflict": conflict,
        "macro_local_pearson": pearson,
        "macro_local_spearman": spearman,
        "distinct_action_count_mean": float(action_counts.mean()),
        "distinct_action_singleton_group_ratio": float(np.mean(action_counts == 1)),
        "distinct_action_ge3_group_ratio": float(np.mean(action_counts >= 3)),
    }
    if "source_type" in df:
        out["source_occurrence_ratio"] = float(n / max(1, len(df)))
    return out


def _gigpo_step(path: Path, token_path: Path, gamma: float, penalty: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [
        "trajectory_id", "online_episode_group_uid", "pre_observation", "raw_env_reward",
        "is_action_valid", "episode_return", "episode_advantage", "step_advantage",
        "final_advantage", "env_step", "parsed_action", "occurrence_id",
    ]
    d = pq.read_table(path, columns=cols).to_pydict()
    n = len(d["trajectory_id"])
    score = np.zeros(n, dtype=float)
    by_traj: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(d["trajectory_id"]):
        by_traj[str(uid)].append(i)
    for inds0 in by_traj.values():
        inds = sorted(inds0, key=lambda i: int(d["env_step"][i]))
        running = 0.0
        for i in reversed(inds):
            running = float(d["raw_env_reward"][i] or 0.0) + gamma * running
            score[i] = running - (penalty if not d["is_action_valid"][i] else 0.0)
    local = np.zeros(n, dtype=float)
    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(n):
        by_group[(str(d["online_episode_group_uid"][i]), str(d["pre_observation"][i]))].append(i)
    for inds in by_group.values():
        local[inds] = _norm(score[inds])
    macro_scores = np.asarray([
        float(d["episode_return"][i]) - (penalty if not d["is_action_valid"][i] else 0.0)
        for i in range(n)
    ])
    macro = np.zeros(n, dtype=float)
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(d["online_episode_group_uid"]):
        by_task[str(uid)].append(i)
    for inds in by_task.values():
        macro[inds] = _norm(macro_scores[inds])
    counts = _token_counts(token_path)
    rows = pd.DataFrame({
        "run": "gigpo", "step": int(path.stem.split("_")[-1]),
        "task_id": [str(x) for x in d["online_episode_group_uid"]],
        "traj_uid": [str(x) for x in d["trajectory_id"]],
        "local_group": [f"{d['online_episode_group_uid'][i]}::{d['pre_observation'][i]}" for i in range(n)],
        "anchor_obs_hash": [_obs_hash(str(d["pre_observation"][i])) for i in range(n)],
        "action_identity": [str(d["parsed_action"][i]) for i in range(n)],
        "source_type": "root", "macro_advantage": macro, "local_advantage": local,
        "response_token_count": [counts.get(str(x), 0) for x in d["occurrence_id"]],
    })
    stored_macro = np.asarray(d["episode_advantage"], dtype=float)
    stored_local = np.asarray(d["step_advantage"], dtype=float)
    check = {
        "reconstruction_macro_max_abs_error": float(np.max(np.abs(macro - stored_macro))),
        "reconstruction_macro_mean_abs_error": float(np.mean(np.abs(macro - stored_macro))),
        "reconstruction_local_max_abs_error": float(np.max(np.abs(local - stored_local))),
        "reconstruction_local_mean_abs_error": float(np.mean(np.abs(local - stored_local))),
        "stored_final_sum_max_abs_error": float(np.max(np.abs(np.asarray(d["final_advantage"], dtype=float) - (stored_macro + stored_local)))),
    }
    return rows, check


def _bace_step(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, bad = _read_jsonl(path)
    if not raw:
        return pd.DataFrame(), {"bad_records": bad}
    rows = []
    for r in raw:
        macro = float(r.get("macro_advantage") or 0.0)
        local = float(r.get("local_advantage") or 0.0)
        rows.append({
            "run": "bace", "step": int(r.get("step", 0)), "task_id": str(r.get("task_id", "")),
            "traj_uid": str(r.get("traj_uid", "")),
            "local_group": f"{r.get('task_id','')}::{r.get('anchor','')}",
            "anchor_obs_hash": _obs_hash(str(r.get("anchor", ""))),
            "action_identity": str(r.get("action_identity", "")),
            "source_type": str(r.get("source_type", "unknown")),
            "macro_advantage": macro, "local_advantage": local,
            "response_token_count": int(r.get("response_token_count") or len(r.get("response_loss_mask") or [])),
            "occurrence_sum": float(r.get("occurrence_advantage") or 0.0),
        })
    df = pd.DataFrame(rows)
    summed = df.macro_advantage + df.local_advantage
    check = {
        "bad_records": bad,
        "usable_records": len(df),
        "occurrence_sum_max_abs_error": float(np.max(np.abs(summed - df.occurrence_sum.to_numpy(float)))),
        "occurrence_sum_mean_abs_error": float(np.mean(np.abs(summed - df.occurrence_sum.to_numpy(float)))),
    }
    return df, check


def _phase(step: int) -> str:
    return "early" if step <= 50 else "middle" if step <= 100 else "late"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gigpo-run", type=Path, default=Path("experiments/gigpo-alfworld-reference/runs/seed_0_v2"))
    ap.add_argument("--bace-run", type=Path, default=Path("experiments/alfworld-qwen2.5-1.5b-exact"))
    ap.add_argument("--bace-name", default="bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry")
    ap.add_argument("--output-dir", type=Path, default=Path("GiGPO_reference_analysis/A0_credit_decomposition_2gpu"))
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--invalid-penalty", type=float, default=0.1)
    args = ap.parse_args()
    out = args.output_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    gigpo_occ = args.gigpo_run / "raw/occurrences"
    gigpo_tok = args.gigpo_run / "raw/token_data"
    bace_occ = args.bace_run / "bace_artifacts" / args.bace_name
    all_rows: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    gfiles = sorted(gigpo_occ.glob("update_*.parquet"))
    coverage["gigpo"] = {"steps_found": len(gfiles), "steps_usable": 0, "missing_fields": [], "token_count_available": False}
    for p in gfiles:
        try:
            df, chk = _gigpo_step(p, gigpo_tok / p.name, args.gamma, args.invalid_penalty)
            all_rows.append(df); checks[f"gigpo/{df.step.iloc[0]}"] = chk
            coverage["gigpo"]["steps_usable"] += 1
            coverage["gigpo"]["token_count_available"] |= bool((df.response_token_count > 0).all())
        except Exception as exc:
            coverage["gigpo"].setdefault("corrupted_steps", []).append({"path": str(p), "error": repr(exc)})

    bfiles = sorted(bace_occ.glob("step_*/trainable_occurrences.jsonl"))
    coverage["bace"] = {"steps_found": len(bfiles), "steps_usable": 0, "missing_fields": [], "token_count_available": True, "corrupted_records": 0}
    for p in bfiles:
        df, chk = _bace_step(p)
        if df.empty:
            coverage["bace"].setdefault("corrupted_steps", []).append(str(p)); continue
        all_rows.append(df); checks[f"bace/{int(df.step.iloc[0])}"] = chk
        coverage["bace"]["steps_usable"] += 1
        coverage["bace"]["corrupted_records"] += int(chk.get("bad_records", 0))

    if not all_rows:
        raise SystemExit("No usable occurrences found")
    occurrences = pd.concat(all_rows, ignore_index=True)
    occurrences["phase"] = occurrences.step.map(_phase)
    occurrences.to_parquet(out / "occurrence_metrics.parquet", index=False)
    (out / "data_coverage_report.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "reconstruction_report.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    resolved = {"gamma": args.gamma, "invalid_penalty": args.invalid_penalty, "epsilon": EPS,
                "gigpo_run": str(args.gigpo_run), "bace_run": str(bace_occ),
                "bace_source": "per-occurrence emitted macro/local components",
                "normalization": "sample std (ddof=1), zero for singleton/zero variance groups",
                "token_weighting": "response_mask count from GiGPO token_data; BACE response_token_count"}
    (out / "resolved_analysis_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")

    metric_rows = []
    for (run, step), df in occurrences.groupby(["run", "step"], sort=True):
        metric_rows.append(_metrics(df, run, int(step)))
        if run == "bace":
            for src in ("root", "branch_origin", "branch_suffix"):
                metric_rows.append(_metrics(df, run, int(step), src))
    step_summary = pd.DataFrame(metric_rows)
    step_summary.to_csv(out / "step_summary.csv", index=False)
    phase_rows = []
    for (run, phase), df in occurrences.groupby(["run", "phase"], sort=True):
        phase_rows.append(_metrics(df, run, -1)) if False else None
        x = _metrics(df, run, -1); x["phase"] = phase; phase_rows.append(x)
    phase_summary = pd.DataFrame(phase_rows).drop(columns=["step"], errors="ignore")
    phase_summary.to_csv(out / "phase_summary.csv", index=False)
    src_rows = []
    bace = occurrences[occurrences.run == "bace"]
    for phase, df in bace.groupby("phase", sort=True):
        for src in ("root", "branch_origin", "branch_suffix"):
            x = _metrics(df, "bace", -1, src); x["phase"] = phase; src_rows.append(x)
    pd.DataFrame(src_rows).drop(columns=["step"], errors="ignore").to_csv(out / "source_summary.csv", index=False)
    act_rows = []
    for (run, phase), df in occurrences.groupby(["run", "phase"], sort=True):
        gs = df.groupby("local_group").action_identity.nunique()
        act_rows.append({"run": run, "phase": phase, "groups": len(gs), "action_1_ratio": float((gs == 1).mean()),
                         "action_2_ratio": float((gs == 2).mean()), "action_ge3_ratio": float((gs >= 3).mean())})
    pd.DataFrame(act_rows).to_csv(out / "action_structure_summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        for metric, name, ylabel in [
            ("local_coverage", "local_coverage_over_steps.png", "local coverage"),
            ("local_magnitude_share", "local_magnitude_share_over_steps.png", "local magnitude share"),
            ("token_local_magnitude_share", "token_local_share_over_steps.png", "token-weighted local share"),
            ("macro_local_sign_conflict", "macro_local_conflict_over_steps.png", "macro/local sign conflict"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for run, grp in step_summary[step_summary.source_filter == "all"].groupby("run"):
                ax.plot(grp.step, grp[metric], label=run, alpha=.35)
                ax.plot(grp.step, grp[metric].rolling(5, min_periods=1).mean(), label=f"{run} (5-step mean)", linewidth=2)
            ax.set(xlabel="training step", ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / name, dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        src = pd.DataFrame(src_rows)
        for source, grp in src.groupby("source_filter"):
            vals = grp.set_index("phase").reindex(["early", "middle", "late"]).local_abs_mass
            ax.plot(vals.index, vals.values, marker="o", label=source)
        ax.set(xlabel="phase", ylabel="local absolute mass"); ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "bace_source_local_mass.png", dpi=160); plt.close(fig)
    except Exception as exc:
        # The login-node Python may not carry matplotlib.  Use the cluster's
        # gnuplot installation so the required audit figures are still
        # generated without changing the training environment.
        import shutil
        import subprocess
        if shutil.which("gnuplot"):
            csv = out / "step_summary.csv"
            plot_specs = [
                ("local_coverage", "local_coverage_over_steps.png", "local coverage", 11),
                ("local_magnitude_share", "local_magnitude_share_over_steps.png", "local magnitude share", 18),
                ("token_local_magnitude_share", "token_local_share_over_steps.png", "token-weighted local share", 21),
                ("macro_local_sign_conflict", "macro_local_conflict_over_steps.png", "macro/local sign conflict", 23),
            ]
            for _, name, ylabel, col in plot_specs:
                script = (
                    "set datafile separator ','; set terminal pngcairo size 1200,600; "
                    f"set output '{out / 'figures' / name}'; set xlabel 'training step'; "
                    f"set ylabel '{ylabel}'; set key outside; "
                    f"plot '{csv}' every ::1 using ((stringcolumn(1) eq 'gigpo' && stringcolumn(3) eq 'all') ? column(2) : 1/0):{col} with lines title 'GiGPO', "
                    f"'{csv}' every ::1 using ((stringcolumn(1) eq 'bace' && stringcolumn(3) eq 'all') ? column(2) : 1/0):{col} with lines title 'BACE'"
                )
                subprocess.run(["gnuplot", "-e", script], check=True)
            source_csv = out / "source_summary.csv"
            script = (
                "set datafile separator ','; set terminal pngcairo size 1200,600; "
                f"set output '{out / 'figures' / 'bace_source_local_mass.png'}'; "
                "set xlabel 'phase'; set ylabel 'local absolute mass'; set key outside; "
                f"plot '{source_csv}' every ::1 using ((stringcolumn(2) eq 'root') ? (stringcolumn(29) eq 'early' ? 1 : stringcolumn(29) eq 'middle' ? 2 : 3) : 1/0):16 with linespoints title 'root', "
                f"'{source_csv}' every ::1 using ((stringcolumn(2) eq 'branch_origin') ? (stringcolumn(29) eq 'early' ? 1 : stringcolumn(29) eq 'middle' ? 2 : 3) : 1/0):16 with linespoints title 'branch origin', "
                f"'{source_csv}' every ::1 using ((stringcolumn(2) eq 'branch_suffix') ? (stringcolumn(29) eq 'early' ? 1 : stringcolumn(29) eq 'middle' ? 2 : 3) : 1/0):16 with linespoints title 'branch suffix'"
            )
            subprocess.run(["gnuplot", "-e", script], check=True)
            (out / "plot_backend.txt").write_text(f"gnuplot fallback; matplotlib error: {exc!r}\n", encoding="utf-8")
        else:
            (out / "plot_error.txt").write_text(repr(exc), encoding="utf-8")

    # Compact narrative with explicit caveats, suitable for handoff.
    def val(run: str, phase: str, col: str) -> float:
        q = phase_summary[(phase_summary.run == run) & (phase_summary.phase == phase)]
        return float(q.iloc[0][col]) if not q.empty and pd.notna(q.iloc[0].get(col)) else float("nan")
    lines = ["# A0：2 卡 GiGPO / BACE 信用结构分解", "", "## 数据与口径", "",
             f"- GiGPO：`{args.gigpo_run}`，{coverage['gigpo']['steps_usable']}/{coverage['gigpo']['steps_found']} steps。",
             f"- BACE：`{bace_occ}`，{coverage['bace']['steps_usable']}/{coverage['bace']['steps_found']} steps；跳过截断 JSONL record：{coverage['bace']['corrupted_records']}。",
             "- GiGPO 按 physical occurrence 重建；BACE 使用 trainer 输出的 macro/local occurrence 组件并检查 occurrence sum。",
             "- gamma=0.95、invalid penalty=0.1、sample std(ddof=1)、singleton/零方差 local group 置零；token share 使用真实 response mask。", "",
             "## Early / middle / late 主结果", "", "|run|phase|occurrences|local coverage|local magnitude share|token local share|sign conflict|mean group size|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in phase_summary.sort_values(["run", "phase"]).iterrows():
        lines.append(f"|{r.run}|{r.phase}|{int(r.n_occurrences)}|{r.local_coverage:.4f}|{r.local_magnitude_share:.4f}|{r.token_local_magnitude_share:.4f}|{r.macro_local_sign_conflict:.4f}|{r.mean_local_group_size:.2f}|")
    lines += ["", "## 解读", "", "- `local_coverage` 回答有多少 occurrence 获得非零 local credit；`local_magnitude_share` 和 token-weighted share 才近似训练信号占比。", "- BACE 的 source 表将 root、branch_origin、branch_suffix 分开；另可用 occurrence_metrics.parquet 过滤 `source_type=root` 做 root-only 对照。", "- 任何 reconstruction mismatch、截断记录和 token 缺失均在 JSON 报告中保留；本报告不把历史 artifact 当作当前 HEAD 的同版本验证。", "", "## 后续判断", "", "仅凭 coverage 不能决定下调 step_advantage_w；若 BACE 后期 magnitude share 和 conflict 同时显著升高，才支持 local-credit overshoot 假设。若 root-only 接近 GiGPO 而 all-data 偏离，则优先检查 branch weighting/acquisition，而不是直接改 local estimator。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "coverage": coverage, "rows": len(occurrences)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
