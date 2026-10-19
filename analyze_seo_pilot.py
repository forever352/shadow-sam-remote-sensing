# -*- coding: utf-8 -*-
"""S-EO 试点结果的事后补充分析（不训练）。

读取 seo_direction_pilot.py 落盘的 CSV，补齐 supervisor 结果审查要求的：
- seed×fold 方差；
- epoch 30 训练 loss / train vs eval 误差；
- 跨域按日期聚合的均值与 bootstrap 描述性区间（不把 100 张当独立样本）；
- 预注册偏离与限制。

用法：
    python analyze_seo_pilot.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "results" / "seo_direction_pilot"
SEEDS = [0, 1, 2]


def read_rows(name: str) -> list[dict]:
    path = OUT_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def boot_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_fold_variance(rows: list[dict]) -> dict:
    per_seed = {s: [] for s in SEEDS}
    per_fold_std = []
    for r in rows:
        vals = []
        for s in SEEDS:
            v = f(r, f"model_err_s{s}_deg")
            if np.isfinite(v):
                per_seed[s].append(v)
                vals.append(v)
        if len(vals) > 1:
            per_fold_std.append(float(np.std(vals, ddof=1)))
    all_vals = np.array([v for s in SEEDS for v in per_seed[s]], dtype=np.float64)
    seed_means = {s: float(np.mean(per_seed[s])) for s in SEEDS if per_seed[s]}
    # 简单方差分解：seed 均值之间的方差 vs 折内 seed 方差的均值
    sm = np.array(list(seed_means.values()), dtype=np.float64)
    between_seed_var = float(np.var(sm, ddof=1)) if len(sm) > 1 else 0.0
    within_var = float(np.mean(np.array(per_fold_std) ** 2)) if per_fold_std else 0.0
    return {
        "n_folds": len(rows),
        "seed_means": seed_means,
        "all_seed_fold_mean": float(all_vals.mean()) if len(all_vals) else float("nan"),
        "all_seed_fold_std": float(all_vals.std(ddof=1)) if len(all_vals) > 1 else float("nan"),
        "between_seed_var": between_seed_var,
        "within_fold_seed_var": within_var,
        "mean_per_fold_seed_std": float(np.mean(per_fold_std)) if per_fold_std else float("nan"),
    }


def epoch_loss_summary() -> dict:
    rows = read_rows("oma_loocv_training_losses.csv")
    if not rows:
        return {}
    main = [r for r in rows if r.get("shuffle") == "0"]
    if not main:
        return {}
    max_ep = max(int(r["epoch"]) for r in main)
    last = [float(r["loss"]) for r in main if int(r["epoch"]) == max_ep]
    first_ep = min(int(r["epoch"]) for r in main)
    first = [float(r["loss"]) for r in main if int(r["epoch"]) == first_ep]
    return {
        "epoch_max": max_ep,
        "loss_first_epoch_mean": float(np.mean(first)),
        "loss_last_epoch_mean": float(np.mean(last)),
        "loss_last_std": float(np.std(last, ddof=1)) if len(last) > 1 else float("nan"),
        "n_loss_curves": len(last),
    }


def cross_date_level(name: str) -> dict:
    rows = read_rows(name)
    if not rows:
        return {}
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)

    def agg(key: str, subset: list[dict] | None = None) -> dict:
        vals = []
        for date, rs in by_date.items():
            if subset is not None and not any(int(r["in_source_support"]) == subset for r in rs):
                continue
            selected = rs if subset is None else [r for r in rs if int(r["in_source_support"]) == subset]
            if selected:
                vals.append(float(np.mean([f(r, key) for r in selected])))
        a = np.array(vals, dtype=np.float64)
        lo, hi = boot_ci(a)
        return {
            "n_dates": int(len(a)),
            "mean_deg": float(a.mean()) if len(a) else float("nan"),
            "median_deg": float(np.median(a)) if len(a) else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
        }

    out = {
        "n_images": len(rows),
        "n_dates": len(by_date),
        "all": {
            "model": agg("model_err_mean_deg"),
            "source_constant": agg("source_const_err_deg"),
            "target_elevation_oracle": agg("target_elevation_oracle_err_deg"),
            "target_oracle_constant": agg("target_oracle_const_err_deg"),
        },
        "in_support": {
            "model": agg("model_err_mean_deg", 1),
            "source_constant": agg("source_const_err_deg", 1),
            "target_elevation_oracle": agg("target_elevation_oracle_err_deg", 1),
            "target_oracle_constant": agg("target_oracle_const_err_deg", 1),
        },
        "out_support": {
            "model": agg("model_err_mean_deg", 0),
            "source_constant": agg("source_const_err_deg", 0),
            "target_elevation_oracle": agg("target_elevation_oracle_err_deg", 0),
            "target_oracle_constant": agg("target_oracle_const_err_deg", 0),
        },
    }
    return out


def write_md() -> None:
    loocv = read_rows("oma_loocv_per_fold.csv")
    jax = read_rows("jax_loao_per_fold.csv")
    lines = ["# S-EO 试点 · 事后补充分析", ""]
    lines.append("> 本文件只读取落盘 CSV，不重新训练；补充 supervisor 结果审查要求的诊断。")
    lines.append("")

    if loocv:
        sf = seed_fold_variance(loocv)
        lines += ["## OMA LOOCV：seed×fold", "",
                  f"- 折数：{sf['n_folds']}",
                  f"- 每 seed 总均值：" + "；".join(
                      f"s{s}={v:.3f}°" for s, v in sf["seed_means"].items()),
                  f"- 全部 seed-fold 误差均值/标准差：{sf['all_seed_fold_mean']:.3f}° / {sf['all_seed_fold_std']:.3f}°",
                  f"- seed 均值间方差：{sf['between_seed_var']:.4f}",
                  f"- 折内 seed 方差均值：{sf['within_fold_seed_var']:.4f}",
                  f"- 平均每折 seed 标准差：{sf['mean_per_fold_seed_std']:.3f}°",
                  ""]
        el = epoch_loss_summary()
        if el:
            lines += ["## OMA LOOCV：训练诊断", "",
                      f"- 训练曲线数：{el['n_loss_curves']}",
                      f"- epoch {el['epoch_max']} 训练 loss 均值：{el['loss_last_epoch_mean']:.4f}（std {el['loss_last_std']:.4f}）",
                      f"- epoch 1 训练 loss 均值：{el['loss_first_epoch_mean']:.4f}",
                      ""]
        # train vs eval
        tr = np.array([f(r, "train_err_mean_deg") for r in loocv], dtype=np.float64)
        ev = np.array([f(r, "model_err_mean_deg") for r in loocv], dtype=np.float64)
        lines += [f"- train error 均值：{np.nanmean(tr):.3f}°；eval error 均值：{np.nanmean(ev):.3f}°；"
                  f"差值（eval−train）：{np.nanmean(ev) - np.nanmean(tr):.3f}°", ""]

    if jax:
        lines += ["## JAX 留一 AOI", "",
                  "| held-out AOI | n | model | constant | elevation oracle | train loss |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in jax:
            lines.append(f"| {r['heldout_aoi']} | {r['n_eval']} | {f(r, 'model_err_mean_deg'):.2f}° | "
                         f"{f(r, 'constant_mean_deg'):.2f}° | {f(r, 'elevation_oracle_mean_deg'):.2f}° | "
                         f"{f(r, 'train_loss_mean'):.4f} |")
        lines += ["", "> 4 折只作 pilot 层面描述；sign-flip p 不作推断。", ""]

    for name, title in [("cross_oma_to_jax_per_image.csv", "OMA→JAX"),
                        ("cross_jax_to_oma_per_image.csv", "JAX→OMA")]:
        cl = cross_date_level(name)
        if not cl:
            continue
        lines += [f"## 跨域 {title}：按日期聚合", "",
                  f"- 图像数 {cl['n_images']}，日期数 {cl['n_dates']}；bootstrap 按日期重采样，仅作描述性区间。",
                  "",
                  "| 分层 | 模型 | 源域常数 | 目标高程 oracle | 目标均值 oracle |",
                  "|---|---:|---:|---:|---:|"]
        for key, label in [("all", "全部"), ("in_support", "支撑内"), ("out_support", "支撑外")]:
            d = cl[key]
            m = d["model"].get("mean_deg", float("nan"))
            c = d["source_constant"].get("mean_deg", float("nan"))
            e = d["target_elevation_oracle"].get("mean_deg", float("nan"))
            o = d.get("target_oracle_constant", {}).get("mean_deg", float("nan"))
            lines.append(f"| {label} | {m:.2f}° (95% {d['model'].get('ci_lo', float('nan')):.2f}–{d['model'].get('ci_hi', float('nan')):.2f}) | "
                         f"{c:.2f}° | {e:.2f}° | {o:.2f}° |")
        lines += ["", f"> 日期级样本数：全部 {cl['all']['model']['n_dates']}，"
                      f"支撑内 {cl['in_support']['model']['n_dates']}，"
                      f"支撑外 {cl['out_support']['model']['n_dates']}。", ""]

    lines += ["## 预注册偏离与限制", "",
              "- 偏置诊断（预测方位角 vs 真值的圆形偏差）未实现：脚本只落盘误差，未保存预测方位角。作为预注册承诺的偏离，在结果中明确报告，不作为主判据。",
              "- 跨域 headline 的 per-image CI/p 仅作 per-image 描述；日期级聚合见上表，日期级才是更合适的推断单位。",
              "- 输入固定 224×224；OMA 原图 1344–1671 px，分辨率损失可能影响阴性结论，448 消融未做。",
              "- 跨域前处理不对称：OMA 原始 RGB，JAX 逐图 2–98 百分位拉伸；结论限定在此预处理下。",
              "- 本试点未接 AISD 下游；S-EO 结果不改变 AISD 主表与方向模块归因。", ""]
    (OUT_DIR / "SUMMARY_EXTRA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT_DIR / 'SUMMARY_EXTRA.md'}")


if __name__ == "__main__":
    write_md()
