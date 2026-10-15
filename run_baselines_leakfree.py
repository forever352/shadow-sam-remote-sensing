"""传统基线 / 零样本 SAM 在【无泄漏划分】上的评估（申报书点名的基准对比）。

## 为什么

申报书（`_shenbaoshu_attached_20261014.txt`）的实验方案点名要对比：
    "与传统 U-Net、SegNet 以及**基于物理模型的阈值法**进行对比"、
    "与……**原始 SAM（Zero-shot 模式）**进行对比"。
而现有 `run_experiment.py` **只在官方 `Test51` 上跑过**（`results/per_image.csv`，
含 42/43 同源泄漏），**不能与主表横比**。本脚本把这些基线落到**同一无泄漏测试集(92 张)**
上，使它们进入可横比的主口径。

## 口径（与主方法逐字对齐）

- 划分：`split_by_base(Train412, 0.2, seed=0)` -> `split_by_base(pool, 0.2, seed=1)`，
  与 `train_lora_leakfree.py:96-97` 完全一致；启动时**断言**测试集 name 集合
  与 `results/lora_leakfree_test_per_image.csv` 一致，不等即抛异常。
- 指标：`eval_shadow.compute_metrics`，**macro（逐图平均）**，与主表同口径。
- 方法实现**原样复用** `methods.py`，不重新实现、不调参（避免"为凑指标而调"）。
- SAM-AMG 超参沿用 `run_experiment.load_amg`（`points_per_side=32` 等），保持与历史可比。

## 用法
    python run_baselines_leakfree.py --methods otsu,c3i          # 纯 CPU，随时可跑
    python run_baselines_leakfree.py --methods sam_amg           # 需 GPU（SAM 推理）
    python run_baselines_leakfree.py                             # 全部

## 落盘
    results/{otsu,c3i,enhanced,sam_amg}_leakfree_test.csv
    results/{otsu,c3i,enhanced,sam_amg}_leakfree_test_per_image.csv
    logs/baselines_leakfree_{ts}.log
"""
import argparse
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import numpy as np
import cv2

from eval_shadow import compute_metrics
from train_direction_net import split_by_base
import methods as M

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
ALL_METHODS = ["otsu", "c3i", "c3i_heuristic", "enhanced", "sam_amg"]

# [!] `c3i` 与 `c3i_heuristic` 的区别（第十三轮审查，H1）：
#     `methods.mask_c3i` 里有一段**本项目自加**的启发式（前景 >40% 则整体取反），
#     它不是 C3I 原方法的一部分。审查实测它在无泄漏测试集 92 张上触发 61 张，
#     去掉后 61/61 全部变好、0 张变差 → 当前数据上 100% 有害。
#     故：`c3i`（**主行**）= 不含该启发式，才是 C3I 方法本身的表现；
#         `c3i_heuristic`（**副行**）= 与既有历史产物同口径，保留以供对照。
#     两行必须同时出现，**不得只报其中一个**。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                    help="逗号分隔，可选 " + "/".join(ALL_METHODS))
    args = ap.parse_args()
    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in want if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"未知方法 {bad}；可选 {ALL_METHODS}")

    LOG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"baselines_leakfree_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 传统基线 / 零样本 SAM 在无泄漏划分上的评估")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  方法: {want}")

    # ── 划分（与 train_lora_leakfree.py:96-97 逐字一致）──
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log(f"划分: 训练 {len(tr_files)} 张({n_tr}原图) / 验证 {len(val_files)} 张({n_va}原图) "
        f"/ 测试 {len(test_files)} 张({n_te}原图)  [仅用测试集评估；基线不含训练]")

    # 断言：测试集与主结果逐图 CSV 的 name 集合一致
    main_csv = RES_DIR / "lora_leakfree_test_per_image.csv"
    if main_csv.exists():
        with open(main_csv, newline="", encoding="utf-8-sig") as fh:
            names_main = {r["name"] for r in csv.DictReader(fh) if r.get("name")}
        names_here = {Path(f).name for f in test_files}
        if names_main != names_here:
            raise RuntimeError(
                f"划分不一致：本脚本测试集({len(names_here)}) 与主结果({len(names_main)}) 的 "
                f"name 集合不等。划分被改动过，拒绝产出不可比的结果。")
        log(f"  [OK] 测试集与主结果逐图 CSV 的 name 集合一致（{len(names_here)} 张）")
    else:
        log(f"  [!] 未找到 {main_csv}，跳过划分一致性断言")

    # ── 惰性加载 AMG（只在需要时占用 GPU）──
    amg = None
    if "sam_amg" in want:
        from run_experiment import load_amg
        log("加载 SAM-AMG（vit_b，超参与 run_experiment.load_amg 一致）...")
        amg = load_amg("cuda" if __import__("torch").cuda.is_available() else "cpu")

    def apply(method, img_bgr):
        if method == "otsu":
            return M.mask_otsu(img_bgr)
        if method == "c3i":
            # 主行：不含本项目自加的 ">40% 反转" 启发式（那才是 C3I 方法本来的表现）
            return M.mask_c3i(img_bgr, invert_if_mostly_white=False)
        if method == "c3i_heuristic":
            # 副行：与既有历史产物同口径（带启发式），保留供对照
            return M.mask_c3i(img_bgr, invert_if_mostly_white=True)
        if method == "enhanced":
            return M.mask_enhanced(img_bgr)
        if method == "sam_amg":
            return M.mask_sam_amg(img_bgr, amg)
        raise ValueError(method)

    summary = []
    for method in want:
        log("")
        log(f"=== {method} ===")
        rows = []
        t0 = time.time()
        for f in test_files:
            img = cv2.imread(f)
            gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            pred = apply(method, img)
            m = compute_metrics(pred, gt)
            m["name"] = Path(f).name
            rows.append(m)
        avg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
        std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
        log(f"  测试 {len(rows)} 张  ({time.time()-t0:.0f}s)")
        for k in METRIC_KEYS:
            log(f"  {k:10s} {avg[k]:.4f} +/- {std[k]:.4f}")

        per_img = RES_DIR / f"{method}_leakfree_test_per_image.csv"
        summ = RES_DIR / f"{method}_leakfree_test.csv"
        with open(per_img, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["name"] + METRIC_KEYS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
        with open(summ, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "n_images"] + METRIC_KEYS)
            w.writerow([f"{method}(leakfree)", len(rows)] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
            w.writerow([f"{method}(leakfree)_std", len(rows)] + [f"{std[k]:.6f}" for k in METRIC_KEYS])
        log(f"  -> {summ}")
        summary.append((method, avg))

    log("")
    log("=== 汇总（无泄漏测试集，macro 逐图平均）===")
    log(f"{'方法':<12}{'IoU':>9}{'BER':>9}")
    for method, avg in summary:
        log(f"{method:<12}{avg['IoU']:>9.4f}{avg['BER']:>9.4f}")
    log("")
    log("对照：主方法 LoRA-SAM(leakfree) IoU 0.787821 / BER 0.064359")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
