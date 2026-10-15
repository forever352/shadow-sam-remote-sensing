"""BER 口径敏感性：macro（逐图平均，本项目现行）vs pooled（全数据集汇总）。

## 为什么要做

现行口径（`评估基座与局限性说明.md` 第 22 行明示）是"**逐图计算再取均值**"（macro）。
而阴影检测文献（如 `xw-hu/Unveiling-Deep-Shadows` 的官方评估脚本）用
"**把全数据集的 TP/TN/P/N 汇总后算一次**"（pooled / micro）。

AISD 图像尺寸跨度大（256~1688），两种口径**必然不同**。本脚本量化该差异，
作为**口径敏感性附录**--**不改主口径、不动 `compute_metrics` 契约**。

## 关键：无需重跑 GPU

TP/FP/FN/TN 可从已有的逐图指标 + 图像像素数**反解**：
    TP + FP + FN = N-(1 - OA)                      （OA 定义）
    FP = TP-(1-P)/P ,  FN = TP-(1-R)/R             （P/R 定义）
  ⇒ TP = N(1-OA) / [ (1-P)/P + (1-R)/R ]
反解出的四项会与 IoU / BER / F1 **交叉校验**，不一致则报警（防止公式推错）。

## 落盘
    results/ber_sensitivity.csv    逐方法的 macro/pooled 六项指标
    logs/ber_sensitivity_{ts}.log
"""
import csv
import math
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

# (显示名, 逐图 CSV, mask 所在目录)
TARGETS = [
    ("LoRA-SAM(leakfree)", "lora_leakfree_test_per_image.csv", AISD / "Train412" / "mask"),
    ("U-Net(leakfree)", "unet_leakfree_test_per_image.csv", AISD / "Train412" / "mask"),
    ("SegNet(leakfree)", "segnet_leakfree_test_per_image.csv", AISD / "Train412" / "mask"),
    ("LoRA-SAM(official)", "lora_test51_per_image.csv", AISD / "Test51" / "mask"),
    ("U-Net(official)", "unet_official_test_per_image.csv", AISD / "Test51" / "mask"),
    ("SegNet(official)", "segnet_official_test_per_image.csv", AISD / "Test51" / "mask"),
]


def reconstruct_counts(N, oa, p, r):
    """由 (像素数, OA, Precision, Recall) 反解 (TP, FP, FN, TN)。"""
    denom = (1.0 - p) / max(p, 1e-12) + (1.0 - r) / max(r, 1e-12)
    if denom <= 0:
        return None
    tp = N * (1.0 - oa) / denom
    fp = tp * (1.0 - p) / max(p, 1e-12)
    fn = tp * (1.0 - r) / max(r, 1e-12)
    tn = N - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    ber = 0.5 * (fp / (fp + tn + eps) + fn / (fn + tp + eps))
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return dict(IoU=iou, BER=ber, F1=f1, OA=oa, Precision=prec, Recall=rec)


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ber_sensitivity_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log("# BER 口径敏感性：macro（逐图平均）vs pooled（全数据集汇总）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# 说明：pooled = 汇总全部图的 TP/TN/P/N 后算一次；macro = 逐图算完再平均")
    log()

    rows = []
    for disp, csvname, maskdir in TARGETS:
        p = RES_DIR / csvname
        if not p.exists():
            log(f"[跳过] {disp}：缺 {csvname}")
            continue
        recs = list(csv.DictReader(open(p, encoding="utf-8")))
        cnt = []
        n_bad = 0
        macro = {k: [] for k in METRIC_KEYS}
        for r in recs:
            mp = maskdir / r["name"]
            gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            N = float(gt.size)
            oa, pr, rc = float(r["OA"]), float(r["Precision"]), float(r["Recall"])
            c = reconstruct_counts(N, oa, pr, rc)
            if c is None:
                n_bad += 1
                continue
            tp, fp, fn, tn = c
            m2 = metrics_from_counts(tp, fp, fn, tn)
            # 交叉校验：反解出的指标应与 CSV 一致（防公式推错）
            if (abs(m2["IoU"] - float(r["IoU"])) > 2e-3 or
                    abs(m2["BER"] - float(r["BER"])) > 2e-3):
                n_bad += 1
                continue
            cnt.append((tp, fp, fn, tn))
            for k in METRIC_KEYS:
                macro[k].append(float(r[k]))
        if not cnt:
            log(f"[跳过] {disp}：无有效样本")
            continue
        TP = sum(c[0] for c in cnt); FP = sum(c[1] for c in cnt)
        FN = sum(c[2] for c in cnt); TN = sum(c[3] for c in cnt)
        pooled = metrics_from_counts(TP, FP, FN, TN)
        mac = {k: float(np.mean(macro[k])) for k in METRIC_KEYS}
        log(f"【{disp}】n={len(cnt)}  反解校验失败数={n_bad}  总像素={int(TP+FP+FN+TN):,}")
        log(f"    {'指标':<10}{'macro(现行)':<16}{'pooled(文献)':<16}{'差值':<12}")
        for k in ["IoU", "BER", "F1", "OA"]:
            log(f"    {k:<10}{mac[k]:<16.6f}{pooled[k]:<16.6f}{pooled[k]-mac[k]:<12.6f}")
        rows.append({"method": disp, "n_images": len(cnt),
                     **{f"macro_{k}": f"{mac[k]:.6f}" for k in METRIC_KEYS},
                     **{f"pooled_{k}": f"{pooled[k]:.6f}" for k in METRIC_KEYS},
                     "delta_BER": f"{pooled['BER']-mac['BER']:.6f}",
                     "delta_IoU": f"{pooled['IoU']-mac['IoU']:.6f}"})
        log("")

    out = RES_DIR / "ber_sensitivity.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log(f"[OK] 明细 -> {out}")
    log("# 纪律：本脚本**不改主口径**（仍为 macro）；pooled 仅作口径敏感性附录")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
