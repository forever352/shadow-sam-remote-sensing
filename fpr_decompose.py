"""FPR 成分分解：主方法的误检像素，按 **HSV 启发式** 分成 水体样 / 植被样 / 其它。

## 要回答什么

申报书把"阴影 vs 水体/深色植被的**异物同谱**"当作核心难题（这在遥感阴影检测里
**是真实存在的经典问题**）。但本项目**从未分析过**主方法误检的**成分**——只知道
FPR 的**总量**（macro 0.0405），不知道它由什么构成。本脚本补这一块。

## [!] 三条必须先声明的边界

1. **这是 HSV 启发式分类，不是地物真值。** 类别名一律带 `hsv_` 前缀
   （`hsv_waterlike` / `hsv_veglike`）——**不得**简写成"水体/植被"。
2. **水体判据要求 `V >= 110`（绝对亮度）**，而"异物同谱"的典型形态是**深色水体**
   → **深水会被归入"其它"，水体成分被系统性低估**。该常量本为 *patch 级伪阴影剔除*
   而设（`direction_utils.py:165-169`），**不是为 FP 地物归类调的**。此偏差**必须随结果一并声明**。
3. **本分析不改变任何主表数字**，也不改变"BER<5% 未达成"的结论；它**不构成**
   "误检无妨 / 非方法问题 / 申报书前提被证实或证伪"的任何论据。

## 口径（第二十一轮审查必须修正 1-2）

- **主口径 = macro FPR 的可加分解**（与主表的 macro FPR **同分母**，可直接相加还原）：
      contribution_c = (1/N) * sum_i  FP_i^c / (FP_i + TN_i)
      则  sum_c contribution_c ≡ macro FPR
  （`FP_i^c` = 第 i 张图中"是误检 **且** 被 HSV 判为 c 类"的像素数。）
- **逐像素级**（不用连通域级）：FP 与参照组必须**同粒度**才可比。连通域级会因
  `非gt` 退化成少数巨块而与 FP 的域划分不可对应。
- **pooled 像素占比**只作**附表**，并写明它与 macro 的差异来源（大图主导）。

## 两个参照（让结果可解释）

- **背景本底成分**：全部非阴影像素（`非gt`）的同类分类占比 → 用来算 **lift = FP占比 / 本底占比**。
  裸比例没有意义，**lift** 才有：lift≈1 表示该类的 FP 只是背景的自然延伸。
- **真阴影参照（TP）**：真阴影像素的同类分类占比 → 直接检验"**FP 是否与真阴影同谱**"。

## 复现闸门（审查必须修正 5）

预测掩膜必须与既有 `results/lora_leakfree_test.csv`（IoU **0.787821**）逐位一致，
不一致即**中止**。单 seed = seed-0，须随结果声明。

## 禁止（审查必须修正 6）

**不得**用本脚本的任何数字回头去调 HSV 阈值或 `min_area`——那是**用测试集调参**。
如要调，必须回到 val 重选。

## 落盘
    results/fpr_composition.csv            主口径分解 + 附表 + lift
    results/fpr_composition_per_image.csv  逐图
    logs/fpr_decompose_{ts}.log
"""
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from direction_experiment import method_boxes
from train_adapter_v4 import apply_lora
from train_direction_net import split_by_base
from direction_utils import (VEG_H_RANGE, VEG_S_MIN, WATER_H_RANGE,
                             WATER_S_MAX, WATER_V_MIN)

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
MAIN_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
CLASSES = ["hsv_waterlike", "hsv_veglike", "other_hsv"]
REF_IOU = 0.787821          # 复现闸门参照（主结果 IoU）
REF_TOL = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


def classify_hsv(hsv):
    """逐像素 HSV 归类。输入 cv2 HSV (H in [0,180), S/V in [0,255])。返回类别索引图。

    [!] 复用项目**已预注册**的常量；本函数不做任何阈值调整。
    [!] 水体判据的 V>=110 会**低估深色水体**（见模块 docstring 边界 2）。
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    veg = (h >= VEG_H_RANGE[0]) & (h <= VEG_H_RANGE[1]) & (s >= VEG_S_MIN)
    wat = (h >= WATER_H_RANGE[0]) & (h <= WATER_H_RANGE[1]) & \
          (s <= WATER_S_MAX) & (v >= WATER_V_MIN)
    out = np.full(h.shape, 2, dtype=np.int8)      # 默认 other
    out[wat] = 0
    out[veg] = 1                                  # 植被优先于水体（互斥）
    return out


def main():
    for d in (RES_DIR, LOG_DIR):
        d.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"fpr_decompose_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# FPR 成分分解（HSV 启发式：hsv_waterlike / hsv_veglike / other_hsv）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 权重: {Path(MAIN_CKPT).name}（seed=0，**单 seed**）")
    log("# 口径: 主口径 = macro FPR 的可加分解 contribution_c = (1/N)*sum_i FP_i^c/(FP_i+TN_i)")
    log("# [!] HSV 启发式分类，非地物真值；水体判据 V>=110 会低估深色水体")
    log("# [!] 本分析不改主表任何数字；不得用于调参；不得作'误检无妨/非方法问题'之论据")

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log(f"划分: 训练 {len(tr_files)}({n_tr}) / 验证 {len(val_files)}({n_va}) / 测试 {len(test_files)}({n_te})")

    if len(test_files) != 92 or n_te != 21:
        raise RuntimeError(f"划分异常: test={len(test_files)}/{n_te}（应 92/21）")
    _mcsv = RES_DIR / "lora_leakfree_test_per_image.csv"
    with open(_mcsv, newline="", encoding="utf-8-sig") as fh:
        _nm = {r["name"] for r in csv.DictReader(fh) if r.get("name")}
    if _nm != {Path(f).name for f in test_files}:
        raise RuntimeError("测试集 name 集合与主结果不一致，拒绝产出不可比结果")

    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    sam.load_state_dict(torch.load(MAIN_CKPT, map_location=device))
    sam.to(device); sam.eval()

    # 主口径累加器
    contrib = {c: 0.0 for c in CLASSES}          # sum_i FP_i^c/(FP_i+TN_i)
    fp_px = {c: 0 for c in CLASSES}              # pooled 附表
    bg_px = {c: 0 for c in CLASSES}
    tp_px = {c: 0 for c in CLASSES}
    tot_fp = tot_bg = tot_tp = 0
    rows = []
    ious = []

    with torch.no_grad():
        for f in test_files:
            img_bgr = cv2.imread(f)
            gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred = method_boxes(sam, img_rgb, img_bgr)
            g = gt > 0
            ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])

            cls = classify_hsv(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
            fp_m = pred & ~g          # 误检
            tn_m = ~pred & ~g         # 真阴
            tp_m = pred & g           # 真阳
            denom = int(fp_m.sum() + tn_m.sum())
            row = {"name": Path(f).name}
            for k, c in enumerate(CLASSES):
                fp_c = int((fp_m & (cls == k)).sum())
                bg_c = int((tn_m & (cls == k)).sum())
                tp_c = int((tp_m & (cls == k)).sum())
                fp_px[c] += fp_c; bg_px[c] += bg_c; tp_px[c] += tp_c
                if denom > 0:
                    contrib[c] += fp_c / denom
                row[f"FP_{c}"] = fp_c
            tot_fp += int(fp_m.sum()); tot_bg += int(tn_m.sum()); tot_tp += int(tp_m.sum())
            rows.append(row)

    # 复现闸门
    got_iou = float(np.mean(ious))
    log("")
    log(f"# 复现闸门: 本脚本 IoU = {got_iou:.6f}  主结果 = {REF_IOU:.6f}  偏差 = {abs(got_iou-REF_IOU):.2e}")
    if abs(got_iou - REF_IOU) > REF_TOL:
        log("  [中止] 与主结果不一致 -> 链路漂移，拒绝产出")
        logf.close()
        raise SystemExit(3)
    log("  [OK] 复现通过（单 seed=seed-0）")

    N = len(rows)
    macro_fpr = sum(contrib.values()) / N if N else float("nan")
    log("")
    log(f"=== 主口径：macro FPR 的可加分解（N={N} 张）===")
    log(f"{'类别':<16}{'对 macro FPR 的贡献':>22}{'占比':>10}")
    for c in CLASSES:
        v = contrib[c] / N
        log(f"{c:<16}{v:>22.6f}{100*v/macro_fpr:>9.1f}%")
    log(f"{'合计(macro FPR)':<16}{macro_fpr:>22.6f}{100.0:>9.1f}%")
    log("  （各项之和 == macro FPR，与主表 FPR 同分母，可加）")

    log("")
    log("=== 附表：pooled 像素占比（**与 macro 口径不同，勿混引**）===")
    for c in CLASSES:
        share = fp_px[c] / max(tot_fp, 1)
        log(f"  {c:<16} FP 像素占比 {100*share:>5.1f}%   （{fp_px[c]}/{tot_fp} px）")

    log("")
    log("=== 参照：lift = FP 占比 / 背景本底占比（lift~1 表示该类只是背景的自然延伸）===")
    for c in CLASSES:
        fp_s = fp_px[c] / max(tot_fp, 1)
        bg_s = bg_px[c] / max(tot_bg, 1)
        lift = fp_s / bg_s if bg_s > 0 else float("nan")
        log(f"  {c:<16} FP {100*fp_s:>5.1f}%   本底 {100*bg_s:>5.1f}%   lift = {lift:>6.2f}")
    log("  【真阴影(TP)参照】检验 FP 是否与真阴影同谱：")
    for c in CLASSES:
        tp_s = tp_px[c] / max(tot_tp, 1)
        log(f"  {c:<16} TP {100*tp_s:>5.1f}%（FP {100*fp_px[c]/max(tot_fp,1):>5.1f}%）")

    log("")
    log("# 限定（强制）：单 seed(seed-0) / HSV 启发式非地物真值 / 水体判据 V>=110 低估深色水体")
    log("# 本分析不改主表数字；不得据此调参；不得作'误检无妨/非方法问题/申报书前提被证实或证伪'之论据")

    # 落盘
    with open(RES_DIR / "fpr_composition.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "contribution_to_macro_FPR", "share_of_macro_FPR",
                    "pooled_FP_pixel_share", "background_share", "lift_vs_background",
                    "TP_pixel_share"])
        for c in CLASSES:
            fp_s = fp_px[c] / max(tot_fp, 1)
            bg_s = bg_px[c] / max(tot_bg, 1)
            w.writerow([c, f"{contrib[c]/N:.6f}", f"{(contrib[c]/N)/macro_fpr:.6f}",
                        f"{fp_s:.6f}", f"{bg_s:.6f}",
                        f"{fp_s/bg_s:.4f}" if bg_s > 0 else "nan",
                        f"{tp_px[c]/max(tot_tp,1):.6f}"])
    with open(RES_DIR / "fpr_composition_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name"] + [f"FP_{c}" for c in CLASSES])
        w.writeheader(); w.writerows(rows)

    log("")
    log(f"汇总 -> {RES_DIR / 'fpr_composition.csv'}")
    log(f"逐图 -> {RES_DIR / 'fpr_composition_per_image.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
