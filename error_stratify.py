"""误差结构分析：阴影面积分箱 / 边界带 / 原始场景。

输入
----
`results/prompt_source_masks/{source}/*.png`（由 prompt_source_eval.py 生成）
+ AISD 无泄漏 test 的 GT 掩膜。

预注册分箱（跑之前定死，事后不增删）
------------------------------------
- 连通域面积（GT component，像素）：<256 / 256-1024 / 1024-4096 / >=4096
- 边界带半径（像素）：1 / 3 / 5（定义为距形态学梯度边界 <= r 的像素，含边界两侧）
- 场景：按原图基名 `*_subN` 聚合；场景内先对子图取均值（macro），并记录 n_sub

两种口径（不得混写）
--------------------
- macro：先逐图/逐连通域算指标再平均；
- micro：先汇总 TP/FP/FN 再算指标。
IoU 对口径敏感；本脚本的连通域表同时给出 macro 与 micro，场景表为 macro。

输出
----
results/error_component_size_{tag}.csv
results/error_boundary_band_{tag}.csv
results/error_by_scene_{tag}.csv
logs/error_stratify_{ts}.log

说明
----
- 只做描述性统计；不做显著性升级，不宣称"细小阴影漏检"已被单独量化。
- 逐连通域的 precision 以"该连通域 bbox 内的预测像素"为分母，避免把图内其它
  区域的预测算进该连通域；这是诊断口径，不是主表口径。
- 边界带指标是带内像素的 precision/recall/F1，不是整图 IoU/BER。

用法
----
python error_stratify.py --source enhanced
"""
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval_shadow import compute_metrics
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
MASK_DIR = RES_DIR / "prompt_source_masks"

AREA_BINS = [(0, 255, "<256"), (256, 1023, "[256,1024)"),
             (1024, 4095, "[1024,4096)"), (4096, 10**12, ">=4096")]
BAND_RADII = [1, 3, 5]
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]


def base_of(name):
    m = re.match(r"^(.*?)_sub\d+", Path(name).stem)
    return m.group(1) if m else Path(name).stem


def test_split():
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    _, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    return test_files


def comp_stats(gt_bin, pred_bin):
    """逐 GT 连通域的 recall/precision/IoU（bbox 内）。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_bin.astype(np.uint8))
    out = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels[y:y + h, x:x + w] == i
        pred = pred_bin[y:y + h, x:x + w]
        tp = int(np.logical_and(comp, pred).sum())
        fp = int(np.logical_and(~comp, pred).sum())
        fn = int(area - tp)
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        out.append((int(area), rec, prec, iou, tp, fn, fp))
    return out


def boundary_band(gt_bin, radius):
    """返回距离 GT 边界 <= radius 的像素掩膜（边界用形态学梯度定义）。"""
    k = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(gt_bin.astype(np.uint8), cv2.MORPH_GRADIENT, k)
    if boundary.sum() == 0:
        return np.zeros_like(gt_bin, bool)
    dist = cv2.distanceTransform((boundary == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return dist <= radius


def band_metrics(gt_bin, pred_bin, band):
    if band.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    g = gt_bin[band]
    p = pred_bin[band]
    tp = int(np.logical_and(g, p).sum())
    fp = int(np.logical_and(~g, p).sum())
    fn = int(np.logical_and(g, ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="enhanced", help="prompt_source_masks 下的来源目录")
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许少于 92 张（仅调试；正式分析不得使用）")
    args = ap.parse_args()

    RES_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"error_stratify_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    files = test_split()
    mask_dir = MASK_DIR / args.source
    log(f"# 误差结构分析 source={args.source}")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# 测试: {len(files)} 张 / 掩膜目录 {mask_dir}")

    comp_acc = {label: [] for _, _, label in AREA_BINS}
    comp_imgs = {label: set() for _, _, label in AREA_BINS}
    band_acc = {r: [] for r in BAND_RADII}
    band_empty = {r: 0 for r in BAND_RADII}
    scene = defaultdict(list)
    n_ok = 0
    for ip in files:
        name = Path(ip).name
        stem = Path(name).stem
        gt = cv2.imread(str(AISD / "Train412" / "mask" / name), cv2.IMREAD_GRAYSCALE)
        mp = mask_dir / (stem + ".png")
        if gt is None or not mp.exists():
            log(f"  [跳过] {name} (gt={gt is not None}, mask={mp.exists()})")
            continue
        pred = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            log(f"  [跳过] {name} (mask read failed)")
            continue
        gt_bin = gt > 0
        pred_bin = pred > 0
        m = compute_metrics(pred, gt)
        scene[base_of(name)].append(m)
        n_ok += 1
        for area, rec, prec, iou, tp, fn, fp in comp_stats(gt_bin, pred_bin):
            for lo, hi, label in AREA_BINS:
                if lo <= area <= hi:
                    comp_acc[label].append((area, rec, prec, iou, tp, fn, fp))
                    comp_imgs[label].add(name)
                    break
        for r in BAND_RADII:
            band = boundary_band(gt_bin, r)
            if band.sum() == 0:
                band_empty[r] += 1
            bm = band_metrics(gt_bin, pred_bin, band)
            bm.update({"n_pixels": int(band.sum()), "name": name})
            band_acc[r].append(bm)
        if n_ok % 10 == 0:
            log(f"  已处理 {n_ok}/{len(files)}")

    # 硬断言：正式分析必须 92 张齐全，且 test 名称与实验 1 一致
    if not args.allow_partial:
        if len(files) != 92 or n_ok != 92:
            missing = [Path(p).name for p in files
                       if not (mask_dir / (Path(p).stem + ".png")).exists()]
            log(f"[中止] 有效结果 {n_ok}/{len(files)}，缺失 {len(missing)} 张")
            for m in missing[:20]:
                log(f"  缺失: {m}")
            logf.close()
            raise SystemExit(2)
        ref_p = RES_DIR / "prompt_source_test_per_image.csv"
        if not ref_p.exists():
            log(f"[中止] 缺少 {ref_p}，无法核对 test 名称")
            logf.close()
            raise SystemExit(3)
        with open(ref_p, newline="", encoding="utf-8-sig") as rf:
            ref_names = {r["name"] for r in csv.DictReader(rf) if r["source"] == args.source}
        if {Path(p).name for p in files} != ref_names:
            log("[中止] test 名称集合与实验 1 逐图 CSV 不一致")
            logf.close()
            raise SystemExit(4)
        log("[断言] 92/92 张齐全；test 名称与实验 1 一致")

    # 连通域分箱
    out1 = RES_DIR / f"error_component_size_{args.source}.csv"
    with open(out1, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "bin", "n_components", "n_images",
                    "n_gt_component_pixels",
                    "recall_macro", "precision_macro", "iou_macro",
                    "recall_micro", "precision_micro", "iou_micro",
                    "tp_sum", "fn_sum", "fp_sum"])
        for _, _, label in AREA_BINS:
            xs = comp_acc[label]
            if not xs:
                w.writerow([args.source, label, 0, 0, 0, "", "", "", "", "", "", 0, 0, 0])
                continue
            rec = np.array([x[1] for x in xs], float)
            pre = np.array([x[2] for x in xs], float)
            iou = np.array([x[3] for x in xs], float)
            tp = int(sum(x[4] for x in xs))
            fn = int(sum(x[5] for x in xs))
            fp = int(sum(x[6] for x in xs))
            rec_mi = tp / (tp + fn) if (tp + fn) else float("nan")
            pre_mi = tp / (tp + fp) if (tp + fp) else float("nan")
            iou_mi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
            w.writerow([args.source, label, len(xs), len(comp_imgs[label]),
                        int(sum(x[0] for x in xs)),
                        f"{np.nanmean(rec):.6f}", f"{np.nanmean(pre):.6f}",
                        f"{np.nanmean(iou):.6f}",
                        f"{rec_mi:.6f}", f"{pre_mi:.6f}", f"{iou_mi:.6f}",
                        tp, fn, fp])

    # 边界带
    out2 = RES_DIR / f"error_boundary_band_{args.source}.csv"
    with open(out2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "radius_px", "n_images", "n_nonempty_bands",
                    "mean_band_pixels", "precision", "recall", "f1",
                    "n_empty_band_images"])
        for r in BAND_RADII:
            xs = band_acc[r]
            if not xs:
                w.writerow([args.source, r, 0, 0, 0, "", "", "", band_empty[r]])
                continue
            w.writerow([args.source, r, len(xs), len(xs) - band_empty[r],
                        f"{np.mean([x['n_pixels'] for x in xs]):.1f}",
                        f"{np.nanmean([x['precision'] for x in xs]):.6f}",
                        f"{np.nanmean([x['recall'] for x in xs]):.6f}",
                        f"{np.nanmean([x['f1'] for x in xs]):.6f}",
                        band_empty[r]])

    # 场景级
    out3 = RES_DIR / f"error_by_scene_{args.source}.csv"
    with open(out3, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "scene", "n_sub", "basis"] + METRIC_KEYS)
        for b in sorted(scene):
            xs = scene[b]
            w.writerow([args.source, b, len(xs), "macro_subimage_mean"] +
                       [f"{np.mean([x[k] for x in xs]):.6f}" for k in METRIC_KEYS])

    log()
    log(f"连通域分箱 -> {out1}")
    log(f"边界带     -> {out2}")
    log(f"场景级     -> {out3}")
    log(f"日志       -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
