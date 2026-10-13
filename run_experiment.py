"""
批量实验：在 AISD 测试集上对比多种阴影检测方法。
用法:
    python run_experiment.py [--split Test51] [--limit N]
输出:
    results/summary.csv     各方法平均指标
    results/per_image.csv   每张图 × 各方法指标
"""
import os
import sys
import time
import csv
import argparse
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from eval_shadow import compute_metrics
from methods import mask_otsu, mask_c3i, mask_sam_amg, mask_enhanced, mask_ours


BASE_DIR = Path(__file__).resolve().parent
SAM_CHECKPOINT = str(BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth")
AISD_ROOT = BASE_DIR / "AISD" / "AISD"

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

# 方法名 -> 是否需要 amg 参数
METHODS = [
    ("Otsu", False),
    ("C3I", False),
    ("SAM-AMG", True),
    ("Enhanced", False),
    ("Ours", False),
]


def load_amg(device):
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    amg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    return amg


def run_method(name, img, amg):
    if name == "Otsu":
        return mask_otsu(img)
    if name == "C3I":
        return mask_c3i(img)
    if name == "SAM-AMG":
        return mask_sam_amg(img, amg)
    if name == "Enhanced":
        return mask_enhanced(img)
    if name == "Ours":
        return mask_ours(img)
    raise ValueError(f"未知方法: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="Test51")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0=全部")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    img_dir = AISD_ROOT / args.split / "shadow"
    gt_dir = AISD_ROOT / args.split / "mask"
    imgs = sorted(img_dir.glob("*.tif"))
    if args.limit > 0:
        imgs = imgs[: args.limit]
    print(f"测试图片数: {len(imgs)}")

    amg = load_amg(device)

    method_names = [m for m, _ in METHODS]
    results = {m: [] for m in method_names}
    rows = []

    for idx, img_path in enumerate(imgs):
        name = img_path.stem
        img = cv2.imread(str(img_path))
        gt = cv2.imread(str(gt_dir / img_path.name), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            print(f"⚠ 读取失败: {name}")
            continue

        row = {"image": name}
        for mname, _ in METHODS:
            t0 = time.time()
            pred = run_method(mname, img, amg)
            dt = time.time() - t0
            m = compute_metrics(pred, gt)
            results[mname].append(m)
            for k in METRIC_KEYS:
                row[f"{mname}_{k}"] = round(m[k], 4)
            row[f"{mname}_s"] = round(dt, 2)
        rows.append(row)
        print(f"[{idx + 1}/{len(imgs)}] {name} 完成", flush=True)

    print("\n========== 平均指标汇总 ==========")
    header = "方法".ljust(10) + "  ".join(k.rjust(9) for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    summary = {}
    for mname in method_names:
        avg = {k: float(np.mean([r[k] for r in results[mname]])) for k in METRIC_KEYS}
        summary[mname] = avg
        print(mname.ljust(10) + "  ".join(f"{avg[k]:9.4f}" for k in METRIC_KEYS))

    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method"] + METRIC_KEYS)
        for mname in method_names:
            w.writerow([mname] + [summary[mname][k] for k in METRIC_KEYS])
    if rows:
        with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
