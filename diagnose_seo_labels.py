# -*- coding: utf-8 -*-
"""S-EO OMA 阴影掩膜轴 vs 太阳方位角：两种诊断口径。

口径 A（逐图）：每张图所有连通域（面积 >= min_area）最小外接矩形长轴 -> 二倍角
圆形均值 -> 与太阳方位角取 180° 周期轴差 -> 对 42 张图统计。
口径 B（逐连通域）：不聚合，直接对每个连通域的长轴与太阳方位角取轴差，
报告连通域级分布。

注意：两种口径回答的问题不同；A 描述“整图主导轴”，B 描述“单个阴影块”。
掩膜极性：1=受光，0=阴影（README 已核验）。

用法：
    python diagnose_seo_labels.py
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import env_setup  # noqa: F401
import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "S-EO_subset"
MANIFEST = DATA / "manifest_all.csv"
OUT_DIR = BASE / "results" / "seo_direction_pilot"
LOG_DIR = BASE / "logs"
MIN_AREA = 200


def axis_diff(axis_deg: float, sun_deg: float) -> float:
    return abs(((axis_deg - sun_deg + 90.0) % 180.0) - 90.0)


def per_image_axis(raw: np.ndarray) -> tuple[float | None, list[float]]:
    m = (raw == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    axes: list[float] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (w, h), ang = cv2.minAreaRect(pts)
        if w < h:
            ang += 90.0  # 让 ang 指向长边
        axes.append(ang % 180.0)
    if not axes:
        return None, []
    a = np.deg2rad(np.array(axes) * 2.0)
    mean_axis = (math.degrees(math.atan2(float(np.sin(a).mean()),
                                         float(np.cos(a).mean()))) / 2.0) % 180.0
    return mean_axis, axes


def main() -> None:
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["domain"] == "OMA"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"diagnose_seo_labels_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# S-EO OMA 掩膜轴 vs 太阳方位角")
    log(f"# time={time.strftime('%Y-%m-%d %H:%M:%S')} min_area={MIN_AREA}")
    per_image = []
    all_comp_diffs = []
    all_comp_rows = []
    for r in rows:
        p = DATA / r["shadow_path"].replace("\\", "/")
        raw = np.array(Image.open(p))
        sun = float(r["sun_azimuth_deg"])
        mean_axis, axes = per_image_axis(raw)
        if mean_axis is None:
            continue
        d = axis_diff(mean_axis, sun)
        per_image.append({
            "image_id": r["image_id"],
            "date": r["acquisition_date"],
            "sun_azimuth_deg": sun,
            "mean_axis_deg": mean_axis,
            "axis_diff_deg": d,
            "n_components": len(axes),
        })
        for a in axes:
            comp_d = axis_diff(a, sun)
            all_comp_diffs.append(comp_d)
            all_comp_rows.append({
                "image_id": r["image_id"],
                "sun_azimuth_deg": sun,
                "component_axis_deg": a,
                "axis_diff_deg": comp_d,
            })

    def stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(len(a)),
            "mean_deg": float(a.mean()),
            "median_deg": float(np.median(a)),
            "std_deg": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "iqr_deg": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "max_deg": float(a.max()),
            "frac_within_15": float((a < 15).mean()),
            "frac_within_30": float((a < 30).mean()),
        }

    sa = stats([r["axis_diff_deg"] for r in per_image])
    sb = stats(all_comp_diffs)
    log("口径 A（逐图圆形均值）: " + str(sa))
    log("口径 B（逐连通域）: " + str(sb))

    with open(OUT_DIR / "label_axis_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    with open(OUT_DIR / "label_axis_per_component.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_comp_rows[0].keys()))
        w.writeheader()
        w.writerows(all_comp_rows)
    with open(OUT_DIR / "label_axis_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["definition", "n", "mean_deg", "median_deg", "std_deg",
                                           "iqr_deg", "max_deg", "frac_within_15", "frac_within_30"])
        w.writeheader()
        w.writerow({"definition": "A_per_image_circular_mean", **sa})
        w.writerow({"definition": "B_per_component", **sb})
    log(f"log -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
