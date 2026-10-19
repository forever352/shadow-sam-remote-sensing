# -*- coding: utf-8 -*-
"""生成单面板主结果对比图（无泄漏划分，3 seed 均值±标准差）。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
SRC = BASE / "results" / "cnn_multiseed_summary.csv"
OUT = BASE / "results" / "figures" / "main_result_clean.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

order = ["LoRA-SAM", "U-Net", "SegNet", "Mask R-CNN"]
label = {"LoRA-SAM": "LoRA-SAM\n(本项目)", "U-Net": "U-Net", "SegNet": "SegNet",
         "Mask R-CNN": "Mask R-CNN"}
rows = {r["method"]: r for r in csv.DictReader(open(SRC, encoding="utf-8"))}

iou_mean = np.array([float(rows[m]["IoU_mean"]) for m in order])
iou_sd = np.array([float(rows[m]["IoU_seed_sd"]) for m in order])
ber_mean = np.array([float(rows[m]["BER_mean"]) for m in order])
ber_sd = np.array([float(rows[m]["BER_seed_sd"]) for m in order])
x = np.arange(len(order))
colors = ["#2E74B5", "#B4C7E7", "#B4C7E7", "#B4C7E7"]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.17, wspace=0.28)

for ax, mean, sd, title, ylim, fmt in [
    (axes[0], iou_mean, iou_sd, "IoU（越高越好）", (0.68, 0.84), "{:.3f}"),
    (axes[1], ber_mean, ber_sd, "BER（越低越好）", (0.0, 0.12), "{:.3f}"),
]:
    bars = ax.bar(x, mean, yerr=sd, capsize=4, color=colors, edgecolor="white",
                  linewidth=0.8, error_kw={"elinewidth": 1.1, "ecolor": "#555555"})
    for bar, m in zip(bars, mean):
        ax.text(bar.get_x() + bar.get_width() / 2, m + (ylim[1] - ylim[0]) * 0.02,
                fmt.format(m), ha="center", va="bottom", fontsize=10, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([label[m] for m in order], fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#1F3B63")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("无泄漏划分主结果：92 张 / 21 独立场景，3 seed 均值 ± 样本标准差",
             x=0.08, ha="left", fontsize=14, fontweight="bold", color="#1F3B63")
fig.text(0.08, 0.045, "注：错误棒为 3 seed 样本标准差；只说“数值上高于”，不宣称 SOTA/显著超越。",
         fontsize=10, color="#555555")
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"written {OUT}")
