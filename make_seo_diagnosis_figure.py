# -*- coding: utf-8 -*-
"""生成第 15 页方向诊断双面板图（旧标签 vs S-EO 预注册试点）。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "seo_direction_pilot" / "direction_diagnostic.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# 口径 B：逐连通域轴差
comp = [float(r["axis_diff_deg"]) for r in csv.DictReader(
    open(BASE / "results/seo_direction_pilot/label_axis_per_component.csv", encoding="utf-8"))]
per_img = [float(r["axis_diff_deg"]) for r in csv.DictReader(
    open(BASE / "results/seo_direction_pilot/label_axis_per_image.csv", encoding="utf-8"))]
comp = np.asarray(comp)
per_img = np.asarray(per_img)

fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.6),
                         gridspec_kw={"height_ratios": [1.0, 1.15]})
fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.12, hspace=0.55)

# (a) 旧标签轴差分布
ax = axes[0]
bins = np.arange(0, 95, 5)
ax.hist(comp, bins=bins, color="#9DC3E6", edgecolor="white", linewidth=0.6)
ax.axvline(float(np.median(comp)), color="#A63A2E", linestyle="--", linewidth=1.8,
           label=f"逐连通域中位 {np.median(comp):.1f}°")
ax.axvline(float(np.median(per_img)), color="#1F3B63", linestyle=":", linewidth=1.8,
           label=f"逐图中位 {np.median(per_img):.1f}°")
ax.set_xlim(0, 90)
ax.set_xlabel("掩膜长轴与太阳方位角轴差（°）")
ax.set_ylabel("连通域数")
ax.set_title("(a) 旧标签：掩膜长轴 vs 真实太阳方位角", loc="left",
             fontsize=12.5, fontweight="bold", color="#1F3B63")
ax.legend(fontsize=9, frameon=False, loc="upper right")
frac = float((comp < 15).mean())
ax.text(0.98, 0.57, f"仅 {frac * 100:.1f}% 在 15° 内", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color="#A63A2E")
ax.grid(axis="y", alpha=0.25)

# (b) S-EO 对比
ax = axes[1]
settings = ["OMA\n同 AOI", "JAX\n留一 AOI", "OMA→JAX\n支撑内", "JAX→OMA\n支撑内"]
model = [2.99, 11.92, 8.93, 17.48]
const = [5.75, 13.67, 3.69, 8.83]
elev = [3.96, 4.16, 3.70, 3.54]
x = np.arange(len(settings))
w = 0.26
b1 = ax.bar(x - w, model, w, label="模型", color="#2E74B5")
b2 = ax.bar(x, const, w, label="常数基线", color="#A6A6A6")
b3 = ax.bar(x + w, elev, w, label="高程-only oracle", color="#C55A11")
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=10)
ax.set_ylabel("平均绝对角误差（°）")
ax.set_ylim(0, 21.5)
ax.set_title("(b) S-EO：模型 vs 常数 vs 高程 oracle", loc="left",
             fontsize=12.5, fontweight="bold", color="#1F3B63")
ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper left")
ax.grid(axis="y", alpha=0.25)

fig.text(0.01, 0.015,
         "注：OMA 日期块 p=0.186 未过主门槛，只能作探索性/不确定；JAX 为 4 折描述性；跨域支撑内均无阳性。",
         fontsize=9, color="#555555")
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"written {OUT}")
