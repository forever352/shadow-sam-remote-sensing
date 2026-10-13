"""
生成指标对比柱状图（IoU / F1），用于结题报告和 PPT。
含传统方法、零样本 SAM、增强预处理、微调 SAM 的完整对比。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 最终结果（AISD Test51 全量 51 张平均）
data = [
    ("Otsu\n(传统阈值)", 0.424, 0.581),
    ("C3I\n(阴影指数)", 0.162, 0.211),
    ("SAM-AMG\n(零样本)", 0.302, 0.442),
    ("增强预处理", 0.512, 0.662),
    ("冻结微调\nSAM", 0.717, 0.833),
    ("U-Net\n(全监督)", 0.805, 0.891),
    ("LoRA微调\nSAM(本项目)", 0.839, 0.912),
]
names = [d[0] for d in data]
iou = [d[1] for d in data]
f1 = [d[2] for d in data]

x = np.arange(len(names))
w = 0.38

fig, ax = plt.subplots(figsize=(9.5, 5.0))
colors_iou = ["#9DB4D0"] * 5 + ["#2E74B5"]   # 最后一根(本项目)高亮
colors_f1 = ["#E8C39E"] * 5 + ["#DD7B3B"]
bars1 = ax.bar(x - w/2, iou, w, label="IoU", color=colors_iou)
bars2 = ax.bar(x + w/2, f1, w, label="F1", color=colors_f1)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylabel("数值", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("AISD 测试集（51 张）各方法阴影检测指标对比", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=9)
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = "results/metrics_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("已保存:", out)
