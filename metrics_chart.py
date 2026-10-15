"""
生成指标对比柱状图（IoU / F1），用于结题报告和 PPT。

⚠️ 口径纪律（重要）：
    左panel = 无泄漏划分（本项目主结果，92 张 / 21 独立场景，best-val）
    右panel = 官方划分（含同源泄漏，仅作历史存档）
两个 panel **不可横向比较**，故分panel呈现，并在图注中标注。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ── 左：无泄漏划分主结果（三方法同一划分重训、best-val 口径）──
main = [
    ("SegNet\n(全监督)", 0.740, 0.847),
    ("U-Net\n(全监督)†", 0.741, 0.846),
    ("LoRA微调\nSAM(本项目)", 0.788, 0.880),
]
# ── 右：官方划分（含同源泄漏，旧 run，仅存档）──
legacy = [
    ("Otsu", 0.424, 0.581),
    ("SAM-AMG\n(零样本)", 0.302, 0.442),
    ("增强\n预处理", 0.512, 0.662),
    ("冻结微调\nSAM", 0.717, 0.833),
    ("SegNet\n(全监督)‡", 0.780, 0.875),
    ("U-Net\n(全监督)‡", 0.805, 0.891),
    ("LoRA微调\nSAM(本项目)", 0.839, 0.912),
]

BLUE, ORANGE = "#2E74B5", "#DD7B3B"
FAINT_BLUE, FAINT_ORANGE = "#9DB4D0", "#E8C39E"


def draw(ax, data, title, highlight_last=True):
    names = [d[0] for d in data]
    iou = [d[1] for d in data]
    f1 = [d[2] for d in data]
    x = np.arange(len(names))
    w = 0.38
    n = len(names)
    ci = [FAINT_BLUE] * n
    cf = [FAINT_ORANGE] * n
    if highlight_last:
        ci[-1], cf[-1] = BLUE, ORANGE
    b1 = ax.bar(x - w / 2, iou, w, label="IoU", color=ci)
    b2 = ax.bar(x + w / 2, f1, w, label="F1", color=cf)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("数值", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))
draw(ax1, main, "无泄漏划分（主结果）\n92 张 / 21 独立场景，best-val", highlight_last=True)
draw(ax2, legacy, "官方划分（含同源泄漏，仅存档）\n! 禁止与左图横比", highlight_last=True)

fig.text(0.5, 0.005,
         "† U-Net 为训练 58/60 中断后第 50 轮 best-val；‡ 旧 run（泄漏 + 增强 bug）"
         "　|　单 seed、未做显著性检验，领先幅度小于逐图标准差，仅表述为“数值上高于”",
         ha="center", fontsize=9, color="#555555")

plt.tight_layout(rect=(0, 0.045, 1, 1))
out = "results/metrics_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("已保存:", out)
