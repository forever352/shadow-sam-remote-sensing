"""
生成指标对比柱状图（IoU / F1），用于结题报告和 PPT。

⚠️ 口径纪律（重要）：
    左panel = 无泄漏划分（本项目主结果，92 张 / 21 独立场景，best-val）
    右panel = AISD 自带划分（含同源泄漏，仅作历史存档）
两个 panel **不可横向比较**，故分panel呈现，并在图注中标注。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ── 左：无泄漏划分主结果（同一划分重训、best-val 口径）──
# 数据源：results/{segnet,unet,maskrcnn_leakfree_res800,lora_leakfree*}_test.csv
# LoRA 为 3 个随机种子的均值（0.7878 / 0.8075 / 0.8016），其余三基线均为单 seed=0
main = [
    ("Mask R-CNN\n(全监督)†", 0.7382, 0.8481),
    ("SegNet\n(全监督)", 0.7403, 0.8466),
    ("U-Net\n(全监督)", 0.7525, 0.8552),
    ("LoRA微调\nSAM(本项目)", 0.7990, 0.8869),
]
# ── 右：AISD 自带划分（含同源泄漏，禁与左图横比）──
# 前 4 项为**旧 run**（原聚合 CSV 已不在库，数字仅存于 README §3.2 / 结题报告 §3.2 表注）；后 3 项为该划分下的完整重跑
# 注意：此处**不用 ‡ 记号** —— README 里 ‡ 指的是"CNN 增强 bug"，与"旧 run"不是同一组，避免同符号两义
legacy = [
    ("Otsu\n(旧run)", 0.424, 0.581),
    ("SAM-AMG\n(零样本,旧run)", 0.302, 0.442),
    ("增强\n预处理(旧run)", 0.512, 0.662),
    ("冻结微调\nSAM(旧run)", 0.717, 0.833),
    ("SegNet\n(全监督)", 0.8089, 0.8939),
    ("U-Net\n(全监督)", 0.8153, 0.8971),
    ("LoRA微调\nSAM(本项目)", 0.8393, 0.9122),
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
draw(ax2, legacy, "AISD 自带划分（含同源泄漏，禁横比）\n51 张，仅作历史对齐", highlight_last=True)

fig.text(0.5, 0.005,
         "† Mask R-CNN 分辨率不对称仅“部分缓解”（输入仍固定 512）；"
         "右图标注“(旧run)”的四项为早期跑数，原聚合 CSV 已不在库，仅作历史对齐"
         "　|　本项目 LoRA 为 3 seed 均值（0.7878/0.8075/0.8016），三基线为单 seed；"
         "未做显著性检验，仅表述为“数值上高于”",
         ha="center", fontsize=9, color="#555555")

plt.tight_layout(rect=(0, 0.045, 1, 1))
out = "results/metrics_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("已保存:", out)
