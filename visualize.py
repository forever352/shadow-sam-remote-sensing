"""
生成对比可视化图：原图 | 真值 | Otsu | 增强预处理 | 零样本SAM | 微调SAM(本项目)
用于结题报告和 PPT。输出到 results/figures/。
"""
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry

from methods import mask_otsu, mask_enhanced
from prompt_experiment import component_boxes, predict_box

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
ZERO_CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
FT_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v2.pth")
AISD_ROOT = BASE / "AISD" / "AISD"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sam(ckpt):
    sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(DEVICE)
    sam.eval()
    return sam


def auto_mask(sam, img_bgr):
    """增强 Otsu 连通域包围框提示 + SAM 分割。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        return np.zeros(img_bgr.shape[:2], bool)
    combined = np.zeros(img_bgr.shape[:2], bool)
    for b in boxes:
        combined |= predict_box(sam, img_rgb, b)
    return combined


def main():
    sam_zero = load_sam(ZERO_CKPT)
    sam_ft = load_sam(FT_CKPT)

    img_dir = AISD_ROOT / "Test51" / "shadow"
    gt_dir = AISD_ROOT / "Test51" / "mask"
    imgs = sorted(img_dir.glob("*.tif"))
    samples = imgs[:2] + imgs[25:27]  # 4 张代表性样本

    panels = ["原图", "真值", "Otsu", "增强预处理", "零样本SAM", "微调SAM(本项目)"]

    out_dir = BASE / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for imp in samples:
        img_bgr = cv2.imread(str(imp))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_dir / imp.name), cv2.IMREAD_GRAYSCALE)

        masks = {
            "Otsu": mask_otsu(img_bgr) > 0,
            "增强预处理": mask_enhanced(img_bgr) > 0,
            "零样本SAM": auto_mask(sam_zero, img_bgr),
            "微调SAM(本项目)": auto_mask(sam_ft, img_bgr),
        }

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
        axes[0].imshow(img_rgb); axes[0].set_title("原图", fontsize=12); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("真值", fontsize=12); axes[1].axis("off")
        for ax, key in zip(axes[2:], ["Otsu", "增强预处理", "零样本SAM", "微调SAM(本项目)"]):
            overlay = np.zeros_like(img_rgb)
            overlay[masks[key]] = [0, 255, 0]
            ax.imshow(cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0))
            ax.set_title(key, fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        out = out_dir / f"{imp.stem}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("已保存:", out)


if __name__ == "__main__":
    main()
