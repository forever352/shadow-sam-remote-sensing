"""
单张影像阴影检测演示：加载微调后的 SAM，全自动检测阴影并保存结果图。
用法:
    python demo.py                     # 检测一张 AISD 标准测试图（默认 austin13_sub2.tif）
    python demo.py 图片路径            # 检测任意图片
"""
import sys
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from prompt_experiment import component_boxes, predict_box  # 框提示
from methods import mask_enhanced

BASE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMG = BASE / "AISD" / "AISD" / "Test51" / "shadow" / "austin13_sub2.tif"


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_IMG)
    ckpt = str(BASE / "checkpoints" / "sam_vit_b_shadow_v2.pth")

    print(f"加载微调模型 ({DEVICE})...")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(DEVICE)
    sam.eval()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"找不到图片: {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. 增强物理粗提取 -> 2. 框提示 -> 3. 微调 SAM 分割
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        print("未检测到明显阴影区域")
        return

    pred = np.zeros(img_bgr.shape[:2], bool)
    for b in boxes:
        pred |= predict_box(sam, img_rgb, b)  # 逐个框提示，合并结果

    # 绿色叠加显示
    overlay = img_bgr.copy()
    overlay[pred] = [0, 255, 0]
    result = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

    out = str(BASE / "demo_result.png")
    cv2.imwrite(out, result)
    ratio = 100.0 * pred.sum() / pred.size
    print(f"检测完成：阴影占图像 {ratio:.2f}%（绿色），结果已保存到 {out}")


if __name__ == "__main__":
    main()
