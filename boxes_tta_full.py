"""
框提示 + TTA 在全量 51 张上的最终测试。
"""
import glob
from pathlib import Path

import env_setup
import cv2
import numpy as np

from eval_shadow import compute_metrics
from methods import mask_enhanced
from prompt_experiment import predict_box, component_boxes, load_sam

BASE = Path(__file__).resolve().parent
AISD_ROOT = BASE / "AISD" / "AISD"


def boxes_tta(sam, img_rgb, img_bgr, mode):
    h, w = img_rgb.shape[:2]
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        return np.zeros((h, w), bool)
    c1 = np.zeros((h, w), bool)
    for b in boxes:
        c1 |= predict_box(sam, img_rgb, b)
    img_flip = img_rgb[:, ::-1, :].copy()
    boxes_flip = [[w - 1 - x2, y1, w - 1 - x1, y2] for (x1, y1, x2, y2) in boxes]
    c2 = np.zeros((h, w), bool)
    for b in boxes_flip:
        c2 |= predict_box(sam, img_flip, b)
    c2 = c2[:, ::-1]
    return (c1 & c2) if mode == "inter" else (c1 | c2)


def main():
    sam = load_sam()
    imgs = sorted((AISD_ROOT / "Test51" / "shadow").glob("*.tif"))
    gt_dir = AISD_ROOT / "Test51" / "mask"
    for mode in ["inter", "union"]:
        ious = []
        for ip in imgs:
            img_bgr = cv2.imread(ip)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(gt_dir / Path(ip).name), cv2.IMREAD_GRAYSCALE)
            pred = boxes_tta(sam, img_rgb, img_bgr, mode)
            ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
        print(f"boxes_tta_{mode} (全量51张): {np.mean(ious):.4f}", flush=True)


if __name__ == "__main__":
    main()
