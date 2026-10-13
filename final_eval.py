"""
最终方法评估：框提示（增强 Otsu 连通域包围框）+ 微调 SAM，全量 51 张完整指标。
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


def main():
    sam = load_sam()
    imgs = sorted((AISD_ROOT / "Test51" / "shadow").glob("*.tif"))
    gt_dir = AISD_ROOT / "Test51" / "mask"
    metrics = []
    for ip in imgs:
        img_bgr = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_dir / Path(ip).name), cv2.IMREAD_GRAYSCALE)
        em = mask_enhanced(img_bgr)
        boxes = component_boxes(em)
        combined = np.zeros(img_bgr.shape[:2], bool)
        for b in boxes:
            combined |= predict_box(sam, img_rgb, b)
        metrics.append(compute_metrics((combined * 255).astype(np.uint8), gt))

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in ["IoU", "BER", "F1", "OA", "Precision", "Recall"]}
    print("=== 框提示 + 微调SAM 全量51张 ===", flush=True)
    for k, v in avg.items():
        print(f"{k:10s} {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
