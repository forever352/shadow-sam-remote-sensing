"""LoRA 微调模型的完整评估：框提示 + Test51，输出六项指标。"""
import glob
from pathlib import Path

import env_setup
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_enhanced
from direction_experiment import component_boxes, predict_box, encode_image
from train_adapter_v4 import apply_lora

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
LORA = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_last.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
apply_lora(sam)
sam.load_state_dict(torch.load(LORA, map_location=device))
sam.to(device)
sam.eval()

imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
gt_dir = AISD / "Test51" / "mask"

allm = []
for ip in imgs:
    img_bgr = cv2.imread(ip)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(gt_dir / Path(ip).name), cv2.IMREAD_GRAYSCALE)
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for b in boxes:
        combined |= predict_box(sam, img_emb, h, w, b)
    allm.append(compute_metrics((combined * 255).astype(np.uint8), gt))

avg = {k: float(np.mean([m[k] for m in allm])) for k in ["IoU", "BER", "F1", "OA", "Precision", "Recall"]}
print("=== LoRA 微调 + 框提示，Test51 全量 51 张 ===")
for k, v in avg.items():
    print(f"{k:10s} {v:.4f}")
