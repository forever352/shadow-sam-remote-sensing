"""计算各方法在 Test51 上的均值±标准差（用于报告严谨性）。

口径提示（2026-10-18）：本脚本为历史 Test51 口径（含 42/43 同源泄漏），
混合了旧 LoRA_last / U-Net / SegNet 权重与 frozen leakfree 对照权重，
不是干净对比；输出不得标成无泄漏 0.5621 或历史 v2 0.717。
"""
import glob
from pathlib import Path

import env_setup
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_otsu, mask_enhanced
from direction_experiment import component_boxes, predict_box, encode_image
from train_adapter_v4 import apply_lora
from train_unet import UNet
from train_segnet import SegNet

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 加载模型 ----
sam_lora = sam_model_registry["vit_b"](checkpoint=str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth"))
apply_lora(sam_lora)
sam_lora.load_state_dict(torch.load(str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_last.pth"), map_location=device))
sam_lora.to(device); sam_lora.eval()

sam_frozen = sam_model_registry["vit_b"](checkpoint=str(BASE / "checkpoints" / "sam_vit_b_shadow_frozen_leakfree.pth")).to(device)
sam_frozen.eval()

unet = UNet().to(device)
unet.load_state_dict(torch.load(str(BASE / "checkpoints" / "unet_shadow.pth"), map_location=device))
unet.eval()

segnet = SegNet().to(device)
segnet.load_state_dict(torch.load(str(BASE / "checkpoints" / "segnet_shadow.pth"), map_location=device))
segnet.eval()


def pred_sam_box(sam, img_rgb, img_bgr):
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for b in boxes:
        combined |= predict_box(sam, img_emb, h, w, b)
    return combined


def pred_cnn(model, img_bgr):
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)[None].to(device) / 255.0
    logits = model(x)[0, 0]
    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return pred > 0


METHODS = {
    "Otsu": lambda b, r: mask_otsu(b) > 0,
    "增强预处理": lambda b, r: mask_enhanced(b) > 0,
    "冻结编码器(无泄漏权重)": lambda b, r: pred_sam_box(sam_frozen, r, b),
    "SegNet": lambda b, r: pred_cnn(segnet, b),
    "U-Net": lambda b, r: pred_cnn(unet, b),
    "LoRA(本项目)": lambda b, r: pred_sam_box(sam_lora, r, b),
}

imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
gt_dir = AISD / "Test51" / "mask"

acc = {m: {k: [] for k in ["IoU", "BER", "F1"]} for m in METHODS}
for ip in imgs:
    img_bgr = cv2.imread(ip)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(gt_dir / Path(ip).name), cv2.IMREAD_GRAYSCALE)
    for mname, fn in METHODS.items():
        pred = fn(img_bgr, img_rgb)
        mm = compute_metrics((pred * 255).astype(np.uint8), gt)
        for k in ["IoU", "BER", "F1"]:
            acc[mname][k].append(mm[k])

print("=== 均值 ± 标准差（Test51 全量 51 张）===")
for mname in METHODS:
    line = f"{mname:12s}"
    for k in ["IoU", "BER", "F1"]:
        arr = np.array(acc[mname][k])
        line += f"  {k}={arr.mean():.4f}±{arr.std():.4f}"
    print(line)
