"""LoRA 微调模型的完整评估：框提示 + Test51，输出六项指标（结果落盘，可核验）。

改动（相对旧版）：把逐图指标写 results/lora_test51_per_image.csv、汇总写
results/lora_test51.csv（含均值±标准差）、运行日志写 logs/eval_lora_*.log。
评估逻辑与旧版完全一致，仅增加落盘，保证数字可比。
"""
import csv
import glob
import time
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
OUT_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
OUT_PER_IMAGE = OUT_DIR / "lora_test51_per_image.csv"
OUT_SUMMARY = OUT_DIR / "lora_test51.csv"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
apply_lora(sam)
sam.load_state_dict(torch.load(LORA, map_location=device))
sam.to(device)
sam.eval()

imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
gt_dir = AISD / "Test51" / "mask"

OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"eval_lora_{ts}.log"
logf = open(log_path, "w", encoding="utf-8")


def log(msg=""):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


log("# LoRA 微调 + 框提示，Test51 全量评估")
log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"# 权重: {LORA}")
log(f"# 提示方式: 增强 Otsu(CLAHE+Otsu+形态学) -> 连通域框提示")
log(f"# 图像数: {len(imgs)}   设备: {device}")
log()

rows = []
allm = []
for ip in imgs:
    name = Path(ip).name
    img_bgr = cv2.imread(ip)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for b in boxes:
        combined |= predict_box(sam, img_emb, h, w, b)
    m = compute_metrics((combined * 255).astype(np.uint8), gt)
    rows.append({"name": name, **m})
    allm.append(m)
    log(f"  {name:24s} IoU={m['IoU']:.4f}  BER={m['BER']:.4f}  F1={m['F1']:.4f}")

avg = {k: float(np.mean([m[k] for m in allm])) for k in METRIC_KEYS}
std = {k: float(np.std([m[k] for m in allm])) for k in METRIC_KEYS}

# 逐图 CSV
with open(OUT_PER_IMAGE, "w", newline="", encoding="utf-8") as f:
    wtr = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
    wtr.writeheader()
    wtr.writerows(rows)

# 汇总 CSV（均值 + 标准差）
with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
    wtr = csv.writer(f)
    wtr.writerow(["method", "n_images"] + METRIC_KEYS)
    wtr.writerow(["LoRA-Finetuned-SAM(box)", len(imgs)] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
    wtr.writerow(["LoRA-Finetuned-SAM(box)_std", len(imgs)] + [f"{std[k]:.6f}" for k in METRIC_KEYS])

log()
log(f"=== 汇总（Test51 全量 {len(imgs)} 张）===")
for k in METRIC_KEYS:
    log(f"{k:10s} {avg[k]:.4f} ± {std[k]:.4f}")
log()
log(f"逐图结果 -> {OUT_PER_IMAGE}")
log(f"汇总结果 -> {OUT_SUMMARY}")
log(f"运行日志 -> {log_path}")
logf.close()
