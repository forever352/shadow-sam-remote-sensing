"""
SAM 参数高效微调 v2（改进版）：
- 150 epoch（原 30）
- 余弦学习率衰减
- 水平翻转数据增强
- Focal Loss + Dice（处理阴影像素少的类别不平衡）
- 点提示 + 框提示 双模式训练（让模型两种提示都会用）

冻结图像编码器，仅训练掩膜解码器 + 提示编码器（4.34% 参数）。
"""
import os
import random
import glob
import time
import math
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from methods import mask_enhanced
from evaluate_finetuned import auto_points, predict
from eval_shadow import compute_metrics

BASE = Path(__file__).resolve().parent
CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v2.pth")

N_EPOCHS = 150
LR = 2e-4
LR_MIN = 1e-5
N_FG = 5
N_BG = 5
VAL_EVERY = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum() + 1e-6
    den = probs.sum() + target.sum() + 1e-6
    return 1 - num / den


def focal_loss(logits, target, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    w = (1 - p_t) ** gamma
    aw = alpha * target + (1 - alpha) * (1 - target)
    return (aw * w * bce).mean()


def sample_points(gt, n_fg=N_FG, n_bg=N_BG):
    ys, xs = np.where(gt > 0)
    yb, xb = np.where(gt == 0)
    coords, labels = [], []
    if len(xs) > 0:
        idx = np.random.choice(len(xs), min(n_fg, len(xs)), replace=False)
        for i in idx:
            coords.append([xs[i], ys[i]]); labels.append(1)
    if len(xb) > 0:
        idx = np.random.choice(len(xb), min(n_bg, len(xb)), replace=False)
        for i in idx:
            coords.append([xb[i], yb[i]]); labels.append(0)
    if not coords:
        return None, None
    return np.array(coords, dtype=float), np.array(labels, dtype=np.int64)


def gt_bbox(gt, margin=5):
    ys, xs = np.where(gt > 0)
    if len(xs) == 0:
        return None
    h, w = gt.shape
    x1 = max(0, int(xs.min()) - margin); y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + margin); y2 = min(h, int(ys.max()) + margin)
    return np.array([x1, y1, x2, y2], dtype=float)


def train_step(sam, img_rgb, gt, opt):
    """单步训练，返回 loss。"""
    h, w = gt.shape
    # 数据增强：水平翻转
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()

    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    # 双提示：70% 点 / 30% 框
    if random.random() < 0.7:
        coords, labels = sample_points(gt)
        if coords is None:
            return None
        c1024 = transform.apply_coords(coords, (h, w))
        c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
        l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
        points, boxes = (c_t, l_t), None
    else:
        box = gt_bbox(gt)
        if box is None:
            return None
        b1024 = transform.apply_boxes(box, (h, w))
        b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
        points, boxes = None, b_t

    x = sam.preprocess(img_t)
    with torch.no_grad():
        img_emb = sam.image_encoder(x)
    sparse, dense = sam.prompt_encoder(points=points, boxes=boxes, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    gt256 = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.as_tensor((gt256 > 0), dtype=torch.float32)[None, None].to(device)
    loss = focal_loss(low_res, gt_t) + dice_loss(low_res, gt_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate_auto(sam, img_paths, gt_paths):
    """用全自动提示（增强 Otsu 布点）在验证集上评估 IoU。"""
    ious = []
    for ip, gp in zip(img_paths, gt_paths):
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        em = mask_enhanced(img_bgr)
        pts, labs = auto_points(em)
        if pts is None:
            ious.append(0.0); continue
        pred = predict(sam, img_rgb, pts, labs)
        ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
    return float(np.mean(ious)) if ious else 0.0


def main():
    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    print(f"训练 {len(train_imgs)} 张，验证 {len(val_imgs)} 张，设备 {device}，共 {N_EPOCHS} epoch", flush=True)

    sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    sam.image_encoder.eval()
    sam.prompt_encoder.train()
    sam.mask_decoder.train()

    params = list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"可训练参数 {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    best_iou = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        # 余弦学习率
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * epoch / N_EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr

        sam.prompt_encoder.train(); sam.mask_decoder.train()
        pairs = list(zip(train_imgs, train_gts))
        random.shuffle(pairs)
        t0 = time.time(); losses = []
        for ip, gp in pairs:
            img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            loss = train_step(sam, img, gt, opt)
            if loss is not None:
                losses.append(loss)
        avg = float(np.mean(losses))
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg:.4f}  lr={lr:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if epoch % VAL_EVERY == 0:
            sam.prompt_encoder.eval(); sam.mask_decoder.eval()
            viou = evaluate_auto(sam, val_imgs, val_gts)
            print(f"    -> 全自动验证 IoU={viou:.4f}", flush=True)
            if viou > best_iou:
                best_iou = viou
                torch.save(sam.state_dict(), OUT)
                print(f"    -> 保存最佳模型 (IoU {viou:.4f})", flush=True)

    torch.save(sam.state_dict(), str(BASE / "checkpoints" / "sam_vit_b_shadow_v2_last.pth"))
    print(f"训练完成，最佳全自动验证 IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
