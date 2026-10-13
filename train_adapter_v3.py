"""
SAM 方向提示微调 v3：训练时就用方向点提示，实现训练/推理提示一致。

核心思路：v2 用"轴对齐框 + 随机点"微调，导致推理时换成方向提示模型不适应。
本脚本在训练时就采样"方向点"（对 gt 每个连通域沿主轴方向撒点），
让模型学会"方向点 → 分割阴影"，推理时也用同样的方向点提示。

与 v2 的差别：
- 训练采样：per-instance 方向点（对 gt 每个连通域沿主轴采样前景点）
- 目标 mask：单连通域（而非整图）
- 评估：方向点提示推理（而非质心点）
- 其余不变：冻结编码器、Focal+Dice、余弦学习率

用法：
    python train_adapter_v3.py --epochs 20   # 快速验证（约 45 分钟）
    python train_adapter_v3.py               # 完整训练（150 epoch，约 6 小时）
"""
import argparse
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

from eval_shadow import compute_metrics
from direction_experiment import method_direction_axis  # 方向点推理（用于评估）

BASE = Path(__file__).resolve().parent
CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v3.pth")

N_EPOCHS = 150
LR = 2e-4
LR_MIN = 1e-5
VAL_EVERY = 10
N_AXIS = 5   # 主轴方向采样点数
N_BG = 3     # 背景点数
MIN_AREA = 200

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


def sample_direction(gt, n_axis=N_AXIS, n_bg=N_BG, min_area=MIN_AREA):
    """
    随机选一个连通域，沿其主轴方向采样前景点，再从真背景采背景点。
    返回 (coords, labels, target_mask)；无合适连通域时返回 None。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return None
    i = random.choice(comps)
    ys, xs = np.where(labels == i)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w < h:
        angle += 90  # 保证 angle 指向长边（主轴）
    theta = np.deg2rad(angle)
    axis = np.array([np.cos(theta), np.sin(theta)])
    half = max(w, h) / 2
    coords, labs = [], []
    for t in np.linspace(-half, half, n_axis):
        px, py = int(round(cx + axis[0] * t)), int(round(cy + axis[1] * t))
        if 0 <= px < gt.shape[1] and 0 <= py < gt.shape[0] and labels[py, px] == i:
            coords.append([px, py]); labs.append(1)
    if len(coords) < 3:
        return None
    # 背景点：真背景（gt == 0）随机采样
    yb, xb = np.where(gt == 0)
    if len(xb) >= n_bg:
        idx = np.random.choice(len(xb), n_bg, replace=False)
        for j in idx:
            coords.append([xb[j], yb[j]]); labs.append(0)
    target = (labels == i).astype(np.float32)
    return np.array(coords, float), np.array(labs, np.int64), target


def train_step(sam, img_rgb, gt, opt):
    """单步训练：方向点提示 + 单连通域目标 mask。"""
    h, w = gt.shape
    if random.random() < 0.5:  # 水平翻转增强
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()

    res = sample_direction(gt)
    if res is None:
        return None
    coords, labels, target = res

    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
    c1024 = transform.apply_coords(coords, (h, w))
    c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
    l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)

    x = sam.preprocess(img_t)
    with torch.no_grad():
        img_emb = sam.image_encoder(x)
    sparse, dense = sam.prompt_encoder(points=(c_t, l_t), boxes=None, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    gt256 = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.as_tensor((gt256 > 0), dtype=torch.float32)[None, None].to(device)
    loss = focal_loss(low_res, gt_t) + dice_loss(low_res, gt_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate_direction(sam, img_paths, gt_paths):
    """用方向点提示（method_direction_axis）在验证集上评估 IoU。"""
    ious = []
    for ip, gp in zip(img_paths, gt_paths):
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = method_direction_axis(sam, img_rgb, img_bgr)
        ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
    return float(np.mean(ious)) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    print(f"训练 {len(train_imgs)} 张，验证 {len(val_imgs)} 张，设备 {device}，共 {args.epochs} epoch", flush=True)

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
    for epoch in range(1, args.epochs + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * epoch / args.epochs))
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
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  lr={lr:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if epoch % VAL_EVERY == 0:
            sam.prompt_encoder.eval(); sam.mask_decoder.eval()
            viou = evaluate_direction(sam, val_imgs, val_gts)
            print(f"    -> 方向提示验证 IoU={viou:.4f}", flush=True)
            if viou > best_iou:
                best_iou = viou
                torch.save(sam.state_dict(), OUT)
                print(f"    -> 保存最佳模型 (IoU {viou:.4f})", flush=True)

    torch.save(sam.state_dict(), str(BASE / "checkpoints" / "sam_vit_b_shadow_v3_last.pth"))
    print(f"训练完成，最佳方向提示验证 IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
