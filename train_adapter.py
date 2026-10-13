"""
SAM 参数高效微调（PEFT）：冻结图像编码器，仅训练轻量掩膜解码器 + 提示编码器。
用于 AISD 遥感阴影检测的领域适配。

训练思路：
- 输入影像 -> 冻结的 ViT 编码器 -> 图像嵌入（固定）
- 从真值掩膜采样前景/背景点作为提示
- 掩膜解码器（可训练）输出阴影掩膜，与真值计算 BCE + Dice 损失
"""
import os
import random
import glob
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

BASE = Path(__file__).resolve().parent
CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow.pth")

N_EPOCHS = 30
LR = 1e-4
N_FG = 5   # 每张图采样的前景点（阴影内）数量
N_BG = 5   # 每张图采样的背景点（阴影外）数量
VAL_EVERY = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum() + 1e-6
    den = probs.sum() + target.sum() + 1e-6
    return 1 - num / den


def sample_points(gt, n_fg=N_FG, n_bg=N_BG):
    """从真值掩膜采样前景点(阴影内, label=1)与背景点(阴影外, label=0)。返回 (coords(N,2), labels(N,))。"""
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


def train_step(sam, img_rgb, gt, opt):
    """单步训练：返回 loss 值。"""
    h, w = gt.shape
    img1024 = transform.apply_image(img_rgb)              # (1024,1024,3)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    coords, labels = sample_points(gt)
    if coords is None:
        return None
    coords1024 = transform.apply_coords(coords, (h, w))   # (N,2) -> 1024 坐标系
    coords_t = torch.as_tensor(coords1024, dtype=torch.float32)[None].to(device)  # (1,N,2)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)        # (1,N)

    x = sam.preprocess(img_t)                            # 归一化 + pad（已 1024，无需 pad）
    with torch.no_grad():
        img_emb = sam.image_encoder(x)                   # (1,256,64,64) 冻结，无梯度

    sparse, dense = sam.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )                                                   # (1,1,256,256)

    gt256 = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.as_tensor((gt256 > 0), dtype=torch.float32)[None, None].to(device)

    loss = F.binary_cross_entropy_with_logits(low_res, gt_t) + dice_loss(low_res, gt_t)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(sam, img_paths, gt_paths):
    """在给定图像上评估 IoU（用真值质心点做提示，验证模型是否学到了阴影分割）。"""
    ious = []
    for ip, gp in zip(img_paths, gt_paths):
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        h, w = gt.shape
        # 用真值质心点做提示（评估"给好提示能分割多准"）
        coords, labels = sample_points(gt, n_fg=1, n_bg=0)
        if coords is None:
            continue
        img1024 = transform.apply_image(img)
        img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
        coords1024 = transform.apply_coords(coords, (h, w))
        coords_t = torch.as_tensor(coords1024, dtype=torch.float32)[None].to(device)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
        x = sam.preprocess(img_t)
        img_emb = sam.image_encoder(x)
        sparse, dense = sam.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))  # (1,1,h,w)
        pred = (mask[0, 0] > 0).cpu().numpy()
        gt_b = gt > 0
        inter = (pred & gt_b).sum(); union = (pred | gt_b).sum()
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def main():
    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    print(f"训练 {len(train_imgs)} 张，验证 {len(val_imgs)} 张，设备 {device}")

    sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(device)
    # 冻结图像编码器
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    sam.image_encoder.eval()
    sam.prompt_encoder.train()
    sam.mask_decoder.train()

    params = list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters())
    n_params = sum(p.numel() for p in params)
    total = sum(p.numel() for p in sam.parameters())
    print(f"可训练参数 {n_params/1e6:.2f}M / 总参数 {total/1e6:.1f}M（仅 {100*n_params/total:.2f}%）")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    best_iou = 0.0
    for epoch in range(1, N_EPOCHS + 1):
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
        print(f"Epoch {epoch:2d}/{N_EPOCHS}  loss={avg:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        if epoch % VAL_EVERY == 0:
            sam.prompt_encoder.eval(); sam.mask_decoder.eval()
            viou = evaluate(sam, val_imgs, val_gts)
            print(f"    -> 验证 IoU={viou:.4f}", flush=True)
            if viou > best_iou:
                best_iou = viou
                torch.save(sam.state_dict(), OUT)
                print(f"    -> 保存最佳模型 (IoU {viou:.4f})", flush=True)

    torch.save(sam.state_dict(), str(BASE / "checkpoints" / "sam_vit_b_shadow_last.pth"))
    print(f"训练完成，最佳验证 IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
