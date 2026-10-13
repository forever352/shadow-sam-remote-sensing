"""
SAM LoRA 微调 v4：在图像编码器上做 LoRA，让 ViT 也适配遥感阴影。

背景：v2 只微调 mask_decoder + prompt_encoder（冻结编码器），Test51 IoU 0.717，
但全监督 U-Net 达 0.805。原因在于 ViT 编码器是自然图像预训练的，特征不完全
适配遥感阴影。本脚本用 LoRA（低秩适配）微调编码器的 attention 层，只需极少
参数就让编码器学到遥感领域特征，是申报书"Adapter 微调"的正规实现。

LoRA 原理：在原权重 W 旁加低秩增量 W + ΔW = W + BA，只训练 A、B（低秩矩阵）。

改动要点（相比 v2）：
- 在 image_encoder 每个 attention 的 qkv、proj 上挂 LoRA（rank=8）
- 冻结原权重，训练 LoRA + mask_decoder + prompt_encoder
- image_encoder 前向不再 no_grad（LoRA 需要梯度）

用法：
    python train_adapter_v4.py --epochs 20   # 快速验证（约 1 小时）
    python train_adapter_v4.py               # 完整训练（150 epoch，约 8 小时）
"""
import argparse
import random
import glob
import time
import math
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from eval_shadow import compute_metrics
from methods import mask_enhanced
from evaluate_finetuned import auto_points, predict
from direction_experiment import method_boxes  # 框提示推理（最终测试用）

BASE = Path(__file__).resolve().parent
CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora.pth")
RESUME_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_resume.pth")

N_EPOCHS = 150
LR = 1e-4
LR_MIN = 1e-5
N_FG = 5
N_BG = 5
VAL_EVERY = 10
LORA_RANK = 8
LORA_ALPHA = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


# ---------- LoRA ----------
class LoRALinear(nn.Module):
    """在原 nn.Linear 旁加低秩增量 W + BA。原权重冻结，只训练 A、B。"""

    def __init__(self, linear, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False  # 冻结原权重
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def apply_lora(sam, rank=LORA_RANK, alpha=LORA_ALPHA):
    """在 image_encoder 每个 attention 的 qkv、proj 上挂 LoRA。"""
    for block in sam.image_encoder.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)


def lora_params(sam):
    """收集所有 LoRA 参数（A、B 矩阵）。"""
    params = []
    for block in sam.image_encoder.blocks:
        for m in (block.attn.qkv, block.attn.proj):
            if isinstance(m, LoRALinear):
                params += [m.lora_A, m.lora_B]
    return params


# ---------- 损失与采样（同 v2） ----------
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


def train_step(sam, img_rgb, gt, opt, scaler):
    h, w = gt.shape
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()

    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

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
    with torch.cuda.amp.autocast():
        img_emb = sam.image_encoder(x)  # 不 no_grad：LoRA 需要梯度
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
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


@torch.no_grad()
def evaluate_auto(sam, img_paths, gt_paths):
    """质心点提示验证（与 v2 同口径，便于对比）。"""
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


@torch.no_grad()
def evaluate_boxes(sam, img_paths, gt_paths):
    """框提示推理（最终测试口径）。"""
    ious = []
    for ip, gp in zip(img_paths, gt_paths):
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = method_boxes(sam, img_rgb, img_bgr)
        ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
    return float(np.mean(ious)) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 续训（模型权重路径）")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    test_imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
    test_gts = [str(AISD / "Test51" / "mask" / Path(p).name) for p in test_imgs]
    print(f"训练 {len(train_imgs)} 张，验证 {len(val_imgs)} 张，设备 {device}，共 {args.epochs} epoch", flush=True)

    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    apply_lora(sam)  # 先挂 LoRA，再 .to(device)，确保 LoRA 参数也移到 GPU
    if args.resume:
        sam.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"已从 {args.resume} 恢复权重", flush=True)
    sam.to(device)

    # 冻结整个 image_encoder，再启用 LoRA
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in lora_params(sam):
        p.requires_grad = True

    sam.prompt_encoder.train()
    sam.mask_decoder.train()

    params = lora_params(sam) + list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters())
    n_params = sum(p.numel() for p in params)
    n_lora = sum(p.numel() for p in lora_params(sam))
    print(f"可训练参数 {n_params/1e6:.2f}M（其中 LoRA {n_lora/1e6:.2f}M）", flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

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
            loss = train_step(sam, img, gt, opt, scaler)
            if loss is not None:
                losses.append(loss)
        avg = float(np.mean(losses))
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  lr={lr:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if epoch % VAL_EVERY == 0:
            sam.prompt_encoder.eval(); sam.mask_decoder.eval()
            viou = evaluate_auto(sam, val_imgs, val_gts)
            print(f"    -> 质心点验证 IoU={viou:.4f}", flush=True)
            if viou > best_iou:
                best_iou = viou
                torch.save(sam.state_dict(), OUT)
                print(f"    -> 保存最佳模型 (IoU {viou:.4f})", flush=True)
        if epoch % 5 == 0:
            torch.save(sam.state_dict(), RESUME_CKPT)
            print(f"    -> 保存续训 checkpoint (epoch {epoch})", flush=True)

    torch.save(sam.state_dict(), str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_last.pth"))
    print(f"训练完成，最佳质心点验证 IoU={best_iou:.4f}")
    # 最终：框提示在 Test51 上评估
    sam.prompt_encoder.eval(); sam.mask_decoder.eval()
    test_iou = evaluate_boxes(sam, test_imgs, test_gts)
    print(f"=== Test51 框提示 IoU = {test_iou:.4f} ===", flush=True)


if __name__ == "__main__":
    main()
