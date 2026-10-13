"""
U-Net 阴影检测基线：用于 SOTA 对比实验。

在 AISD 上训练一个经典 U-Net，与 SAM 微调方法对比，
验证"基础模型 + 参数高效微调"是否优于传统全监督分割网络。

用法：
    python train_unet.py --epochs 60           # 训练
    python train_unet.py --evaluate            # 用已训权重在 Test51 上评估
"""
import argparse
import glob
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from eval_shadow import compute_metrics

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT = str(BASE / "checkpoints" / "unet_shadow.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """经典 4 层 U-Net。输入 (B,3,H,W)，输出 (B,1,H,W) 阴影概率 logits。"""

    def __init__(self, in_c=3, out_c=1, base=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(in_c, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        e4 = self.d4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.c4(torch.cat([self.u4(b), e4], 1))
        d3 = self.c3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.c2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.c1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum() + 1e-6
    den = probs.sum() + target.sum() + 1e-6
    return 1 - num / den


def load_image_tensor(path, augment=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if augment and random.random() < 0.5:
        img = img[:, ::-1, :].copy()
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return x


def load_mask_tensor(path, augment=False):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if augment and random.random() < 0.5:
        m = m[:, ::-1].copy()
    m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
    return torch.as_tensor((m > 0), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, img_paths, gt_paths):
    model.eval()
    ious = []
    for ip, gp in zip(img_paths, gt_paths):
        x = load_image_tensor(ip)[None].to(device)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        logits = model(x)[0, 0]
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        # 预测结果缩放回原始尺寸，与原始 mask 对齐后算指标（与 SAM 方法同口径）
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
    return float(np.mean(ious)) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net 参数量 {n_params/1e6:.2f}M，设备 {device}", flush=True)

    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    test_imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
    test_gts = [str(AISD / "Test51" / "mask" / Path(p).name) for p in test_imgs]

    if args.evaluate:
        model.load_state_dict(torch.load(CKPT, map_location=device))
        print(f"Test51 IoU = {evaluate(model, test_imgs, test_gts):.4f}")
        return

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pairs = list(zip(train_imgs, train_gts))
        random.shuffle(pairs)
        t0 = time.time(); losses = []
        for ip, gp in pairs:
            x = load_image_tensor(ip, augment=True)[None].to(device)
            t = load_mask_tensor(gp, augment=True)[None, None].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, t) + dice_loss(logits, t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if epoch % 10 == 0:
            viou = evaluate(model, val_imgs, val_gts)
            print(f"    -> 验证 IoU={viou:.4f}", flush=True)
            if viou > best:
                best = viou
                torch.save(model.state_dict(), CKPT)
                print(f"    -> 保存最佳模型", flush=True)

    if Path(CKPT).exists():
        model.load_state_dict(torch.load(CKPT, map_location=device))
    print(f"训练完成，Test51 IoU = {evaluate(model, test_imgs, test_gts):.4f}")


if __name__ == "__main__":
    main()
