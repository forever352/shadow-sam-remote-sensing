"""
SegNet 阴影检测基线：用于 SOTA 对比实验。

SegNet 是经典语义分割网络，与 U-Net 的区别在于没有跳跃连接，
靠对称的编解码结构 + 转置卷积上采样恢复分辨率。

用法：
    python train_segnet.py --epochs 60     # 训练
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
CKPT = str(BASE / "checkpoints" / "segnet_shadow.pth")
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


class SegNet(nn.Module):
    """对称编解码结构（无跳跃连接）。输入 (B,3,H,W)，输出 (B,1,H,W) 阴影 logits。"""

    def __init__(self, in_c=3, out_c=1, base=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = DoubleConv(in_c, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.d4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 8, base * 8)
        self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 4, base * 4)
        self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 2, base * 2)
        self.d1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base, base)
        self.out = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        x = self.e1(x); x = self.pool(x)
        x = self.e2(x); x = self.pool(x)
        x = self.e3(x); x = self.pool(x)
        x = self.e4(x); x = self.pool(x)
        x = self.bottleneck(x)
        x = self.c4(self.d4(x))
        x = self.c3(self.d3(x))
        x = self.c2(self.d2(x))
        x = self.c1(self.d1(x))
        return self.out(x)


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
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])
    return float(np.mean(ious)) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    model = SegNet().to(device)
    print(f"SegNet 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M，设备 {device}", flush=True)

    train_imgs = sorted(glob.glob(str(AISD / "Train412" / "shadow" / "*.tif")))
    train_gts = [str(AISD / "Train412" / "mask" / Path(p).name) for p in train_imgs]
    val_imgs = sorted(glob.glob(str(AISD / "Val51" / "shadow" / "*.tif")))
    val_gts = [str(AISD / "Val51" / "mask" / Path(p).name) for p in val_imgs]
    test_imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
    test_gts = [str(AISD / "Test51" / "mask" / Path(p).name) for p in test_imgs]

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
    test_iou = evaluate(model, test_imgs, test_gts)
    print(f"训练完成，Test51 IoU = {test_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
