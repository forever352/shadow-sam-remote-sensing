"""阶段2：方向回归头 —— 从"翻转判别"骨干微调出"阴影方向向量"。

背景：申报书要求"自监督方向感知网络自动提取全局太阳方位角特征向量"。
阶段1（train_direction_net.py）已用自监督翻转任务训练出方向感知骨干；
本脚本在其基础上加回归头，输出全局阴影方向 (cos2θ, sin2θ)。

标签来源：AISD 自带逐像素阴影掩膜 —— 对每个阴影连通域求主轴方向，
用二倍角圆形均值聚合成全局主导方向（无需任何人工方向标注）。

关键设计：
- 用 cos2θ/sin2θ 表示方向（阴影是"轴"不是"箭头"，180°周期）
- 损失：1 - cos_sim（角度损失）
- 评估：角度误差（度）= acos(cos2θ向量夹角)/2
- 划分与下游方法完全同一基座（68/16/21，测试场景训练中未出现）

用法：
    python train_direction_regress.py --epochs 1     # 冒烟
    python train_direction_regress.py --epochs 40
"""
import argparse
import math
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

from train_direction_net import split_by_base, IMG_SIZE, MEAN, STD

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
BACKBONE = str(BASE / "checkpoints" / "direction_resnet18_flip_basis.pth")
OUT = str(BASE / "checkpoints" / "direction_resnet18_regress.pth")
LOG_DIR = BASE / "logs"

device = "cuda" if torch.cuda.is_available() else "cpu"


def gt_direction(mask_gray, min_area=200):
    """从 GT 阴影掩膜算全局主导方向，返回 (cos2θ, sin2θ) 单位向量，或 None。

    对每个连通域求主轴方向（二倍角），再做圆形均值 → 全局方向。
    """
    m = (mask_gray > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    angs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 5:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            angle += 90  # 让 angle 指向长边（主轴）
        angs.append(math.radians(angle))
    if not angs:
        return None
    a = np.array(angs) * 2.0  # 二倍角，消除 180° 周期
    c = float(np.cos(a).mean())
    s = float(np.sin(a).mean())
    theta2 = math.atan2(s, c)
    return np.array([math.cos(theta2), math.sin(theta2)], dtype=np.float32)


class DirectionDataset(torch.utils.data.Dataset):
    """返回 (图像张量, 方向单位向量)。标签从 GT 掩膜算，无方向的图被过滤。"""

    def __init__(self, files, train=True, no_aug=False):
        self.train = train
        # ⚠️ 关键：默认训练用 RandomResizedCrop(scale=0.4~1.0) 随机裁剪，
        #    而标签是**整图**方向 → 输入只是局部、标签却是整体，**系统性错配**
        #    （第五轮审查判定）。`no_aug=True` 时训练也用全图变换，消除该错配。
        if train and not no_aug:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])
        # 预筛：只保留能算出方向的图
        self.items = []
        for f in files:
            g = str(AISD / "Train412" / "mask" / Path(f).name)
            mask = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            d = gt_direction(mask)
            if d is not None:
                self.items.append((f, d))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, d = self.items[idx]
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(img))
        return x, torch.as_tensor(d)


def build_model():
    """ResNet-18 + 回归头(2维)；骨干加载阶段1的翻转预训练权重。"""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if Path(BACKBONE).exists():
        sd = torch.load(BACKBONE, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        model.load_state_dict(sd, strict=False)  # 骨干加载，fc 用新回归头
        print(f"已加载翻转预训练骨干: {BACKBONE}", flush=True)
    else:
        print(f"⚠️ 未找到 {BACKBONE}，用 ImageNet 初始化", flush=True)
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@torch.no_grad()
def evaluate_angle(model, files):
    """返回平均角度误差(度) + 样本数。"""
    model.eval()
    errs = []
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    for f in files:
        mask = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        d = gt_direction(mask)
        if d is None:
            continue
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        x = tf(Image.fromarray(img))[None].to(device)
        p = F.normalize(model(x)[0], dim=0)
        cos = float(torch.clamp((p * torch.as_tensor(d).to(device)).sum(), -1, 1))
        # 二倍角：向量夹角 = 2Δθ → 实际误差 = 夹角/2
        errs.append(math.degrees(math.acos(cos)) / 2.0)
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-aug", action="store_true",
                    help="训练也用全图变换（不做随机裁剪）。用于消除「随机裁剪 vs 整图标签」错配")
    ap.add_argument("--tag", type=str, default="",
                    help="输出标签后缀，避免不同运行互相覆盖（冒烟请务必指定）")
    args = ap.parse_args()
    out_path = str(BASE / "checkpoints" / (f"direction_resnet18_regress_{args.tag}.pth" if args.tag else "direction_resnet18_regress.pth"))

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_regress_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log("# 方向回归头（从翻转预训练骨干微调出阴影方向向量 cos2θ/sin2θ）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  epochs={args.epochs} seed=0")

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, _, _ = split_by_base(train_pool, 0.2, seed=1)
    log(f"# 划分=basis（与方法同基座）：训练 {len(tr_files)} / 验证 {len(val_files)} / 测试 {len(test_files)}")

    train_ds = DirectionDataset(tr_files, train=True, no_aug=args.no_aug)
    val_ds = DirectionDataset(val_files, train=False)
    log(f"# 有效样本（能算出方向）：训练 {len(train_ds)} / 验证 {len(val_ds)}")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_err = 999.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); losses = []
        for x, d in train_loader:
            x, d = x.to(device), d.to(device)
            p = F.normalize(model(x), dim=1)
            loss = (1 - (p * d).sum(dim=1)).mean()  # 角度损失
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        msg = f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)"
        if epoch % 5 == 0 or epoch == args.epochs:
            verr, vn = evaluate_angle(model, val_files)
            msg += f"  | 验证角度误差={verr:.2f}° (n={vn})"
            if verr < best_err:
                best_err = verr
                torch.save(model.state_dict(), out_path)
                msg += "  -> 保存最佳"
        log(msg)

    # 最终：测试集角度误差
    if Path(out_path).exists():
        model.load_state_dict(torch.load(out_path, map_location=device))
    terr, tn = evaluate_angle(model, test_files)
    valr, vn = evaluate_angle(model, val_files)
    # ★ 训练集角度误差（**直接测量**，此前只能由 loss 反推 —— 第五/六轮审查要求补）
    trerr, trn = evaluate_angle(model, tr_files)

    # ★ 基线（第六轮审查要求补）：
    #   ① 常数预测器：对所有图都预测「训练标签的圆形均值方向」——"什么都不学"的水平
    #   ② 随机水平：180° 周期下均匀随机的期望绝对角误差 = 45°
    tr_ds = DirectionDataset(tr_files, train=False)
    if tr_ds.items:
        D = np.array([d for _, d in tr_ds.items], dtype=np.float64)
        mc, ms = float(D[:, 0].mean()), float(D[:, 1].mean())
        nrm = math.hypot(mc, ms)
        const_d = np.array([mc / nrm, ms / nrm]) if nrm > 0 else np.array([1.0, 0.0])
        const_errs = []
        for fs in (val_files, test_files):
            ds = DirectionDataset(fs, train=False)
            for _, d in ds.items:
                c = float(np.clip(np.dot(const_d, np.asarray(d, dtype=np.float64)), -1, 1))
                const_errs.append(math.degrees(math.acos(c)) / 2.0)
        const_err = float(np.mean(const_errs)) if const_errs else float("nan")
        const_deg = math.degrees(math.atan2(const_d[1], const_d[0])) / 2.0
    else:
        const_err, const_deg = float("nan"), float("nan")

    log("")
    log("=== 方向回归结果 ===")
    log(f"训练集角度误差: {trerr:.2f}° (n={trn})   ← 直接测量")
    log(f"验证集角度误差: {valr:.2f}° (n={vn})")
    log(f"测试集角度误差: {terr:.2f}° (n={tn})   ← 无泄漏")
    log(f"最佳验证误差: {best_err:.2f}°，模型 -> {out_path}")
    log("")
    log("--- 基线对照 ---")
    log(f"常数预测器（恒预测训练集圆形均值方向 θ={const_deg:.1f}°）: {const_err:.2f}°")
    log(f"随机水平（180° 周期均匀分布的期望绝对角误差）        : 45.00°")
    log("")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
