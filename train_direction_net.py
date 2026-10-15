"""
阶段1：阴影方向判别自监督预训练（复现导师论文"阴影方向先验"）。

核心任务：原图 vs 竖直翻转 二分类。
阴影有确定朝向（沿太阳方位反方向），把图竖直翻转会反转阴影的垂直方向分量，
所以网络要分辨"这张图有没有被上下翻转"，就必须学会阴影的方向先验。
这是【纯自监督】——标签由"翻转"这个变换自动生成，零人工标注。

关键设计（对照论文《基于阴影方向先验的高分辨率遥感影像地物阴影检测》）：
1. 按"原始大图基名"重新划分 train/val，杜绝 AISD 官方划分的子图同源泄漏
   （同一张原图的所有子图，太阳方向相同；若跨 train/val 会让指标虚高）。
2. 数据增强：随机裁剪 + 色彩抖动（对应论文"对比剪裁 + 色彩变换"），
   但【不做水平/竖直翻转】——竖直翻转就是标签，几何翻转会破坏方向语义。
3. 每轮打印 val 翻转准确率，<85% 报警（说明方向先验没学好）。

用法：
    python train_direction_net.py --epochs 3     # 快速冒烟测试
    python train_direction_net.py                # 完整训练（默认 30 epoch）
    python train_direction_net.py --from-scratch # 不加载 ImageNet 权重（更"纯"自监督）
"""
import argparse
import random
import re
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前，修复 Windows CUDA DLL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "direction_resnet18_flip.pth")
LOG_DIR = BASE / "logs"

N_EPOCHS = 30
LR = 1e-3
BATCH = 32
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 按原始大图基名划分（防泄漏） ----------
def base_name(fname):
    """austin13_sub2.tif -> austin13；无法解析则退回原文件名。"""
    m = re.match(r"(.+?)_sub\d+", fname)
    return m.group(1) if m else fname


def split_by_base(files, val_frac=0.2, seed=0):
    """把文件按原始大图基名分组，再按组划分 train/val，保证同一原图的子图不跨集合。"""
    groups = {}
    for f in files:
        groups.setdefault(base_name(f), []).append(f)
    names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_frac))
    val_bases = set(names[:n_val])
    train_files, val_files = [], []
    for b in names:
        (val_files if b in val_bases else train_files).extend(groups[b])
    return train_files, val_files, len(names) - n_val, n_val


# ---------- 数据集：每张图生成"原图(label 0) / 竖直翻转(label 1)"两样本 ----------
class FlipDataset(torch.utils.data.Dataset):
    def __init__(self, files, train=True):
        self.files = files
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return 2 * len(self.files)  # 原图 + 翻转图各一份

    def __getitem__(self, idx):
        f = self.files[idx // 2]
        flipped = (idx % 2 == 1)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if flipped:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 竖直翻转（上下镜像）
        x = self.transform(img)
        y = torch.tensor(1.0 if flipped else 0.0)
        return x, y


def build_model(from_scratch=False):
    weights = None if from_scratch else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 单 logit + BCE
    return model


@torch.no_grad()
def evaluate_flip_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x).squeeze(1)) > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--val-frac", type=float, default=0.2, help="验证集占原始大图的比例")
    parser.add_argument("--from-scratch", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--split", choices=["v1", "basis"], default="basis",
                        help="v1=84/21(旧)；basis=68/16/21（与方法同一基座，不碰21个测试场景）")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 落盘日志（证据优先于终端声明；审计要求每个数字可核验）
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_pretrain_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 阴影方向判别自监督预训练（原图 vs 竖直翻转 二分类）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# 骨干: ResNet-18  {'从零训练' if args.from_scratch else 'ImageNet预训练'}")
    log(f"# epochs={args.epochs} batch={args.batch_size} lr={LR} seed=0")
    log(f"# 增强: RandomResizedCrop + ColorJitter（禁几何翻转，翻转即标签）")

    out_path = str(BASE / "checkpoints" / ("direction_resnet18_flip_basis.pth" if args.split == "basis" else "direction_resnet18_flip.pth"))
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    if args.split == "v1":
        # 旧划分：单次划分 84 训练 / 21 验证
        train_files, val_files, n_tr_base, n_va_base = split_by_base(files, args.val_frac, seed=0)
        log(f"划分=v1（84/21）：Train412 共 {len(files)} 张")
    else:
        # 与方法(train_lora_leakfree/train_cnn_leakfree)完全同一基座：
        # 先划出 21 个测试场景，再从剩余 84 里划出 16 验证 → 68/16/21
        train_pool, _test_files, _, _n_te = split_by_base(files, 0.2, seed=0)
        train_files, val_files, n_tr_base, n_va_base = split_by_base(train_pool, 0.2, seed=1)
        log(f"划分=basis（68/16/21，与方法同基座）：21 个测试场景({_n_te}原图)不参与训练")
    log(f"  训练 {len(train_files)} 张（{n_tr_base} 张原图） / 验证 {len(val_files)} 张（{n_va_base} 张原图）")
    log(f"  train/val 无同源子图泄漏（按原图基名划分）")

    train_loader = torch.utils.data.DataLoader(
        FlipDataset(train_files, train=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        FlipDataset(val_files, train=False), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.from_scratch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logit = model(x).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        acc = evaluate_flip_acc(model, val_loader)
        flag = "" if acc >= 0.85 else "   <== 警告：方向先验没学好(低于85%)"
        log(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  "
            f"val翻转准确率={acc:.4f}  ({time.time()-t0:.0f}s){flag}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)

    log(f"训练完成，最佳验证翻转准确率={best_acc:.4f}，模型已保存到 {out_path}")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
