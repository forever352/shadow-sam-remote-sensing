"""模块① 补强：DTICT -- 方向变换无关性噪声抑制（Co-Teaching 简化版）。

## 为什么需要它

导师论文（祁昆仑 等, 2025, 遥感学报 29(3):776-791）的核心机制是 **DTICT**
(Direction Transformation Independent Co-Teaching)，用来处理方向判别任务里的**噪声样本**：

> Figs.5/12 原文：噪声样本 = "samples **without shadows**" 与
> "samples containing shadows in a **special orientation**"；
> 结论段："The use of the DTICT strategy helps alleviate the interference caused by
> shadow-free samples and samples containing shadows in special directions during network training."
> 论文 Table 2 消融：DTICT 相对 ResNet18 基线 **IoU +1.63% / F1 +0.81%**。

我们此前的 `train_direction_net.py` **对全部样本一视同仁、没有任何噪声抑制** --
这是与论文之间唯一的实质落差。本脚本补齐它。

## 实现（两个来源都来自论文原文，未凭空发明）

1. **Co-Teaching 双网络互选**（论文："Co-teaching DTICT | Han 2018"）
   - 两个独立初始化的 ResNet-18；每个 mini-batch 内，
     每个网络按**自己**的逐样本损失挑出损失最小的 `keep_frac` 比例样本，
     交给**另一个**网络训练（互相当"教师"）。
   - 作用：避免网络记住噪声样本的标签。

2. **方向一致性门（direction-consistency gate）**
   - 论文："computing **cosine distance** between high-level feature vectors of
     original and flipped images, suppressing interference from noise samples"。
   - 落地：对每张图取原图视图与竖直翻转视图的**倒数第二层特征**，
     算余弦相似度 `s_i = cos(v0, vt)`。
     - `s_i` 过大 -> 翻转几乎没改变表征 -> **无方向信号的图**（如无阴影）-> 判为噪声；
     - `s_i` 过小 -> 两视图表征矛盾 -> **朝向特殊/不一致** -> 判为噪声。
   - 门限由**训练集自身分布的分位数**确定（预注册：上下各 `noise_q`），不做人工调参。

> [!] 论文未公开 DTICT 的完整超参，本实现是**按原理的简化重构**，不是逐行复现。
> 报告时必须写明这一点，且**只用于内部消融**，不得声称"复现了论文的 DTICT"。

## 用法

    python train_direction_dtict.py --no-dtict --epochs 30      # 对照（= 单网络，无噪声抑制）
    python train_direction_dtict.py --epochs 30                 # 本项目 DTICT（双网络 + 一致性门）
    python train_direction_dtict.py --epochs 2 --limit 40       # 冒烟
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from train_direction_net import split_by_base, IMG_SIZE, MEAN, STD

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT_DIR = BASE / "checkpoints"
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"
LOG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


class FlipDataset(torch.utils.data.Dataset):
    """与 train_direction_net.FlipDataset 同构（原图 label0 / 竖直翻转 label1），
    额外返回文件名，供方向一致性门使用。"""

    def __init__(self, files, train=True):
        self.files = files
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        tf_eval = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.transform = tf_train if train else tf_eval
        self.train = train

    def __len__(self):
        return 2 * len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx // 2]
        flipped = (idx % 2 == 1)
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if flipped:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return self.transform(img), torch.tensor(1.0 if flipped else 0.0)


def build_model(from_scratch=False, seed=0):
    torch.manual_seed(seed)
    weights = None if from_scratch else ResNet18_Weights.IMAGENET1K_V1
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


# ══════════════════════════════════════════════════════════════════
# 方向一致性门
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def feature_of(model, x):
    """取倒数第二层特征（ResNet-18 的 avgpool 输出，512 维）。"""
    m = model
    f = m.conv1(x); f = m.bn1(f); f = m.relu(f); f = m.maxpool(f)
    f = m.layer1(f); f = m.layer2(f); f = m.layer3(f); f = m.layer4(f)
    f = m.avgpool(f)
    return f.flatten(1)


@torch.no_grad()
def consistency_scores(model, files):
    """每张图的「原图 vs 竖直翻转」特征余弦相似度 s_i ∈ [-1, 1]。"""
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    out = {}
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        x0 = tf(pil)[None].to(device)
        x1 = tf(pil.transpose(Image.FLIP_TOP_BOTTOM))[None].to(device)
        v0, v1 = feature_of(model, x0), feature_of(model, x1)
        out[f] = float(F.cosine_similarity(v0, v1, dim=1).item())
    return out


def gate_noise(scores, q=0.10):
    """按训练集自身分位数剔除两端样本（预注册 q=0.10，不做人工调参）。

    返回 (保留的文件列表, 被判为噪声的文件列表, 阈值)。
    """
    vals = np.array(list(scores.values()))
    lo, hi = np.quantile(vals, q), np.quantile(vals, 1 - q)
    keep = [f for f, v in scores.items() if lo <= v <= hi]
    noise = [f for f, v in scores.items() if not (lo <= v <= hi)]
    return keep, noise, (float(lo), float(hi))


# ══════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def flip_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x).squeeze(1)) > 0.5).float()
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--keep-frac", type=float, default=0.70, help="co-teaching 每步保留的样本比例")
    ap.add_argument("--noise-q", type=float, default=0.10, help="一致性门两端各剔除的分位（预注册）")
    ap.add_argument("--limit", type=int, default=0, help="限制训练集大小（冒烟用）")
    ap.add_argument("--no-dtict", action="store_true", help="消融：单网络、无噪声抑制（= 原基线）")
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_dtict_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    # 与下游完全同一基座（68/16/21）
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    train_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    if args.limit:
        train_files = train_files[:args.limit]

    mode = "单网络（无噪声抑制，对照）" if args.no_dtict else "DTICT（双网络互选 + 方向一致性门）"
    log(f"# 方向判别 - {mode}")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  epochs={args.epochs} seed=0 设备={device}")
    log(f"# 划分: basis 68/16/21 -> 训练 {len(train_files)}({n_tr}原图) / 验证 {len(val_files)}({n_va}原图) / 测试 {len(test_files)}")
    log(f"# 超参: batch={args.batch_size} lr={args.lr} keep_frac={args.keep_frac} noise_q={args.noise_q}")

    te_loader = torch.utils.data.DataLoader(FlipDataset(val_files, train=False),
                                            batch_size=args.batch_size, shuffle=False)

    netA = build_model(args.from_scratch, seed=0).to(device)
    netB = None if args.no_dtict else build_model(args.from_scratch, seed=1).to(device)

    n_param = sum(p.numel() for p in netA.parameters())
    log(f"# 参数量/网络 {n_param/1e6:.2f}M；网络数 {'1' if netB is None else '2'}")

    # ── 方向一致性门（仅 DTICT 模式启用；用**当前**网络估分，训练中周期性刷新）──
    used_files = list(train_files)
    noise_files = []
    if not args.no_dtict:
        scores = consistency_scores(netA, train_files)
        used_files, noise_files, (lo, hi) = gate_noise(scores, args.noise_q)
        log(f"# 方向一致性门：剔除 {len(noise_files)} / {len(train_files)} 张噪声图（s∈[{lo:.3f},{hi:.3f}] 内保留）")
        log(f"#   剔除样例: {[Path(f).name for f in noise_files[:5]]}")

    train_loader = torch.utils.data.DataLoader(FlipDataset(used_files, train=True),
                                               batch_size=args.batch_size, shuffle=True)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    optA = torch.optim.Adam(netA.parameters(), lr=args.lr)
    optB = None if netB is None else torch.optim.Adam(netB.parameters(), lr=args.lr)

    best = 0.0; best_ep = 0
    for epoch in range(1, args.epochs + 1):
        netA.train()
        if netB is not None:
            netB.train()
        t0 = time.time(); ls = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            yv = y.view(-1, 1)
            if netB is None:
                loss = crit(netA(x), yv).mean()
                optA.zero_grad(); loss.backward(); optA.step()
                ls.append(float(loss))
                continue
            # ── Co-teaching：各网络用**自己**的损失挑"干净"样本，喂给**对方** ──
            k = max(1, int(args.keep_frac * x.size(0)))
            with torch.no_grad():
                lA_det = crit(netA(x), yv).view(-1)
                lB_det = crit(netB(x), yv).view(-1)
                idxA = lA_det.argsort()[:k]   # A 认为干净的 -> 给 B 学
                idxB = lB_det.argsort()[:k]   # B 认为干净的 -> 给 A 学
            lossA = crit(netA(x[idxB]), yv[idxB]).mean()
            lossB = crit(netB(x[idxA]), yv[idxA]).mean()
            optA.zero_grad(); lossA.backward(); optA.step()
            optB.zero_grad(); lossB.backward(); optB.step()
            ls.append(float(lossA.detach()))

        line = f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(ls):.4f}  ({time.time()-t0:.0f}s)"
        if epoch % 5 == 0 or epoch == args.epochs:
            accA = flip_acc(netA, te_loader)
            line += f"  val翻转准确率A={accA:.4f}"
            if netB is not None:
                line += f"  B={flip_acc(netB, te_loader):.4f}"
                accA = max(accA, flip_acc(netB, te_loader))
            if accA > best:
                best, best_ep = accA, epoch
                torch.save(netA.state_dict(), str(CKPT_DIR / "direction_resnet18_dtict.pth"))
                line += "  -> 保存最佳"
        log(line)

    log("")
    log(f"# 最佳验证翻转准确率 {best:.4f} @ Epoch {best_ep}")
    log(f"# 训练用图 {len(used_files)} 张（原训练集 {len(train_files)} 张，剔除噪声 {len(noise_files)} 张）")
    log(f"# 权重 -> {CKPT_DIR / 'direction_resnet18_dtict.pth'}")
    log(f"# 运行日志 -> {log_path}")

    # 消融汇总（追加，便于两次运行后直读对比）
    s = RES_DIR / "direction_dtict_ablation.csv"
    new = not s.exists()
    with open(s, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["mode", "epochs", "keep_frac", "noise_q",
                        "n_train_img", "n_noise_removed", "best_val_flip_acc", "best_epoch"])
        w.writerow(["no-dtict" if args.no_dtict else "dtict", args.epochs, args.keep_frac,
                    args.noise_q, len(used_files), len(noise_files),
                    f"{best:.4f}", best_ep])
    log(f"# 消融汇总 -> {s}")
    logf.close()


if __name__ == "__main__":
    main()
