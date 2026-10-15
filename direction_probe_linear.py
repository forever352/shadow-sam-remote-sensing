"""方向编码探针：翻转预训练骨干的特征里，到底有没有"阴影方向"？

## 要回答的问题

项目（以及导师论文）的核心假设是：**竖直翻转自监督能让骨干学到阴影方向**。
但本项目此前**从未检验过**：骨干特征里能否读出方向。翻转判别存在**无需方向即可解出**的可能
（例如"亮部在上还是在下"这类全局上下不对称），故 100% 准确率**不能单独证明**方向先验存在。
本脚本是**检验该问题的第一步尝试**，但其口径（GAP+线性）**不足以给出结论**（见下"能力边界"）。

本脚本用**线性探针**直接检验：
    冻结骨干 -> 取 avgpool 特征(512维) -> **线性**预测 (cos2θ, sin2θ) -> 测角度误差

## 三组对照（缺一不可）

| # | 特征来源 | 预测目标 | 期望 | 含义 |
|---|---|---|---|---|
| A | **翻转预训练骨干** | 真实方向 | ? | 核心问题 |
| B | **ImageNet 骨干**（未翻转预训练） | 真实方向 | ? | **A 是否优于 B** = 翻转训练**是否带来**方向信息 |
| C | 翻转预训练骨干 | **随机方向**（阴性对照） | 应~=常数基线 | 验证探针方法本身可靠（防"探针总能拟合出点什么"） |

## [!] 本脚本的**能力边界**（第十一轮审查裁定，务必先读）

本脚本**只检验一条口径**：全局平均池化(512维) 特征 + **岭回归线性**读出。
因此它**只能**给出描述性结果，**不能**据此判定"骨干是否编码方向"。原因：
  - **GAP 抹平空间布局** —— 方向本质是空间/朝向属性，全局平均池化后，
    无论骨干是否编码方向都可能读不出（**最要害**）；
  - 线性不可读 != 无信息（非线性读出可能可行）；
  - 标签取自**整图 mask**，特征取自 `Resize(256)+CenterCrop(224)`，存在**输入-标签空间错配**；
  - 样本量小（262/58/92）、单划分；A 与 B 各选各的 lambda，未统一。

**尤其不可引用 A vs B 的对比**：必须同时看 val 与 test —— 若两集符号相反，
则差异属**划分噪声**而非特征差异（本项目首次运行即出现该情形）。

**要下任何实质结论，须补做**：空间特征（feature-map 展平 / mask 区域池化）
+ 非线性读出（kNN/MLP）+ **>=5 次随机划分**，报均值±std 与 CI。

## 落盘
    results/direction_probe_linear.csv
    logs/direction_probe_{ts}.log
"""
import csv
import math
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from train_direction_net import split_by_base, IMG_SIZE, MEAN, STD
from direction_utils import gt_direction

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CK = BASE / "checkpoints"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
FLIP_CKPT = CK / "direction_resnet18_flip_basis.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_backbone(kind):
    """kind='flip' 加载翻转预训练骨干；kind='imagenet' 用 ImageNet 权重。"""
    m = resnet18(weights=None)
    if kind == "imagenet":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    if kind == "flip":
        if not FLIP_CKPT.exists():
            raise FileNotFoundError(f"缺 {FLIP_CKPT}（T0.3 方向网络 basis 权重）")
        sd = torch.load(str(FLIP_CKPT), map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f"  [flip] 加载 {n_loaded} 个键；missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def extract(model, files):
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    X, names = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = tf(Image.fromarray(rgb))[None].to(device)
        X.append(model(x)[0].cpu().numpy())
        names.append(Path(f).name)
    return np.array(X, dtype=np.float64), names


def labels_for(files, rng_seed=None):
    """真实方向标签；rng_seed 给定时返回**随机方向**（阴性对照）。"""
    Y, keep = [], []
    rng = np.random.RandomState(rng_seed) if rng_seed is not None else None
    for f in files:
        if rng is not None:
            a = rng.uniform(0, math.pi)
            Y.append([math.cos(2 * a), math.sin(2 * a)])
            keep.append(Path(f).name)
            continue
        gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        d = gt_direction(gt)
        if d is None:
            continue
        Y.append([float(d[0]), float(d[1])])
        keep.append(Path(f).name)
    return np.array(Y, dtype=np.float64), keep


def ridge_fit(X, Y, lam=1.0):
    """闭式解 W = (X'X + lam I)^-1 X'Y；X 已中心化。返回 (W, xmean, ymean)。"""
    xm = X.mean(0, keepdims=True)
    ym = Y.mean(0, keepdims=True)
    Xc = X - xm
    Yc = Y - ym
    d = X.shape[1]
    A = Xc.T @ Xc + lam * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, xm, ym


def predict(X, W, xm, ym):
    P = (X - xm) @ W + ym
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return P / n


def angle_err(P, Y):
    cos = np.clip(np.sum(P * Y, axis=1), -1, 1)
    return float(np.mean(np.degrees(np.arccos(cos)) / 2.0)), len(cos)


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_probe_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log("# 方向编码线性探针：翻转预训练骨干的特征里有没有方向？")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log()

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    pool, test, _, _ = split_by_base(files, 0.2, seed=0)
    tr, va, _, _ = split_by_base(pool, 0.2, seed=1)
    log(f"# 划分 basis: 训练 {len(tr)} / 验证 {len(va)} / 测试 {len(test)}")

    # ── 标签 ──
    Ytr, ntr = labels_for(tr)
    Yva, nva = labels_for(va)
    Yte, nte = labels_for(test)
    log(f"# 有效方向标签: 训练 {len(Ytr)} / 验证 {len(Yva)} / 测试 {len(Yte)}")
    # 常数基线：恒预测训练集圆形均值
    mdir = Ytr.mean(0); mdir /= max(np.linalg.norm(mdir), 1e-12)
    ce_va, _ = angle_err(np.tile(mdir, (len(Yva), 1)), Yva)
    ce_te, _ = angle_err(np.tile(mdir, (len(Yte), 1)), Yte)
    log(f"# 常数基线（训练集圆形均值方向）: 验证 {ce_va:.2f}° / 测试 {ce_te:.2f}°")
    log(f"# 随机水平: 45.00°")
    log()

    runs = []
    for kind in ["flip", "imagenet"]:
        log(f"=== 骨干 = {kind} ===")
        model = build_backbone(kind)
        Xtr, _ = extract(model, tr)
        Xva, _ = extract(model, va)
        Xte, _ = extract(model, test)
        log(f"  特征维度 {Xtr.shape[1]}  样本 {Xtr.shape[0]}/{Xva.shape[0]}/{Xte.shape[0]}")
        # 选择 λ 用验证集
        best = None
        for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
            W, xm, ym = ridge_fit(Xtr, Ytr, lam)
            e, _ = angle_err(predict(Xva, W, xm, ym), Yva)
            if best is None or e < best[1]:
                best = (lam, e)
        lam = best[0]
        W, xm, ym = ridge_fit(Xtr, Ytr, lam)
        eva, _ = angle_err(predict(Xva, W, xm, ym), Yva)
        ete, _ = angle_err(predict(Xte, W, xm, ym), Yte)
        etr, _ = angle_err(predict(Xtr, W, xm, ym), Ytr)
        log(f"  λ*={lam}  训练 {etr:.2f}°  验证 {eva:.2f}°  测试 {ete:.2f}°")
        runs.append({"feature_source": kind, "lambda": lam,
                     "train_deg": f"{etr:.2f}", "val_deg": f"{eva:.2f}", "test_deg": f"{ete:.2f}"})
        del model
        torch.cuda.empty_cache()
        log("")

    # ── 阴性对照：随机目标 ──
    log("=== 阴性对照：翻转骨干 -> 随机方向（探针应失败）===")
    model = build_backbone("flip")
    Xtr, _ = extract(model, tr)
    Xte, _ = extract(model, test)
    Yr_tr, _ = labels_for(tr, rng_seed=12345)
    Yr_te, _ = labels_for(test, rng_seed=54321)
    W, xm, ym = ridge_fit(Xtr, Yr_tr, 1.0)
    ec, _ = angle_err(predict(Xte, W, xm, ym), Yr_te)
    log(f"  随机目标测试角度误差 = {ec:.2f}°  （若明显低于 45°，说明探针方法有问题）")
    runs.append({"feature_source": "flip->随机目标(阴性对照)", "lambda": 1.0,
                 "train_deg": "-", "val_deg": "-", "test_deg": f"{ec:.2f}"})
    log("")

    # ── 结果汇总（**只陈述，不作因果判断**）──
    a = next(r for r in runs if r["feature_source"] == "flip")
    b = next(r for r in runs if r["feature_source"] == "imagenet")
    log("=== 结果（描述性，本脚本不作因果判断）===")
    log(f"  A 翻转骨干 -> 真实方向 : 验证 {a['val_deg']}°  测试 {a['test_deg']}°")
    log(f"  B ImageNet  -> 真实方向 : 验证 {b['val_deg']}°  测试 {b['test_deg']}°")
    log(f"  C 翻转骨干 -> 随机方向 : 测试 {ec:.2f}° （阴性对照；若明显低于 45 则方法有问题）")
    log(f"  常数基线 {ce_te:.2f}° / 随机 45.00°")
    log("")
    log("  [!] 无论上述数字如何，**均不可**据此推断'骨干是否编码方向'，因为：")
    log("      (i) **GAP 抹平空间布局** —— 方向本质是空间/朝向属性，全局平均池化后")
    log("          无论骨干是否编码方向都可能读不出（最要害）；")
    log("      (ii) 线性不可读 != 无信息；")
    log("      (iii) 标签取自整图 mask，特征取自 Resize(256)+CenterCrop(224)，存在输入-标签空间错配；")
    log("      (iv) 样本量小(262/58/92)、单划分；A 与 B 的 lambda 亦可能不同（本脚本各选各的，未统一）。")
    log("  [!] **A 与 B 的对比尤其不可引用**：请同时看 val 与 test 两列 ——")
    log("      若两集符号相反，则差异属划分噪声而非特征差异。")
    log("  要用本探针下任何结论，须补：**空间特征 + 非线性读出 + >=5 次随机划分**。")

    out = RES_DIR / "direction_probe_linear.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature_source", "lambda", "train_deg", "val_deg", "test_deg"])
        w.writeheader(); w.writerows(runs)
    log("")
    log(f"  [注] 常数基线 测试 {ce_te:.2f}°；随机水平 45.00°；阴性对照 {ec:.2f}°")
    log(f"明细 -> {out}")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
