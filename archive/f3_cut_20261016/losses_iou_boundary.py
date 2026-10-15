"""F3 用的损失：IoU Loss + Boundary Loss（申报书点名的"混合损失函数"）。

## 出处与实现声明（**必须随结果一并出现**）

申报书原文只写："设计混合损失函数：结合 **IoU Loss**（关注整体重合度）和
**Boundary Loss**（关注阴影边缘的精细度）"，**未给出公式**。
故本模块是**按原理实现、非逐行复现申报书**（同 DTICT 的处理）。

## 公式（预注册，开跑前定死，此后不得再改）

- **IoU Loss**：`1 - (Σ p·g + ε) / (Σ p + Σ g - Σ p·g + ε)`，`p=sigmoid(logits)`；
  在 256×256 上**逐图**算再对该 batch 平均（与 Dice 同尺度）。
- **Boundary Loss**（Kervadec et al. 2019，**有符号**距离图）：
  `φ(q) = -D_G(q) if q∈G else +D_G(q)`，`D_G` = 到 GT 边界的欧氏距离；
  `L_B = Σ_q φ(q)·p(q) / (Σ_q |φ(q)| + ε)`，逐图算再平均。
- **组合**：`L = Focal + IoU + λ·Boundary`，**λ = 1.0 固定**（不用 Kervadec 的渐进 α）。

## 为什么保留 Focal（第十八+轮审查裁定）

阴影是少数类；纯"区域+边界"项缺密集逐像素梯度、易训练不稳。
申报书写"结合 IoU 和 Boundary"**并不排斥**保留 Focal，故保留。

## 硬约束（审查强制）

1. **损失必须在 `autocast()` 之外、以 fp32 计算** —— IoU 的 `Σp` 在 256²=65536 个元素上
   若落进 fp16 会**溢出**。（调用方 `train_lora_leakfree_ioubound.py` 已保证。）
2. **距离图在调用处对"已翻转+已 resize 的 gt256"现算**，本模块不做缓存 ——
   避免与 `train_step` 的随机翻转**错位**（那是静默错误，日志看不出来）。
3. **空 GT / 全满 GT 的 guard**：无边界可言，该样本的边界项跳过（不产生 NaN）。
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

LAMBDA_BOUNDARY = 1.0   # 预注册固定值。改动须重送审。


def iou_loss(logits, target, eps=1e-6):
    """软 IoU 损失（逐图算再平均）。logits/target: (B,1,H,W)。"""
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dim=dims)
    union = p.sum(dim=dims) + target.sum(dim=dims) - inter
    return (1.0 - (inter + eps) / (union + eps)).mean()


def _signed_distance(gt_bool):
    """有符号距离图：GT 内为负、外为正，量纲=像素。"""
    d_out = distance_transform_edt(~gt_bool)   # 外部像素 -> 到 GT 的距离；内部为 0
    d_in = distance_transform_edt(gt_bool)     # 内部像素 -> 到边界的距离；外部为 0
    return np.where(gt_bool, -d_in, d_out).astype(np.float32)


def boundary_loss(logits, target, eps=1e-6):
    """Kervadec 有符号边界损失，逐图算再平均。

    [!] 调用方必须传入**与图像同一次翻转之后**的 target，否则距离图与预测错位。
    """
    p = torch.sigmoid(logits)
    b = target.shape[0]
    total = None
    n = 0
    for i in range(b):
        g = target[i, 0].detach().cpu().numpy() > 0.5
        s = int(g.sum())
        if s == 0 or s == g.size:        # guard：无边界
            continue
        phi = _signed_distance(g)
        phi_t = torch.as_tensor(phi, dtype=torch.float32, device=logits.device)
        term = (phi_t * p[i, 0]).sum() / (phi_t.abs().sum() + eps)
        total = term if total is None else total + term
        n += 1
    if n == 0:
        return torch.zeros((), device=logits.device)
    return total / n


def combined_loss(logits, target, focal_loss_fn):
    """L = Focal + IoU + λ·Boundary。**必须在 autocast 之外调用（fp32）。**"""
    return (focal_loss_fn(logits, target)
            + iou_loss(logits, target)
            + LAMBDA_BOUNDARY * boundary_loss(logits, target))
