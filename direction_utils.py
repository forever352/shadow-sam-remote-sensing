"""公共方向工具模块（direction_utils）。

定位：**re-export + 新增方向数学**。
不搬迁任何已被现有脚本 import 的函数（`split_by_base`/`mask_enhanced`/`component_*`
分散在 5+ 个已审查脚本里，移动会波及它们）。本模块只作新脚本
（方向引导提示 `direction_prompt.py` / 方向特征 Adapter `direction_adapter.py`）的单一 import 面。

方向表示约定（全项目统一）：
    阴影是"轴"不是"箭头"，180° 周期 -> 用 **二倍角单位向量 (cos2θ, sin2θ)**。
    θ 为轴角（弧度），物理角度误差 = 向量夹角 / 2。

坐标系：图像像素坐标，x 右 / y 下，角度沿用 `cv2.minAreaRect` 约定。
"""
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import env_setup  # 必须在 import torch 之前（Windows DLL 修复）
import numpy as np
import cv2

# ── re-export：现成可复用函数，只汇聚 import 面，不搬迁 ──
from train_direction_net import split_by_base, base_name, IMG_SIZE, MEAN, STD
from methods import mask_enhanced, mask_c3i, _rm_small, _direction_filter
from direction_experiment import (
    component_direction_corners,
    component_direction_axis,
    component_box_and_direction,
    encode_image,
    predict_points,
    predict_box,
    predict_box_points,
)
from prompt_experiment import component_boxes
from evaluate_finetuned import auto_points

__all__ = [
    # re-export
    "split_by_base", "base_name", "IMG_SIZE", "MEAN", "STD",
    "mask_enhanced", "mask_c3i", "_rm_small", "_direction_filter",
    "component_direction_corners", "component_direction_axis", "component_box_and_direction",
    "encode_image", "predict_points", "predict_box", "predict_box_points",
    "component_boxes", "auto_points",
    # 新增
    "gt_direction", "axis_from_vector", "vector_from_axis",
    "flip_direction_horizontal", "angle_deviation_deg",
    "component_items", "hsv_pseudo_shadow_mask",
    "load_direction_net", "predict_direction",
]


# ══════════════════════════════════════════════════════════════════
# 一、方向标签：从 GT 掩膜算全局主导方向
# ══════════════════════════════════════════════════════════════════

def gt_direction(mask_gray, min_area: int = 200) -> Optional[np.ndarray]:
    """从 GT 阴影掩膜算全局主导方向，返回 (cos2θ, sin2θ) 单位向量，或 None。

    对每个连通域求主轴方向（二倍角），再做圆形均值 -> 全局方向。

    [!] 本函数是 `train_direction_regress.py:47-72` 的**逐字副本**（不改动原文件，
    靠 `python direction_utils.py` 的自测保证两者输出完全一致）。
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


# ══════════════════════════════════════════════════════════════════
# 二、向量 <-> 角度 互转（此前项目里全是内联写法，无统一工具）
# ══════════════════════════════════════════════════════════════════

def axis_from_vector(d) -> float:
    """(cos2θ, sin2θ) 单位向量 -> 轴角 θ  in  [0, π)。"""
    return float(0.5 * math.atan2(float(d[1]), float(d[0])) % math.pi)


def vector_from_axis(theta: float) -> np.ndarray:
    """轴角 θ（弧度）-> (cos2θ, sin2θ) 单位向量 float32。"""
    return np.array([math.cos(2.0 * theta), math.sin(2.0 * theta)], dtype=np.float32)


def angle_deviation_deg(a: float, b: float) -> float:
    """两个**轴角**（弧度）的最小偏差（度），模 180°。与 methods.py:99 同口径。"""
    dev = abs(math.degrees(a) - math.degrees(b)) % 180.0
    return min(dev, 180.0 - dev)


def flip_direction_horizontal(d) -> np.ndarray:
    """水平镜像（`img[:, ::-1]`）下的方向变换。

    推导：水平镜面把轴角 θ->π−θ，二倍角 2θ->2π−2θ，
        故 (cos2θ, sin2θ) -> (cos(2π−2θ), sin(2π−2θ)) = (cos2θ, −sin2θ)。
    注：竖直镜面 θ->−θ，二倍角后同为 (cos2θ, −sin2θ)（轴 180° 周期）。

    [!] 训练时**必须**对翻转后的图同步调用本函数，否则方向条件与图像错位；
    且因为 Adapter 初始恒等（scale=shift=0），这种错位**不会**被等价性自检发现。
    """
    return np.array([float(d[0]), -float(d[1])], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# 三、连通域几何（供方向提示使用）
# ══════════════════════════════════════════════════════════════════

def component_items(mask, min_area: int = 150) -> List[dict]:
    """把粗提取掩膜拆成连通域列表，每个域带几何信息。

    抽取自 `direction_experiment.py:133-157 / 192-218` 的 minAreaRect 逻辑
    （`w < h` 则 `angle += 90` 校正到长轴），统一成 dict 便于下游使用。

    返回：list of {label, area, centroid(x,y), rect(cx,cy,w,h,angle_deg),
                   axis_rad, axis_vec(cos2θ,sin2θ)}
    """
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    items = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 5:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            angle += 90.0
        axis = math.radians(angle)
        items.append({
            "label": i,
            "area": area,
            "centroid": (float(cx), float(cy)),
            "rect": (float(cx), float(cy), float(w), float(h), float(angle)),
            "axis_rad": axis,
            "axis_vec": vector_from_axis(axis),
        })
    return items


# ══════════════════════════════════════════════════════════════════
# 四、伪阴影剔除（光度）：水体 / 深色植被
# ══════════════════════════════════════════════════════════════════

# 阈值**预注册**为默认常量（见 review_log 要求：不得为凑指标而调）
VEG_H_RANGE = (30, 85)     # OpenCV H  in  [0,180)
VEG_S_MIN = 50
WATER_H_RANGE = (85, 130)
WATER_S_MAX = 90
WATER_V_MIN = 110
DARK_RATIO = 0.85          # 阴影 patch 亮度须 < DARK_RATIO x 全图均值


def hsv_pseudo_shadow_mask(patch_bgr, img_v_mean: float = 0.0,
                           patch_radius: int = 5,
                           require_dark: bool = False) -> Tuple[bool, str]:
    """判断一个小 patch 是否**像伪阴影**（水体 / 深色植被）。

    判据只保留申报书明确要求的"水体 / 深色植被"两类（HSV 色彩判据）。
    `require_dark=True` 时才额外启用"须比全图暗"（默认**关闭**）：
    该判据不在申报书内，且实测会杀掉 74%~97% 的候选点（因为它与"点已在阴影掩膜内"
    这一前置条件重复），故默认不用，仅留作消融选项。

    返回：(is_pseudo, reason)
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return True, "empty"
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = float(hsv[..., 0].mean())
    s = float(hsv[..., 1].mean())
    v = float(hsv[..., 2].mean())

    if VEG_H_RANGE[0] <= h <= VEG_H_RANGE[1] and s >= VEG_S_MIN:
        return True, "vegetation"
    if WATER_H_RANGE[0] <= h <= WATER_H_RANGE[1] and s <= WATER_S_MAX and v >= WATER_V_MIN:
        return True, "water"
    if require_dark and img_v_mean > 0 and v > DARK_RATIO * img_v_mean:
        return True, "not-dark"
    return False, "ok"


# ══════════════════════════════════════════════════════════════════
# 五、方向回归网络的加载与推理（模块①的产物，模块②③消费）
# ══════════════════════════════════════════════════════════════════

REGRESS_CKPT = str(Path(__file__).resolve().parent / "checkpoints" / "direction_resnet18_regress.pth")


def load_direction_net(path: str = REGRESS_CKPT, device: str = "cpu"):
    """加载方向回归头（ResNet-18 + fc(512,2)），返回 eval 模式的模型。

    [!] 若文件不存在会**抛 FileNotFoundError**（不静默回退 ImageNet）--
    静默回退会让"预测方向"悄悄变成随机初始化输出，属于最隐蔽的失效。
    """
    import torch
    from torchvision.models import resnet18
    import torch.nn as nn

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"方向回归权重不存在：{p}\n"
            f"请先跑 train_direction_regress.py（它依赖 direction_resnet18_flip_basis.pth，"
            f"由后台队列产出）。"
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(str(p), map_location="cpu"))
    model.to(device).eval()
    return model


_PREDICT_TF = None


def predict_direction(model, img_bgr, device: str = "cpu") -> Optional[np.ndarray]:
    """对单张 BGR 图预测全局方向，返回 (cos2θ, sin2θ) 单位向量，或 None。

    预处理**必须**与 `train_direction_regress.py:134-137` 的评估变换一致：
        Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(MEAN, STD)
    """
    global _PREDICT_TF
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    if _PREDICT_TF is None:
        _PREDICT_TF = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
    if img_bgr is None or img_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = _PREDICT_TF(Image.fromarray(rgb))[None].to(device)
    with torch.no_grad():
        d = F.normalize(model(x)[0], dim=0)
    return d.detach().cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 自测（python direction_utils.py）
# ══════════════════════════════════════════════════════════════════

def _self_test(n_masks: int = 20) -> None:
    import train_direction_regress as tdr

    AISD = Path(__file__).resolve().parent / "AISD" / "AISD"
    masks = sorted((AISD / "Train412" / "mask").glob("*.tif"))[:n_masks]
    if not masks:
        print("[!] 未找到 AISD 掩膜，跳过 gt_direction 等价性测试")
    else:
        max_dev, n_valid = 0.0, 0
        for p in masks:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            a = gt_direction(g)
            b = tdr.gt_direction(g)
            if a is None or b is None:
                assert a is None and b is None, f"{p.name}: None 判定不一致"
                continue
            n_valid += 1
            max_dev = max(max_dev, float(np.abs(a - b).max()))
        print(f"[1] gt_direction 与 train_direction_regress 等价性："
              f"{n_valid} 张有效掩膜，max|Δ| = {max_dev:.3e}")
        assert max_dev == 0.0, "gt_direction 副本与原实现不一致！"

    # 角度 <-> 向量 往返
    worst = 0.0
    for k in range(360):
        th = math.pi * k / 360.0
        d = vector_from_axis(th)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-6
        back = axis_from_vector(d)
        worst = max(worst, angle_deviation_deg(th, back))
    print(f"[2] axis<->vector 往返最大偏差：{worst:.3e} 度")
    assert worst < 1e-4

    # 水平翻转语义：对一张图左右镜像后，其方向应等于原方向做 flip 变换
    if masks:
        g = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        d0 = gt_direction(g)
        d1 = gt_direction(cv2.flip(g, 1))
        if d0 is not None and d1 is not None:
            pred = flip_direction_horizontal(d0)
            dev = math.degrees(math.acos(float(np.clip(np.dot(pred, d1), -1, 1)))) / 2.0
            print(f"[3] 翻转方向变换一致性：偏差 {dev:.2f} 度（应~=0，掩膜噪声下 <15 可接受）")

    # 角度偏差工具
    assert abs(angle_deviation_deg(0.0, math.radians(170)) - 10.0) < 1e-6
    assert abs(angle_deviation_deg(0.0, math.radians(10)) - 10.0) < 1e-6
    print("[4] angle_deviation_deg 通过")
    print("\n[OK] direction_utils 自测全部通过")


if __name__ == "__main__":
    _self_test()
