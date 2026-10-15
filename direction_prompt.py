"""模块②：方向引导动态提示生成（Direction-Guided Dynamic Prompt Generator）。

对应申报书原文：
  "利用色彩空间变换（如RGB转HSV）结合改进的阴影指数（C3I），快速生成粗糙的疑似阴影区域。
   其次，结合上述提取的'方向特征向量'，设计一种'射线投射采样法'。具体而言，算法将计算
   疑似阴影区域的形心（Centroid），并结合方向向量构建扇形搜索区域。在此区域内，根据像素
   梯度筛选高置信度的前景点作为Prompt，从而剔除因水体或深色植被造成的'伪阴影'干扰。"

## 算法链路

    ① 粗提取：mask_enhanced(CLAHE+Otsu+形态学) & mask_c3i(LAB 阴影指数) -> 清理小连通域
    ② 逐连通域：取形心 + 主轴角（component_items）
    ③ 伪阴影剔除①（几何）：主轴角与全局方向偏差 > dev_thresh -> 丢弃
    ④ 扇形搜索：以形心为中心、沿方向轴 ±phi 张角发射 n_rays 条射线
    ⑤ 射线采样：每条射线上取 n_samples 个点，记录梯度
    ⑥ 高置信筛选：保留梯度 top grad_keep，且落在该连通域/粗掩膜内
    ⑦ 伪阴影剔除②（光度）：HSV 判据剔除水体/植被/不够暗的点
    ⑧ 每域均匀下采样 <= n_fg_max 个前景点；掩膜外随机采 n_bg 个背景点

## [!] 参数预注册（审查要求）

`CFG` 里的阈值是**预注册默认值**。按审查纪律：
  - **调参只允许在验证集（58 张 / 16 场景）**做；
  - **主表只报这套默认参数的结果**；
  - 消融三条件（none / random / project）除"方向来源"外**其余参数完全一致**。
以此避免"调参调到好看"的嫌疑。

用法：
    python direction_prompt.py --image <path> --direction 0.6,0.8      # 单图打印采样点
    python direction_prompt.py --image <path> --direction-source oracle
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2

from direction_utils import (
    mask_enhanced, mask_c3i, _rm_small, gt_direction, component_items,
    axis_from_vector, vector_from_axis, angle_deviation_deg, hsv_pseudo_shadow_mask,
)

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"


@dataclass
class DirPromptConfig:
    """参数**预注册**于此，主表只报这套默认值（见模块 docstring 的审查纪律）。"""
    min_area: int = 150        # 连通域最小面积
    dev_thresh_deg: float = 45.0   # 伪阴影剔除①：主轴与全局方向最大允许偏差（度）
    phi_deg: float = 30.0      # 扇形半张角
    n_rays: int = 7            # 每条扇区的射线条数
    n_samples: int = 16        # 每条射线的采样点数
    radius_k: float = 1.6      # 搜索半径 R = radius_k * max(w,h)/2
    grad_keep: float = 0.30    # 保留梯度最高的比例
    n_fg_max: int = 5          # 每个连通域最多保留的前景点
    n_bg: int = 10             # 背景点数
    sector_mode: str = "axis"  # axis / perp / cross
    sector_source: str = "global"  # global（整图方向，申报书口径）/ component（逐域自身主轴）
    apply_dir_filter: bool = True  # 是否启用"方向一致性过滤"（丢弃与全局方向不符的连通域）
                                   # [!] 这是**项目自加项**，不在申报书描述内；关掉它才能把
                                   #    "扇形的作用"与"过滤的作用"分开（第八轮审查要求）
    patch_radius: int = 5      # 光度判据的邻域半径
    require_dark: bool = False  # 额外启用"须比全图暗"（默认关；不在申报书内，见 hsv_pseudo_shadow_mask）
    fallback_to_none: bool = True  # 方向过滤后前景点为 0 时，退用无方向版（防空掩膜）
    seed: int = 0              # 背景点采样随机种子（可复现）


# ══════════════════════════════════════════════════════════════════
# 粗提取（来源**写死**，保证可复现）
# ══════════════════════════════════════════════════════════════════

def coarse_shadow_mask(img_bgr, cfg: DirPromptConfig):
    """粗糙疑似阴影区域 = mask_enhanced & mask_c3i，再去小连通域。

    两个来源都写死在此（审查要求：不得含糊"粗提取 mask 来自哪"）：
      - `methods.mask_enhanced`：CLAHE + 反相 Otsu + 形态学开运算
      - `methods.mask_c3i`    ：LAB 空间 C3I 阴影指数 + 阈值
    交集兼顾"形态学干净"与"光谱判据"，是申报书"HSV/C3I 快速粗提取"的落地。

    **兜底**：交集为空时退用 `mask_enhanced` 单路（实测存在两路交集为空的图，
    若不兜底该图会直接输出空掩膜、指标为 0）。
    """
    a = mask_enhanced(img_bgr)
    b = mask_c3i(img_bgr)
    inter = cv2.bitwise_and(a, b)
    m = _rm_small(inter, cfg.min_area)
    if m.sum() == 0:
        m = _rm_small(a, cfg.min_area)
    return m


# ══════════════════════════════════════════════════════════════════
# 扇形射线采样
# ══════════════════════════════════════════════════════════════════

def _sector_bases(theta_g: float, mode: str) -> List[float]:
    """返回扇形中心角列表。"""
    if mode == "axis":
        return [theta_g]
    if mode == "perp":
        return [theta_g + math.pi / 2]
    if mode == "cross":
        return [theta_g, theta_g + math.pi / 2]
    raise ValueError(f"未知 sector_mode: {mode}")


def ray_sample(centroid, half_len: float, theta_g: float, cfg: DirPromptConfig,
               shape) -> List[Tuple[int, int]]:
    """从形心沿扇形方向发射射线，返回落在图内的采样点 (x, y) 列表。"""
    h, w = shape[:2]
    cx, cy = centroid
    r = cfg.radius_k * half_len
    phi = math.radians(cfg.phi_deg)
    pts: List[Tuple[int, int]] = []
    for base in _sector_bases(theta_g, cfg.sector_mode):
        for alpha in np.linspace(base - phi, base + phi, cfg.n_rays):
            for t in np.linspace(0.15 * r, r, cfg.n_samples):
                x = int(round(cx + t * math.cos(alpha)))
                y = int(round(cy + t * math.sin(alpha)))
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
    # 去重（不同射线可能落在同一像素）
    return list(dict.fromkeys(pts))


# ══════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════

def build_direction_points(img_bgr, d_global,
                           cfg: Optional[DirPromptConfig] = None,
                           mode: str = "project",
                           stats: Optional[dict] = None):
    """生成方向引导的前景/背景点提示。

    参数：
        img_bgr : (H,W,3) uint8
        d_global: (cos2θ, sin2θ) 单位向量；mode='random' / sector_source='component' 时可忽略
        mode    : 'project'（方向引导）| 'none'（同点数、无扇区）| 'random'（固定随机方向）
        stats   : 可选 dict，函数会把**漏斗中间量**填进去（供失效模式 #4/#5 盯防）：
                  n_comp_total / n_dropped_dir / n_cand / n_after_grad /
                  n_dropped_photo / n_fg / n_fg_before_downsample / used_fallback
    返回：
        (coords (M,2) float64, labels (M,) int64)  与 `predict_points` 兼容；无候选返回 (None, None)
    """
    cfg = cfg or DirPromptConfig()
    if stats is not None:
        stats.clear()
        stats.update(dict(n_comp_total=0, n_dropped_dir=0, n_cand=0, n_after_grad=0,
                          n_dropped_photo=0, n_fg_before_downsample=0, n_fg=0,
                          used_fallback=False))
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)

    coarse = coarse_shadow_mask(img_bgr, cfg)
    if coarse.sum() == 0:
        return None, None

    # 背景点候选：粗掩膜之外
    rng = np.random.RandomState(cfg.seed)
    bg_ys, bg_xs = np.where(coarse == 0)
    if len(bg_xs) > 0:
        sel = rng.choice(len(bg_xs), size=min(cfg.n_bg, len(bg_xs)), replace=False)
        bg_pts = list(zip(bg_xs[sel].tolist(), bg_ys[sel].tolist()))
    else:
        bg_pts = []

    img_v_mean = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 2].mean())

    if mode == "random":
        # 固定随机方向：由全局种子 + 图像尺寸派生，可复现且与图像内容无关
        rr = np.random.RandomState(cfg.seed + 12345)
        ang = float(rr.uniform(0, math.pi))
        theta_g = ang
    else:
        theta_g = axis_from_vector(d_global)

    fg_pts: List[Tuple[int, int]] = []
    n_dropped_dir = n_dropped_photo = 0
    n_comp_total = n_cand_total = n_after_grad_total = n_fg_before_ds = 0

    # 连通域标签图（供 mode='none' 在同一连通域内取点；与 component_items 同掩膜、同 8-连通）
    _, labels_img = cv2.connectedComponents((coarse > 0).astype(np.uint8), 8)

    for item in component_items(coarse, cfg.min_area):
        n_comp_total += 1
        cx, cy = item["centroid"]
        _, _, bw, bh, _ = item["rect"]
        half_len = max(bw, bh) / 2.0

        # 扇形中心角 θ_i 的来源：
        #   global    -- 用整图的全局方向（申报书原文口径；依赖模块①的方位角向量）
        #   component -- 用**该连通域自身的主轴**（几何可得、无需网络；每域内部良定义）
        if mode == "project":
            if cfg.sector_source == "component":
                theta_i = item["axis_rad"]
            else:
                # ③ 伪阴影剔除①（几何）：与全局方向不一致的连通域丢弃
                #    [!] 仅当 apply_dir_filter=True；关掉它是为了把"扇形"与"过滤"两个变量分开
                if cfg.apply_dir_filter and \
                        angle_deviation_deg(item["axis_rad"], theta_g) > cfg.dev_thresh_deg:
                    n_dropped_dir += 1
                    continue
                theta_i = theta_g
        else:
            theta_i = theta_g

        if mode == "none":
            # 对照：同样点数，但**不建扇形**--在该连通域内均匀取点，隔离"方向"与"点数"
            ys, xs = np.where(labels_img == item["label"])
            if len(xs) == 0:
                continue
            k = min(cfg.n_fg_max * cfg.n_rays, len(xs))
            sel = rng.choice(len(xs), size=k, replace=False)
            cand = list(zip(xs[sel].tolist(), ys[sel].tolist()))
        else:
            cand = ray_sample((cx, cy), half_len, theta_i, cfg, img_bgr.shape)

        if not cand:
            continue
        n_cand_total += len(cand)

        # ④ 高置信：**只在落在粗掩膜内的候选点之间**按梯度排名取 top grad_keep。
        #    [!] 必须先在掩膜内过滤再排名：射线末端位于阴影**边界外**、梯度反而最高，
        #    若在全候选上取分位数，选出的全是边界外点，随后会被 in-mask 判据全部丢弃
        #    （实测 552 个候选 -> 仅 19 个通过，是本模块最初的 bug）。
        inside = [(x, y) for (x, y) in cand if coarse[y, x] > 0]
        if not inside:
            continue
        grads_in = np.array([grad[y, x] for x, y in inside], dtype=np.float32)
        thr = np.quantile(grads_in, 1.0 - cfg.grad_keep)
        keep = [p for p, g in zip(inside, grads_in) if g >= thr]
        n_after_grad_total += len(keep)

        # ⑤ 伪阴影剔除②（光度）：水体 / 深色植被（申报书明确要求的两类）
        kept2 = []
        for (x, y) in keep:
            r = cfg.patch_radius
            patch = img_bgr[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            is_pseudo, _ = hsv_pseudo_shadow_mask(
                patch, img_v_mean, cfg.patch_radius, require_dark=cfg.require_dark)
            if is_pseudo:
                n_dropped_photo += 1
                continue
            kept2.append((x, y))

        # 均匀下采样到 <= n_fg_max
        n_fg_before_ds += len(kept2)
        if len(kept2) > cfg.n_fg_max:
            idx = np.linspace(0, len(kept2) - 1, cfg.n_fg_max).round().astype(int)
            kept2 = [kept2[i] for i in idx]
        fg_pts.extend(kept2)

    def _fill_stats(used_fallback=False):
        if stats is None:
            return
        stats.update(dict(n_comp_total=n_comp_total, n_dropped_dir=n_dropped_dir,
                          n_cand=n_cand_total, n_after_grad=n_after_grad_total,
                          n_dropped_photo=n_dropped_photo,
                          n_fg_before_downsample=n_fg_before_ds,
                          n_fg=len(fg_pts), used_fallback=used_fallback))

    if not fg_pts and not bg_pts:
        _fill_stats()
        return None, None

    # 兜底：方向过滤把所有连通域都判掉时（实测存在极端图），前景点会为 0 ->
    # 该图将输出空掩膜、IoU=0。此时退用 mode='none'（同样点数的无方向版），
    # 保证方法在任一图上都能给出非空预测。**此兜底必须记入日志/统计**。
    if not fg_pts and mode == "project" and cfg.fallback_to_none:
        _fill_stats(used_fallback=True)
        c2, l2 = build_direction_points(img_bgr, d_global, cfg, mode="none")
        if c2 is not None:
            return c2, l2
        return None, None

    _fill_stats()
    coords = np.array(fg_pts + bg_pts, dtype=np.float64) if (fg_pts or bg_pts) else None
    labels = np.array([1] * len(fg_pts) + [0] * len(bg_pts), dtype=np.int64)
    return coords, labels


# ══════════════════════════════════════════════════════════════════
# 自测 / 单图调试
# ══════════════════════════════════════════════════════════════════

def _self_test():
    masks = sorted((AISD / "Train412" / "mask").glob("*.tif"))[:5]
    if not masks:
        print("[!] 未找到 AISD 数据，跳过")
        return
    cfg = DirPromptConfig()
    for mp in masks:
        ip = AISD / "Train412" / "shadow" / mp.name
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        d = gt_direction(gt)
        if d is None:
            continue
        print(f"--- {mp.name}  方向 d=({d[0]:.3f},{d[1]:.3f})  θ={math.degrees(axis_from_vector(d)):.1f}°")
        for mode in ("project", "none", "random"):
            coords, labels = build_direction_points(img, d, cfg, mode=mode)
            if coords is None:
                print(f"    {mode:<8}: 无候选点")
            else:
                nf = int((labels == 1).sum()); nb = int((labels == 0).sum())
                print(f"    {mode:<8}: 前景 {nf} 点 / 背景 {nb} 点")
    print("\n[OK] direction_prompt 运行正常")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--direction", type=str, default=None, help="如 0.6,0.8")
    ap.add_argument("--direction-source", choices=["oracle", "regress", "random"],
                    default="oracle")
    ap.add_argument("--mode", choices=["project", "none", "random"], default="project")
    args = ap.parse_args()

    if args.image is None:
        _self_test()
    else:
        img = cv2.imread(args.image)
        if args.direction:
            d = np.array([float(v) for v in args.direction.split(",")], dtype=np.float32)
            d = d / np.linalg.norm(d)
        else:
            gt_path = AISD / "Train412" / "mask" / Path(args.image).name
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            d = gt_direction(gt) if gt is not None else None
        if d is None:
            print("[!] 无法确定方向")
        else:
            c, l = build_direction_points(img, d, mode=args.mode)
            print(f"方向 d={d}  mode={args.mode}  点数={0 if c is None else len(c)}")
            if c is not None:
                print("前景点:", c[l == 1][:10].tolist())
