"""方向标签一致性探针（不训练、不占 GPU、分钟级）。

## 背景

第五/六轮审查判定：此前对 T1.1 的定性（"任务不可学"、"过拟合"）**均属过度声明**。
根因之一是**方向标签本身的定义**：现用标签 = 阴影连通域的 `cv2.minAreaRect` **最小外接矩形长轴**，
它取决于「**建筑朝向 + 太阳方位角**」两者，是**几何混杂量**，未必是太阳方位角的好代理。

本探针**只算指标、不训练**，比较 4 种候选方向估计量的**一致性 R**，
以判断"标签定义"是不是主要噪声源：

  1. `rect_axis`  -- 现状：minAreaRect 长轴（基线）
  2. `pca_axis`   -- 连通域像素 PCA 主轴
  3. `to_bright`  -- **物理量**：阴影质心 -> 邻域最亮区块质心 的连线
                     （阴影被投向背光侧，故该向量指向光源侧；更接近真实方位角）
  4. `edge_grad`  -- 阴影边界上强度梯度方向的二倍角均值

对每种估计量，同时报告：
  - 子图级 R 中位 / 基名级 R 中位
  - **随机基线**（均匀分布的二倍角圆形 R ~= 0，对应角误差 ~=45°）
  - **样本量偏差对照**：把 `rect_axis` 降采样到与 `to_bright` 相同样本数后再算 R，隔离"样本越少 R 越高"的偏差

## 落盘
  results/direction_label_probe.csv
  logs/direction_label_probe_{ts}.log
"""
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2

from train_direction_net import split_by_base, base_name

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
MIN_AREA = 200
RING_K = 15          # to_bright：向外扩的环宽


def _comps(mask, min_area=MIN_AREA):
    m = (mask > 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 5:
            continue
        out.append({"label": i, "area": int(st[i, cv2.CC_STAT_AREA]),
                    "centroid": (float(cen[i][0]), float(cen[i][1])),
                    "xs": xs, "ys": ys})
    return out, m, lab


def est_rect_axis(c, _):
    pts = np.column_stack([c["xs"], c["ys"]]).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        ang += 90
    return math.radians(ang)          # 轴角 ∈ [0,180)


def est_pca_axis(c, _):
    pts = np.column_stack([c["xs"], c["ys"]]).astype(np.float32)
    pts = pts - pts.mean(0)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    d = v[:, -1]                       # 最大特征向量
    return math.atan2(d[1], d[0])      # 轴角，稍后统一 mod 180°


def est_to_bright(c, ctx):
    """阴影质心 -> 邻域(环内)最亮连通块质心 的连线。指向光源侧，更接近方位角。"""
    img = ctx["gray"]
    # [!] 必须用**当前连通域自己**的掩膜，不能用整图掩膜（否则所有域算出同一个角度）
    own = (ctx["lab"] == c["label"]).astype(np.uint8)
    ring = cv2.dilate(own, np.ones((2 * RING_K + 1,) * 2, np.uint8)) - own
    ys, xs = np.where(ring > 0)
    if len(xs) < 10:
        return None
    vals = img[ys, xs]
    thr = np.quantile(vals, 0.90)      # 环内最亮 10%
    sel = vals >= thr
    if sel.sum() < 5:
        return None
    bx, by = float(xs[sel].mean()), float(ys[sel].mean())
    cx, cy = c["centroid"]
    return math.atan2(by - cy, bx - cx)   # 有向角


def est_edge_grad(c, ctx):
    """阴影边界处强度梯度方向的二倍角均值（梯度从暗到亮）。"""
    gx, gy = ctx["gx"], ctx["gy"]
    own = (ctx["lab"] == c["label"]).astype(np.uint8)   # [!] 同上：用本域掩膜
    k = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(own, k) - cv2.erode(own, k)
    ys, xs = np.where(edge > 0)
    if len(xs) < 10:
        return None
    ang = np.arctan2(gy[ys, xs], gx[ys, xs])
    a = ang * 2
    return math.atan2(np.sin(a).mean(), np.cos(a).mean()) / 2   # 折回轴角


def circ_R(angles_rad):
    """二倍角圆形集中度 R ∈ [0,1]。"""
    if len(angles_rad) < 2:
        return float("nan")
    a = np.array(angles_rad) * 2.0
    return float(math.hypot(np.cos(a).mean(), np.sin(a).mean()))


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_label_probe_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    tr, va, _, _ = split_by_base(train_pool, 0.2, seed=1)
    allf = tr + va + test_files

    log("# 方向标签一致性探针（不训练）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  样本: {len(allf)} 张  最小连通域={MIN_AREA}")
    log()

    ests = {"rect_axis": est_rect_axis, "pca_axis": est_pca_axis,
            "to_bright": est_to_bright, "edge_grad": est_edge_grad}
    per_img = defaultdict(list)     # est -> [R_img, ...]
    per_img_n = defaultdict(list)   # est -> [该图连通域数, ...]（供样本量匹配基线）
    per_base = defaultdict(lambda: defaultdict(list))
    n_ang = defaultdict(int)
    DEGEN_CHECKED = [0]             # 退化检测只看前 5 张图，避免拖慢

    for f in allf:
        gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gt is None or img is None:
            continue
        comps, m, lab = _comps(gt)
        if not comps:
            continue
        gray = cv2.GaussianBlur(img, (5, 5), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ctx = {"gray": gray, "mask": m, "lab": lab, "gx": gx, "gy": gy}

        # ── 退化检测：防止"所有连通域算出同一个角度"这类 bug（R 会假性=1）──
        if DEGEN_CHECKED[0] < 5:
            for nm, fn2 in ests.items():
                vals = [fn2(c, ctx) for c in comps[:8]]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 3:
                    spread = float(np.std(np.array(vals) * 2))
                    if spread < 1e-6:
                        raise RuntimeError(
                            f"退化检测失败：估计量 '{nm}' 在同一图的不同连通域上给出完全相同角度"
                            f" -- 极可能用错了掩膜（应为本域掩膜）。拒绝产出 R~=1 的假结果。")
            DEGEN_CHECKED[0] += 1

        bn = base_name(f)
        for name, fn in ests.items():
            angs = []
            for c in comps:
                a = fn(c, ctx)
                if a is not None:
                    angs.append(a)
            if len(angs) < 2:
                continue
            per_img[name].append(circ_R(angs))
            per_img_n[name].append(len(angs))
            per_base[bn][name].extend(angs)
            n_ang[name] += len(angs)

    # 汇总
    rows = []
    log(f"{'估计量':<12}{'角度总数':<10}{'子图级R中位':<14}{'基名级R中位':<14}{'基名级R均值':<12}")
    base_R_med = {}
    for name in ests:
        bi = np.array(per_img[name])
        bR = np.array([circ_R(v[name]) for v in per_base.values() if len(v[name]) >= 2])
        bR = bR[~np.isnan(bR)]
        med_i = float(np.median(bi)) if len(bi) else float("nan")
        med_b = float(np.median(bR)) if len(bR) else float("nan")
        mean_b = float(np.mean(bR)) if len(bR) else float("nan")
        base_R_med[name] = med_b
        log(f"{name:<12}{n_ang[name]:<10}{med_i:<14.3f}{med_b:<14.3f}{mean_b:<12.3f}")
        rows.append({"estimator": name, "n_angles": n_ang[name],
                     "R_img_median": round(med_i, 4), "R_img_mean": round(float(np.mean(bi)), 4),
                     "R_base_median": round(med_b, 4), "R_base_mean": round(mean_b, 4)})

    # 样本量偏差对照：R 是有偏估计，样本越少越偏高 -- 必须做样本量匹配
    log()
    log("【样本量匹配的随机基线】（关键：R 随样本量变小而系统性偏高）")
    log("  对每张图，用**该图实际的连通域数 n** 生成 n 个随机方向算 R，重复 200 次取均值；")
    log("  只有观测 R 显著高于该基线，才能说存在方向信号。")
    log()
    rng = np.random.RandomState(0)
    matched_baseline = {}
    log(f"{'估计量':<12}{'观测R(均值)':<14}{'匹配随机基线':<14}{'超出':<10}")
    for name in ests:
        obs_arr = np.array(per_img[name], dtype=float)
        ns = np.array(per_img_n[name], dtype=int)
        if len(obs_arr) == 0:
            continue
        base = []
        for nn in ns:
            base.append(np.mean([circ_R(rng.uniform(0, math.pi, int(nn))) for _ in range(200)]))
        base = float(np.mean(base))
        matched_baseline[name] = base
        rows_by = [r for r in rows if r["estimator"] == name]
        if rows_by:
            rows_by[0]["R_img_mean"] = round(float(np.mean(obs_arr)), 4)
        obs = float(np.mean(obs_arr))
        log(f"{name:<12}{obs:<14.3f}{base:<14.3f}{obs-base:<10.3f}")
    log()
    log("  解析近似（n 个均匀随机角度的 R 期望 ~= sqrt(pi/(4n))）：")
    for nn in [5, 10, 20, 40, 80]:
        log(f"    n={nn:<4} -> {math.sqrt(math.pi / (4 * nn)):.3f}")
    log(f"  本项目每图连通域数中位 ~= {int(np.median([n for v in per_img_n.values() for n in v]))}")

    log()
    log(f"运行日志 -> {log_path}")
    # 把样本量匹配基线并入 rows 再落盘
    for r in rows:
        r["R_matched_random_baseline"] = round(matched_baseline.get(r["estimator"], float("nan")), 4)
        r["R_excess_over_baseline"] = round(r["R_img_mean"] - matched_baseline.get(r["estimator"], float("nan")), 4)

    p = RES_DIR / "direction_label_probe.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["estimator", "n_angles", "R_img_median", "R_img_mean",
                                           "R_base_median", "R_base_mean",
                                           "R_matched_random_baseline", "R_excess_over_baseline"])
        w.writeheader(); w.writerows(rows)
    log(f"明细 -> {p}")
    logf.close()


if __name__ == "__main__":
    main()
