"""方向标签质量诊断（落盘证据版）。

## 为什么需要它

第五轮审查判定：此前"T1.1 任务不可学"的结论**无落盘证据、且逻辑不成立**
（训练用 RandomResizedCrop 随机裁剪，标签却是整图方向 -> 输入-标签系统性错配）。
本脚本把诊断**正式落盘**，供后续引用。

## 核心假设（监督智能体提出，本脚本验证）

**太阳方位角是"原图级"常量**：同一张**原始大图**切出的所有 `_subN` 子图，
其太阳方位角**必然相同**。因此：
  - 逐子图各算方向 -> 每图仅 20 个连通域、方向一致性 R 中位仅 0.34（噪声大）
  - **按原图基名汇总**所有子图的连通域再取方向 -> 样本量 xN，噪声应显著下降

若"基名级 R"显著高于"子图级 R"，则说明**此前的标签定义是主要噪声源**，
而非"任务不可学"。

## 落盘
  results/direction_label_quality.csv   逐图/逐基名的 R、连通域数、阴影占比
  logs/direction_label_diag_{ts}.log    汇总统计 + 结论
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


def component_angles(mask_gray, min_area=MIN_AREA, min_ar=0.0):
    """返回 [(angle_rad, aspect_ratio, area, cx, cy), ...]，与 gt_direction 同口径。"""
    m = (mask_gray > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 5:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            angle += 90
            w, h = h, w
        ar = w / max(h, 1e-6)
        if ar < min_ar:
            continue
        out.append((math.radians(angle), ar, area, float(cx), float(cy)))
    return out


def concentration(angles_rad):
    """二倍角圆形集中度 R ∈ [0,1]（0=杂乱, 1=全同向）。"""
    if not angles_rad:
        return 0.0
    a = np.array(angles_rad) * 2.0
    return float(math.hypot(np.cos(a).mean(), np.sin(a).mean()))


def mean_dir(angles_rad):
    """二倍角圆形均值 -> (cos2θ, sin2θ)。"""
    a = np.array(angles_rad) * 2.0
    c, s = float(np.cos(a).mean()), float(np.sin(a).mean())
    n = math.hypot(c, s)
    return (c / n, s / n) if n > 0 else (0.0, 0.0)


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_label_diag_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    # 与下游完全同一划分
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    tr_files, va_files, _, _ = split_by_base(train_pool, 0.2, seed=1)

    log("# 方向标签质量诊断")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  最小连通域面积={MIN_AREA}")
    log(f"# 划分: basis 68/16/21 -> 训练 {len(tr_files)} / 验证 {len(va_files)} / 测试 {len(test_files)}")
    log()

    rows = []
    per_img_R, per_img_n = [], []
    all_angles = []
    base_angles = defaultdict(list)
    base_nimg = defaultdict(int)

    for split_name, fs in [("train", tr_files), ("val", va_files), ("test", test_files)]:
        for f in fs:
            name = Path(f).name
            gt = cv2.imread(str(AISD / "Train412" / "mask" / name), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            cs = component_angles(gt)
            angs = [a for a, _, _, _, _ in cs]
            R = concentration(angs)
            ratio = float((gt > 0).mean())
            bn = base_name(f)
            base_angles[bn].extend(angs)
            base_nimg[bn] += 1
            per_img_R.append(R); per_img_n.append(len(angs))
            all_angles.extend(angs)
            rows.append({"level": "image", "split": split_name, "key": name,
                         "n_comp": len(angs), "R": round(R, 4),
                         "shadow_ratio": round(ratio, 5)})

    # 基名级
    base_R = []
    for bn, angs in base_angles.items():
        R = concentration(angs)
        base_R.append(R)
        rows.append({"level": "base", "split": "-", "key": bn,
                     "n_comp": len(angs), "R": round(R, 4),
                     "shadow_ratio": ""})

    # ── 汇总 ──
    per_img_R = np.array(per_img_R); per_img_n = np.array(per_img_n)
    base_R = np.array(base_R)
    log(f"【子图级】{len(per_img_R)} 张")
    log(f"  连通域数: 中位 {np.median(per_img_n):.0f} (范围 {per_img_n.min()}~{per_img_n.max()})")
    log(f"  方向一致性 R: 中位 {np.median(per_img_R):.3f}  均值 {per_img_R.mean():.3f}")
    log(f"    R<0.5 占比 {(per_img_R < 0.5).mean()*100:.0f}%   R<0.3 占比 {(per_img_R < 0.3).mean()*100:.0f}%")
    log(f"  全局集中度 Rg = {concentration(all_angles):.3f}")
    log()
    log(f"【原图基名级】{len(base_R)} 个大图（假设：同原图太阳方位角相同）")
    log(f"  每基名连通域数: 中位 {np.median([len(v) for v in base_angles.values()]):.0f}")
    log(f"  方向一致性 R: 中位 {np.median(base_R):.3f}  均值 {base_R.mean():.3f}")
    log(f"  平均每基名含 {np.mean(list(base_nimg.values())):.1f} 张子图")
    log()
    delta = float(np.median(base_R) - np.median(per_img_R))
    log(f"* 结论：基名级 R 中位 - 子图级 R 中位 = {delta:+.3f}")
    if delta > 0.15:
        log("  -> 基名级显著更一致：**此前的标签定义（逐子图）是主要噪声源**，")
        log("    应改用「按原图基名汇总」的方向标签。")
    else:
        log("  -> 两者接近：方向噪声来自图像本身，而非标签聚合层级。")
    log(f"运行日志 -> {log_path}")

    p = RES_DIR / "direction_label_quality.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "split", "key", "n_comp", "R", "shadow_ratio"])
        w.writeheader(); w.writerows(rows)
    log(f"明细 -> {p}")
    logf.close()


if __name__ == "__main__":
    main()
