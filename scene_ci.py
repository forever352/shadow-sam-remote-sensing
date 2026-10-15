"""主表的场景级统计：按【原始大图】聚合，出配对比较的置信区间与检验。

## 为什么必须做

主表是 92 张测试子图的均值，但这 92 张只来自 **21 张原始大图（场景）**。
**独立单元是场景（21），不是子图（92）**--同场景的 3~5 张子图高度相关。
直接对 92 张算 std / t 检验会**低估不确定性、夸大显著性**。

本脚本：
  1. 按原图基名聚合 -> 每场景 1 个均值（n=21）
  2. 对 `LoRA vs U-Net`、`LoRA vs SegNet` 做**配对**比较（同一场景配对）
  3. 给出**场景级**差值均值、bootstrap 95% CI、配对 t 检验 p 值
  4. 同时给出**子图级**的同口径数字作对照（展示差异）

## 落盘
    results/scene_level_ci.csv
    logs/scene_ci_{ts}.log
"""
import csv
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import env_setup
import numpy as np

BASE = Path(__file__).resolve().parent
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

SOURCES = {
    "LoRA-SAM": "lora_leakfree_test_per_image.csv",
    "U-Net": "unet_leakfree_test_per_image.csv",
    "SegNet": "segnet_leakfree_test_per_image.csv",
}


def base_of(name):
    m = re.match(r"(.+?)_sub\d+", Path(name).stem)
    return m.group(1) if m else Path(name).stem


def bootstrap_ci(x, n_boot=20000, alpha=0.05, seed=0):
    """bootstrap 均值置信区间。"""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan")
    means = np.array([x[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def paired_t(d):
    """配对 t 检验（单样本 t，双尾 p）。返回 (t, p, df)。"""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), 0
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf"), 0.0, n - 1
    t = d.mean() / (sd / math.sqrt(n))
    # 双侧 p：用正态近似（n=21 时与 t 分布接近；保守起见同时给 t 值）
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p), n - 1


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"scene_ci_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log("# 主表统计：场景级（独立单元=21 个原始大图）vs 子图级（92 张）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    # 读入
    per_img = {}
    scenes = {}
    for disp, fn in SOURCES.items():
        p = RES_DIR / fn
        if not p.exists():
            log(f"[跳过] {disp}: 缺 {fn}")
            continue
        recs = list(csv.DictReader(open(p, encoding="utf-8")))
        per_img[disp] = {r["name"]: {k: float(r[k]) for k in METRIC_KEYS} for r in recs}
        sc = defaultdict(list)
        for r in recs:
            sc[base_of(r["name"])].append({k: float(r[k]) for k in METRIC_KEYS})
        scenes[disp] = {b: {k: float(np.mean([x[k] for x in v])) for k in METRIC_KEYS}
                        for b, v in sc.items()}

    common_scenes = sorted(set.intersection(*[set(s) for s in scenes.values()]))
    log(f"# 参与比较的场景数 = {len(common_scenes)}；子图数 = {len(per_img['LoRA-SAM'])}")
    log()

    rows = []
    log("=== 各方法：子图级 vs 场景级 均值+-std（IoU / BER）===")
    log(f"{'方法':<12}{'子图级 IoU':<22}{'场景级 IoU':<22}{'子图级 BER':<20}{'场景级 BER':<20}")
    for disp in scenes:
        imgv = np.array([per_img[disp][n]["IoU"] for n in per_img[disp]])
        scv = np.array([scenes[disp][b]["IoU"] for b in common_scenes])
        img_b = np.array([per_img[disp][n]["BER"] for n in per_img[disp]])
        sc_b = np.array([scenes[disp][b]["BER"] for b in common_scenes])
        log(f"{disp:<12}{imgv.mean():.4f}+-{imgv.std():.4f}    "
            f"{scv.mean():.4f}+-{scv.std():.4f}    "
            f"{img_b.mean():.4f}+-{img_b.std():.4f}  {sc_b.mean():.4f}+-{sc_b.std():.4f}")
        rows.append({"level": "image", "method": disp, "n": len(imgv),
                     "IoU_mean": f"{imgv.mean():.6f}", "IoU_std": f"{imgv.std():.6f}",
                     "BER_mean": f"{img_b.mean():.6f}", "BER_std": f"{img_b.std():.6f}"})
        rows.append({"level": "scene", "method": disp, "n": len(scv),
                     "IoU_mean": f"{scv.mean():.6f}", "IoU_std": f"{scv.std():.6f}",
                     "BER_mean": f"{sc_b.mean():.6f}", "BER_std": f"{sc_b.std():.6f}"})

    log()
    log("=== 配对比较（同一场景配对；独立单元 = 场景）===")
    for a, b in [("LoRA-SAM", "U-Net"), ("LoRA-SAM", "SegNet"), ("U-Net", "SegNet")]:
        if a not in scenes or b not in scenes:
            continue
        # 场景级配对差
        d_sc = np.array([scenes[a][s]["IoU"] - scenes[b][s]["IoU"] for s in common_scenes])
        d_img = np.array([per_img[a][n]["IoU"] - per_img[b][n]["IoU"] for n in per_img[a]])
        m_sc, lo_sc, hi_sc = bootstrap_ci(d_sc)
        m_img, lo_img, hi_img = bootstrap_ci(d_img)
        t_sc, p_sc, df = paired_t(d_sc)
        t_img, p_img, _ = paired_t(d_img)
        log(f"\n  {a} - {b}  (ΔIoU)")
        log(f"    场景级(n={len(d_sc)}): 均值={m_sc:+.4f}  95%CI=[{lo_sc:+.4f}, {hi_sc:+.4f}]  "
            f"配对t={t_sc:.2f} p~={p_sc:.4f}")
        log(f"    子图级(n={len(d_img)}): 均值={m_img:+.4f}  95%CI=[{lo_img:+.4f}, {hi_img:+.4f}]  "
            f"配对t={t_img:.2f} p~={p_img:.4f}")
        log(f"    -> 差值 std：场景级 {d_sc.std():.4f} / 子图级 {d_img.std():.4f}"
            f"（子图级偏小 = 同场景相关导致，会**夸大**显著性）")
        verdict = "显著" if (lo_sc > 0 or hi_sc < 0) else "不显著"
        log(f"    -> 场景级 95%CI {'不跨 0 -> 显著' if verdict == '显著' else '跨 0 -> **不显著**'}")
        rows.append({"level": "paired_scene", "method": f"{a}-{b}", "n": len(d_sc),
                     "IoU_mean": f"{m_sc:+.6f}", "IoU_std": f"{d_sc.std():.6f}",
                     "BER_mean": "", "BER_std": "",
                     "ci_lo": f"{lo_sc:+.6f}", "ci_hi": f"{hi_sc:+.6f}",
                     "t": f"{t_sc:.3f}", "p": f"{p_sc:.4f}"})
        rows.append({"level": "paired_image", "method": f"{a}-{b}", "n": len(d_img),
                     "IoU_mean": f"{m_img:+.6f}", "IoU_std": f"{d_img.std():.6f}",
                     "BER_mean": "", "BER_std": "",
                     "ci_lo": f"{lo_img:+.6f}", "ci_hi": f"{hi_img:+.6f}",
                     "t": f"{t_img:.3f}", "p": f"{p_img:.4f}"})

    keys = ["level", "method", "n", "IoU_mean", "IoU_std", "BER_mean", "BER_std",
            "ci_lo", "ci_hi", "t", "p"]
    out = RES_DIR / "scene_level_ci.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log("")
    log(f"[OK] 明细 -> {out}")
    log("# 注：p 用正态近似（n=21）；结论以 95%CI 是否跨 0 为准，p 仅供参考")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
