"""RSISD 与 AISD 的**任务难度对照**（落盘件）。

## 为什么要这个

G 阶段显示：LoRA-SAM 在 RSISD 上的 IoU(0.7842) 与在 AISD 上(0.7878) 几乎持平。
"跨域不掉点"**看似是泛化好**，但 G/B 审查指出：**不能仅凭这一点下结论**——
更可能的解释是 **RSISD 这个任务本身更易**。本脚本**把该判断所需的证据落盘**，
使其可被独立复算与引用。

[!] 这是**提示性证据**，不是证明。跨数据集**绝对 IoU 不可直接横比**（第十七轮审查先例）。
本脚本只提供：**同一实现**下平凡方法在两集上的表现 + 两集 GT 的统计分布。
**成因（为何更易）仍需另行分析**，本脚本不作因果结论。

## 口径

- 两集用**同一个** `methods.mask_otsu`（不另写实现，避免"尺子不同"）
- 平凡基线 = "全预测为阴影"时的 IoU = 该图 GT 阴影占比（逐图），再 macro 平均
- 指标口径同 `eval_shadow.compute_metrics`（macro，逐图算再平均）
- 逐图明细一并落盘，便于独立复算

## 用法
    python rsisd_difficulty.py

## 落盘
    results/rsisd_difficulty.csv
    results/rsisd_difficulty_per_image.csv
    logs/rsisd_difficulty_{ts}.log
"""
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前（本脚本其实不用 torch，保持一致）

import numpy as np
import cv2

from eval_shadow import compute_metrics
from methods import mask_otsu

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
RSISD = BASE / "datasets" / "RSISD"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"


def run_set(name, img_dir, mask_dir, aisd_test_names=None):
    """返回 (逐图 rows, 汇总 dict)。"""
    if aisd_test_names is not None:
        items = [(AISD / "Train412" / "shadow" / n, AISD / "Train412" / "mask" / n,
                  Path(n).stem, n) for n in sorted(aisd_test_names)]
    else:
        imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.tif"))
        items = [(p, mask_dir / p.name, p.stem, p.name) for p in imgs]

    rows = []
    for ip, mp, stem, fn in items:
        img = cv2.imread(str(ip))
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        gt_ratio = float((gt > 0).mean())
        m = compute_metrics(mask_otsu(img), gt)
        rows.append({"set": name, "name": fn, "gt_shadow_ratio": round(gt_ratio, 6),
                     "trivial_allshadow_IoU": round(gt_ratio, 6),
                     "otsu_IoU": round(m["IoU"], 6), "otsu_BER": round(m["BER"], 6)})
    if not rows:
        return [], {}
    r = np.array([x["gt_shadow_ratio"] for x in rows])
    summary = {
        "set": name, "n_images": len(rows),
        "gt_shadow_ratio_mean": float(r.mean()),
        "gt_shadow_ratio_median": float(np.median(r)),
        "gt_shadow_ratio_p5": float(np.percentile(r, 5)),
        "gt_shadow_ratio_p95": float(np.percentile(r, 95)),
        "pct_images_over_50pct_shadow": float((r > 0.5).mean()),
        "trivial_allshadow_IoU_macro": float(np.mean([x["trivial_allshadow_IoU"] for x in rows])),
        "otsu_IoU_macro": float(np.mean([x["otsu_IoU"] for x in rows])),
        "otsu_BER_macro": float(np.mean([x["otsu_BER"] for x in rows])),
    }
    return rows, summary


def main():
    for d in (RES_DIR, LOG_DIR):
        d.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"rsisd_difficulty_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# RSISD vs AISD 任务难度对照（落盘件）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# 口径: 同一 methods.mask_otsu；macro 逐图平均；trivial = '全预测为阴影' 的 IoU = GT 阴影占比")
    log("# [!] 提示性证据，非证明。跨集绝对 IoU 不可直接横比（第十七轮先例）。成因需另析。")

    # AISD 侧：严格用无泄漏 test92（与主结果同一集合）
    aisd_names = None
    pic = RES_DIR / "lora_leakfree_test_per_image.csv"
    if pic.exists():
        with open(pic, newline="", encoding="utf-8-sig") as fh:
            aisd_names = {r["name"] for r in csv.DictReader(fh) if r.get("name")}
        log(f"AISD 侧用无泄漏 test{len(aisd_names)}（取自 {pic.name}）")
    else:
        log(f"！未找到 {pic}，跳过 AISD 侧")

    all_rows, summaries = [], []
    if aisd_names:
        rows, s = run_set("AISD_leakfree_test", None, None, aisd_test_names=aisd_names)
        all_rows += rows
        if s:
            summaries.append(s)
    rows, s = run_set("RSISD_test200", RSISD / "test200" / "shadow", RSISD / "test200" / "mask")
    all_rows += rows
    if s:
        summaries.append(s)

    log("")
    log(f"{'数据集':<22}{'n':>5}{'GT占比均值':>12}{'中位':>8}{'p95':>8}{'%>50%':>8}{'平凡IoU':>10}{'Otsu IoU':>10}")
    for s in summaries:
        log(f"{s['set']:<22}{s['n_images']:>5}{s['gt_shadow_ratio_mean']:>12.4f}"
            f"{s['gt_shadow_ratio_median']:>8.4f}{s['gt_shadow_ratio_p95']:>8.4f}"
            f"{100*s['pct_images_over_50pct_shadow']:>7.1f}%"
            f"{s['trivial_allshadow_IoU_macro']:>10.4f}{s['otsu_IoU_macro']:>10.4f}")

    log("")
    log("# 读法：Otsu（普通阈值法）在两集上的差距，是'哪个任务更易'的一个可比尺度。")
    log("# 本节不做因果结论；如需下'更易'的判断，须补成因分析（对比度/阴影尺度/边界清晰度/标注松紧）。")

    with open(RES_DIR / "rsisd_difficulty.csv", "w", newline="", encoding="utf-8") as fh:
        keys = list(summaries[0].keys()) if summaries else ["set"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in s.items()})
    with open(RES_DIR / "rsisd_difficulty_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "name", "gt_shadow_ratio",
                                           "trivial_allshadow_IoU", "otsu_IoU", "otsu_BER"])
        w.writeheader(); w.writerows(all_rows)

    log("")
    log(f"汇总 -> {RES_DIR / 'rsisd_difficulty.csv'}")
    log(f"逐图 -> {RES_DIR / 'rsisd_difficulty_per_image.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
