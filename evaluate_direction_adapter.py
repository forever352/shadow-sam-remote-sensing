"""模块③ 评估：在同一 checkpoint 上对比 "装 / 不装 方向 Adapter"，并出消融数字。

这是模块③ 最干净的口径：**同一个权重、同一批测试图、同一套框提示**，
唯一变量是"方向条件是否注入"。因此两行结果之差就是"方向 Adapter 的净效果"。

输出三行（同一 CSV，便于直读）：
  - `LoRA-only (dir off)`    ：不传方向（等价于主方法链路，应复现 ~=0.787821）
  - `LoRA+DirAdapter(oracle-dir)`：传 **GT 算的方向** -- [!] 仅作诊断/上界
  - `LoRA+DirAdapter(regress-dir)`：传**预测**方向 -- 主方法口径

落盘：results/diradapter_eval_{tag}.csv / _per_image.csv、logs/diradapter_eval_{ts}.log

用法：
    python evaluate_direction_adapter.py --ckpt checkpoints/sam_vit_b_shadow_v4_dir_adapter_leakfree.pth
    python evaluate_direction_adapter.py --limit 5        # 冒烟
"""
import argparse
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch

from train_direction_adapter import (
    build_sam, leakfree_split, eval_box_full, directions_for, AISD, BASE, RES_DIR, LOG_DIR,
    METRIC_KEYS, device,
)
from direction_utils import load_direction_net

DEFAULT_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_dir_adapter_main_leakfree.pth")
REF_LORA_ONLY = (0.787821, 0.064359)  # 主结果（无泄漏，best-val，测试集 92 张）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--regress-ckpt", type=str, default=None)
    ap.add_argument("--tag", type=str, default="main",
                    help="输出标签，避免不同运行互相覆盖（须与被评估 ckpt 的 tag 一致）")
    args = ap.parse_args()

    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"diradapter_eval_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log("# 模块③ 方向 Adapter 评估（同权重、同测试集、同提示，唯一变量=方向注入）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 待评权重: {args.ckpt}")

    if not Path(args.ckpt).exists():
        log(f"！权重不存在：{args.ckpt}（需先跑 train_direction_adapter.py）")
        logf.close()
        return
    log(f"# 对照：主结果 LoRA-only IoU={REF_LORA_ONLY[0]:.6f} / BER={REF_LORA_ONLY[1]:.6f}")

    _, _, te = leakfree_split()
    if args.limit:
        te = te[:args.limit]
    gts = [str(AISD / "Train412" / "mask" / Path(f).name) for f in te]

    sam, missing, unexpected = build_sam(args.ckpt, with_adapter=True)
    log(f"# 加载：missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        raise SystemExit(f"unexpected keys 非空：{unexpected[:5]}")
    sam.to(device)

    dmodel = load_direction_net(args.regress_ckpt, device) if args.regress_ckpt \
        else load_direction_net(device=device)

    runs = [
        ("LoRA-only(dir off)", "none", None),
        ("LoRA+DirAdapter(oracle-dir)", "oracle", None),
        ("LoRA+DirAdapter(regress-dir)", "regress", dmodel),
    ]

    rows_all, summary = [], []
    for name, src, dm in runs:
        dirs = directions_for(te, src, dm) if src != "none" else None
        avg, std, rows = eval_box_full(sam, te, gts, dirs)
        log(f"  {name:32s} IoU={avg['IoU']:.6f}  BER={avg['BER']:.6f}  F1={avg['F1']:.6f}  (n={len(rows)})")
        summary.append((name, avg, std, len(rows)))
        for r in rows:
            rows_all.append({"method": name, **{k: r[k] for k in ["name"] + METRIC_KEYS}})

    # 逐图 CSV（含 method 列，长表）
    p = RES_DIR / f"diradapter_eval_{args.tag}_per_image.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        wt = csv.DictWriter(f, fieldnames=["method", "name"] + METRIC_KEYS)
        wt.writeheader(); wt.writerows(rows_all)

    s = RES_DIR / f"diradapter_eval_{args.tag}.csv"
    with open(s, "w", newline="", encoding="utf-8") as f:
        wt = csv.writer(f)
        wt.writerow(["method", "n_images"] + METRIC_KEYS)
        for name, avg, std, n in summary:
            wt.writerow([name, n] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
            wt.writerow([name + "_std", n] + [f"{std[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log("=== 净效果（相对 LoRA-only 基线）===")
    base = summary[0][1]["IoU"]
    for name, avg, std, n in summary[1:]:
        log(f"  {name:32s} ΔIoU={avg['IoU']-base:+.4f}  ΔBER={avg['BER']-summary[0][1]['BER']:+.4f}")
    log("")
    log("# [!] oracle 行方向来自 GT，**仅作诊断/上界**，禁作可汇报主方法")
    log("# 铁律：新方法 >= 0.7878 才替换主方法；否则保留 LoRA-only，本条如实报告为'尝试'")
    log(f"逐图 -> {p}")
    log(f"汇总 -> {s}")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
