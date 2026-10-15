"""无泄漏版 LoRA 微调：按原始大图基名重划 Train412，训练/选择/报告口径统一。

背景：官方 AISD 划分存在同源泄漏——Test51 的 51 张里 50 张来自训练场景
（Test51 与 Train412 共享 42/43 个原始大图基名），因此官方划分下 IoU 偏高。

本脚本【只在 Train412 的 105 张原始大图内】按基名重划 train/val/test，
使测试场景在训练中完全未出现，得到无泄漏的诚实指标（代价：训练场景变少，数字会低于官方划分）。

同时修正旧版口径不一致：旧版用 Val51 质心点提示选模型、却用 Test51 框提示报告结果；
本版【选择集与报告集都用框提示】，口径统一。

用法：
    python train_lora_leakfree.py --epochs 1     # 冒烟测试
    python train_lora_leakfree.py --epochs 20    # 快速
    python train_lora_leakfree.py                # 完整(默认 150)
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry

from eval_shadow import compute_metrics
from direction_experiment import method_boxes
from train_adapter_v4 import apply_lora, lora_params, train_step, LR, LR_MIN, CKPT, AISD
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")
RESUME_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree_resume.pth")
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

VAL_EVERY = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_box_full(sam, img_paths, gt_paths):
    """框提示评估，返回六项指标均值/标准差/逐图。口径与最终报告一致。"""
    rows = []
    for ip, gp in zip(img_paths, gt_paths):
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = method_boxes(sam, img_rgb, img_bgr)
        m = compute_metrics((pred * 255).astype(np.uint8), gt)
        m["name"] = Path(ip).name
        rows.append(m)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
    return avg, std, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.2, help="验证集占原图比例（从训练池内再划）")
    ap.add_argument("--test-frac", type=float, default=0.2, help="测试集占原图比例")
    ap.add_argument("--resume", type=str, default=None, help="从 checkpoint 续训（跳过已完成 epoch）")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    LOG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"lora_leakfree_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 无泄漏版 LoRA 微调 SAM（按原图基名重划 Train412）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# epochs={args.epochs} seed=0  口径=框提示（选择与报告一致）")

    # ---- 按原图基名划分：先划 test，再从剩余划 val ----
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    gt_of = lambda f: str(AISD / "Train412" / "mask" / Path(f).name)

    train_pool, test_files, n_pool, n_te = split_by_base(files, args.test_frac, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, args.val_frac, seed=1)

    log(f"Train412 {len(files)} 张 / {n_tr + n_va + n_te} 张原始大图")
    log(f"  训练 {len(tr_files)} 张({n_tr}原图) / 验证 {len(val_files)} 张({n_va}原图) / 测试 {len(test_files)} 张({n_te}原图)")
    log(f"  三集原图基名互不重叠（测试场景训练中未出现）")

    val_gts = [gt_of(f) for f in val_files]
    test_gts = [gt_of(f) for f in test_files]
    train_pairs = list(zip(tr_files, [gt_of(f) for f in tr_files]))

    # ---- 建模 ----
    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    apply_lora(sam)
    sam.to(device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in lora_params(sam):
        p.requires_grad = True

    params = lora_params(sam) + list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters())
    log(f"可训练参数 {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            sam.load_state_dict(ck["model"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            best = float(ck.get("best", 0.0))
        else:
            sam.load_state_dict(ck)
        log(f"从 {args.resume} 续训，start_epoch={start_epoch}, best={best:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * epoch / args.epochs))
        for g in opt.param_groups:
            g["lr"] = lr
        sam.prompt_encoder.train()
        sam.mask_decoder.train()
        random.shuffle(train_pairs)
        t0 = time.time()
        losses = []
        for ip, gp in train_pairs:
            img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            loss = train_step(sam, img, gt, opt, scaler)
            if loss is not None:
                losses.append(loss)
        if losses:
            log(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  lr={lr:.1e}  ({time.time()-t0:.0f}s)")

        if epoch % VAL_EVERY == 0:
            sam.prompt_encoder.eval()
            sam.mask_decoder.eval()
            vavg, _, _ = eval_box_full(sam, val_files, val_gts)
            log(f"    -> 验证(框提示) IoU={vavg['IoU']:.4f}  BER={vavg['BER']:.4f}")
            if vavg["IoU"] > best:
                best = vavg["IoU"]
                torch.save(sam.state_dict(), OUT)
                log(f"    -> 保存最佳 (val IoU {best:.4f})")
            torch.save({"epoch": epoch, "model": sam.state_dict(), "best": best}, RESUME_CKPT)
            log(f"    -> 续训点已存 (epoch {epoch})")

    # ---- 最终：无泄漏测试集六项指标 ----
    if Path(OUT).exists():
        sam.load_state_dict(torch.load(OUT, map_location=device))
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()
    tavg, tstd, trows = eval_box_full(sam, test_files, test_gts)
    log("")
    log(f"=== 无泄漏测试集（{len(trows)} 张 / {n_te} 原图，测试场景训练中未出现）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {tavg[k]:.4f} ± {tstd[k]:.4f}")

    with open(RES_DIR / "lora_leakfree_test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        w.writeheader()
        for r in trows:
            w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
    with open(RES_DIR / "lora_leakfree_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_images"] + METRIC_KEYS)
        w.writerow(["LoRA-Finetuned-SAM(box)_leakfree", len(trows)] + [f"{tavg[k]:.6f}" for k in METRIC_KEYS])
        w.writerow(["LoRA-Finetuned-SAM(box)_leakfree_std", len(trows)] + [f"{tstd[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"逐图 -> {RES_DIR / 'lora_leakfree_test_per_image.csv'}")
    log(f"汇总 -> {RES_DIR / 'lora_leakfree_test.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
