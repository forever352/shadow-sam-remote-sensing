"""无泄漏版**冻结编码器**微调：补申报书"Adapter 微调"链条上缺的那一环。

## 为什么写这个

申报书模块 3 写的是"**冻结 SAM 图像编码器**，仅对其掩膜解码器改进 / 训练 Adapter"。
但项目实际主线用的是 **LoRA 微调**（`train_lora_leakfree.py`），而且
`结题报告.md` 里那个 **0.717**（"冻结编码器微调 SAM"）**只有 AISD 自带划分（含同源泄漏）
口径**、没有无泄漏版；"零样本 -> 冻结 -> LoRA"这条递进故事因此缺了一格干净证据。

本脚本产出**无泄漏划分**下的冻结编码器版本结果，把这一格补上。

## 与 LoRA 版的**唯一差异**

| | LoRA 版 (`train_lora_leakfree.py`) | 本脚本 |
|---|---|---|
| 图像编码器 | 挂 LoRA（低秩旁路），LoRA 参数可训 | **完全冻结**，且前向放 `no_grad` |
| 可训练参数 | LoRA + prompt_encoder + mask_decoder | **仅 prompt_encoder + mask_decoder** |
| 其余 | 同一划分 / 同一增强 / 同一损失 / 同一余弦 LR 日程 / 每 5 轮验证 / 同一框提示评估 | **完全相同** |

故两行结果的差异**可归因于"是否对编码器做 LoRA 适配"**（单变量）。
注意：这也意味着本行**不是**"零样本"——它仍然训练了解码器与提示编码器。

**为什么冻结版更快**：LoRA 版每步都要穿过编码器反传；本版编码器前向在 `no_grad` 下，
不建图、不反传，故预计 2-3 倍加速（LoRA 版实测约 500s/轮）。

## 用法
    python train_frozen_leakfree.py --epochs 1      # 冒烟
    python train_frozen_leakfree.py --epochs 40     # 与 LoRA 版同轮数（推荐）
    python train_frozen_leakfree.py --epochs 40 --resume checkpoints/sam_vit_b_shadow_frozen_leakfree_last.pth

## 落盘
    checkpoints/sam_vit_b_shadow_frozen_leakfree.pth          best-val 权重
    checkpoints/sam_vit_b_shadow_frozen_leakfree_last.pth     续训点 {"epoch","model","best"}
    results/frozen_leakfree_test.csv / _per_image.csv
    logs/frozen_leakfree_{ts}.log
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
from train_adapter_v4 import (CKPT, AISD, LR, LR_MIN, device, transform,
                              dice_loss, focal_loss, sample_points, gt_bbox)
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_frozen_leakfree.pth")
RESUME_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_frozen_leakfree_last.pth")
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
VAL_EVERY = 5


def train_step_frozen(sam, img_rgb, gt, opt, scaler):
    """与 train_adapter_v4.train_step 同逻辑，**唯一区别**：编码器前向放 no_grad。"""
    h, w = gt.shape
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()

    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    if random.random() < 0.7:
        coords, labels = sample_points(gt)
        if coords is None:
            return None
        c1024 = transform.apply_coords(coords, (h, w))
        c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
        l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
        points, boxes = (c_t, l_t), None
    else:
        box = gt_bbox(gt)
        if box is None:
            return None
        b1024 = transform.apply_boxes(box, (h, w))
        b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
        points, boxes = None, b_t

    x = sam.preprocess(img_t)
    # [!] 冻结版的关键：编码器不建图、不反传。
    # [!] autocast 作用域必须与 train_adapter_v4.train_step **完全一致** —— 那里编码器
    #     前向也在 autocast 内（fp16）。若把本版编码器前向放到 autocast 之外（fp32），
    #     就会在"有无 LoRA"之外**多出一个 fp32/fp16 的差异**，破坏单变量可比性，
    #     而且会白白损失速度。故此处两者合并。
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_emb = sam.image_encoder(x)
    with torch.cuda.amp.autocast():
        sparse, dense = sam.prompt_encoder(points=points, boxes=boxes, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    gt256 = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.as_tensor((gt256 > 0), dtype=torch.float32)[None, None].to(device)
    loss = focal_loss(low_res, gt_t) + dice_loss(low_res, gt_t)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


def eval_box_full(sam, img_paths, gt_paths):
    """框提示评估，口径与最终报告一致（与 LoRA 版同一函数体）。"""
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
    ap.add_argument("--epochs", type=int, default=40, help="与 LoRA 版同轮数才可比（推荐 40）")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--resume", type=str, default=None, help="从续训点恢复（跳过已完成 epoch）")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    LOG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"frozen_leakfree_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 无泄漏版 **冻结编码器** 微调（补申报书 Adapter 链条 + 结题报告 0.717 的无泄漏缺口）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# epochs={args.epochs} seed=0  口径=框提示（选择与报告一致）")
    log("# [!] 与 LoRA 版**唯一差异**：编码器完全冻结且前向 no_grad；其余完全相同（单变量）")

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    gt_of = lambda f: str(AISD / "Train412" / "mask" / Path(f).name)

    train_pool, test_files, n_pool, n_te = split_by_base(files, args.test_frac, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, args.val_frac, seed=1)
    log(f"Train412 {len(files)} 张 / {n_tr + n_va + n_te} 张原始大图")
    log(f"  训练 {len(tr_files)} 张({n_tr}原图) / 验证 {len(val_files)} 张({n_va}原图) / 测试 {len(test_files)} 张({n_te}原图)")

    val_gts = [gt_of(f) for f in val_files]
    test_gts = [gt_of(f) for f in test_files]
    train_pairs = list(zip(tr_files, [gt_of(f) for f in tr_files]))

    # ---- 建模：不挂 LoRA，整块冻结编码器 ----
    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    sam.to(device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    params = list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters())
    n_train = sum(p.numel() for p in params)
    n_img = sum(p.numel() for p in sam.image_encoder.parameters())
    log(f"可训练参数 {n_train/1e6:.2f}M（仅 prompt_encoder + mask_decoder）；"
        f"冻结图像编码器 {n_img/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        sam.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", 0.0))
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
            loss = train_step_frozen(sam, img, gt, opt, scaler)
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

    # ---- 最终：无泄漏测试集六项 ----
    if Path(OUT).exists():
        sam.load_state_dict(torch.load(OUT, map_location=device))
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()
    tavg, tstd, trows = eval_box_full(sam, test_files, test_gts)
    log("")
    log(f"=== 无泄漏测试集（{len(trows)} 张 / {n_te} 原图）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {tavg[k]:.4f} +/- {tstd[k]:.4f}")

    with open(RES_DIR / "frozen_leakfree_test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        w.writeheader()
        for r in trows:
            w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
    with open(RES_DIR / "frozen_leakfree_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_images"] + METRIC_KEYS)
        w.writerow(["Frozen-Encoder-SAM(box)_leakfree", len(trows)] + [f"{tavg[k]:.6f}" for k in METRIC_KEYS])
        w.writerow(["Frozen-Encoder-SAM(box)_leakfree_std", len(trows)] + [f"{tstd[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"逐图 -> {RES_DIR / 'frozen_leakfree_test_per_image.csv'}")
    log(f"汇总 -> {RES_DIR / 'frozen_leakfree_test.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
