"""Mask R-CNN 阴影检测基线（无泄漏划分，与 LoRA-SAM / U-Net / SegNet 同口径）。

## 为什么有这个脚本

申报书（`_shenbaoshu_attached_20261014.txt`）的硬判据之一是：
    "交并比（IoU）优于当前主流全监督网络（如 Mask R-CNN）。"
但本项目**从未跑过 Mask R-CNN**（见 `申报书问题清单.md` 3.2）。本脚本补齐该项。

## 口径（与主方法完全对齐，确保可比）

- **划分**：与 `train_lora_leakfree.py` / `train_cnn_leakfree.py` **逐字相同**：
    split_by_base(Train412, 0.2, seed=0) -> train_pool + test(92/21 原图)
    split_by_base(train_pool, 0.2, seed=1) -> train(262/68) + val(58/16)
  **注意：本脚本内的划分 seed 恒为 0/1，不随任何参数变化**（划分一变，结果全部不可比）。
  启动时会**断言**测试集 name 集合与 `results/lora_leakfree_test_per_image.csv` 一致，不等即抛异常。
- **输入**：整图 resize 到 512x512（`INTER_LINEAR`），与 U-Net/SegNet 一致。
- **实例定义**：GT 二值掩膜的连通域（`area >= 200`，与 `direction_utils.gt_direction` 同阈值）。
- **评估**：保留 `score >= tau` 的实例 -> 软掩膜 `>0.5` 并集 -> **最近邻** resize 回 GT 原尺寸
  -> `eval_shadow.compute_metrics` 出六项指标。**tau 与最佳 epoch 只在验证集(58)上选**。
- **提示来源说明（公平性，审查 M3）**：主方法（LoRA-SAM）的框提示来自 `mask_enhanced`
  （Otsu+CLAHE 粗提取），**不是 GT**；本脚本的 Mask R-CNN 亦为 COCO 预训练 + 全自动推理。
  两者皆为**全自动、无 GT 泄漏**的可比对手。

## 收敛判据（审查 M2 要求，必须落盘）

逐次验证落盘 IoU/BER 曲线（`results/maskrcnn_leakfree_val_curve.csv`），并记录
"最佳出现在第几次评估"。若**最佳出现在最后一次评估**，日志会明确标注"基线可能仍未收敛" ——
此时"LoRA 优于 Mask R-CNN"是对**未收敛基线**取优，报告须如实标注为下界。

## 用法
    python train_maskrcnn_leakfree.py --epochs 1 --limit-train 8 --limit-val 4 --eval-every 1
    python train_maskrcnn_leakfree.py --epochs 60
    python train_maskrcnn_leakfree.py --resume
    python train_maskrcnn_leakfree.py --evaluate

## 落盘
    checkpoints/maskrcnn_shadow_leakfree.pth / _last.pth
    results/maskrcnn_leakfree_test.csv / _test_per_image.csv / _val_curve.csv
    logs/maskrcnn_leakfree_{ts}.log
"""
import argparse
import csv
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前（Windows CUDA DLL 修复）

import torch
import numpy as np
import cv2

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from eval_shadow import compute_metrics
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT_DIR = BASE / "checkpoints"
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
MIN_AREA = 200          # 连通域最小面积（与 gt_direction 同阈值）
IN_SIZE = 512           # 输入分辨率（与 U-Net/SegNet 一致）
MASK_THR = 0.5          # 实例软掩膜二值化阈值（审查要求写死）
TAU_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # 含 0.05，避免最优解落在网格边界
device = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────── 数据 ──────────────────────────

def load_rgb512(path):
    """读图 -> RGB uint8 (512,512,3)。"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)


def instance_targets(gt_path, stats_out=None):
    """GT 掩膜 -> 实例 (boxes xyxy@512, masks bool@512)。

    连通域在**原尺寸**上求（阈值语义与 gt_direction 一致），标签图再最近邻缩放到 512，
    这样实例边界不会因缩放而伪造出新连通域。
    """
    g = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 0).astype(np.uint8), 8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    if stats_out is not None:
        stats_out.append(len(keep))
    if not keep:
        return None
    # [!] 用 float32 做最近邻缩放：cv2.resize 不支持 CV_32S；标签为整数，float32 表示精确
    lab512 = cv2.resize(lab.astype(np.float32), (IN_SIZE, IN_SIZE),
                        interpolation=cv2.INTER_NEAREST).round().astype(np.int64)
    boxes, masks = [], []
    for i in keep:
        m = (lab512 == i)
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)])
        masks.append(m)
    if not boxes:
        return None
    return (torch.as_tensor(np.array(boxes), dtype=torch.float32),
            torch.as_tensor(np.array(masks), dtype=torch.uint8))


def to_target(boxes, masks):
    # [!] labels 必须与 boxes 同 device：torchvision 的 ROIHead 会用 GPU 上的
    #     matched_idxs 去索引 labels，若 labels 留在 CPU 会抛
    #     "indices should be either on cpu or on the same device as the indexed tensor"。
    return {"boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64, device=boxes.device),
            "masks": masks}


# ────────────────────────── 模型 ──────────────────────────

def build_model(min_size, max_size):
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
                              min_size=min_size, max_size=max_size)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)      # 背景 + 阴影
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, 2)
    return m


# ────────────────────────── 评估 ──────────────────────────
# 关键：每个 tau 都重新前向会浪费 9 倍算力；这里**每图只前向一次**，
# 预先算好 (masks > 0.5) 的布尔张量，再对 9 个 tau 各取并集算指标。

@torch.no_grad()
def eval_tau_curve(model, imgs, gts):
    """一次前向覆盖全部 tau。返回 {tau: rows}（rows 为逐图指标 dict）。"""
    model.eval()
    acc = {t: [] for t in TAU_GRID}
    for img512, gp in zip(imgs, gts):
        if img512 is None:
            continue
        gt = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        x = torch.as_tensor(img512, dtype=torch.float32).permute(2, 0, 1) / 255.0
        out = model([x.to(device)])[0]
        scores = out["scores"].cpu().numpy()
        binm = (out["masks"].cpu().numpy()[:, 0] > MASK_THR)   # (N,512,512) bool
        del out
        for tau in TAU_GRID:
            sel = scores >= tau
            union = binm[sel].any(axis=0) if sel.any() else np.zeros((IN_SIZE, IN_SIZE), dtype=bool)
            pred = cv2.resize(union.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            m = compute_metrics((pred * 255).astype(np.uint8), gt)
            m["name"] = Path(gp).name
            m["n_inst"] = int(sel.sum())
            acc[tau].append(m)
    return acc


def avg_of(rows):
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}


def select_tau(acc):
    """在验证集曲线里按 IoU 选 tau（严禁用测试集）。返回 (tau, [(tau, IoU, BER), ...])。"""
    curve, best = [], (None, -1.0)
    for tau in TAU_GRID:
        a = avg_of(acc[tau])
        if a is None:
            continue
        curve.append((tau, a["IoU"], a["BER"]))
        if a["IoU"] > best[1]:
            best = (tau, a["IoU"])
    return best[0], curve


# ────────────────────────── 结果落盘 ──────────────────────────

def write_results(rows, tau, log, tag=""):
    """tag != "" 时文件名加后缀，**绝不覆盖其它设置的产物**。
    注意 `tag` 有两种来源：冒烟（`_smoke`，由 --limit-train/--limit-val 触发）
    与自定义（`--tag`，如 `_res800`，属**正式跑**）。二者措辞不同，不得混用。"""
    avg = avg_of(rows)
    std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
    log("")
    log(f"=== 测试集（{len(rows)} 张，tau={tau}）" + ("  [冒烟模式]" if "_smoke" in tag else "") + " ===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {avg[k]:.4f} +/- {std[k]:.4f}")
    per_img = RES_DIR / f"maskrcnn_leakfree{tag}_test_per_image.csv"
    summ = RES_DIR / f"maskrcnn_leakfree{tag}_test.csv"
    with open(per_img, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS + ["n_inst"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS + ["n_inst"]})
    with open(summ, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_images", "tau"] + METRIC_KEYS)
        w.writerow([f"MaskRCNN(leakfree){tag}", len(rows), tau] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
        w.writerow([f"MaskRCNN(leakfree){tag}_std", len(rows), tau] + [f"{std[k]:.6f}" for k in METRIC_KEYS])
    log(f"汇总 -> {summ}")
    # [!] 只对**冒烟**产物打警告。自定义 --tag（如 _res800）是正式跑，不得误标为冒烟
    #     （2026-10-15 修：此前 tag 非空即打印"冒烟产物"，把 _res800 也误标了）
    if "_smoke" in tag:
        log(f"[!] 冒烟产物（{tag}），**不得作为结果引用**")
    elif tag:
        log(f"[注] 已加 tag '{tag}'（正式跑；与其它设置的产物互不覆盖）")
    return avg, std


# ────────────────────────── 主流程 ──────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--max-size", type=int, default=800)
    ap.add_argument("--limit-train", type=int, default=0, help="冒烟用：只用前 N 张训练")
    ap.add_argument("--limit-val", type=int, default=0, help="冒烟用：只用前 N 张验证")
    ap.add_argument("--tag", type=str, default="",
                    help="输出文件名后缀（如 _res800）。同一脚本跑不同分辨率/设置时必须给不同 tag，"
                         "否则会**互相覆盖**，且旧结果会被静默改写")
    ap.add_argument("--evaluate", action="store_true", help="只评估已有 best 权重")
    ap.add_argument("--resume", action="store_true", help="从 _last.pth 续训")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    for d in (CKPT_DIR, LOG_DIR, RES_DIR):
        d.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"maskrcnn_leakfree_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# Mask R-CNN 阴影检测基线（无泄漏划分 68/16/21）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 输入 {IN_SIZE}x{IN_SIZE}  实例=GT连通域(area>={MIN_AREA})  "
        f"模型 min_size={args.min_size} max_size={args.max_size}  掩膜阈值={MASK_THR}")
    log(f"# epochs={args.epochs} lr={args.lr} eval_every={args.eval_every}  全局seed=0")
    log("# 划分 seed 恒为 0/1（与主方法逐字相同，不随任何参数变化）")

    # ── 划分（与 train_lora_leakfree.py:96-97 逐字一致）──
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log(f"Train412 {len(files)} 张")
    log(f"  训练 {len(tr_files)} 张({n_tr}原图) / 验证 {len(val_files)} 张({n_va}原图) "
        f"/ 测试 {len(test_files)} 张({n_te}原图)")

    def gt_of(f):
        return AISD / "Train412" / "mask" / Path(f).name

    # 断言：测试集与主结果逐图 CSV 的 name 集合一致（防止划分被静默改动）
    main_csv = RES_DIR / "lora_leakfree_test_per_image.csv"
    if main_csv.exists():
        with open(main_csv, newline="", encoding="utf-8-sig") as fh:
            names_main = {r["name"] for r in csv.DictReader(fh) if r.get("name")}
        names_here = {Path(f).name for f in test_files}
        if names_main != names_here:
            raise RuntimeError(
                f"划分不一致：本脚本测试集({len(names_here)}) 与主结果({len(names_main)}) 的 "
                f"name 集合不等（多 {len(names_here-names_main)} / 少 {len(names_main-names_here)}）。"
                f"划分被改动过，拒绝产出不可比的结果。")
        log(f"  [OK] 测试集与主结果逐图 CSV 的 name 集合一致（{len(names_here)} 张）")
    else:
        log(f"  [!] 未找到 {main_csv}，跳过划分一致性断言")

    # ── 预读图像到内存（512 uint8，262 张约 200MB）──
    t0 = time.time()
    log("读取图像到内存 ...")
    tr_imgs = [load_rgb512(f) for f in tr_files]
    val_imgs = [load_rgb512(f) for f in val_files]
    val_gts = [gt_of(f) for f in val_files]
    log(f"  完成（{time.time()-t0:.0f}s）")

    # ── 实例统计落盘（审查 L1：用代码重算，不引用二手数字）──
    stats, tr_targets = [], []
    for f in tr_files:
        tr_targets.append(instance_targets(gt_of(f), stats_out=stats))
    keep = [i for i, t in enumerate(tr_targets) if t is not None]
    n_drop = len(tr_targets) - len(keep)
    tr_files = [tr_files[i] for i in keep]
    tr_imgs = [tr_imgs[i] for i in keep]
    tr_targets = [tr_targets[i] for i in keep]
    log(f"# 训练集实例统计: n={len(stats)} mean={np.mean(stats):.1f} "
        f"median={np.median(stats):.0f} min={min(stats)} max={max(stats)}")
    log(f"# 有效训练图 {len(tr_files)}（丢弃无实例图 {n_drop} 张）")

    if args.limit_train:
        tr_files, tr_imgs, tr_targets = (tr_files[:args.limit_train],
                                         tr_imgs[:args.limit_train],
                                         tr_targets[:args.limit_train])
    if args.limit_val:
        val_imgs, val_gts = val_imgs[:args.limit_val], val_gts[:args.limit_val]

    model = build_model(args.min_size, args.max_size).to(device)
    log(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 冒烟模式（--limit-train/--limit-val）绝不污染生产权重与结果；
    # 自定义 --tag（如 _res800）用于同一脚本跑不同设置时**互不覆盖**，属**正式跑**。
    tag = ("_smoke" if (args.limit_train or args.limit_val) else "") + args.tag
    if "_smoke" in tag:
        log(f"[!] 冒烟模式：权重与结果均写入 *{tag} 后缀文件，不覆盖生产产物")
    elif tag:
        log(f"[注] 自定义 tag '{tag}'（正式跑；与其它设置的产物互不覆盖）")
    ckpt = CKPT_DIR / f"maskrcnn_shadow_leakfree{tag}.pth"
    last_ckpt = CKPT_DIR / f"maskrcnn_shadow_leakfree{tag}_last.pth"
    curve_path = RES_DIR / f"maskrcnn_leakfree{tag}_val_curve.csv"

    # ── 只评估模式 ──
    if args.evaluate:
        if not ckpt.exists():
            log(f"！未找到 {ckpt}，无法评估")
            logf.close()
            return
        model.load_state_dict(torch.load(ckpt, map_location=device))
        log(f"评估模式：加载 {ckpt}（不训练）")
        tau, vcurve = select_tau(eval_tau_curve(model, val_imgs, val_gts))
        log(f"验证集选得 tau={tau}")
        for t, i, b in vcurve:
            log(f"    tau={t:g}  val IoU={i:.4f}  val BER={b:.4f}")
        tacc = eval_tau_curve(model, [load_rgb512(f) for f in test_files],
                              [gt_of(f) for f in test_files])
        write_results(tacc[tau], tau, log, tag)
        log(f"日志 -> {log_path}")
        logf.close()
        return

    # ── 续训 ──
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    start_epoch, best, curve = 1, -1.0, []
    if args.resume and last_ckpt.exists():
        try:
            st = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["optim"])
            start_epoch = st["epoch"] + 1; best = st["best"]; curve = st.get("curve", [])
            log(f"续训：从 {last_ckpt} 恢复（已完成 epoch {st['epoch']}，best val IoU {best:.4f}）")
        except Exception as e:
            log(f"！{last_ckpt} 读取失败（{type(e).__name__}: {e}），从 epoch 1 重新开始")
    if start_epoch > args.epochs:
        log(f"！已训练 {start_epoch-1}/{args.epochs} 轮，跳过训练（如需评估请用 --evaluate）")
        logf.close()
        return

    # ── 训练 ──
    order = list(range(len(tr_files)))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        random.shuffle(order)
        t0 = time.time()
        losses = []
        for idx in order:
            img = tr_imgs[idx]
            boxes, masks = tr_targets[idx]
            # 一次抽签同时作用于图像/框/掩膜（防图文错位，与 train_cnn_leakfree.py 同纪律）
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                x1 = IN_SIZE - boxes[:, 2]; x2 = IN_SIZE - boxes[:, 0]
                boxes = torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=1)
                masks = torch.flip(masks, dims=[2])
            x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            tgt = to_target(boxes.to(device), masks.to(device))
            loss_d = model([x.to(device)], [tgt])
            loss = sum(loss_d.values())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        sched.step()
        log(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # 只前向一次：一次 eval_tau_curve 同时给出全部 tau 的逐图指标
            vacc = eval_tau_curve(model, val_imgs, val_gts)
            tau, vcurve = select_tau(vacc)
            vavg = avg_of(vacc[tau])
            if vavg is not None:
                curve.append({"epoch": epoch, "tau": tau,
                              "val_IoU": round(vavg["IoU"], 6), "val_BER": round(vavg["BER"], 6)})
                log(f"    -> val: tau={tau}  IoU={vavg['IoU']:.4f}  BER={vavg['BER']:.4f}")
                if vavg["IoU"] > best:
                    best = vavg["IoU"]
                    torch.save(model.state_dict(), ckpt)
                    log(f"    -> 保存最佳 (val IoU {best:.4f})")
            with open(curve_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["epoch", "tau", "val_IoU", "val_BER"])
                w.writeheader(); w.writerows(curve)

        tmp = Path(str(last_ckpt) + ".tmp")
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "epoch": epoch, "best": best, "curve": curve}, tmp)
        tmp.replace(last_ckpt)

    # ── 收敛判据（审查 M2）──
    if len(curve) >= 3:
        best_ep = max(curve, key=lambda r: r["val_IoU"])["epoch"]
        last_ep = curve[-1]["epoch"]
        log("")
        log(f"# 收敛判据：最佳 val IoU={best:.4f} 在 epoch {best_ep}；最后一次评估在 epoch {last_ep}")
        if best_ep == last_ep:
            log("  [警告] 最佳出现在**最后一次**评估 -> 基线可能**仍未收敛**。")
            log("  [警告] 此时'LoRA 优于 Mask R-CNN'属对**未收敛基线**取优，报告必须如实标注为下界。")
        else:
            log(f"  [OK] 最佳出现在最后一次评估之前（相隔 {last_ep - best_ep} 轮），曲线已见顶/回落。")
    elif curve:
        log("")
        log(f"！验证曲线只有 {len(curve)} 个点（<3），**无法判定收敛**；"
            f"不得据此声称基线已充分训练。")
    else:
        log("！没有产生验证曲线，无法判定收敛")

    # ── 最终评估（best 权重 + 验证集选出的 tau）──
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    log("")
    log("用 best-val 权重在验证集上重选 tau（严禁用测试集选）...")
    vacc = eval_tau_curve(model, val_imgs, val_gts)
    tau, vcurve = select_tau(vacc)
    for t, i, b in vcurve:
        log(f"    tau={t:g}  val IoU={i:.4f}  val BER={b:.4f}")
    tacc = eval_tau_curve(model, [load_rgb512(f) for f in test_files],
                          [gt_of(f) for f in test_files])
    write_results(tacc[tau], tau, log, tag)
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
