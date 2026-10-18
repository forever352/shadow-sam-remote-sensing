"""无泄漏版 U-Net / SegNet 基线：与 LoRA 无泄漏版用【同一划分】，做公平对比。

背景：官方划分下 U-Net(0.805)/SegNet(0.780) 与 LoRA(0.839) 都评在官方 Test51 上，
且都用全部 Train412 训练——存在同源泄漏。为与 LoRA 无泄漏版(IoU 0.788)公平对比，
本脚本在【同一 68/16/21 原图划分】上重训 U-Net/SegNet，并在同一无泄漏测试集上报告六项指标。

划分逻辑与 train_lora_leakfree.py 完全一致：
    split_by_base(Train412文件, 0.2, seed=0) -> train_pool + test
    split_by_base(train_pool,   0.2, seed=1) -> train + val

用法：
    python train_cnn_leakfree.py --arch unet
    python train_cnn_leakfree.py --arch segnet
    python train_cnn_leakfree.py --arch unet --epochs 60
    python train_cnn_leakfree.py --arch unet --split official --epochs 60   # 官方划分（含泄漏）
    python train_cnn_leakfree.py --arch unet --split official --resume      # 崩溃后续训

变更留痕（provenance）：
    2026-10-14  新增 --resume 与每轮 `_last.pth` 落盘（含 epoch/best/优化器状态）。
                该改动【不影响任何数值计算】：不消耗随机数、不改变训练/评估路径、
                不改变 split_by_base 的数据划分（划分仍由固定 seed 决定）与 best-val 选模型口径。
                故此前产出的 leakfree 结果 CSV 与本版本脚本结果一致，无需重跑。
    2026-10-14  加三类守卫（supervisor 复审要求）：
                (i) `_last.pth` 原子写（写 .tmp 再 replace），防休眠/断电写成半截损坏文件；
                (ii) `torch.load(_last.pth)` 包 try/except，损坏时告警并回退 epoch 1；
                (iii) `start_epoch > epochs` 时告警并**跳过训练与最终评估**，防止空转后
                      拿旧 ckpt 静默写出一张误导性的结果表。
                注意：续训不恢复 RNG 状态，故 **resumed 运行无法逐步复现**；from-scratch 运行不受影响。
"""
import argparse
import csv
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from eval_shadow import compute_metrics
from train_unet import UNet, load_image_tensor, load_mask_tensor, dice_loss
from train_segnet import SegNet
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
LOG_DIR = BASE / "logs"
RES_DIR = BASE / "results"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
VAL_EVERY = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def eval_full(model, img_paths, gt_paths):
    """整图评估，返回六项指标均值/标准差/逐图。口径与原 U-Net/SegNet 评估一致（512 输入）。"""
    model.eval()
    rows = []
    for ip, gp in zip(img_paths, gt_paths):
        x = load_image_tensor(ip)[None].to(device)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        logits = model(x)[0, 0]
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = compute_metrics((pred * 255).astype(np.uint8), gt)
        m["name"] = Path(ip).name
        rows.append(m)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
    return avg, std, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["unet", "segnet"], required=True)
    ap.add_argument("--split", choices=["leakfree", "official"], default="leakfree",
                    help="leakfree=按原图基名重划(68/16/21)；official=官方 Train412/Val51/Test51")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, required=True,
                    help="只控制模型初始化/数据 shuffle/增强 RNG；划分 seed 恒为 0/1。seed=0 请沿用既有结果")
    ap.add_argument("--evaluate", action="store_true", help="只评估已有检查点，不训练")
    ap.add_argument("--resume", action="store_true",
                    help="从 <arch>_shadow_<split>_s<seed>_last.pth 续训（恢复 epoch/best/优化器）。"
                         "注意：不改变数据划分（划分由 split_by_base 的固定 seed 决定）")
    args = ap.parse_args()
    if args.split != "leakfree":
        raise SystemExit("本脚本仅支持 --split leakfree；official 划分请用原脚本")
    if args.seed == 0:
        raise SystemExit("seed=0 请沿用既有 results/{arch}_leakfree_test*.csv，不用本脚本重跑")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    LOG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{args.arch}_{args.split}_s{args.seed}_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"# {args.arch.upper()} 基线（划分={args.split}，seed={args.seed}）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  epochs={args.epochs}  seed={args.seed}")
    log(f"# 超参: VAL_EVERY={VAL_EVERY}  lr=1e-3  loss=BCE+Dice  input=512  aug=horizontal_flip")

    def _imgs(split):
        return sorted(str(p) for p in (AISD / split / "shadow").glob("*.tif"))

    def _gts(split, fs):
        return [str(AISD / split / "mask" / Path(f).name) for f in fs]

    if args.split == "leakfree":
        # 与 train_lora_leakfree.py 完全相同的划分
        files = _imgs("Train412")
        gt_of = lambda f: str(AISD / "Train412" / "mask" / Path(f).name)
        train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
        tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
        log(f"Train412 {len(files)} 张")
        log(f"  训练 {len(tr_files)} 张({n_tr}原图) / 验证 {len(val_files)} 张({n_va}原图) / 测试 {len(test_files)} 张({n_te}原图)")
        log(f"  与 LoRA 无泄漏版同一划分（三集基名互不重叠）")
        # 划分与 seed-0 完全一致；任何不一致都必须立即停下
        if not (len(tr_files) == 262 and len(val_files) == 58):
            raise RuntimeError(f"train/val 数量异常: {len(tr_files)}/{len(val_files)} (期望 262/58)")
        if not (len(test_files) == 92 and n_te == 21):
            raise RuntimeError(f"test 数量异常: {len(test_files)}/{n_te} (期望 92/21)")
        def _base(p):
            return Path(p).stem.split("_sub")[0]
        b_tr = {_base(f) for f in tr_files}
        b_va = {_base(f) for f in val_files}
        b_te = {_base(f) for f in test_files}
        if (b_tr & b_va) or (b_tr & b_te) or (b_va & b_te):
            raise RuntimeError("三集基名存在交集（泄漏）")
        test_names = {Path(f).name for f in test_files}
        for ref_fn in (f"{args.arch}_leakfree_test_per_image.csv",
                       "lora_leakfree_test_per_image.csv"):
            ref_p = RES_DIR / ref_fn
            with open(ref_p, newline="", encoding="utf-8-sig") as rf:
                ref_names = {r["name"] for r in csv.DictReader(rf)}
            if test_names != ref_names:
                raise RuntimeError(f"test 名称集合与 {ref_fn} 不一致")
        import hashlib
        name_hash = hashlib.sha256("\n".join(sorted(test_names)).encode("utf-8")).hexdigest()[:16]
        log(f"  断言通过: 262/58/92 张、21 场景、三集基名不重叠、test 名称与 seed-0 一致")
        log(f"  test 名称 sha256[:16]: {name_hash}")
        log(f"  split={args.split} seed={args.seed}")
        train_pairs = list(zip(tr_files, [gt_of(f) for f in tr_files]))
        val_gts = [gt_of(f) for f in val_files]
        test_gts = [gt_of(f) for f in test_files]
    else:
        # 官方划分：Train412 训练 / Val51 验证 / Test51 测试（含同源泄漏，用于与原基线对齐）
        tr_files = _imgs("Train412")
        val_files = _imgs("Val51")
        test_files = _imgs("Test51")
        log(f"官方划分  训练 {len(tr_files)} / 验证 {len(val_files)} / 测试 {len(test_files)}（Test51 与 Train412 同源，含泄漏）")
        train_pairs = list(zip(tr_files, _gts("Train412", tr_files)))
        val_gts = _gts("Val51", val_files)
        test_gts = _gts("Test51", test_files)

    model = (UNet() if args.arch == "unet" else SegNet()).to(device)
    log(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M  设备 {device}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ckpt = str(BASE / "checkpoints" / f"{args.arch}_shadow_{args.split}_s{args.seed}.pth")
    last_ckpt = str(BASE / "checkpoints" / f"{args.arch}_shadow_{args.split}_s{args.seed}_last.pth")

    if args.evaluate:
        if not Path(ckpt).exists():
            log(f"⚠️ 未找到检查点 {ckpt}，无法评估")
            logf.close()
            return
        model.load_state_dict(torch.load(ckpt, map_location=device))
        log(f"评估模式：加载 {ckpt}（不训练）")
        tavg, tstd, trows = eval_full(model, test_files, test_gts)
        log("")
        log(f"=== 测试集（{len(trows)} 张，split={args.split}）===")
        for k in METRIC_KEYS:
            log(f"{k:10s} {tavg[k]:.4f} ± {tstd[k]:.4f}")
        with open(RES_DIR / f"{args.arch}_{args.split}_s{args.seed}_test_per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
            w.writeheader()
            for r in trows:
                w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
        with open(RES_DIR / f"{args.arch}_{args.split}_s{args.seed}_test.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "n_images"] + METRIC_KEYS)
            w.writerow([f"{args.arch.upper()}({args.split})", len(trows)] + [f"{tavg[k]:.6f}" for k in METRIC_KEYS])
            w.writerow([f"{args.arch.upper()}({args.split})_std", len(trows)] + [f"{tstd[k]:.6f}" for k in METRIC_KEYS])
        log(f"汇总 -> {RES_DIR / (args.arch + '_' + args.split + '_s' + str(args.seed) + '_test.csv')}")
        logf.close()
        return

    best = 0.0
    start_epoch = 1
    if args.resume:
        st = None
        if Path(last_ckpt).exists():
            try:
                st = torch.load(last_ckpt, map_location=device)
            except Exception as e:
                # 文件损坏（如休眠/断电中断写入）→ 保守回退，不静默用坏数据
                log(f"！{last_ckpt} 读取失败（{type(e).__name__}: {e}），回退到 epoch 1 重新开始")
                st = None
        if st is not None:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optim"])
            start_epoch = st["epoch"] + 1
            best = st["best"]
            log(f"续训：从 {last_ckpt} 恢复 —— 已完成 epoch {st['epoch']}，best val IoU {best:.4f}")
        else:
            log(f"！--resume 生效但无可用恢复点，将从 epoch 1 重新开始")

    if start_epoch > args.epochs:
        # 防止"range 空转 → 直接拿现有 ckpt 评估并写结果表"这种静默误导
        log(f"！已训练 {start_epoch - 1}/{args.epochs} 轮，无可训练轮次，**跳过训练与最终评估**")
        log(f"！如需评估现有权重，请显式运行：--evaluate（或提高 --epochs）")
        logf.close()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        t0 = time.time()
        losses = []
        for ip, gp in train_pairs:
            # 修正：图像与掩膜用【同一次】翻转判定，避免图文错位。
            # （原 train_unet.py/train_segnet.py 各自独立抽签是 bug，会压低 CNN 基线）
            flip = random.random() < 0.5
            x = load_image_tensor(ip)[None]
            t = load_mask_tensor(gp)[None, None]
            if flip:
                x = torch.flip(x, dims=[3])
                t = torch.flip(t, dims=[3])
            x = x.to(device)
            t = t.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, t) + dice_loss(logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)")
        if epoch % VAL_EVERY == 0:
            vavg, _, _ = eval_full(model, val_files, val_gts)
            log(f"    -> 验证 IoU={vavg['IoU']:.4f}  BER={vavg['BER']:.4f}")
            if vavg["IoU"] > best:
                best = vavg["IoU"]
                torch.save(model.state_dict(), ckpt)
                log(f"    -> 保存最佳 (val IoU {best:.4f})")
        # 每轮落盘 last 状态，供崩溃后续训（不改变 best-val 选模型口径）
        # 原子写：先写 .tmp 再替换，避免休眠/断电时把 _last.pth 写成半截损坏文件
        tmp = Path(str(last_ckpt) + ".tmp")
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "epoch": epoch, "best": best}, tmp)
        tmp.replace(last_ckpt)

    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    tavg, tstd, trows = eval_full(model, test_files, test_gts)
    log("")
    log(f"=== 无泄漏测试集（{len(trows)} 张 / {n_te} 原图）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {tavg[k]:.4f} ± {tstd[k]:.4f}")

    with open(RES_DIR / f"{args.arch}_{args.split}_s{args.seed}_test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        w.writeheader()
        for r in trows:
            w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
    with open(RES_DIR / f"{args.arch}_{args.split}_s{args.seed}_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_images"] + METRIC_KEYS)
        w.writerow([f"{args.arch.upper()}({args.split})", len(trows)] + [f"{tavg[k]:.6f}" for k in METRIC_KEYS])
        w.writerow([f"{args.arch.upper()}({args.split})_std", len(trows)] + [f"{tstd[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"汇总 -> {RES_DIR / (args.arch + '_' + args.split + '_s' + str(args.seed) + '_test.csv')}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
