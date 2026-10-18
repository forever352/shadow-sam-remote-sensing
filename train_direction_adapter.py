"""模块③ 训练：方向特征 Adapter 与 LoRA 联合微调 SAM（无泄漏划分）。

对应申报书模块3："冻结 SAM 图像编码器，改进掩膜解码器，在解码器中嵌入方向特征
适配器（Direction Adapter），将方向向量作为条件信息注入网络。"

## 关键设计（详见 direction_adapter.py 的模块 docstring）

- 注入点在 **PromptEncoder 输出端**（= 解码器入口），既有调用点全不用改；
- **恒等初始化**（末层零初始化 -> FiLM γ=1, β=0）：装上但未训练时，
  输出与不装**逐比特相同** -> 可安全从主结果权重增量加载；
- **不做"拼接零 token"**：Transformer 会对全部 token 做 softmax，零 token 的 Q/K
  非零会改变归一化 -> 初始不等价。
- **禁止 gate=0 + 末层零初始化并存**：二者相乘使全部梯度恒为 0（adapter 永不更新，
  "指标没变"会被误读为"方向无效"）。本脚本首步即做**梯度自检**断言。

## 用法

    # ① 只做守卫检查（不训练）：断言恒等 + 评估 IoU 仍 = 0.787821
    python train_direction_adapter.py --check-only

    # ② 小样本冒烟
    python train_direction_adapter.py --epochs 1 --limit-train 8 --limit-val 3 --direction-source oracle

    # ③ 正式训练（先只训 adapter，再联合微调）
    python train_direction_adapter.py --epochs 20 --freeze-base --direction-source oracle
    python train_direction_adapter.py --epochs 20 --direction-source oracle
"""
import argparse
import csv
import glob
import math
import random
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from eval_shadow import compute_metrics
from methods import mask_enhanced
from train_adapter_v4 import (
    apply_lora, lora_params, focal_loss, dice_loss, sample_points, gt_bbox,
)
from train_direction_net import split_by_base

from direction_utils import gt_direction, predict_direction, load_direction_net, \
    component_boxes, flip_direction_horizontal, predict_box
from direction_adapter import (
    install_direction_adapter, direction_adapter_params,
    trainable_params_with_adapter, grad_self_check,
)

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
# 主结果权重（无泄漏 LoRA，best-val）。**只读**，绝不覆盖。
BASE_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")
CKPT_DIR = BASE / "checkpoints"
RES_DIR = BASE / "results"
OUT_DIR = BASE / "results" / "sprint20261018"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
LR = 1e-4
LR_MIN = 1e-5
VAL_EVERY = 1
BOX_MIN_AREA = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


# ══════════════════════════════════════════════════════════════════
# 划分：与 train_lora_leakfree / train_cnn_leakfree 完全同一调用序列
# ══════════════════════════════════════════════════════════════════

def leakfree_split():
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    tr, va, _, _ = split_by_base(train_pool, 0.2, seed=1)
    return tr, va, test_files


def write_val_summary(tag, adapter_mode, direction_source, epochs, seed,
                      best_ep, best_iou, best_ber, ckpt, log_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"diradapter_{tag}_val_summary.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "tag", "adapter_mode", "direction_source", "epochs", "seed",
            "best_epoch", "best_val_IoU", "best_val_BER", "ckpt", "log"])
        w.writeheader()
        w.writerow({
            "tag": tag, "adapter_mode": adapter_mode,
            "direction_source": direction_source, "epochs": epochs,
            "seed": seed,
            "best_epoch": best_ep, "best_val_IoU": f"{best_iou:.6f}",
            "best_val_BER": f"{best_ber:.6f}", "ckpt": ckpt,
            "log": Path(log_path).name,
        })
    return p


def build_sam(ckpt_override=None, with_adapter=True, hidden=64, adapter_mode="film"):
    """按正确顺序构建：官方权重 -> apply_lora -> install_adapter ->（可选）增量加载。

    注意顺序：LoRA 与 adapter 都必须在 `sam.to(device)` **之前**挂上。
    `ckpt_override` 指定要增量加载的权重（默认主结果权重）。
    """
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    if with_adapter:
        install_direction_adapter(sam, hidden=hidden, mode=adapter_mode)
    src = ckpt_override if ckpt_override else BASE_CKPT
    if Path(src).exists():
        sd = torch.load(src, map_location="cpu")
        missing, unexpected = sam.load_state_dict(sd, strict=False)
        return sam, missing, unexpected
    return sam, [], []


# ══════════════════════════════════════════════════════════════════
# 评估：框提示 + 方向（与主结果同口径，仅多一个 direction 入参）
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_box_full(sam, img_paths, gt_paths, directions=None):
    sam.eval()
    rows = []
    for i, (ip, gp) in enumerate(zip(img_paths, gt_paths)):
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if gt is None or img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img1024 = transform.apply_image(img_rgb)
        x = sam.preprocess(torch.as_tensor(img1024, dtype=torch.float32)
                           .permute(2, 0, 1)[None].to(device))
        img_emb = sam.image_encoder(x)
        d = None if directions is None else directions[i]
        d_t = None if d is None else torch.as_tensor(d, dtype=torch.float32)[None].to(device)

        boxes = component_boxes(mask_enhanced(img_bgr), BOX_MIN_AREA)
        comb = np.zeros(gt.shape, bool)
        for b in boxes:
            b1024 = transform.apply_boxes(np.array([b], float), (h, w))
            b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
            sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None,
                                               direction=d_t)
            low_res, _ = sam.mask_decoder(
                image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            comb |= (sam.postprocess_masks(low_res, (1024, 1024), (h, w))[0, 0] > 0).cpu().numpy()
        m = compute_metrics((comb * 255).astype(np.uint8), gt)
        m["name"] = Path(ip).name
        rows.append(m)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
    return avg, std, rows


def directions_for(files, source, dmodel):
    """为一批图准备方向向量。"""
    out = []
    for f in files:
        if source == "none":
            out.append(None)
        elif source == "random":
            rr = np.random.RandomState(hash(Path(f).name) % (2 ** 31))
            a = rr.uniform(0, math.pi)
            out.append(np.array([math.cos(2 * a), math.sin(2 * a)], dtype=np.float32))
        elif source == "oracle":
            gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
            out.append(gt_direction(gt) if gt is not None else None)
        else:  # regress
            img = cv2.imread(f)
            out.append(predict_direction(dmodel, img, device) if img is not None else None)
    return out


# ══════════════════════════════════════════════════════════════════
# 训练步（复制 train_adapter_v4.train_step，差异：多一个方向条件）
# ══════════════════════════════════════════════════════════════════

def train_step_direction(sam, img_rgb, gt, d, opt, scaler):
    h, w = gt.shape
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()
        # [!] 水平镜像必须同步变换方向：轴角 θ->π−θ ⇒ (cos2θ, sin2θ)->(cos2θ, −sin2θ)。
        #    漏做不会有任何报错，且因 adapter 恒等初始化，等价性自检也抓不到。
        if d is not None:
            d = flip_direction_horizontal(d)

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

    d_t = None
    if d is not None:
        d_t = torch.as_tensor(np.asarray(d, dtype=np.float32), dtype=torch.float32)[None].to(device)

    x = sam.preprocess(img_t)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        img_emb = sam.image_encoder(x)
        sparse, dense = sam.prompt_encoder(points=points, boxes=boxes, masks=None,
                                           direction=d_t)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--direction-source", choices=["regress", "oracle", "random", "none"],
                    default="regress",
                    help="方向来源。**主方法必须用 regress（预测方向）**；"
                         "oracle 用 GT 方向，仅作诊断/上界，训练+评估须同源否则构成隐性泄漏")
    ap.add_argument("--freeze-base", action="store_true",
                    help="只训练 adapter（冻结 LoRA 与解码器）--应急预案 D1 的第二步")
    ap.add_argument("--check-only", action="store_true",
                    help="只做守卫检查：断言恒等 + 评估 IoU 仍等于主结果 0.787821，不训练")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--adapter-mode", choices=["film", "cross_attn"], default="film",
                    help="film=原方向 Adapter；cross_attn=简化版 Feature Aggregator")
    ap.add_argument("--val-only", action="store_true",
                    help="训练结束只保留 val 结果，不评 test（冲刺预注册要求）")
    ap.add_argument("--seed", type=int, default=0,
                    help="只影响随机数/初始化；数据划分不变")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="初始化权重（默认主结果权重）。warm-start 时传上一次运行的产物："
                         "--ckpt checkpoints/sam_vit_b_shadow_v4_dir_adapter_<tag>.pth")
    ap.add_argument("--tag", type=str, default="main",
                    help="输出标签：决定权重与结果文件名。**不同运行必须用不同 tag**，"
                         "否则会互相覆盖（冒烟请用 --tag smoke）")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"diradapter_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"# 模块③ 方向特征 Adapter 训练（方向来源={args.direction_source}"
        f"{'，只训 adapter' if args.freeze_base else '，联合微调'}）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 基线权重（只读）: {args.ckpt or BASE_CKPT}")
    log(f"# adapter_mode={args.adapter_mode}  seed={args.seed}")

    tr, va, te = leakfree_split()
    if args.limit_train:
        tr = tr[:args.limit_train]
    if args.limit_val:
        va = va[:args.limit_val]
    val_imgs = va

    dmodel = None
    if args.direction_source == "regress":
        dmodel = load_direction_net(device=device)

    sam, missing, unexpected = build_sam(args.ckpt, with_adapter=True,
                                         hidden=args.hidden,
                                         adapter_mode=args.adapter_mode)
    log(f"# 增量加载：missing={len(missing)} keys, unexpected={len(unexpected)} keys")
    if unexpected:
        log(f"！unexpected 非空（键名/结构不符）：{unexpected[:5]}")
        raise SystemExit("unexpected keys 非空，停止")
    bad_missing = [k for k in missing if "direction_adapter" not in k]
    if bad_missing:
        log(f"！missing 里出现非 adapter 的键（基座权重不匹配）：{bad_missing[:5]}")
        raise SystemExit("missing keys 异常，停止")
    log(f"#  missing 仅含 adapter 新增键（{len(missing)} 个），符合预期")

    # 模型上设备（必须在任何推理/训练之前）
    sam.to(device)

    # ── 守卫 1：恒等性（方向注入前后逐比特相同）──
    from direction_adapter import equivalence_check
    ad = sam.prompt_encoder.direction_adapter
    delta = equivalence_check(ad)
    log(f"# 恒等性检查：max|Δ| = {delta:.3e}（要求 < 1e-5）")
    assert delta < 1e-5, "恒等初始化失效：注入改变了输出"

    # ── 守卫 2：--check-only 断言评估 IoU 仍等于主结果（**在测试集上**）──
    # [!] 必须用测试集 te（92 张），因为 0.787821 是测试集口径；
    #    若在验证集 va（58 张）上评估，其 best-val 是 ~=0.8128，断言会**假报警**。
    if args.check_only:
        log("# check-only：在**测试集(92张)**上评估未训练状态（不做任何训练）")
        te_gts = [str(AISD / "Train412" / "mask" / Path(f).name) for f in te]
        for src in ["none", "oracle"]:
            dirs = directions_for(te, src, dmodel) if src != "none" else None
            avg, std, _ = eval_box_full(sam, te, te_gts, dirs)
            log(f"    方向源={src:<7} IoU={avg['IoU']:.6f}  BER={avg['BER']:.6f}")
            if src == "none":
                assert abs(avg["IoU"] - 0.787821) < 1e-3, (
                    f"守卫失败：装上 adapter 后 IoU 变了！期望 0.787821，实际 {avg['IoU']:.6f}。"
                    f"说明恒等初始化被破坏，会污染主结果--**立即停止**")
                log("    -> [PASS] 与主结果 0.787821 一致（差 <1e-3），恒等初始化成立")
        log(f"运行日志 -> {log_path}")
        logf.close()
        return

    sam.to(device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in lora_params(sam):
        p.requires_grad = True
    sam.prompt_encoder.train(); sam.mask_decoder.train()

    params = trainable_params_with_adapter(sam, lora_params)
    n_ad = sum(p.numel() for p in direction_adapter_params(sam))
    if args.freeze_base:
        for p in params:
            p.requires_grad = False
        for p in direction_adapter_params(sam):
            p.requires_grad = True
        params = direction_adapter_params(sam)
        log(f"# --freeze-base：只训练 adapter（{n_ad} 个参数）")
    else:
        log(f"# 可训练参数 {sum(p.numel() for p in params)/1e6:.2f}M（其中 adapter {n_ad} 参数）")

    # ── 守卫 3：梯度自检（首步后 adapter 参数须有非零梯度，防"零梯度死锁"）──
    gopt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    gscaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    tmp_rep = grad_self_check(ad, use_autocast=(device == "cuda"))
    if tmp_rep["__dead__"]:
        raise SystemExit(f"！零梯度死锁：{tmp_rep['__dead__']}--adapter 将永不更新，请检查初始化")
    log("# 梯度自检：adapter 每个参数都曾拿到非零梯度（无死锁）")

    opt = gopt
    scaler = gscaler
    ckpt_out = str(CKPT_DIR / f"sam_vit_b_shadow_v4_dir_adapter_{args.tag}_leakfree.pth")
    best = 0.0
    best_ep = 0
    best_ber = float("nan")

    for epoch in range(1, args.epochs + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * epoch / args.epochs))
        for g in opt.param_groups:
            g["lr"] = lr
        sam.prompt_encoder.train(); sam.mask_decoder.train()
        pairs = list(zip(tr, [str(AISD / "Train412" / "mask" / Path(f).name) for f in tr]))
        random.shuffle(pairs)
        t0 = time.time(); losses = []
        dir_cache = {}
        for ip, gp in pairs:
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            if args.direction_source == "none":
                d = None
            elif args.direction_source == "oracle":
                if ip not in dir_cache:
                    dir_cache[ip] = gt_direction(gt)
                d = dir_cache[ip]
            elif args.direction_source == "random":
                rr = np.random.RandomState(hash(Path(ip).name) % (2 ** 31))
                a = rr.uniform(0, math.pi)
                d = np.array([math.cos(2 * a), math.sin(2 * a)], dtype=np.float32)
            else:
                d = predict_direction(dmodel, cv2.imread(ip), device)
            img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            loss = train_step_direction(sam, img, gt, d, opt, scaler)
            if loss is not None:
                losses.append(loss)
        log(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses) if losses else float('nan'):.4f}"
            f"  lr={lr:.2e}  ({time.time()-t0:.0f}s)")

        if epoch % VAL_EVERY == 0:
            dirs = directions_for(val_imgs, args.direction_source, dmodel)
            vavg, _, _ = eval_box_full(
                sam, val_imgs,
                [str(AISD / "Train412" / "mask" / Path(f).name) for f in val_imgs], dirs)
            log(f"    -> 验证 IoU={vavg['IoU']:.4f}  BER={vavg['BER']:.4f}")
            if vavg["IoU"] > best:
                best, best_ep = vavg["IoU"], epoch
                best_ber = vavg["BER"]
                torch.save(sam.state_dict(), ckpt_out)
                log(f"    -> 保存最佳 (val IoU {best:.4f})")

    if args.val_only:
        vp = write_val_summary(args.tag, args.adapter_mode, args.direction_source,
                               args.epochs, args.seed, best_ep, best, best_ber,
                               ckpt_out, log_path)
        log("")
        log("[SKIP] --val-only: test not evaluated")
        log(f"val summary -> {vp}")
        log(f"权重 -> {ckpt_out}")
        log(f"运行日志 -> {log_path}")
        logf.close()
        return

    # 最终评估（测试集）
    if Path(ckpt_out).exists():
        sd = torch.load(ckpt_out, map_location=device)
        sam.load_state_dict(sd, strict=False)
    dirs_te = directions_for(te, args.direction_source, dmodel)
    tavg, tstd, trows = eval_box_full(
        sam, te, [str(AISD / "Train412" / "mask" / Path(f).name) for f in te], dirs_te)

    log("")
    log(f"=== 无泄漏测试集（{len(trows)} 张，best-val@Epoch{best_ep}）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {tavg[k]:.4f} ± {tstd[k]:.4f}")

    mname = f"LoRA+DirAdapter({args.direction_source}-dir)_{args.tag}_leakfree"
    p = RES_DIR / f"diradapter_{args.tag}_test_per_image.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        wt = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        wt.writeheader()
        for r in trows:
            wt.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
    s = RES_DIR / f"diradapter_{args.tag}_test.csv"
    with open(s, "w", newline="", encoding="utf-8") as f:
        wt = csv.writer(f)
        wt.writerow(["method", "n_images"] + METRIC_KEYS)
        wt.writerow([mname, len(trows)] + [f"{tavg[k]:.6f}" for k in METRIC_KEYS])
        wt.writerow([mname + "_std", len(trows)] + [f"{tstd[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"# 基线对照（主结果 LoRA-only）: IoU 0.787821 / BER 0.064359")
    log(f"# 铁律：新方法必须 >= 0.7878 才替换主方法，否则保留 LoRA-only 并如实报告为'尝试'")
    log(f"逐图 -> {p}")
    log(f"汇总 -> {s}")
    vp = write_val_summary(args.tag, args.adapter_mode, args.direction_source,
                           args.epochs, args.seed, best_ep, best, best_ber,
                           ckpt_out, log_path)
    log(f"val summary -> {vp}")
    log(f"权重 -> {ckpt_out}")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
