"""模块② 评估：方向引导提示 vs 对照，在同一无泄漏划分上出六项指标（结果落盘）。

方法集（一次遍历内全部跑完，图像只编码一次）：
  - `DirPrompt(project-dir)`  本项目：方向引导提示（主方法）
  - `DirPrompt(random-dir)`   对照：固定随机方向（隔离"方向是否有信息"）
  - `DirPrompt(no-dir)`       对照：同样点数但**不建扇形**（隔离"方向"与"点数/位置"）
  - `LoRA-SAM(axis-box)`      参考基线：= evaluate_lora 的框提示链路（用于**校验链路**：
                              在无泄漏划分 + 无泄漏 LoRA 下应复现 ~=0.7878）
  - `LoRA-SAM(centroid-points)` 参考基线：连通域质心点提示

方向来源（`--direction-source`）：
  - `oracle` ：从 **GT 掩膜**算方向 -- [!] **仅作"诊断/上界"**，方法名会带 `(oracle-dir)`，
               **禁止**当作可汇报主方法（用 GT 即标签泄漏）
  - `regress`：用方向回归网络的**预测**方向 -- 主方法口径
  - `random` ：固定随机方向

落盘（对齐项目规范）：
  results/dirprompt_{Tag}_test.csv            （method,n_images,六指标 + _std 行，:.6f，utf-8 无 BOM）
  results/dirprompt_{Tag}_test_per_image.csv  （name,六指标，原始浮点）
  logs/dirprompt_{YYYYMMDD_HHmmss}.log

用法：
    python direction_prompt_eval.py --limit 3 --direction-source oracle          # 冒烟
    python direction_prompt_eval.py --direction-source oracle                    # 全量诊断
    python direction_prompt_eval.py --direction-source regress                   # 主方法（需先训好回归头）
"""
import argparse
import csv
import dataclasses
import random
import time
from collections import defaultdict
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_enhanced
from train_adapter_v4 import apply_lora
from train_direction_net import split_by_base
from evaluate_finetuned import auto_points

from direction_utils import (
    gt_direction, predict_direction, load_direction_net,
    component_boxes, component_items,
)
from direction_prompt import build_direction_points, DirPromptConfig

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
LORA_LEAKFREE = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
BOX_MIN_AREA = 300   # 与 evaluate_lora 一致，保证 axis_box 参考行可复现 0.7878

device = "cuda" if torch.cuda.is_available() else "cpu"


def leakfree_split():
    """无泄漏划分（与 train_lora_leakfree / train_cnn_leakfree 同一调用序列）。"""
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, _ = split_by_base(files, 0.2, seed=0)
    _, _, _, _ = split_by_base(train_pool, 0.2, seed=1)
    return test_files


@torch.no_grad()
def predict_points_batch(sam, img_emb, h, w, coords, labels, transform):
    """点提示推理（与 direction_experiment.predict_points 同口径）。"""
    c1024 = transform.apply_coords(coords, (h, w))
    c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
    l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
    sparse, dense = sam.prompt_encoder(points=(c_t, l_t), boxes=None, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return (mask[0, 0] > 0).cpu().numpy()


@torch.no_grad()
def predict_box_single(sam, img_emb, h, w, box, transform):
    b1024 = transform.apply_boxes(np.array([box], float), (h, w))
    b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
    sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return (sam.postprocess_masks(low_res, (1024, 1024), (h, w))[0, 0] > 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-source", choices=["oracle", "regress", "random"], default="oracle")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--ckpt", type=str, default=LORA_LEAKFREE)
    ap.add_argument("--regress-ckpt", type=str, default=None)
    ap.add_argument("--phi", type=float, default=None)
    ap.add_argument("--rays", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--sector-mode", choices=["axis", "perp", "cross"], default=None)
    ap.add_argument("--sector-source", choices=["global", "component"], default=None,
                    help="扇形中心角来源：global=整图全局方向（申报书口径）；"
                         "component=逐连通域自身主轴（几何可得，无需网络）")
    ap.add_argument("--dev-thresh", type=float, default=None,
                    help="方向一致性过滤阈值（度）。默认 45；配合 project(有过滤) vs "
                         "project_nofilter(无过滤) 分离「过滤」的净作用")

    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"dirprompt_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    cfg = DirPromptConfig()
    for k, v in [("phi_deg", args.phi), ("n_rays", args.rays),
                 ("n_samples", args.samples), ("sector_mode", args.sector_mode),
                 ("sector_source", args.sector_source), ("dev_thresh_deg", args.dev_thresh)]:
        if v is not None:
            setattr(cfg, k, v)

    src = args.direction_source
    # tag 里带上扇形来源，避免 global/component 两次运行互相覆盖
    tag = f"{src}" if cfg.sector_source == "global" else f"{src}_{cfg.sector_source}"
    # [!] 冒烟（--limit）时加 _smoke 后缀，**禁止冒烟覆盖生产 CSV**
    #     （第九轮审查：这是失效模式 #4 的复发，此前 --limit 直接写了生产路径）
    if args.limit:
        tag += f"_smoke{args.limit}"

    log(f"# 模块② 方向引导提示评估（方向来源={src}）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 权重: {args.ckpt}")
    log(f"# 划分: 无泄漏 basis 68/16/21（与主结果同一基座）")
    log(f"# 提示参数（预注册默认）: phi={cfg.phi_deg} rays={cfg.n_rays} "
        f"samples={cfg.n_samples} radius_k={cfg.radius_k} grad_keep={cfg.grad_keep} "
        f"dev_thresh={cfg.dev_thresh_deg} sector_mode={cfg.sector_mode} "
        f"require_dark={cfg.require_dark}")
    # [!] 必须显式打印这两个：第九轮审查指出，此前只打 sector_mode=axis，
    #     导致把 sector_source=component 的运行误读为 global（追了一次不存在的"异常"）
    log(f"# sector_source={cfg.sector_source}  apply_dir_filter(随条件变化)  "
        f"fallback_to_none={cfg.fallback_to_none}")
    if src == "oracle":
        log("# [!] oracle：方向由 GT 掩膜计算，**仅作诊断/上界**，不得作为可汇报主方法")
    log()

    # 模型
    from segment_anything.utils.transforms import ResizeLongestSide
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    sam.load_state_dict(torch.load(args.ckpt, map_location=device))
    sam.to(device); sam.eval()
    transform = ResizeLongestSide(1024)

    dmodel = None
    if src == "regress":
        dmodel = load_direction_net(args.regress_ckpt or None, device) if args.regress_ckpt \
            else load_direction_net(device=device)

    files = leakfree_split()
    if args.limit:
        files = files[:args.limit]
    log(f"# 图像数: {len(files)}")

    methods = ["project", "project_nofilter", "random", "none", "axis_box", "centroid_points"]
    dir_label = f"{src}-dir" if cfg.sector_source == "global" else f"{src}-dir/{cfg.sector_source}-sector"
    # [!] 对照分解（第八轮审查必须修正 1）：
    #   project          = 扇形 + 方向一致性过滤
    #   project_nofilter = 扇形，**不过滤** <- 与 project 比，隔离"过滤"的作用
    #   none             = 无扇形、无过滤       <- 与 project_nofilter 比，隔离"扇形(方向)"的作用
    point_variants = [
        ("project", "project", True),
        ("project_nofilter", "project", False),
        ("random", "random", True),
        ("none", "none", True),
    ]
    method_names = {
        "project": f"DirPrompt({dir_label})_leakfree",
        "project_nofilter": f"DirPrompt({dir_label}/no-filter)_leakfree",
        "random": "DirPrompt(random-dir)_leakfree",
        "none": "DirPrompt(no-dir)_leakfree",
        "axis_box": "LoRA-SAM(axis-box)_leakfree",
        "centroid_points": "LoRA-SAM(centroid-points)_leakfree",
    }

    acc = {m: [] for m in methods}
    per_image = {m: [] for m in methods}
    n_fallback = 0
    funnel = {m: defaultdict(int) for m, _, _ in point_variants}   # 逐方法累计漏斗

    for ip in files:
        name = Path(ip).name
        img_bgr = cv2.imread(ip)
        gt = cv2.imread(str(AISD / "Train412" / "mask" / name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        em = mask_enhanced(img_bgr)

        # 方向
        if src == "oracle":
            d = gt_direction(gt)
            if d is None:
                log(f"  {name:26s} 跳过（GT 无有效方向）")
                continue
        elif src == "regress":
            d = predict_direction(dmodel, img_bgr, device)
            if d is None:
                continue
        else:
            d = None  # random 模式在 build_direction_points 内部处理

        # 图像编码一次复用
        # [!] 必须 no_grad：SAM 参数 requires_grad=True，若不带 no_grad，
        #    每张图都会建一张计算图并保留 -> 显存随图像数累积 -> OOM（实测踩过）
        h, w = img_rgb.shape[:2]
        img1024 = transform.apply_image(img_rgb)
        with torch.no_grad():
            x = sam.preprocess(torch.as_tensor(img1024, dtype=torch.float32)
                               .permute(2, 0, 1)[None].to(device))
            img_emb = sam.image_encoder(x)

        line = f"  {name:26s}"

        # ── 方向提示的四种条件（见 point_variants 的对照分解）──
        for mname, mode, use_filter in point_variants:
            cfg_m = dataclasses.replace(cfg, apply_dir_filter=use_filter)
            st = {}
            c, l = build_direction_points(img_bgr, d, cfg_m, mode=mode, stats=st)
            for k, v in st.items():
                if isinstance(v, bool):
                    funnel[mname][k] += int(v)
                else:
                    funnel[mname][k] += int(v)
            if c is None:
                pred = np.zeros(gt.shape, bool)
            else:
                pred = predict_points_batch(sam, img_emb, h, w, c, l, transform)
            if st.get("used_fallback"):
                n_fallback += 1
            met = compute_metrics((pred * 255).astype(np.uint8), gt)
            acc[mname].append(met); per_image[mname].append({"name": name, **met})
            line += f"  {mname[:6]}={met['IoU']:.4f}"

        # 漏斗中间量（冒烟时打印首图，防"近乎失效"的实现静默通过）
        if args.limit and len(per_image["project"]) == 1:
            for mname, _, _ in point_variants:
                log(f"      [漏斗/{mname}] 连通域={funnel[mname]['n_comp_total']} "
                    f"方向过滤丢弃={funnel[mname]['n_dropped_dir']} 候选={funnel[mname]['n_cand']} "
                    f"过梯度={funnel[mname]['n_after_grad']} 光度剔除={funnel[mname]['n_dropped_photo']} "
                    f"最终前景={funnel[mname]['n_fg']} 兜底={funnel[mname]['used_fallback']}")

        # ── 参考基线：框提示（= evaluate_lora 链路）──
        boxes = component_boxes(em, BOX_MIN_AREA)
        comb = np.zeros(gt.shape, bool)
        for b in boxes:
            comb |= predict_box_single(sam, img_emb, h, w, b, transform)
        met = compute_metrics((comb * 255).astype(np.uint8), gt)
        acc["axis_box"].append(met); per_image["axis_box"].append({"name": name, **met})
        line += f"  box={met['IoU']:.4f}"

        # ── 参考基线：质心点 ──
        c2, l2 = auto_points(em)
        if c2 is None or len(c2) == 0:
            pred2 = np.zeros(gt.shape, bool)
        else:
            pred2 = predict_points_batch(sam, img_emb, h, w,
                                         np.asarray(c2, float), np.asarray(l2, np.int64), transform)
        met = compute_metrics((pred2 * 255).astype(np.uint8), gt)
        acc["centroid_points"].append(met); per_image["centroid_points"].append({"name": name, **met})
        line += f"  ctr={met['IoU']:.4f}"

        log(line)

    # ── 落盘 ──
    log()
    log("=== 汇总（无泄漏划分）===")
    for m in methods:
        ms = acc[m]
        if not ms:
            continue
        avg = {k: float(np.mean([r[k] for r in ms])) for k in METRIC_KEYS}
        std = {k: float(np.std([r[k] for r in ms])) for k in METRIC_KEYS}
        p = RES_DIR / f"dirprompt_{m}_{tag}_test_per_image.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            wt = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
            wt.writeheader(); wt.writerows(per_image[m])
        s = RES_DIR / f"dirprompt_{m}_{tag}_test.csv"
        with open(s, "w", newline="", encoding="utf-8") as f:
            wt = csv.writer(f)
            wt.writerow(["method", "n_images"] + METRIC_KEYS)
            wt.writerow([method_names[m], len(ms)] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
            wt.writerow([method_names[m] + "_std", len(ms)] + [f"{std[k]:.6f}" for k in METRIC_KEYS])
        log(f"{method_names[m]:38s} IoU={avg['IoU']:.4f}±{std['IoU']:.4f}  "
            f"BER={avg['BER']:.4f}  F1={avg['F1']:.4f}  (n={len(ms)})")
        log(f"    逐图 -> {p}")
        log(f"    汇总 -> {s}")

    log()
    log(f"# project 模式触发无方向兜底的图像数：{n_fallback}")
    log("")
    log("--- 漏斗累计（全部图像求和）---")
    for mname, _, _ in point_variants:
        f = funnel[mname]
        log(f"  {mname:18s} 连通域={f['n_comp_total']:6d} 方向过滤丢弃={f['n_dropped_dir']:6d} "
            f"候选={f['n_cand']:7d} 过梯度={f['n_after_grad']:6d} 光度剔除={f['n_dropped_photo']:6d} "
            f"最终前景={f['n_fg']:6d} 兜底图数={f['used_fallback']}")
    log("")
    log("--- 对照分解（判读用）---")
    log("  (1) project_nofilter vs none  -> 「扇形(方向)」的净作用")
    log("  (2) project         vs project_nofilter -> 「方向一致性过滤」的净作用")
    log("  [!] 只看 project vs none 会把两者混在一起，结论无效")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
