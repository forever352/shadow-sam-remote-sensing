"""AISD 版单图多方法可视对比（n=1，非 benchmark）。

与 RSISD 版同一套方法/权重/参数，只换域内 AISD 测试图与复现门禁来源。
选图与预注册见 `results/visual_compare/SELECTION_LOG_AISD.md`。

用法：
    python visual_compare_aisd.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import env_setup  # noqa: F401

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import visual_compare_methods as ovm
from visual_compare_methods import (
    BOX_MIN_AREA, CKPT, DEVICE, IN_SIZE, MRCNN_TAU, METRIC_KEYS,
    load_frozen_sam, load_lora_sam, load_official_sam,
    predict_maskrcnn, predict_sam_amg, predict_sam_boxes,
    predict_unet_segnet, sha256,
)
from eval_shadow import compute_metrics
from methods import mask_c3i, mask_enhanced, mask_otsu
from train_unet import UNet
from train_segnet import SegNet
from train_maskrcnn_leakfree import build_model

BASE = Path(__file__).resolve().parent
CAVEAT = (
    "n=1 case study; in-domain AISD test image from the leak-free 68/16/21 split "
    "(92-image / 21-scene test set); this image is a member of that test set, not an independent new test; "
    "not a random or representative sample; "
    "seed-0; no TTA/ensemble/significance; do not rank, do not mark best, no generalization claim; "
    "do not compare this single-image metric with the AISD main table (3-seed mean 0.7990 / BER 0.0612), "
    "the 92-image aggregate, or literature; Mask R-CNN input 512 with model min800/max1333 and tau=0.05 "
    "(val-selected); C3I without inversion heuristic; Enhanced is the project's internal coarse extractor, "
    "not an external baseline; AISD test split has no visibly obvious water body, this image was chosen "
    "for vegetation and dark-surface content."
)


def overlay(img_bgr, mask):
    ov = img_bgr.copy()
    ov[mask] = [0, 255, 0]
    return cv2.addWeighted(img_bgr, 0.6, ov, 0.4, 0)


def save_figure(img_bgr, gt, preds, metrics, out_png, stem):
    panels = [
        ("Input", None, img_bgr, None),
        ("Ground truth (reference only)", None, gt, None),
        ("Enhanced coarse (internal module)", "Enhanced", img_bgr, preds["Enhanced"]),
        ("Otsu", "Otsu", img_bgr, preds["Otsu"]),
        ("C3I (no inversion heuristic)", "C3I", img_bgr, preds["C3I"]),
        ("SAM-AMG (zero-shot)", "SAM-AMG", img_bgr, preds["SAM-AMG"]),
        ("SAM zero-shot (box prompt)", "SAM-zero", img_bgr, preds["SAM-zero"]),
        ("Frozen SAM (box prompt)", "Frozen-SAM", img_bgr, preds["Frozen-SAM"]),
        ("LoRA-SAM seed0 (box prompt)", "LoRA-SAM", img_bgr, preds["LoRA-SAM"]),
        ("U-Net seed0", "U-Net", img_bgr, preds["U-Net"]),
        ("SegNet seed0", "SegNet", img_bgr, preds["SegNet"]),
        ("Mask R-CNN seed0 (res800, tau=0.05)", "Mask R-CNN", img_bgr, preds["Mask R-CNN"]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 14.5), constrained_layout=True)
    for ax, (title, key, base, mask) in zip(axes.ravel(), panels):
        if title.startswith("Ground truth"):
            ax.imshow(gt, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(overlay(base, mask), cv2.COLOR_BGR2RGB) if mask is not None
                      else cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        sub = (f"\nIoU {metrics[key]['IoU']:.3f} / BER {metrics[key]['BER']:.3f} / "
               f"F1 {metrics[key]['F1']:.3f}") if key else ""
        ax.set_title(title + sub, fontsize=9, pad=8)
    fig.suptitle(f"Single-image case study (n=1), AISD test / {stem} — "
                 "in-domain (AISD-trained and tested), seed-0, no TTA/ensemble", fontsize=13)
    fig.supxlabel(
        "Not a benchmark. n=1 case study; in-domain AISD test image from the leak-free 92-image / 21-scene "
        "test set, not an independent new test; not a random or representative sample.\n"
        "seed-0; no TTA/ensemble/significance; do not rank, do not mark best, no generalization claim; "
        "do not compare with the AISD main table (3-seed mean 0.7990 / BER 0.0612), the 92-image aggregate, "
        "or literature.\n"
        "Mask R-CNN input 512 with model min800/max1333 and tau=0.05 (val-selected); C3I without inversion "
        "heuristic; Enhanced is the project's internal coarse extractor, not an external baseline; "
        "AISD test split has no visibly obvious water body.",
        fontsize=7,
    )
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(BASE / "AISD/AISD/Train412/shadow/chicago22_sub9.tif"))
    ap.add_argument("--gt", default=str(BASE / "AISD/AISD/Train412/mask/chicago22_sub9.tif"))
    ap.add_argument("--out-dir", default=str(BASE / "results/visual_compare"))
    args = ap.parse_args()

    img_path, gt_path = Path(args.image), Path(args.gt)
    out_dir = Path(args.out_dir)
    stem = img_path.stem
    mask_dir = out_dir / f"AISD_{stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = BASE / "logs" / f"visual_compare_aisd_{ts}.log"
    log_path.parent.mkdir(exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"读不到影像: {img_path}")
    h, w = img_bgr.shape[:2]
    log("# AISD 版单图多方法可视对比（n=1，非 benchmark）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
    log(f"# 命令: {' '.join(sys.argv)}")
    log(f"# 脚本: {Path(__file__).resolve()}  SHA256: {sha256(Path(__file__).resolve())}")
    ovm_path = Path(ovm.__file__).resolve()
    log(f"# 复用模块: {ovm_path}  SHA256: {sha256(ovm_path)}")
    log(f"# 依赖: torch {torch.__version__} / cv2 {cv2.__version__} / numpy {np.__version__}")
    log(f"# 影像: {img_path}")
    log(f"# 影像 SHA256: {sha256(img_path)}  尺寸: {w}x{h}")
    sel_log = out_dir / "SELECTION_LOG_AISD.md"
    log(f"# 选图与预注册: {sel_log}"
        + (f"  SHA256: {sha256(sel_log)}" if sel_log.exists() else "  [缺失]"))
    log("# [!] 域内: 学习模型 AISD 训练; 本图 AISD 无泄漏测试集")
    log("# [!] n=1 / seed-0 / 无 TTA / 无集成 / 无显著性; 不排序、不作泛化结论")
    log("# [!] AISD 测试集无可见水体; 本图代表植被/暗色地物")
    log("# [!] 推理只读输入影像; GT 在全部预测落盘后才读取, 仅用于指标")

    weights = [
        "sam_vit_b_01ec64.pth", "sam_vit_b_shadow_frozen_leakfree.pth",
        "sam_vit_b_shadow_v4_lora_leakfree.pth", "unet_shadow_leakfree.pth",
        "segnet_shadow_leakfree.pth", "maskrcnn_shadow_leakfree_res800.pth",
    ]
    log("\n# 权重清单（size / mtime / sha256）")
    for name in weights:
        p = CKPT / name
        st = p.stat()
        log(f"  {name:42s} {st.st_size:>12d}  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}  {sha256(p)}")

    preds = {}

    log("\n[1/6] 零样本 SAM（同框提示）+ SAM-AMG")
    sam = load_official_sam()
    preds["SAM-zero"] = predict_sam_boxes(sam, img_bgr)
    log(f"  SAM-zero 前景像素: {int(preds['SAM-zero'].sum())}")
    preds["SAM-AMG"] = predict_sam_amg(sam, img_bgr)
    log(f"  SAM-AMG 前景像素: {int(preds['SAM-AMG'].sum())}")
    del sam
    torch.cuda.empty_cache()

    log("[2/6] 冻结编码器 SAM")
    sam = load_frozen_sam()
    preds["Frozen-SAM"] = predict_sam_boxes(sam, img_bgr)
    log(f"  Frozen-SAM 前景像素: {int(preds['Frozen-SAM'].sum())}")
    del sam
    torch.cuda.empty_cache()

    log("[3/6] LoRA-SAM seed0（主方法）")
    sam = load_lora_sam()
    preds["LoRA-SAM"] = predict_sam_boxes(sam, img_bgr)
    log(f"  LoRA-SAM 前景像素: {int(preds['LoRA-SAM'].sum())}")
    del sam
    torch.cuda.empty_cache()

    log("[4/6] 传统方法（无模型）")
    preds["Enhanced"] = mask_enhanced(img_bgr) > 0
    preds["Otsu"] = mask_otsu(img_bgr) > 0
    preds["C3I"] = mask_c3i(img_bgr, invert_if_mostly_white=False) > 0
    for k in ("Enhanced", "Otsu", "C3I"):
        log(f"  {k} 前景像素: {int(preds[k].sum())}")

    log("[5/6] U-Net / SegNet seed0")
    for arch, cls, ckpt_name in [
        ("U-Net", UNet, "unet_shadow_leakfree.pth"),
        ("SegNet", SegNet, "segnet_shadow_leakfree.pth"),
    ]:
        model = cls().to(DEVICE)
        model.load_state_dict(torch.load(str(CKPT / ckpt_name), map_location=DEVICE))
        model.eval()
        preds[arch] = predict_unet_segnet(model, img_path, (h, w))
        log(f"  {arch} 前景像素: {int(preds[arch].sum())}")
        del model
        torch.cuda.empty_cache()

    log("[6/6] Mask R-CNN seed0（res800）")
    model = build_model(800, 1333).to(DEVICE)
    model.load_state_dict(torch.load(str(CKPT / "maskrcnn_shadow_leakfree_res800.pth"),
                                     map_location=DEVICE))
    model.eval()
    log(f"  模型 transform: min_size={model.transform.min_size} max_size={model.transform.max_size}")
    preds["Mask R-CNN"] = predict_maskrcnn(model, img_bgr, MRCNN_TAU)
    log(f"  Mask R-CNN 前景像素: {int(preds['Mask R-CNN'].sum())}"
        f"  (input {IN_SIZE}x{IN_SIZE}, tau={MRCNN_TAU})")
    del model
    torch.cuda.empty_cache()

    # 预测全部完成 -> 先落盘掩膜（GT 仍未读取）
    for k, m in preds.items():
        cv2.imwrite(str(mask_dir / f"{k.replace(' ', '_').replace('(', '').replace(')', '')}.png"),
                    m.astype(np.uint8) * 255)
    log(f"\n# predictions written to disk before GT is read: {mask_dir}")

    log("# GT not used for inference; reading GT only now for metrics")
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise SystemExit(f"读不到 GT: {gt_path}")
    if not np.all(np.isin(np.unique(gt), [0, 255])):
        raise SystemExit(f"GT 非二值: {np.unique(gt)[:10]}")
    if gt.shape != (h, w):
        raise SystemExit(f"GT 尺寸不符: {gt.shape} vs {(h, w)}")

    metrics = {k: compute_metrics((m * 255).astype(np.uint8), gt) for k, m in preds.items()}

    ref = None
    with open(BASE / "results/lora_leakfree_test_per_image.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("name") == img_path.name:
                ref = {k: float(r[k]) for k in ("IoU", "BER", "F1")}
                break
    if ref is None:
        raise SystemExit(f"{img_path.name} 不在 lora_leakfree_test_per_image.csv，无法做复现门禁")

    dev = max(abs(metrics["LoRA-SAM"][k] - ref[k]) for k in ref)
    log(f"\n# LoRA 单图复现门禁: max|diff| = {dev:.3e}（阈值 1e-6；参照 "
        f"lora_leakfree_test_per_image.csv 的 {img_path.name} 行）")
    if dev > 1e-6:
        log("  [中止] 与主结果逐图 CSV 不一致 -> 链路漂移，不报告")
        logf.close()
        raise SystemExit(3)
    log("  [OK] 复现通过（seed-0，同框提示，同 BOX_MIN_AREA=300）")

    log("\n# 单图指标（n=1；不排序、不写 best、不作泛化结论）")
    log(f"{'method':22s}{'IoU':>9s}{'BER':>9s}{'F1':>9s}{'OA':>9s}{'P':>9s}{'R':>9s}")
    for k in preds:
        v = metrics[k]
        log(f"{k:22s}" + "".join(f"{v[x]:9.4f}" for x in METRIC_KEYS))

    csv_path = out_dir / f"AISD_{stem}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["method"] + METRIC_KEYS + ["n", "note"])
        for k in preds:
            wr.writerow([k] + [f"{metrics[k][x]:.6f}" for x in METRIC_KEYS] + [1, CAVEAT])

    png_path = out_dir / f"AISD_{stem}_compare.png"
    save_figure(img_bgr, gt, preds, metrics, png_path, stem)
    log(f"\n图 -> {png_path}")
    log(f"指标 -> {csv_path}")
    log(f"掩膜 -> {mask_dir}")
    log(f"日志 -> {log_path}")
    log(f"# 限定: {CAVEAT}")
    logf.close()


if __name__ == "__main__":
    main()
