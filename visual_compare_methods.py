"""一次性单图多方法可视对比（n=1，非 benchmark）。

人类要求：选一张带水体和植被的影像，把项目已存各方法模型都跑一遍，做直观对比。
选图与预注册见 `results/visual_compare/SELECTION_LOG.md`；本脚本不负责选图。

重要限定（同时写入图、CSV、日志）：
- 跨数据集：学习模型均在 AISD 训练，本图来自 RSISD test200。
- n=1、seed-0、无 TTA / 无集成 / 无显著性；不排序、不写 best、不作泛化结论。
- 不與 AISD 主表、RSISD 200 张聚合或文献数字横比。
- 推理只读输入影像；GT 在全部预测落盘后才读取，仅用于指标。

用法：
    python visual_compare_methods.py
    python visual_compare_methods.py --image <path> --gt <path> --out-dir results/visual_compare
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_c3i, mask_enhanced, mask_otsu, mask_sam_amg
from prompt_experiment import component_boxes, predict_box
from train_adapter_v4 import apply_lora
from train_unet import UNet, load_image_tensor
from train_segnet import SegNet
from train_maskrcnn_leakfree import MASK_THR, build_model

BASE = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_SIZE = 512
BOX_MIN_AREA = 300
MRCNN_TAU = 0.05
REF = {"IoU": 0.6035804626168885, "BER": 0.08598179847007913, "F1": 0.7527909431246824}
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
CAVEAT = (
    "n=1 case study; cross-dataset (AISD-trained -> RSISD test200); this image is a member of the "
    "RSISD test200 200-image aggregate, not an independent new test; seed-0; no TTA/ensemble/"
    "significance; do not rank, do not mark best, no generalization claim; do not compare with "
    "AISD main table 0.7878, RSISD aggregate 0.7842, or literature; Mask R-CNN input 512 with "
    "model min800/max1333 and tau=0.05 (val-selected); C3I without inversion heuristic; Enhanced "
    "is the project's internal coarse extractor, not an external baseline"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_sam():
    sam = sam_model_registry["vit_b"](checkpoint=str(CKPT / "sam_vit_b_01ec64.pth"))
    return sam.to(DEVICE).eval()


def load_frozen_sam():
    sam = load_official_sam()
    sam.load_state_dict(torch.load(str(CKPT / "sam_vit_b_shadow_frozen_leakfree.pth"),
                                   map_location=DEVICE))
    return sam.to(DEVICE).eval()


def load_lora_sam():
    sam = load_official_sam()
    apply_lora(sam)
    sam.load_state_dict(torch.load(str(CKPT / "sam_vit_b_shadow_v4_lora_leakfree.pth"),
                                   map_location=DEVICE))
    return sam.to(DEVICE).eval()


@torch.no_grad()
def predict_sam_boxes(sam, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = component_boxes(mask_enhanced(img_bgr), BOX_MIN_AREA)
    out = np.zeros(img_bgr.shape[:2], bool)
    for b in boxes:
        out |= predict_box(sam, img_rgb, b)
    return out


@torch.no_grad()
def predict_sam_amg(sam, img_bgr):
    amg = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.88,
        stability_score_thresh=0.95, min_mask_region_area=100,
    )
    return mask_sam_amg(img_bgr, amg, dark_ratio=0.7) > 0


@torch.no_grad()
def predict_unet_segnet(model, img_path, shape):
    x = load_image_tensor(str(img_path))[None].to(DEVICE)
    logits = model(x)[0, 0]
    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred.astype(bool)


@torch.no_grad()
def predict_maskrcnn(model, img_bgr, tau):
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    x = torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    out = model([x.to(DEVICE)])[0]
    scores = out["scores"].cpu().numpy()
    binm = out["masks"].cpu().numpy()[:, 0] > MASK_THR
    sel = scores >= tau
    union = binm[sel].any(axis=0) if sel.any() else np.zeros((IN_SIZE, IN_SIZE), bool)
    pred = cv2.resize(union.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return pred.astype(bool)


def overlay(img_bgr, mask):
    ov = img_bgr.copy()
    ov[mask] = [0, 255, 0]
    return cv2.addWeighted(img_bgr, 0.6, ov, 0.4, 0)


def save_figure(img_bgr, gt, preds, metrics, out_png):
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
    fig.suptitle("Single-image case study (n=1), RSISD test200 / GuiZhou_37.png — "
                 "AISD-trained models, cross-dataset, seed-0, no TTA/ensemble",
                 fontsize=13)
    fig.supxlabel(
        "Not a benchmark. n=1 case study; cross-dataset (AISD-trained -> RSISD test200); "
        "this image is a member of the RSISD test200 200-image aggregate, not an independent new test.\n"
        "seed-0; no TTA/ensemble/significance; do not rank, do not mark best, no generalization claim; "
        "do not compare with AISD main table 0.7878, RSISD aggregate 0.7842, or literature.\n"
        "Mask R-CNN input 512 with model min800/max1333 and tau=0.05 (val-selected); "
        "C3I without inversion heuristic; Enhanced is the project's internal coarse extractor, "
        "not an external baseline.",
        fontsize=7,
    )
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(BASE / "datasets/RSISD/test200/shadow/GuiZhou_37.png"))
    ap.add_argument("--gt", default=str(BASE / "datasets/RSISD/test200/mask/GuiZhou_37.png"))
    ap.add_argument("--out-dir", default=str(BASE / "results/visual_compare"))
    args = ap.parse_args()

    img_path, gt_path = Path(args.image), Path(args.gt)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "GuiZhou_37_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = BASE / "logs" / f"visual_compare_{ts}.log"
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
    log("# 单图多方法可视对比（n=1，非 benchmark）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
    log(f"# 命令: {' '.join(sys.argv)}")
    log(f"# 脚本: {Path(__file__).resolve()}  SHA256: {sha256(Path(__file__).resolve())}")
    log(f"# 依赖: torch {torch.__version__} / cv2 {cv2.__version__} / numpy {np.__version__}")
    log(f"# 影像: {img_path}")
    log(f"# 影像 SHA256: {sha256(img_path)}  尺寸: {w}x{h}")
    sel_log = out_dir / "SELECTION_LOG.md"
    log(f"# 选图与预注册: {sel_log}"
        + (f"  SHA256: {sha256(sel_log)}" if sel_log.exists() else "  [缺失]"))
    log("# [!] 跨数据集: 学习模型均 AISD 训练; 本图 RSISD test200")
    log("# [!] n=1 / seed-0 / 无 TTA / 无集成 / 无显著性; 不排序、不作泛化结论")
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
        log(f"  {name:42s} {st.st_size:>12d}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}  {sha256(p)}")

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

    # 预测全部完成 -> 先把预测掩膜落盘（此时 GT 仍未读取）
    for k, m in preds.items():
        cv2.imwrite(str(mask_dir / f"{k.replace(' ', '_').replace('(', '').replace(')', '')}.png"),
                    m.astype(np.uint8) * 255)
    log(f"\n# predictions written to disk before GT is read: {mask_dir}")

    # 再读 GT，仅用于指标
    log("# GT not used for inference; reading GT only now for metrics")
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise SystemExit(f"读不到 GT: {gt_path}")
    if not np.all(np.isin(np.unique(gt), [0, 255])):
        raise SystemExit(f"GT 非二值: {np.unique(gt)[:10]}")
    if gt.shape != (h, w):
        raise SystemExit(f"GT 尺寸不符: {gt.shape} vs {(h, w)}")

    metrics = {}
    for k, m in preds.items():
        metrics[k] = compute_metrics((m * 255).astype(np.uint8), gt)

    # LoRA 复现门禁（必须与 200 张聚合评估中的该图逐位一致）
    got = metrics["LoRA-SAM"]
    dev = max(abs(got[x] - REF[x]) for x in REF)
    log(f"\n# LoRA 单图复现门禁: max|diff| = {dev:.3e}（阈值 1e-6）")
    if dev > 1e-6:
        log("  [中止] 与 cross_rsisd_test200_per_image.csv 不一致 -> 链路漂移，不报告")
        logf.close()
        raise SystemExit(3)
    log("  [OK] 复现通过（seed-0，同框提示，同 BOX_MIN_AREA=300）")

    log("\n# 单图指标（n=1；不排序、不写 best、不作泛化结论）")
    log(f"{'method':22s}{'IoU':>9s}{'BER':>9s}{'F1':>9s}{'OA':>9s}{'P':>9s}{'R':>9s}")
    for k in preds:
        v = metrics[k]
        log(f"{k:22s}" + "".join(f"{v[x]:9.4f}" for x in METRIC_KEYS))

    csv_path = out_dir / "GuiZhou_37_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["method"] + METRIC_KEYS + ["n", "note"])
        for k in preds:
            wr.writerow([k] + [f"{metrics[k][x]:.6f}" for x in METRIC_KEYS] + [1, CAVEAT])

    png_path = out_dir / "GuiZhou_37_compare.png"
    save_figure(img_bgr, gt, preds, metrics, png_path)
    log(f"\n图 -> {png_path}")
    log(f"指标 -> {csv_path}")
    log(f"掩膜 -> {mask_dir}")
    log(f"日志 -> {log_path}")
    log(f"# 限定: {CAVEAT}")
    logf.close()


if __name__ == "__main__":
    main()
