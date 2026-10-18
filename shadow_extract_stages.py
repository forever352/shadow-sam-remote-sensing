"""阴影提取全流程分阶段可视化：一张图 -> 每个阶段的 PNG + 总览拼图。

输出目录（默认 results/stages/<影像名>/）
    01_original.png          原图
    02_gray.png              灰度
    03_clahe.png             CLAHE 对比度增强
    04_inverted.png          反相（255 - CLAHE）
    05_otsu.png              反相图上的 Otsu 阈值
    06_morph_open.png        形态学开运算（= mask_enhanced 输出）
    07_boxes.png             原图 + 连通域框提示（编号）
    08_sam_union.png         逐框 SAM 预测并集（白=阴影）
    09_final_overlay.png     最终绿色叠加
    10_error_map.png         仅当找到 GT：TP 绿 / FP 红 / FN 蓝
    00_stages_montage.png    3x3 总览拼图（中文标签）
    06b_box_prompts_montage.png  前 12 个框的逐框预测拼图
    boxes/box_XXX.png        每个框的局部原图 + SAM 预测叠加
    stages_summary.txt       计数/指标/路径

用法：
    python shadow_extract_stages.py --image AISD/AISD/Train412/shadow/chicago30_sub1.tif
    python shadow_extract_stages.py --image 你的图.tif --out-dir results/stages/mycase
    python shadow_extract_stages.py --image x.tif --gt x_mask.png
"""
import argparse
import time
from pathlib import Path

# noinspection PyUnusedImports
import env_setup  # noqa: F401  必须在 import torch 之前

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval_shadow import compute_metrics
from methods import mask_enhanced
from prompt_experiment import component_boxes, predict_box
from shadow_extract import DEFAULT_CKPT, load_sam

BASE = Path(__file__).resolve().parent
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def draw_boxes(img_bgr, boxes, color=(0, 255, 255)):
    out = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, str(i), (int(x1) + 3, int(y1) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.4):
    out = img_bgr.copy()
    out[mask] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, out, alpha, 0)


def error_map(img_bgr, pred, gt):
    """TP 绿 / FP 红 / FN 蓝，叠加到原图。"""
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    color = np.zeros_like(img_bgr)
    color[tp] = (0, 200, 0)    # BGR green
    color[fp] = (0, 0, 255)    # BGR red
    color[fn] = (255, 0, 0)    # BGR blue
    return cv2.addWeighted(img_bgr, 0.45, color, 0.55, 0)


def find_gt(image_path):
    """AISD 结构：.../shadow/xxx.tif -> .../mask/xxx.tif"""
    p = Path(image_path)
    parts = list(p.parts)
    if "shadow" in parts:
        i = parts.index("shadow")
        cand = Path(*parts[:i], "mask", *parts[i + 1:])
        if cand.exists():
            return cand
    return None


def make_montage(panels, out_path, cols=3, cell=512):
    """panels: [(label, bgr_image)] -> 带中文标签的总览拼图。"""
    try:
        font = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = ImageFont.load_default()
    rows = (len(panels) + cols - 1) // cols
    bar_h = 34
    W, H = cols * cell, rows * (cell + bar_h)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img_bgr) in enumerate(panels):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell, r * (cell + bar_h)
        small = cv2.resize(img_bgr, (cell, cell), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(rgb), (x0, y0 + bar_h))
        draw.text((x0 + 8, y0 + 6), label, fill=(0, 0, 0), font=font)
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + bar_h + cell - 1],
                       outline=(180, 180, 180), width=1)
    canvas.save(out_path)


def box_montage(img_bgr, boxes, box_masks, out_path, limit=12, cell=256):
    """前 N 个框的局部原图 + SAM 预测，拼成网格。"""
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except Exception:
        font = ImageFont.load_default()
    items = list(zip(boxes, box_masks))[:limit]
    if not items:
        return
    cols = 4
    rows = (len(items) + cols - 1) // cols
    bar_h = 28
    canvas = Image.new("RGB", (cols * cell, rows * (cell + bar_h)), "white")
    draw = ImageDraw.Draw(canvas)
    h, w = img_bgr.shape[:2]
    for idx, ((x1, y1, x2, y2), m) in enumerate(items, 1):
        mx = max(8, int(0.08 * (x2 - x1)))
        my = max(8, int(0.08 * (y2 - y1)))
        cx1, cy1 = max(0, int(x1) - mx), max(0, int(y1) - my)
        cx2, cy2 = min(w, int(x2) + mx), min(h, int(y2) + my)
        crop = img_bgr[cy1:cy2, cx1:cx2].copy()
        mcrop = m[cy1:cy2, cx1:cx2]
        crop = overlay_mask(crop, mcrop, color=(0, 255, 0), alpha=0.45)
        crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx - 1, cols)
        x0, y0 = c * cell, r * (cell + bar_h)
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                     (x0, y0 + bar_h))
        draw.text((x0 + 6, y0 + 5), f"box {idx}", fill=(0, 0, 0), font=font)
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + bar_h + cell - 1],
                       outline=(180, 180, 180), width=1)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="输入影像")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 results/stages/<影像名>）")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="模型权重")
    ap.add_argument("--min-area", type=int, default=300, help="连通域框最小面积")
    ap.add_argument("--gt", default=None, help="GT 掩膜；不填则按 AISD 目录自动查找")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"找不到影像: {img_path}")
    out_dir = Path(args.out_dir) if args.out_dir else BASE / "results" / "stages" / img_path.stem
    (out_dir / "boxes").mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"cv2 无法读取: {img_path}")
    h, w = img_bgr.shape[:2]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"# 分阶段可视化: {img_path.name}  {w}x{h}")
    print(f"# 权重: {Path(args.checkpoint).name}")
    print(f"# 输出: {out_dir}")

    # 1-6: 物理粗提取各阶段
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    inv = 255 - enh
    _, otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = mask_enhanced(img_bgr)
    boxes = component_boxes(morph, args.min_area)
    boxes_img = draw_boxes(img_bgr, boxes)

    cv2.imwrite(str(out_dir / "01_original.png"), img_bgr)
    cv2.imwrite(str(out_dir / "02_gray.png"), gray)
    cv2.imwrite(str(out_dir / "03_clahe.png"), enh)
    cv2.imwrite(str(out_dir / "04_inverted.png"), inv)
    cv2.imwrite(str(out_dir / "05_otsu.png"), otsu)
    cv2.imwrite(str(out_dir / "06_morph_open.png"), morph)
    cv2.imwrite(str(out_dir / "07_boxes.png"), boxes_img)

    # 7: 逐框 SAM 预测
    sam = load_sam(args.checkpoint)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    box_masks = []
    union = np.zeros((h, w), bool)
    for i, b in enumerate(boxes, 1):
        m = predict_box(sam, img_rgb, b)
        box_masks.append(m)
        union |= m
        crop = img_bgr[max(0, int(b[1])):min(h, int(b[3]) + 1),
                       max(0, int(b[0])):min(w, int(b[2]) + 1)].copy()
        mcrop = m[max(0, int(b[1])):min(h, int(b[3]) + 1),
                  max(0, int(b[0])):min(w, int(b[2]) + 1)]
        cv2.imwrite(str(out_dir / "boxes" / f"box_{i:03d}.png"),
                    overlay_mask(crop, mcrop, color=(0, 255, 0), alpha=0.45))
    union_u8 = (union.astype(np.uint8)) * 255
    final = overlay_mask(img_bgr, union, color=(0, 255, 0), alpha=0.4)
    cv2.imwrite(str(out_dir / "08_sam_union.png"), union_u8)
    cv2.imwrite(str(out_dir / "09_final_overlay.png"), final)

    # GT / 误差图
    gt_path = Path(args.gt) if args.gt else find_gt(img_path)
    gt = None
    metrics = None
    err_img = None
    if gt_path is not None and Path(gt_path).exists():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None and gt.shape[:2] == (h, w):
            gt_bin = gt > 0
            err_img = error_map(img_bgr, union, gt_bin)
            cv2.imwrite(str(out_dir / "10_error_map.png"), err_img)
            metrics = compute_metrics(union_u8, gt)

    # 总览拼图
    panels = [
        ("1 原图", img_bgr),
        ("2 灰度", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("3 CLAHE 增强", cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)),
        ("4 反相 + Otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        ("5 形态学开运算", cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)),
        ("6 连通域框提示", boxes_img),
        ("7 逐框 SAM 并集", cv2.cvtColor(union_u8, cv2.COLOR_GRAY2BGR)),
        ("8 最终叠加", final),
    ]
    if err_img is not None:
        panels.append(("9 误差图 TP绿/FP红/FN蓝", err_img))
    make_montage(panels, str(out_dir / "00_stages_montage.png"), cols=3, cell=512)
    box_montage(img_bgr, boxes, box_masks, str(out_dir / "06b_box_prompts_montage.png"))

    # 摘要
    lines = [
        "# 分阶段可视化摘要",
        f"时间: {ts}",
        f"影像: {img_path}",
        f"尺寸: {w} x {h}",
        f"权重: {args.checkpoint}",
        f"连通域框数: {len(boxes)}",
        f"阴影像素占比: {100.0 * union.mean():.2f}%",
        f"输出目录: {out_dir}",
    ]
    if metrics:
        lines.append(f"GT: {gt_path}")
        for k in ["IoU", "BER", "F1", "OA", "Precision", "Recall"]:
            lines.append(f"{k}: {metrics[k]:.6f}")
    else:
        lines.append("GT: 未找到（跳过误差图与指标）")
    (out_dir / "stages_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"# 框数: {len(boxes)}  阴影占比: {100.0 * union.mean():.2f}%")
    if metrics:
        print(f"# IoU={metrics['IoU']:.4f}  BER={metrics['BER']:.4f}  F1={metrics['F1']:.4f}")
    print(f"[OK] 总览: {out_dir / '00_stages_montage.png'}")
    print(f"[OK] 摘要: {out_dir / 'stages_summary.txt'}")


if __name__ == "__main__":
    main()
