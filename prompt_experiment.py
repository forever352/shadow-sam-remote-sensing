"""
提示优化 + TTA 实验：在微调模型上对比多种推理策略，看能否超过当前 0.69。
策略：
- points          当前基线（增强Otsu质心点 + 背景点）
- boxes           连通域包围框提示
- points_tta_inter 点提示 + 水平翻转 TTA（取交集）
- points_tta_union 点提示 + 水平翻转 TTA（取并集）
- iterative       点提示 -> 用预测结果的质心再精修一次
"""
import glob
import argparse
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from eval_shadow import compute_metrics
from methods import mask_enhanced
from evaluate_finetuned import auto_points, predict

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)
FT_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v2.pth")


def load_sam():
    sam = sam_model_registry["vit_b"](checkpoint=FT_CKPT).to(device)
    sam.eval()
    return sam


@torch.no_grad()
def predict_box(sam, img_rgb, box):
    h, w = img_rgb.shape[:2]
    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
    b1024 = transform.apply_boxes(np.array([box], float), (h, w))
    b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
    x = sam.preprocess(img_t)
    img_emb = sam.image_encoder(x)
    sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return (mask[0, 0] > 0).cpu().numpy()


def component_boxes(mask, min_area=300):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, wb, hb = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        boxes.append([x, y, x + wb, y + hb])
    return boxes


def method_points(sam, img_rgb, img_bgr):
    em = mask_enhanced(img_bgr)
    pts, labs = auto_points(em)
    if pts is None:
        return np.zeros(img_bgr.shape[:2], bool)
    return predict(sam, img_rgb, pts, labs)


def method_boxes(sam, img_rgb, img_bgr):
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        return np.zeros(img_bgr.shape[:2], bool)
    combined = np.zeros(img_bgr.shape[:2], bool)
    for b in boxes:
        combined |= predict_box(sam, img_rgb, b)
    return combined


def method_points_tta(sam, img_rgb, img_bgr, mode="inter"):
    h, w = img_rgb.shape[:2]
    em = mask_enhanced(img_bgr)
    pts, labs = auto_points(em)
    if pts is None:
        return np.zeros((h, w), bool)
    pred1 = predict(sam, img_rgb, pts, labs)
    img_flip = img_rgb[:, ::-1, :].copy()
    pts_flip = pts.copy()
    pts_flip[:, 0] = w - 1 - pts[:, 0]
    pred2 = predict(sam, img_flip, pts_flip, labs)[:, ::-1]
    return (pred1 & pred2) if mode == "inter" else (pred1 | pred2)


def method_iterative(sam, img_rgb, img_bgr):
    em = mask_enhanced(img_bgr)
    pts, labs = auto_points(em)
    if pts is None:
        return np.zeros(img_bgr.shape[:2], bool)
    pred1 = predict(sam, img_rgb, pts, labs)
    n, _, stats, centroids = cv2.connectedComponentsWithStats((pred1 * 255).astype(np.uint8))
    coords, lab2 = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        coords.append([centroids[i][0], centroids[i][1]])
        lab2.append(1)
    if not coords:
        return pred1
    return predict(sam, img_rgb, np.array(coords, float), np.array(lab2))


METHODS = {
    "points": method_points,
    "boxes": method_boxes,
    "points_tta_inter": lambda s, i, b: method_points_tta(s, i, b, "inter"),
    "points_tta_union": lambda s, i, b: method_points_tta(s, i, b, "union"),
    "iterative": method_iterative,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0=全部51张")
    args = parser.parse_args()

    sam = load_sam()
    imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
    if args.limit:
        imgs = imgs[: args.limit]
    gt_dir = AISD / "Test51" / "mask"

    acc = {m: [] for m in METHODS}
    for ip in imgs:
        img_bgr = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_dir / Path(ip).name), cv2.IMREAD_GRAYSCALE)
        for mname, fn in METHODS.items():
            pred = fn(sam, img_rgb, img_bgr)
            acc[mname].append(compute_metrics((pred * 255).astype(np.uint8), gt)["IoU"])

    print(f"=== 各策略平均 IoU（{len(imgs)} 张）===")
    for m in METHODS:
        print(f"{m:18s} {np.mean(acc[m]):.4f}")


if __name__ == "__main__":
    main()
