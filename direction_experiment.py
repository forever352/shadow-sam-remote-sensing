"""
方向提示消融实验：验证"方向先验"是否真正有效。

核心问题：项目标题是"方向提示"，但方向信息目前只做了后处理过滤（反而掉分）。
本实验把"方向"真正注入 SAM 的提示，用消融对比证明其价值。

三种提示方式（都在微调 SAM 上）：
- centroid_points    质心点提示（现有基线：每块 1 个质心点 + 背景点）
- axis_box           轴对齐框提示（现有基线，IoU 0.717 的来源）
- direction_corners  方向感知提示（新）：minAreaRect 求旋转包围框，
                     取 4 个旋转角点作点提示，让提示携带阴影朝向

对比逻辑：axis_box 是"2 个轴对齐角点"，direction_corners 是"4 个旋转角点"，
二者唯一区别就是"方向"。若 direction_corners 更优，即证明方向先验有效。

用法:
    python direction_experiment.py --limit 5   # 先跑 5 张快速验证
    python direction_experiment.py             # 全量 51 张

口径提示（2026-10-18）：本脚本的 load_sam() 来自 prompt_experiment，现加载
frozen leakfree（冻结编码器 + 无泄漏对照），不是已删除历史 v2；评估集为
AISD 自带 Test51（含 42/43 同源泄漏），输出不得标成无泄漏 0.5621/历史 0.717。
"""
import argparse
import glob
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import numpy as np
import cv2

from segment_anything.utils.transforms import ResizeLongestSide

from eval_shadow import compute_metrics
from methods import mask_enhanced
from evaluate_finetuned import auto_points
from prompt_experiment import component_boxes, load_sam

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)

# ---- 复用图像编码：图像只进一次编码器，多个提示共用同一个 embedding ----
@torch.no_grad()
def encode_image(sam, img_rgb):
    h, w = img_rgb.shape[:2]
    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
    x = sam.preprocess(img_t)
    return sam.image_encoder(x), h, w


@torch.no_grad()
def predict_points(sam, img_emb, h, w, points, labels):
    c1024 = transform.apply_coords(points, (h, w))
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
def predict_box(sam, img_emb, h, w, box):
    b1024 = transform.apply_boxes(np.array([box], float), (h, w))
    b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
    sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return (mask[0, 0] > 0).cpu().numpy()


def component_direction_corners(mask, min_area=100):
    """返回每个连通域最小外接旋转矩形的 4 个顶点（list of (4,2) float arrays）。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 5:
            continue
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        boxes.append(cv2.boxPoints(rect))  # 4 个顶点，顺时针，编码旋转朝向
    return boxes


def method_points(sam, img_rgb, img_bgr):
    """质心点提示（现有基线）：所有连通域质心 + 背景点，一次性传入。"""
    em = mask_enhanced(img_bgr)
    pts, labs = auto_points(em)
    if pts is None:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    return predict_points(sam, img_emb, h, w, pts, labs)


def method_boxes(sam, img_rgb, img_bgr):
    """轴对齐框提示（现有基线，0.717 的来源）。"""
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em)
    if not boxes:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for b in boxes:
        combined |= predict_box(sam, img_emb, h, w, b)
    return combined


def method_direction(sam, img_rgb, img_bgr):
    """方向感知提示：每块取旋转框 4 顶点作前景点，逐块预测后合并。"""
    em = mask_enhanced(img_bgr)
    boxes = component_direction_corners(em)
    if not boxes:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for box in boxes:
        labs = np.ones(len(box), dtype=np.int64)  # 4 个点全为前景
        combined |= predict_points(sam, img_emb, h, w, box, labs)
    return combined


def component_direction_axis(mask, min_area=100, n_axis=5):
    """对每个连通域，沿主轴方向在连通域内采样 n_axis 个像素点。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    pts_list = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 5:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            angle += 90  # 保证 angle 指向长边（主轴）
        theta = np.deg2rad(angle)
        axis = np.array([np.cos(theta), np.sin(theta)])
        half = max(w, h) / 2
        samples = []
        for t in np.linspace(-half, half, n_axis):
            px, py = int(round(cx + axis[0] * t)), int(round(cy + axis[1] * t))
            if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and labels[py, px] == i:
                samples.append([px, py])
        if len(samples) >= 3:
            pts_list.append(np.array(samples, float))
    return pts_list


def method_direction_axis(sam, img_rgb, img_bgr):
    """方向采样点提示：沿主轴方向撒一串点（比 4 顶点更密集地表达方向）。"""
    em = mask_enhanced(img_bgr)
    pts_list = component_direction_axis(em)
    if not pts_list:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for pts in pts_list:
        labs = np.ones(len(pts), dtype=np.int64)
        combined |= predict_points(sam, img_emb, h, w, pts, labs)
    return combined


@torch.no_grad()
def predict_box_points(sam, img_emb, h, w, box, points, labels):
    """框 + 额外方向点联合提示。"""
    b1024 = transform.apply_boxes(np.array([box], float), (h, w))
    b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
    c1024 = transform.apply_coords(points, (h, w))
    c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
    l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
    sparse, dense = sam.prompt_encoder(points=(c_t, l_t), boxes=b_t, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return (mask[0, 0] > 0).cpu().numpy()


def component_box_and_direction(mask, min_area=100, n_axis=3):
    """返回每个连通域的 (轴对齐框, 主轴方向采样点) 对，保证一一对应。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    items = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, wb, hb = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        box = [x, y, x + wb, y + hb]
        ys, xs = np.where(labels == i)
        dir_pts = None
        if len(xs) >= 5:
            (cx, cy), (w_, h_), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            if w_ < h_:
                angle += 90
            theta = np.deg2rad(angle)
            axis = np.array([np.cos(theta), np.sin(theta)])
            half = max(w_, h_) / 2
            samples = []
            for t in np.linspace(-half, half, n_axis):
                px, py = int(round(cx + axis[0] * t)), int(round(cy + axis[1] * t))
                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and labels[py, px] == i:
                    samples.append([px, py])
            if samples:
                dir_pts = np.array(samples, float)
        items.append((box, dir_pts))
    return items


def method_box_plus_direction(sam, img_rgb, img_bgr):
    """框 + 方向点联合提示：框负责定位，方向点负责朝向。"""
    em = mask_enhanced(img_bgr)
    items = component_box_and_direction(em)
    if not items:
        return np.zeros(img_bgr.shape[:2], bool)
    img_emb, h, w = encode_image(sam, img_rgb)
    combined = np.zeros((h, w), bool)
    for box, dir_pts in items:
        if dir_pts is None:
            combined |= predict_box(sam, img_emb, h, w, box)
        else:
            labs = np.ones(len(dir_pts), dtype=np.int64)
            combined |= predict_box_points(sam, img_emb, h, w, box, dir_pts, labs)
    return combined


METHODS = {
    "centroid_points": method_points,
    "axis_box": method_boxes,
    "direction_corners": method_direction,
    "direction_axis": method_direction_axis,
    "box+direction": method_box_plus_direction,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0=全部 51 张")
    args = parser.parse_args()

    np.random.seed(0)  # 固定随机种子，保证背景点采样可复现

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

    print(f"=== 各提示方式平均 IoU（{len(imgs)} 张）===")
    for m in METHODS:
        print(f"{m:20s} {np.mean(acc[m]):.4f}", flush=True)


if __name__ == "__main__":
    main()
