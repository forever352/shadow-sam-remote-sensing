"""
评估微调后的 SAM：用"增强 Otsu 自动点提示 + SAM"做阴影检测，在 AISD Test51 上算指标。
可与零样本 SAM 对比，验证微调是否带来提升。

口径提示（2026-10-18）：本脚本是点提示 + AISD 自带 Test51（含 42/43 同源泄漏），
不是主方法的框提示 / LoRA；frozen leakfree 在 Test51 上的输出不得标成
无泄漏 92 张的 0.5621，也不得标成已删除历史 v2/v0 的 0.717。

用法:
    python evaluate_finetuned.py                  # 零样本
    python evaluate_finetuned.py --checkpoint checkpoints/sam_vit_b_shadow_frozen_leakfree.pth   # 冻结编码器对照
    python evaluate_finetuned.py --limit 5        # 先跑前 5 张
"""
import argparse
import glob
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from eval_shadow import compute_metrics
from methods import mask_enhanced

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


def load_sam(ckpt):
    sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(device)
    sam.eval()
    return sam


def auto_points(enhanced_mask, n_bg=10):
    """从增强 Otsu 掩膜生成自动点提示：前景点=连通域质心，背景点=掩膜外随机采样。"""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_mask)
    coords, labs = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        cx, cy = centroids[i]
        coords.append([cx, cy]); labs.append(1)
    yb, xb = np.where(enhanced_mask == 0)
    if len(xb) > 0 and n_bg > 0:
        idx = np.random.choice(len(xb), min(n_bg, len(xb)), replace=False)
        for i in idx:
            coords.append([xb[i], yb[i]]); labs.append(0)
    if not coords:
        return None, None
    return np.array(coords, dtype=float), np.array(labs, dtype=np.int64)


@torch.no_grad()
def predict(sam, img_rgb, points, labels):
    h, w = img_rgb.shape[:2]
    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
    c1024 = transform.apply_coords(points, (h, w))
    c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
    l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
    x = sam.preprocess(img_t)
    img_emb = sam.image_encoder(x)
    sparse, dense = sam.prompt_encoder(points=(c_t, l_t), boxes=None, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return (mask[0, 0] > 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="微调权重路径，缺省=零样本")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ckpt = args.checkpoint or str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
    sam = load_sam(ckpt)

    imgs = sorted(glob.glob(str(AISD / "Test51" / "shadow" / "*.tif")))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"评估 {len(imgs)} 张 | checkpoint = {ckpt}", flush=True)

    allm = []
    for ip in imgs:
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(AISD / "Test51" / "mask" / Path(ip).name), cv2.IMREAD_GRAYSCALE)
        em = mask_enhanced(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        pts, labs = auto_points(em)
        if pts is None:
            allm.append(compute_metrics(np.zeros_like(gt), gt))
            continue
        pred = predict(sam, img, pts, labs)
        allm.append(compute_metrics((pred * 255).astype(np.uint8), gt))

    avg = {k: float(np.mean([m[k] for m in allm])) for k in ["IoU", "BER", "F1", "OA", "Precision", "Recall"]}
    print("=== 平均指标 ===")
    for k, v in avg.items():
        print(f"{k:10s} {v:.4f}")


if __name__ == "__main__":
    main()
