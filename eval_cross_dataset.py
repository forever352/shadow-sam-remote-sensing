"""T3.2 跨数据集评估：AISD 上训练的 LoRA-SAM 直接迁移到 RSISD（零样本，设计 A）。

## 为什么做这个

申报书要求"验证模型在建筑物阴影检测中的泛化能力"（原文用的是 WHU，属**用途错位**，
详见结题报告的说明）。本项目改用 **RSISD**--SARU 提出的**真正的阴影检测基准**，
其掩膜经核验为二值（像素值仅 {0,255}），与本项目 AISD 流程同构。

**设计 A（零样本迁移）**：**不重训、不微调**，直接拿 AISD 上训好的 LoRA-SAM，
在 RSISD 测试集上评估。这是最严格的泛化证据（也是申报书"泛化能力"的直接回应）。

## 口径

- 提示方式与 AISD 主结果**完全一致**：增强 Otsu（CLAHE+Otsu+形态学）-> 连通域框提示
- 指标六项同 `eval_shadow.compute_metrics`
- **跨域掉点是预期现象**，如实报告；不做任何为"好看"的挑选

## 用法

    python eval_cross_dataset.py --limit 10          # 冒烟
    python eval_cross_dataset.py                     # 全量 test200
    python eval_cross_dataset.py --split val86       # 换验证集
"""
import argparse
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from eval_shadow import compute_metrics
from methods import mask_enhanced
from train_adapter_v4 import apply_lora
from direction_utils import component_boxes

BASE = Path(__file__).resolve().parent
RSISD_DIR = BASE / "datasets" / "RSISD"      # 由用户从 SARU 网盘下载后放入
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
LORA_LEAKFREE = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
BOX_MIN_AREA = 300   # 与 evaluate_lora / AISD 主结果一致

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = ResizeLongestSide(1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test200", "val86", "train1000"], default="test200")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=LORA_LEAKFREE)
    ap.add_argument("--tag", type=str, default=None, help="结果文件名标签（默认用 split）")
    args = ap.parse_args()

    # [M1 修正] 冒烟（--limit）必须与全量**不同 tag**：否则冒烟文件会冒充完整结果
    tag = (args.tag or args.split) + ("_smoke" if args.limit else "")
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"cross_{tag}_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log("# 跨数据集评估（RSISD，零样本迁移，设计 A）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 权重（AISD 上训练，未在 RSISD 上微调）: {args.ckpt}")
    log(f"# 提示方式: 增强 Otsu(CLAHE+Otsu+形态学) -> 连通域框提示（与 AISD 主结果同口径）")

    img_dir = RSISD_DIR / args.split / "shadow"
    gt_dir = RSISD_DIR / args.split / "mask"
    if not img_dir.exists():
        log(f"！未找到 {img_dir}")
        log("  请把 SARU 网盘里的 RSISD 数据集放到 " + str(RSISD_DIR))
        logf.close()
        return

    imgs = sorted(str(p) for p in img_dir.glob("*.png")) + sorted(str(p) for p in img_dir.glob("*.tif"))
    if args.limit:
        imgs = imgs[:args.limit]
    log(f"# 数据集: RSISD/{args.split}  图像数: {len(imgs)}")

    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    sam.load_state_dict(torch.load(args.ckpt, map_location=device))
    sam.to(device); sam.eval()

    rows, allm = [], []
    n_empty = 0
    n_gt_checked = 0   # [M3] GT 二值核验计数（必须落盘）
    with torch.no_grad():
        for ip in imgs:
            name = Path(ip).name
            img_bgr = cv2.imread(ip)
            gt = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt is None:
                log(f"  {name:26s} 读取失败，跳过")
                continue
            # [M3 修正] GT 二值核验必须**由脚本自身落盘**（审查要求："已核验"要有证据）
            _u = np.unique(gt)
            if not np.all(np.isin(_u, [0, 255])):
                raise RuntimeError(f"GT {name} 非二值！unique={_u[:10]} -> 中止（口径不符）")
            n_gt_checked += 1
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            img1024 = transform.apply_image(img_rgb)
            x = sam.preprocess(torch.as_tensor(img1024, dtype=torch.float32)
                               .permute(2, 0, 1)[None].to(device))
            img_emb = sam.image_encoder(x)
            boxes = component_boxes(mask_enhanced(img_bgr), BOX_MIN_AREA)
            comb = np.zeros(gt.shape, bool)
            for b in boxes:
                b1024 = transform.apply_boxes(np.array([b], float), (h, w))
                b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
                sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None)
                low_res, _ = sam.mask_decoder(
                    image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                comb |= (sam.postprocess_masks(low_res, (1024, 1024), (h, w))[0, 0] > 0).cpu().numpy()
            if not boxes:
                n_empty += 1
            m = compute_metrics((comb * 255).astype(np.uint8), gt)
            rows.append({"name": name, **m}); allm.append(m)
            log(f"  {name:26s} IoU={m['IoU']:.4f}  BER={m['BER']:.4f}  F1={m['F1']:.4f}")

    if not allm:
        log("！无有效样本"); logf.close(); return

    # [M2 修正] 硬断言样本数：防止"跑到一半被杀"留下**名为全量、实为半截**的汇总 CSV（AGENTS #8）
    EXPECT_N = {"test200": 200, "val86": 86, "train1000": 1000}
    if not args.limit:
        exp = EXPECT_N[args.split]
        if len(allm) != exp:
            log(f"！样本数 {len(allm)} != 期望 {exp} -> **拒绝写汇总 CSV**（防止半截结果冒充完整）")
            logf.close()
            raise SystemExit(3)
    log(f"# [M3] GT 二值核验：{n_gt_checked}/{len(allm)} 张通过（unique 全部落在 0/255）")
    log(f"# [M2] 样本数校验：{'冒烟模式（跳过）' if args.limit else f'{len(allm)} == {EXPECT_N[args.split]} 通过'}")

    avg = {k: float(np.mean([m[k] for m in allm])) for k in METRIC_KEYS}
    std = {k: float(np.std([m[k] for m in allm])) for k in METRIC_KEYS}

    p = RES_DIR / f"cross_rsisd_{tag}_per_image.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        wt = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        wt.writeheader(); wt.writerows(rows)
    s = RES_DIR / f"cross_rsisd_{tag}.csv"
    mname = f"LoRA-SAM(AISD-trained)_on_RSISD-{tag}"
    with open(s, "w", newline="", encoding="utf-8") as f:
        wt = csv.writer(f)
        wt.writerow(["method", "n_images"] + METRIC_KEYS)
        wt.writerow([mname, len(allm)] + [f"{avg[k]:.6f}" for k in METRIC_KEYS])
        wt.writerow([mname + "_std", len(allm)] + [f"{std[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"=== RSISD/{args.split}（{len(allm)} 张，零样本迁移）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {avg[k]:.4f} ± {std[k]:.4f}")
    log("")
    log(f"# 对照：AISD 无泄漏划分上同为框提示的 IoU = 0.787821")
    log(f"# 跨域下降是**预期现象**，如实报告，不据此做任何挑选或调参")
    log(f"# 无连通域（无候选框）的图像数: {n_empty}")
    log(f"逐图 -> {p}")
    log(f"汇总 -> {s}")
    log(f"运行日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
