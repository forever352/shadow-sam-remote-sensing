"""任务 B：「同提示零样本」对照 —— 与主方法**只差"是否微调"**的干净单变量对照。

## 为什么要这个

现有零样本行（`results/sam_amg_leakfree_test.csv`）与主方法对比时**一次改了两个因素**：
  ① 提示方式：AMG 自动掩膜 → 框提示
  ② 权重：零样本 → LoRA 微调
→ 0.52 IoU 的差距**无法归因**（第十七轮审查 M2）。

本脚本固定 ①（用**与主方法完全相同**的提示函数 `direction_experiment.method_boxes`），
只改 ②（权重换成**原生 SAM**，**不 apply_lora**）→ **单变量对照**。

**能支撑的断言**（互不重叠，各只需一次比较）：
  - `原生+框`（本脚本） vs `LoRA+框`（主结果）→ 隔离**「是否微调」**
  - `原生+AMG`（已有 sam_amg 行） vs `原生+框`（本脚本）→ 隔离**「提示方式」**

## 四条硬约束（第十九轮审查 M4）
1. 以**字面量**调用划分函数（`frac=0.2` 配 `seed=0`，再从训练池 `frac=0.2` 配 `seed=1`）
   —— 注意：**本段不得出现 `split_by_base(` 字面式示例**，否则 `preflight_check.py` 的
   静态扫描会把文档里的示例也当成真实调用，从而误报 WARN；
2. 落盘前硬断言：测试集 92 张 / 21 原图，**且 name 集合 == `results/lora_leakfree_test_per_image.csv`**；
3. **严禁 `apply_lora`** —— 否则就不是原生 SAM，单变量破坏；
4. 提示只经 `mask_enhanced` / `component_boxes(300)`，**不读 GT 做推理**。

## 标签与表位（审查问题 3）
行名写「**SAM（本项目框提示，零样本/无微调）_leakfree**」；
**不得**与 `sam_amg` 行混同，**不得**写作裸"原始 SAM 无任何处理"（它用了本项目 Otsu 框提示，须在表注写明）。

## 预期（先说死）
我预估**明显低于**主方法。**若意外偏高，照实报**（那是诚实性要求）。
无论高低**只用"数值上高于/低于"**，禁"提升/改进"。

## 用法
    python sam_box_zeroshot_eval.py

## 落盘
    results/sam_box_zeroshot_leakfree_test.csv + _test_per_image.csv
    logs/sam_box_zeroshot_{ts}.log
"""
import csv
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from direction_experiment import method_boxes          # 与主方法**同一个**推理函数
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OFFICIAL = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    RES_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"sam_box_zeroshot_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log("# 任务 B：同提示零样本对照（原生 SAM + 本项目框提示，无微调）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {device}")
    log(f"# 权重: {Path(OFFICIAL).name}（**原生 SAM，未 apply_lora**，单变量对照）")
    log(f"# 提示: method_boxes（mask_enhanced -> component_boxes(300)，框提示并集）"
        f" —— 与主结果**同一函数**")

    # [M4-1] 字面量调用划分（preflight 静态扫描要求）
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log(f"划分: 训练 {len(tr_files)}({n_tr}原图) / 验证 {len(val_files)}({n_va}原图) / 测试 {len(test_files)}({n_te}原图)")

    # [M4-2] 划分硬断言
    if len(test_files) != 92 or n_te != 21:
        raise RuntimeError(f"划分异常: test={len(test_files)}/{n_te}原图（应 92/21），拒绝产出不可比结果")
    _mcsv = RES_DIR / "lora_leakfree_test_per_image.csv"
    if _mcsv.exists():
        with open(_mcsv, newline="", encoding="utf-8-sig") as fh:
            _nm = {r["name"] for r in csv.DictReader(fh) if r.get("name")}
        _nh = {Path(f).name for f in test_files}
        if _nm != _nh:
            raise RuntimeError(f"划分与主结果不一致（{len(_nh)} vs {len(_nm)}），拒绝产出不可比结果")
        log("  [OK] 测试集 name 集合 == 主结果逐图 CSV（92 张）")

    # [M4-3] **不 apply_lora** —— 原生 SAM
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    sam.to(device); sam.eval()
    log("# [OK] 已加载**原生** SAM（未 apply_lora）")

    rows = []
    with torch.no_grad():   # [!] SAM 推理必须 no_grad，否则显存累积 OOM（本项目踩过）
        for f in test_files:
            img_bgr = cv2.imread(f)
            gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # [M4-4] 只用图像出提示，不读 GT
            pred = method_boxes(sam, img_rgb, img_bgr)
            m = compute_metrics((pred * 255).astype(np.uint8), gt)
            m["name"] = Path(f).name
            rows.append(m)

    if len(rows) != 92:
        log(f"！有效样本 {len(rows)} != 92 -> 拒绝写汇总 CSV")
        logf.close(); raise SystemExit(3)

    avg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}

    per_img = RES_DIR / "sam_box_zeroshot_leakfree_test_per_image.csv"
    summ = RES_DIR / "sam_box_zeroshot_leakfree_test.csv"
    with open(per_img, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name"] + METRIC_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_images"] + METRIC_KEYS)
        for tag, d in [("", avg), ("_std", std)]:
            w.writerow([f"SAM(box-prompt,zero-shot/no-finetune)_leakfree{tag}", len(rows)]
                       + [f"{d[k]:.6f}" for k in METRIC_KEYS])

    log("")
    log(f"=== 无泄漏测试集（{len(rows)} 张 / {n_te} 原图，原生 SAM + 框提示）===")
    for k in METRIC_KEYS:
        log(f"{k:10s} {avg[k]:.4f} +/- {std[k]:.4f}")
    log("")
    log("# 对照（同一 92 张、同一提示函数，**只差是否微调**）：")
    log("#   LoRA 微调 SAM（主结果）: IoU 0.787821 / BER 0.064359")
    log("# 限定（强制）：单 seed、21 独立场景、未做显著性检验；只用'数值上高于/低于'")
    log(f"逐图 -> {per_img}")
    log(f"汇总 -> {summ}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
