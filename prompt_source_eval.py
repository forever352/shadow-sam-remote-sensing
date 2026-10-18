"""提示来源归因实验（仅推理，不改权重）。

目的
----
固定 seed-0 主权重与无泄漏 92 张 test，只替换"提示框来源"，回答：
主方法的性能有多少来自提示质量、多少来自 LoRA 适配？

统一推理路径（保证"只换提示来源"）
----------------------------------
对每个来源：mask -> component_boxes(mask, min_area=300) -> predict_box(...) 的并集。
enhanced 行必须与 results/lora_leakfree_test.csv 在 1e-6 内一致（复现门禁），
否则中止。不使用另一套概率阈值实现。

预注册的 6 组提示来源
--------------------
1. enhanced : mask_enhanced (Otsu+CLAHE+形态学)  -> 现行主方法
2. otsu     : mask_otsu                          -> 传统阈值
3. c3i      : mask_c3i(..., invert_if_mostly_white=False) -> C3I 原始版
4. oracle   : GT 掩膜连通域框                    -> 仅诊断/上界，单独落盘，禁入主表
5. random   : 与 enhanced 逐图同数量、同尺寸分布、位置均匀随机（按文件名确定种子）
6. jitter10 : enhanced 框 x1/x2 各按宽 10%、y1/y2 各按高 10% 独立抖动（按文件名确定种子）

护栏
----
- 不训练、不改权重、不改数据划分、不改指标函数。
- test 只评估一次；不按 test 结果选择来源。
- oracle 只写 prompt_source_oracle_diagnostic.csv，方法名带 (oracle-box)，禁入主表。
- 空框保持空掩膜并记录 n_empty_images；不 fallback、不跳过该图。
- FPR/FNR 复用 ber_sweep_eval.metrics_at 的恒等式，不另写定义。

落盘
----
results/prompt_source_test.csv            主汇总（5 个非 oracle 来源，mean + std）
results/prompt_source_test_per_image.csv  主逐图
results/prompt_source_oracle_diagnostic.csv  oracle 诊断（单独）
results/prompt_source_masks/{source}/*.png 预测二值掩膜（供误差结构分析）
logs/prompt_source_{ts}.log

用法
----
python prompt_source_eval.py --smoke 3   # 冒烟：只跑 3 张，结果加 _smoke 后缀，跳过门禁
python prompt_source_eval.py             # 全量：先 enhanced + 门禁，再跑其余 5 组
"""
import argparse
import csv
import hashlib
import random
import re
import sys
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import cv2
import numpy as np
import torch

from methods import mask_c3i, mask_enhanced, mask_otsu
from prompt_experiment import component_boxes
from direction_experiment import encode_image, predict_box
from train_direction_net import split_by_base
from ber_sweep_eval import load_sam, metrics_at

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT = BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
MASK_DIR = RES_DIR / "prompt_source_masks"

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall", "FPR", "FNR"]
MAIN_SOURCES = ["enhanced", "otsu", "c3i", "random", "jitter10"]
ORACLE_SOURCE = "oracle"
SEED_RANDOM = 20261018
JITTER = 0.10
MIN_AREA = 300
GATE_TOL = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_open(ts):
    LOG_DIR.mkdir(exist_ok=True)
    p = LOG_DIR / f"prompt_source_{ts}.log"
    f = open(p, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()

    return log, f, p


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def names_hash(names):
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def build_test_split():
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    _, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    return test_files, n_te


def boxes_for(source, img_bgr, gt, name):
    h, w = img_bgr.shape[:2]
    if source == "enhanced":
        return component_boxes(mask_enhanced(img_bgr), MIN_AREA)
    if source == "otsu":
        return component_boxes(mask_otsu(img_bgr), MIN_AREA)
    if source == "c3i":
        return component_boxes(mask_c3i(img_bgr, invert_if_mostly_white=False), MIN_AREA)
    if source == ORACLE_SOURCE:
        return component_boxes((gt > 0).astype(np.uint8), MIN_AREA)
    if source == "jitter10":
        base = component_boxes(mask_enhanced(img_bgr), MIN_AREA)
        rng = random.Random(f"{SEED_RANDOM}:jitter10:{name}")
        out = []
        for x1, y1, x2, y2 in base:
            bw, bh = x2 - x1, y2 - y1
            # x1/x2 各按宽 10%、y1/y2 各按高 10% 独立抖动；逐框独立、裁剪、保持框数
            nx1 = int(round(min(max(x1 + rng.uniform(-JITTER, JITTER) * bw, 0), w - 2)))
            nx2 = int(round(min(max(x2 + rng.uniform(-JITTER, JITTER) * bw, nx1 + 2), w - 1)))
            ny1 = int(round(min(max(y1 + rng.uniform(-JITTER, JITTER) * bh, 0), h - 2)))
            ny2 = int(round(min(max(y2 + rng.uniform(-JITTER, JITTER) * bh, ny1 + 2), h - 1)))
            out.append([nx1, ny1, nx2, ny2])
        return out
    if source == "random":
        base = component_boxes(mask_enhanced(img_bgr), MIN_AREA)
        rng = random.Random(f"{SEED_RANDOM}:random:{name}")
        sizes = [(x2 - x1, y2 - y1) for x1, y1, x2, y2 in base]
        rng.shuffle(sizes)
        out = []
        for bw, bh in sizes:
            bw = max(2, min(int(bw), w - 1))
            bh = max(2, min(int(bh), h - 1))
            x1 = rng.randint(0, max(0, w - bw - 1))
            y1 = rng.randint(0, max(0, h - bh - 1))
            out.append([x1, y1, x1 + bw, y1 + bh])
        return out
    raise ValueError(source)


@torch.no_grad()
def predict_source(sam, img_rgb, img_bgr, gt, source, name):
    h, w = img_bgr.shape[:2]
    boxes = boxes_for(source, img_bgr, gt, name)
    if not boxes:
        return np.zeros((h, w), np.uint8), 0
    img_emb, hh, ww = encode_image(sam, img_rgb)
    combined = np.zeros((hh, ww), bool)
    for b in boxes:
        combined |= predict_box(sam, img_emb, hh, ww, b)
    return (combined.astype(np.uint8) * 255), len(boxes)


def read_ref():
    p = RES_DIR / "lora_leakfree_test.csv"
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not r["method"].endswith("_std"):
                return {k: float(r[k]) for k in METRIC_KEYS[:6]}
    raise RuntimeError("no reference row")


def run_sources(sam, sources, test_files, log):
    rows_main, rows_oracle, per_main, per_oracle = [], [], [], []
    for source in sources:
        (MASK_DIR / source).mkdir(parents=True, exist_ok=True)
        src_rows = []
        n_empty = 0
        t0 = time.time()
        for i, ip in enumerate(test_files):
            name = Path(ip).name
            img_bgr = cv2.imread(ip)
            gt = cv2.imread(str(AISD / "Train412" / "mask" / name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt is None:
                log(f"  [跳过] {name}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred, nbox = predict_source(sam, img_rgb, img_bgr, gt, source, name)
            if nbox == 0:
                n_empty += 1
            m = metrics_at(pred.astype(np.float32) / 255.0, gt, 0.5)
            m.update({"name": name, "n_boxes": nbox,
                      "is_oracle": int(source == ORACLE_SOURCE)})
            src_rows.append(m)
            cv2.imwrite(str(MASK_DIR / source / (Path(name).stem + ".png")), pred)
            if (i + 1) % 10 == 0:
                log(f"  {source}: {i + 1}/{len(test_files)}  ({time.time() - t0:.0f}s)")
        if not src_rows:
            log(f"[警告] {source}: 无有效结果")
            continue
        mean = {k: float(np.mean([r[k] for r in src_rows])) for k in METRIC_KEYS}
        std = {k: float(np.std([r[k] for r in src_rows])) for k in METRIC_KEYS}
        label = f"prompt_source_{source}" + ("(oracle-box)" if source == ORACLE_SOURCE else "")
        base = {"method": label, "n_images": len(src_rows),
                "n_boxes_mean": float(np.mean([r["n_boxes"] for r in src_rows])),
                "n_empty_images": n_empty}
        if source == ORACLE_SOURCE:
            rows_oracle.extend([{**base, **mean, "is_oracle": 1},
                                {**base, "method": label + "_std", **std, "is_oracle": 1}])
        else:
            rows_main.extend([{**base, **mean, "is_oracle": 0},
                              {**base, "method": label + "_std", **std, "is_oracle": 0}])
        for r in src_rows:
            r["source"] = source
            if source == ORACLE_SOURCE:
                per_oracle.append(r)
            else:
                per_main.append(r)
        log(f"[完成] {source}: n={len(src_rows)} empty={n_empty} "
            f"IoU={mean['IoU']:.4f} BER={mean['BER']:.4f} "
            f"F1={mean['F1']:.4f} ({time.time() - t0:.0f}s)")
    return rows_main, rows_oracle, per_main, per_oracle


def write_summary(path, rows):
    fields = ["method", "n_images", "n_boxes_mean", "n_empty_images",
              "is_oracle"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_per_image(path, rows):
    fields = ["source", "name", "n_boxes", "is_oracle"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])


def gate(rows_main, log):
    ref = read_ref()
    enh = next((r for r in rows_main if r["method"] == "prompt_source_enhanced"), None)
    if enh is None:
        log("[中止] 缺少 enhanced 结果，无法复现闸门")
        return False
    diffs = {k: abs(enh[k] - ref[k]) for k in METRIC_KEYS[:6]}
    worst = max(diffs, key=diffs.get)
    log(f"[复现闸门] enhanced vs results/lora_leakfree_test.csv: "
        f"max |diff| = {diffs[worst]:.3e} ({worst})，容差 {GATE_TOL:.0e}")
    if any(v > GATE_TOL for v in diffs.values()):
        log("[中止] 复现门禁失败；不采纳本次结果。先在 val/固定子集调试，不得反复看 test")
        return False
    log("[复现闸门] 通过")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 张，结果加 _smoke 后缀")
    ap.add_argument("--gate-only", action="store_true", help="只跑 enhanced 门禁，通过后停止")
    ap.add_argument("--rest-only", action="store_true",
                    help="复用已落盘的 enhanced 结果，只跑其余 5 组（不重复评估 enhanced）")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log, logf, log_path = log_open(ts)
    suffix = f"_smoke{args.smoke}" if args.smoke else ""

    test_files, n_te = build_test_split()
    if args.smoke:
        test_files = test_files[: args.smoke]
    else:
        # 硬断言：防 test 子集/名称漂移
        if len(test_files) != 92 or n_te != 21:
            raise RuntimeError(f"test 数量异常: {len(test_files)}/{n_te} (期望 92/21)")
        ref_p = RES_DIR / "lora_leakfree_test_per_image.csv"
        with open(ref_p, newline="", encoding="utf-8-sig") as rf:
            ref_names = {r["name"] for r in csv.DictReader(rf)}
        if {Path(p).name for p in test_files} != ref_names:
            raise RuntimeError("test 名称集合与 results/lora_leakfree_test_per_image.csv 不一致")
    names = [Path(p).name for p in test_files]
    w_hash = sha256_file(CKPT)[:16]
    n_hash = names_hash(names)[:16]

    log("# 提示来源归因实验（仅推理）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# 命令: {' '.join(sys.argv)}")
    log(f"# 权重: {CKPT}")
    log(f"# 权重 sha256[:16]: {w_hash}")
    log(f"# 测试: {len(test_files)} 张 / {n_te} 原图；名称 sha256[:16]: {n_hash}")
    log(f"# 设备: {DEVICE}")
    log(f"# 统一路径: mask -> component_boxes(min_area={MIN_AREA}) -> predict_box")
    log(f"# 常量: MIN_AREA={MIN_AREA} SEED_RANDOM={SEED_RANDOM} JITTER={JITTER}")
    log("# jitter10: x1/x2 各按宽 10%、y1/y2 各按高 10% 独立抖动；合成宽度约 0.8-1.2 倍原宽")
    log()

    sam = load_sam(str(CKPT))

    if args.rest_only:
        # 复用已落盘的 enhanced 结果；再跑一次门禁（不重新推理）
        p_sum = RES_DIR / "prompt_source_enhanced.csv"
        p_per = RES_DIR / "prompt_source_enhanced_per_image.csv"
        if not (p_sum.exists() and p_per.exists()):
            raise SystemExit("--rest-only 需要先跑 --gate-only 生成 enhanced 结果")
        with open(p_sum, newline="", encoding="utf-8-sig") as fh:
            rows_main = list(csv.DictReader(fh))
        for r in rows_main:
            for k in METRIC_KEYS:
                r[k] = float(r[k])
            r["n_images"] = int(r["n_images"])
            r["n_boxes_mean"] = float(r["n_boxes_mean"])
            r["n_empty_images"] = int(r["n_empty_images"])
            r["is_oracle"] = int(r["is_oracle"])
        with open(p_per, newline="", encoding="utf-8-sig") as fh:
            per_image = list(csv.DictReader(fh))
        for r in per_image:
            for k in METRIC_KEYS:
                r[k] = float(r[k])
            r["n_boxes"] = int(r["n_boxes"])
            r["is_oracle"] = int(r["is_oracle"])
        if not gate(rows_main, log):
            raise SystemExit(2)
    else:
        # 第一阶段：enhanced + 门禁
        rows_main, _, per_image, _ = run_sources(sam, ["enhanced"], test_files, log)
        if not args.smoke and len(per_image) != 92:
            raise RuntimeError(f"enhanced 有效结果数异常: {len(per_image)} (期望 92)")
        write_summary(RES_DIR / f"prompt_source_enhanced{suffix}.csv", rows_main)
        write_per_image(RES_DIR / f"prompt_source_enhanced_per_image{suffix}.csv", per_image)
        if args.smoke:
            log("[冒烟] 跳出门禁")
            logf.close()
            return
        if not gate(rows_main, log):
            # 门禁失败：正式结果名改名为 .gate_failed，避免被误当有效结果
            for fn in (f"prompt_source_enhanced.csv", f"prompt_source_enhanced_per_image.csv"):
                p = RES_DIR / fn
                if p.exists():
                    p.replace(p.with_suffix(p.suffix + ".gate_failed"))
            logf.close()
            raise SystemExit(2)

    if args.gate_only:
        log("[门禁模式] 通过；按指示停止，未运行其余 5 组")
        logf.close()
        return

    # 第二阶段：其余 5 组（含 oracle 单独落盘）
    rest_sources = [s for s in MAIN_SOURCES if s != "enhanced"] + [ORACLE_SOURCE]
    rows_rest, rows_oracle, per_rest, per_oracle = run_sources(sam, rest_sources, test_files, log)
    rows_all = rows_main + rows_rest

    out_main = RES_DIR / "prompt_source_test.csv"
    out_oracle = RES_DIR / "prompt_source_oracle_diagnostic.csv"
    out_oracle_per = RES_DIR / "prompt_source_oracle_per_image.csv"
    out_per = RES_DIR / "prompt_source_test_per_image.csv"
    write_summary(out_main, rows_all)
    write_summary(out_oracle, rows_oracle)
    write_per_image(out_per, per_image + per_rest)
    write_per_image(out_oracle_per, per_oracle)

    log()
    log(f"主汇总 -> {out_main}")
    log(f"oracle 诊断 -> {out_oracle}")
    log(f"oracle 逐图 -> {out_oracle_per}")
    log(f"逐图 -> {out_per}")
    log(f"掩膜 -> {MASK_DIR}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
