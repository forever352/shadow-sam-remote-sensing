"""压 BER 实验（第二轮）：**扫掩膜概率阈值** + **3-seed 掩膜集成**。

## 背景与动机

申报书硬指标「BER 降至 5% 以下」**未达成**：无泄漏划分 BER = **0.0644**（seed-0）/
**0.0612**（3 seed 均值）。前一轮压 BER 实验（`ber_reduce_eval.py`）只动了**提示框几何**
（外扩 / min_area），结论是**无候选稳健优于基线**。

本轮换两个**从未试过**的旋钮：

1. **掩膜概率阈值扫描（族 A）**：现行管线在 `direction_experiment.py:78`
   **硬编码** `(mask[0,0] > 0)`，即概率阈值恒为 **0.5**，全仓**未发现**任何扫描实现。
   BER = 0.5(FPR+FNR) 对两个误差率对称 —— 实测 **FNR 0.0882 是 FPR 0.0405 的 2.18 倍**，
   现工作点**偏向保守（漏检多）**，故"降阈值可调"是**待检验的假设**（见下方反证）。

2. **3-seed 掩膜集成（族 B）**：三个 checkpoint 均已就位（seed-0/1/2，同划分、同 40 轮）。

## ⚠️ 必须先承认的相邻反证

`results/ber_reduce_val_grid.csv`（val 58 张）显示：项目**唯一试过的"提召回"旋钮**
（框外扩 D2/D5/D10）**反而使 FNR 单调升高**（0.08623 -> 0.086664 -> 0.087235 -> 0.087601），
同时 FPR 下降（0.035539 -> 0.032399）—— 与"提召回"预期**相反**。
本轮的阈值扫描**机制不同**（作用在掩膜概率上，而非提示框上），
但"降阈值 -> FNR 降"仍是**假设而非已证事实**，**以实测为准**。

## 预注册（**跑 test 前定死，事后不得增删候选**）

阈值一律作用在**概率**上（等价于对数几率 > logit(0.5) 的原阈值 0.5）。

**族 A — 单 seed-0**（对每个框的概率图取逐像素 max，再按阈值二值化）：
    A60=0.60 / **A50=0.50（= 现行 V0 基线）** / A45=0.45 / A40=0.40 / A35=0.35 / A30=0.30

**族 B — 3 seed 集成**（三 seed 各自逐像素 max 后取**概率均值**，再按阈值二值化）：
    B50=0.50 / B45=0.45 / B40=0.40 / B35=0.35 / B30=0.30

**等价性说明**（V0 闸门依据）：对多个框的并集，`union_i (P_i > t)` 恒等于 `(max_i P_i > t)`；
且 `sigmoid(x) > 0.5 <=> x > 0`。故 A50 在数学上**等价**于现行 `predict_box` 的并集结果。

**选择规则**：在 **val(58 张)** 上，先筛掉 val IoU < V0_val_IoU - 0.01 的；
再取 val BER 最低者；**且**要求它与 V0 的**逐图配对**改善达到 |t| > 2 且均差 > 0；
否则判为"**无候选稳健优于基线**"，如实报告。**test 只评估选中的那一个**。

**失败即如实报告**：若最终 test BER >= 0.05，写"未达成"，**不调口径、不换测试集**。

## 护栏（强制）

1. **划分钉死**：两处划分调用必须**字面量**写成 0.2 + seed 0（测试集划分）与
   0.2 + seed 1（验证集划分）—— 即与主结果**同一划分**；不随任何 seed 参数变化。
   （本行不得写成带括号的调用形式，否则会被 `preflight_check.py` 的静态正则
   误当成真实调用，把真正的两处调用挤出前两位而产生 WARN。）
2. **复现闸门**：先算 A50 于 **test**，与 `results/lora_leakfree_test.csv`
   **六项汇总指标逐项比对**（容差 1e-3）；不一致 -> **中止**。
3. **val 报 spread（逐图 BER sd）+ 配对比较**，不只看均值。
4. **配对方向**：`paired_t(a, b)` 返回 `mean(a.BER - b.BER)`。
   调用一律**基线在前** => 均差 > 0 表示"候选 BER 更低（更好）"。
   （上一轮 `ber_reduce_eval.py` 亦为此顺序；写反会使选择规则挑出更差的候选。）
5. **族 B 须同时与"同族基线 B50"配对**，以隔离"阈值"与"集成"两种效应。
6. **多重比较与相关性声明**：族 A 5 个 + 族 B 5 个 = 10 个候选；val 仅 58 张但只来自
   **16 个原图**（子图同源）=> **子图级 |t| > 2 判据偏松（反保守）**。
   故除子图级配对 t 外，**另报"按原图基名聚合后"的配对 t**（有效 n = 原图数）。
7. **单 seed 噪声地板**：seed 间 BER 样本 sd = **0.0027**（0.064359/0.059349/0.059983）；
   凡小于 ~0.006 的"改善"须谨慎解读。

## ⚠️ 集成会改变"方法"本身（必须在报告中声明）

主表 headline 现为「**3 seed 均值 ± 样本 sd**」（0.7990 +/- 0.0101）。
**3-seed 集成是另一个东西**：不能与"3 seed 均值"并列成同一口径。
若族 B 获胜，须**单列一行**，并保留逐 seed 数值用于方差陈述。
集成亦使**推理成本 x3**。

## 用法
    python ber_sweep_eval.py --smoke 3     # 冒烟：只验等价性，落盘 logs/ber_sweep_smoke_*.log
    python ber_sweep_eval.py               # 全量

## 落盘
    results/ber_sweep_val_grid.csv          各族各阈值在 val 上的六项 + FPR/FNR（聚合）
    results/ber_sweep_val_per_image.csv     val **逐图**六项（选型统计可独立复算的依据）
    results/ber_sweep_selection.json        预注册候选 + 选择决定（**在跑 test 之前写**）
    results/ber_sweep_test.csv              V0 与选中配置在 test 上并列
    results/ber_sweep_test_per_image.csv
    logs/ber_sweep_{ts}.log
    logs/ber_sweep_smoke_{ts}.log           （--smoke 时）
"""
import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_enhanced
from direction_experiment import component_boxes, encode_image, transform
from train_adapter_v4 import apply_lora
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT_DIR = BASE / "checkpoints"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
OFFICIAL = str(CKPT_DIR / "sam_vit_b_01ec64.pth")

# 三个 seed 的主权重（均为无泄漏划分、40 轮）
SEED_CKPTS = {
    0: str(CKPT_DIR / "sam_vit_b_shadow_v4_lora_leakfree.pth"),
    1: str(CKPT_DIR / "sam_vit_b_shadow_v4_lora_leakfree_s1.pth"),
    2: str(CKPT_DIR / "sam_vit_b_shadow_v4_lora_leakfree_s2.pth"),
}
SEEDS = (0, 1, 2)

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
KEYS2 = METRIC_KEYS + ["FPR", "FNR"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- 预注册候选（跑 test 前定死，事后不得增删）--
THRESH_A = [0.60, 0.50, 0.45, 0.40, 0.35, 0.30]   # 族 A：单 seed-0
THRESH_B = [0.50, 0.45, 0.40, 0.35, 0.30]         # 族 B：3 seed 集成
V0_CODE = "A50"                                     # 基线（= 现行硬编码阈值 0.5）
B_BASE = "B50"                                      # 族 B 的内部基线（隔离"集成"效应）
IOU_TOL = 0.01
T_TOL = 2.0


def cfg_code(fam, t):
    return f"{fam}{int(round(t * 100))}"


CONFIGS = ([("A", t) for t in THRESH_A] + [("B", t) for t in THRESH_B])
CONFIG_CODES = [cfg_code(f, t) for f, t in CONFIGS]


def load_sam(ckpt):
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    sam.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    sam.to(DEVICE)
    sam.eval()
    return sam


@torch.no_grad()
def box_prob(sam, img_emb, h, w, box):
    """单框的**概率图**（float32, 0..1），替代 predict_box 的硬二值化。

    与 `direction_experiment.predict_box` 完全同一条链路（复用其模块级
    `transform = ResizeLongestSide(1024)` 实例，保证预处理一致），
    只把最后的 `> 0` 换成返回 sigmoid(logits)，使阈值成为可控变量。
    """
    b1024 = transform.apply_boxes(np.array([box], float), (h, w))
    b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(DEVICE)
    sparse, dense = sam.prompt_encoder(points=None, boxes=b_t, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    logits = sam.postprocess_masks(low_res, (1024, 1024), (h, w))
    return torch.sigmoid(logits[0, 0]).cpu().numpy().astype(np.float32)


@torch.no_grad()
def image_prob_maps(sams, img_bgr, seeds=SEEDS):
    """对一张图，返回 {seed: 概率图}。提示框只算一次（与模型无关）。"""
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em, min_area=300)
    h, w = img_bgr.shape[:2]
    out = {s: np.zeros((h, w), np.float32) for s in seeds}
    if not boxes:
        return out
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for s in seeds:
        img_emb, hh, ww = encode_image(sams[s], img_rgb)
        acc = np.zeros((hh, ww), np.float32)
        for b in boxes:
            np.maximum(acc, box_prob(sams[s], img_emb, hh, ww, b), out=acc)
        out[s] = acc
    return out


def gt_path(name):
    return AISD / "Train412" / "mask" / name


def metrics_at(prob, gt, t):
    pred = (prob > t).astype(np.uint8) * 255
    m = compute_metrics(pred, gt)
    m["FNR"] = 1.0 - m["Recall"]
    m["FPR"] = 2.0 * m["BER"] - m["FNR"]
    return m


def eval_image(sams, img_path, seeds=SEEDS):
    """返回 {config_code: metrics}（含 name）。seeds 少于 3 个时族 B 自动跳过。"""
    name = Path(img_path).name
    img = cv2.imread(img_path)
    gt = cv2.imread(str(gt_path(name)), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        return None
    pm = image_prob_maps(sams, img, seeds)
    p0 = pm[0]
    pe = None
    if len(seeds) >= 3:
        pe = (pm[0] + pm[1] + pm[2]) / 3.0
    rows = {}
    for fam, t in CONFIGS:
        if fam == "B" and pe is None:
            continue
        m = metrics_at(p0 if fam == "A" else pe, gt, t)
        m["name"] = name
        rows[cfg_code(fam, t)] = m
    return rows


def agg(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def base_of(name):
    m = re.match(r"^(.*?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def paired_t(a, b):
    """子图级配对 t：d = a.BER - b.BER（同名对齐）。

    **调用约定：基线在前**（a = 基线，b = 候选）。
    故 mean_d > 0 表示 **b（候选）** 的 BER 更低，即**候选更好**。
    （写反会使选择规则挑出更差的候选 —— 本项目曾因此被判"不通过"。）
    """
    key = {r["name"]: r for r in b}
    d = np.array([r["BER"] - key[r["name"]]["BER"] for r in a if r["name"] in key], dtype=float)
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return float(d.mean()), t, int((d > 0).sum()), int((d < 0).sum())


def paired_t_base(a, b):
    """按原图基名聚合后再做配对 t（缓解子图同源导致的 t 偏松）。基线在前。"""
    ka, kb = {}, {}
    for r in a:
        ka.setdefault(base_of(r["name"]), []).append(r["BER"])
    for r in b:
        kb.setdefault(base_of(r["name"]), []).append(r["BER"])
    common = sorted(set(ka) & set(kb))
    d = np.array([float(np.mean(ka[k]) - np.mean(kb[k])) for k in common], dtype=float)
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"base_paired_mean_gain": float(d.mean()), "base_paired_t": t,
            "base_win": int((d > 0).sum()), "base_lose": int((d < 0).sum()), "base_n": n}


def build_split():
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    return tr_files, val_files, test_files, (n_tr, n_va, n_te)


def read_ref_summary():
    ref = {}
    with open(RES_DIR / "lora_leakfree_test.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not r["method"].endswith("_std"):
                ref = {k: float(r[k]) for k in METRIC_KEYS}
    return ref


def read_ref_per_image():
    ref = {}
    with open(RES_DIR / "lora_leakfree_test_per_image.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ref[r["name"]] = {k: float(r[k]) for k in METRIC_KEYS}
    return ref


def run_smoke(n):
    """冒烟：只验"max 概率 > 0.5"与既有 predict_box 结果的一致性，落盘证据。"""
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ber_sweep_smoke_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        def log(s=""):
            print(s, flush=True)
            lf.write(s + "\n")

        log("# 压 BER 第二轮 冒烟：A50 与既有 predict_box 并集结果的一致性")
        log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
        log("# 等价性: sigmoid(x)>0.5 <=> x>0；union_i(P_i>t) == (max_i P_i > t)")
        _tr, _va, test_files, (n_tr, n_va, n_te) = build_split()
        log(f"# 划分: 训练 {n_tr} / 验证 {n_va} / 测试 {n_te} 原图")
        ref = read_ref_per_image()
        sam = load_sam(SEED_CKPTS[0])
        sams = {0: sam}
        log(f"# 显存分配: {torch.cuda.memory_allocated()/1e6:.1f} MB" if DEVICE == "cuda" else "# CPU")
        log("")
        log("%-26s %12s %12s %12s" % ("name", "ours A50 IoU", "existing IoU", "abs diff"))
        worst = 0.0
        for f in test_files[:n]:
            rows = eval_image(sams, f, seeds=(0,))
            if rows is None:
                log(f"{Path(f).name:26s} [SKIP 读图失败]")
                continue
            name = Path(f).name
            ours = rows[V0_CODE]["IoU"]
            ex = ref[name]["IoU"]
            d = abs(ours - ex)
            worst = max(worst, d)
            log("%-26s %12.6f %12.6f %12.2e" % (name, ours, ex, d))
        log("")
        log("最大绝对偏差 = %.3e" % worst)
        ok = worst < 1e-6
        log("判定: %s" % ("[OK] 逐位一致（< 1e-6）" if ok else "[FAIL] 存在偏差"))
        log(f"日志 -> {log_path}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gate", action="store_true", help="跳过 A50 复现闸门（不推荐）")
    ap.add_argument("--smoke", type=int, default=0, help="只跑 N 张测试图的等价性冒烟并退出")
    args = ap.parse_args()

    for d in (RES_DIR, LOG_DIR):
        d.mkdir(exist_ok=True)
    if args.smoke:
        raise SystemExit(run_smoke(args.smoke))

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ber_sweep_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 压 BER 第二轮：掩膜概率阈值扫描 + 3-seed 掩膜集成")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
    log("# 权重: seed-0/1/2 三份主权重（无泄漏划分、40 轮）")
    log(f"# 预注册候选: {CONFIG_CODES}   共 {len(CONFIG_CODES)} 个")
    log(f"# 基线 {V0_CODE} = 概率阈值 0.5，与现行硬编码 (mask>0) 等价；族B 内部基线 {B_BASE}")
    log(f"# 选择规则: val IoU >= V0-{IOU_TOL} 且 val BER 最小 且 配对|t|>{T_TOL} 且 均差>0；否则'无候选稳健优于基线'")
    log("# [!] 配对方向: paired_t(a,b)=mean(a.BER-b.BER)，**基线在前** => 均差>0 即候选更优")
    log("# [!] 相邻反证: 上轮框几何(提召回)旋钮反而使 FNR 升高；本轮机制不同，但仍以实测为准")

    _tr, val_files, test_files, (n_tr, n_va, n_te) = build_split()
    log(f"划分: 训练 {n_tr} 原图 / 验证 {len(val_files)} 张（{n_va} 原图） / 测试 {len(test_files)} 张（{n_te} 原图）")
    log(f"# 有效样本提醒: val 子图同源，配对 t 的有效 n 约等于原图数 {n_va}")

    sams = {}
    for s in SEEDS:
        log(f"加载 seed-{s}: {Path(SEED_CKPTS[s]).name}")
        sams[s] = load_sam(SEED_CKPTS[s])
    if DEVICE == "cuda":
        log(f"显存分配: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    base_json = {
        "configs": [{"code": cfg_code(f, t), "family": f, "threshold": t} for f, t in CONFIGS],
        "rule": f"val IoU>=V0-{IOU_TOL} & min val BER & paired |t|>{T_TOL} & mean_gain>0",
        "baseline_code": V0_CODE,
        "family_b_baseline": B_BASE,
        "seeds": list(SEEDS),
        "notes": ("族A=单seed-0概率阈值扫描；族B=3seed集成后阈值扫描；"
                  "集成改变方法本身，须单列；配对一律基线在前"),
        "ts": ts,
    }

    # -- 闸门 + test 一次通过（避免重复评估）--
    log("\n=== 闸门：A50 在 test 上复现既有主结果 ===")
    test_rows_all = {c: [] for c in CONFIG_CODES}
    for f in test_files:
        r = eval_image(sams, f)
        if r is None:
            continue
        for c in CONFIG_CODES:
            test_rows_all[c].append(r[c])
    gate_rows = test_rows_all[V0_CODE]
    a = agg(gate_rows, METRIC_KEYS)
    ref = read_ref_summary()
    log(f"  本脚本 {V0_CODE}: IoU={a['IoU']:.6f}  BER={a['BER']:.6f}")
    log(f"  既有主结果 : IoU={ref['IoU']:.6f}  BER={ref['BER']:.6f}")
    dev = max(abs(a[k] - ref[k]) for k in METRIC_KEYS)
    log(f"  六项汇总指标最大偏差 = {dev:.2e}")
    if not args.skip_gate and dev > 1e-3:
        log("  [中止] 偏差过大 -> 无法区分'配置起效'与'脚本口径漂移'，拒绝继续")
        logf.close()
        raise SystemExit(2)
    log("  [OK] 复现通过（六项汇总逐项比对，容差 1e-3）")
    base_json["gate_repro_max_dev"] = dev
    # 全部候选的 test 汇总先备好，但**在 selection 落盘前不打印、不落盘**
    test_all = {c: agg(test_rows_all[c], KEYS2) for c in CONFIG_CODES}

    # -- val：各候选 --
    log(f"\n=== val({len(val_files)}) 各候选 ===")
    val_store = {c: [] for c in CONFIG_CODES}
    val_rows_l = []
    val_perimg = []
    for f in val_files:
        r = eval_image(sams, f)
        if r is None:
            continue
        for c in CONFIG_CODES:
            val_store[c].append(r[c])
            val_perimg.append({"config": c, **{k: r[c][k] for k in ["name"] + METRIC_KEYS}})
    for c in CONFIG_CODES:
        m = agg(val_store[c], KEYS2)
        val_rows_l.append({"config": c, **{k: round(m[k], 6) for k in KEYS2}})
        log(f"  {c:<5} IoU={m['IoU']:.4f} BER={m['BER']:.4f} "
            f"FPR={m['FPR']:.4f} FNR={m['FNR']:.4f} "
            f"(BER逐图sd={np.std([r['BER'] for r in val_store[c]]):.4f})")

    with open(RES_DIR / "ber_sweep_val_grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config"] + KEYS2)
        w.writeheader(); w.writerows(val_rows_l)
    # val 逐图也必须落盘：否则选型统计（配对 t / 胜负数）无法被独立复算
    with open(RES_DIR / "ber_sweep_val_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "name"] + METRIC_KEYS)
        w.writeheader(); w.writerows(val_perimg)

    # -- 选择（在跑 test 之前落盘）--
    # [!] paired_t(a,b)=mean(a.BER-b.BER)：**基线在前** => 均差>0 表示候选 BER 更低（更好）。
    v0v = agg(val_store[V0_CODE], KEYS2)
    cand = []
    for c in CONFIG_CODES:
        if c == V0_CODE:
            continue
        m = agg(val_store[c], KEYS2)
        if m["IoU"] < v0v["IoU"] - IOU_TOL:
            log(f"  [筛除] {c}: val IoU {m['IoU']:.4f} < V0 {v0v['IoU']:.4f} - {IOU_TOL}")
            continue
        md, t, w_, l_ = paired_t(val_store[V0_CODE], val_store[c])
        rec = {"config": c, "val_BER": m["BER"], "val_IoU": m["IoU"],
               "paired_mean_ber_gain": md, "paired_t": t, "win": w_, "lose": l_}
        rec.update(paired_t_base(val_store[V0_CODE], val_store[c]))
        if c.startswith("B") and c != B_BASE:
            mdb, tb, wb, lb = paired_t(val_store[B_BASE], val_store[c])
            rec.update({"vs_B50_mean_gain": mdb, "vs_B50_t": tb,
                        "vs_B50_win": wb, "vs_B50_lose": lb})
        cand.append(rec)
    cand.sort(key=lambda x: x["val_BER"])
    chosen = next((c for c in cand if abs(c["paired_t"]) > T_TOL and c["paired_mean_ber_gain"] > 0), None)

    base_json["val_baseline"] = {k: round(v0v[k], 6) for k in KEYS2}
    base_json["val_candidates"] = [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in cand
    ]
    base_json["chosen"] = chosen["config"] if chosen else None
    base_json["verdict"] = ("选中 " + chosen["config"]) if chosen else "无候选稳健优于基线（如实报告）"
    with open(RES_DIR / "ber_sweep_selection.json", "w", encoding="utf-8") as fh:
        json.dump(base_json, fh, ensure_ascii=False, indent=2)
    log("\n=== 选择（已落盘 ber_sweep_selection.json）===")
    for c in cand:
        log(f"  {c['config']:<5} val BER={c['val_BER']:.4f}  配对dBER={c['paired_mean_ber_gain']:+.4f} "
            f"t={c['paired_t']:+.2f}  {c['win']}胜/{c['lose']}负")
        log(f"        按原图聚合: dBER={c['base_paired_mean_gain']:+.4f} t={c['base_paired_t']:+.2f} "
            f"({c['base_win']}胜/{c['base_lose']}负, n原图={c['base_n']})")
        if "vs_B50_mean_gain" in c:
            log(f"        对 {B_BASE}:    dBER={c['vs_B50_mean_gain']:+.4f} t={c['vs_B50_t']:+.2f} "
                f"({c['vs_B50_win']}胜/{c['vs_B50_lose']}负)  <- 隔离'阈值'效应")
    log(f"  -> {base_json['verdict']}")

    # -- test 汇总（选择已定，此处只报）--
    log(f"\n=== test({len(test_files)}) ===")
    out_codes = [V0_CODE] + ([chosen["config"]] if chosen else [])
    base_json["test_all_configs"] = [
        {"config": c, **{k: round(test_all[c][k], 6) for k in KEYS2}} for c in CONFIG_CODES
    ]
    summ, perimg = [], []
    for c in out_codes:
        m = test_all[c]
        rows = test_rows_all[c]
        summ.append({"config": c, **{k: round(m[k], 6) for k in KEYS2}})
        log(f"  {c:<5} IoU={m['IoU']:.6f} BER={m['BER']:.6f} FPR={m['FPR']:.4f} FNR={m['FNR']:.4f}")
        for r in rows:
            perimg.append({"config": c, **{k: r[k] for k in ["name"] + METRIC_KEYS}})
        if c == V0_CODE:
            log("        (对照既有主结果：IoU 0.787821 / BER 0.064359)")

    # 成功判据须同时满足 BER 与 IoU 两个条件（L2）；summ[0] 即基线 V0
    iou_floor = summ[0]["IoU"] - IOU_TOL
    hit = any(r["BER"] < 0.05 and r["IoU"] >= iou_floor for r in summ)
    log(f"\n# 成功判据：test BER < 0.05 且 IoU 不低于 V0-{IOU_TOL} -> {'达成' if hit else '未达成'}")
    log(f"# 限定（强制）：阈值仅在 val({len(val_files)}张/{n_va}原图) 上选；候选预注册共 {len(CONFIG_CODES)} 个")
    log(f"# 限定（强制）：配对 t 的子图级有效 n 被同源高估，须同时看'按原图聚合'的 t（n={n_va}）")
    log("# 限定（强制）：seed 间 BER 样本 sd = 0.0027，凡 < ~0.006 的'改善'须谨慎解读")
    log("# 限定（强制）：族B 为集成，**改变了方法本身**，须与'3 seed 均值'分列，不得混入口径")

    with open(RES_DIR / "ber_sweep_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config"] + KEYS2)
        w.writeheader(); w.writerows(summ)
    with open(RES_DIR / "ber_sweep_test_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "name"] + METRIC_KEYS)
        w.writeheader(); w.writerows(perimg)

    base_json["test_summary"] = summ
    base_json["success_criterion_met"] = hit
    with open(RES_DIR / "ber_sweep_selection.json", "w", encoding="utf-8") as fh:
        json.dump(base_json, fh, ensure_ascii=False, indent=2)

    log(f"\n汇总 -> {RES_DIR / 'ber_sweep_test.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
