"""压 BER 实验：**只动"提示框"，不动权重**。

## 目标

申报书硬指标"BER 降低至 5% 以下"当前**未达成**（无泄漏 6.44%）。
诊断（代数精确，非估计）：BER = 0.5(FPR+FNR)，实测 **FPR 0.0405 / FNR 0.0882** ——
**短板在召回（漏检），不在误检**。故本实验只试"提召回"的手段。

## 预注册（第十八轮方案审查要求，**跑 test 前定死，事后不得增删候选**）

候选**全部单变量**，只改提示框构造，其余（权重/编码器/解码器/合并方式）完全不动：

| 代号 | 改动 | 动机 |
|---|---|---|
| V0 | 基线：`component_boxes(em, min_area=300)`，不外扩 | 对照 |
| D2 | 每个框按自身宽高**外扩 2%** | 提召回 |
| D5 | 外扩 **5%** | 同上 |
| D10 | 外扩 **10%** | 同上（幅度上界） |
| M100 | `min_area` 300 → **100**（多发现小阴影） | 提召回 |

**选择规则**：在 **val(58 张)** 上，先筛掉 val IoU < V0_val_IoU − 0.01 的；
再取 val BER 最低者；**且**要求它与 V0 的**逐图配对**改善达到显著
（配对 t 的 |t| > 2）—— 否则判为"**无候选稳健优于基线**"，如实报告。

**失败即如实报告**：若最终 test BER ≥ 0.05，写"未达成"，**不调口径、不换测试集**。

## 护栏（审查必须修正 1-5）

1. **钉死 seed**：只加载主结果权重 `sam_vit_b_shadow_v4_lora_leakfree.pth`（seed=0，40 轮）。
   **单 seed 声明**：seed-1 的 BER 已是 0.0593，与 seed-0 差约 0.005 →
   **凡小于 ~0.006 的"改善"都落在 seed 噪声内**，报告须明写。
2. **复现闸门**：先跑 V0 于 **test**，与 `results/lora_leakfree_test.csv` **逐位比对**；
   不一致 → **中止**（否则无法区分"配置起效"与"脚本口径漂移"）。
3. **val 报 spread + 配对比较**，不只看均值。
4. **全部单变量**（V3 的"外扩+填洞"已按审查拆掉，填洞因与 BER 机制弱而弃用）。
5. **跑 test 之前**先把"候选表 + 选择结果"落盘（`results/ber_reduce_selection.json`）。

## 落盘
    results/ber_reduce_val_grid.csv        各候选在 val 上的六项 + FPR/FNR + 配对检验
    results/ber_reduce_selection.json      预注册候选 + 选择决定（**在跑 test 之前写**）
    results/ber_reduce_test.csv            基线 V0 与选中配置在 test 上并列
    results/ber_reduce_test_per_image.csv
    logs/ber_reduce_{ts}.log
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry
from eval_shadow import compute_metrics
from methods import mask_enhanced
from direction_experiment import component_boxes, predict_box, encode_image
from train_adapter_v4 import apply_lora
from train_direction_net import split_by_base

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CKPT_DIR = BASE / "checkpoints"
RES_DIR = BASE / "results"
LOG_DIR = BASE / "logs"
OFFICIAL = str(CKPT_DIR / "sam_vit_b_01ec64.pth")
MAIN_CKPT = str(CKPT_DIR / "sam_vit_b_shadow_v4_lora_leakfree.pth")

METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 预注册候选（**跑 test 前定死，事后不得增删**）──
CONFIGS = [
    ("V0",   dict(min_area=300, dilate=0.00)),
    ("D2",   dict(min_area=300, dilate=0.02)),
    ("D5",   dict(min_area=300, dilate=0.05)),
    ("D10",  dict(min_area=300, dilate=0.10)),
    ("M100", dict(min_area=100, dilate=0.00)),
]
IOU_TOL = 0.01      # 选中配置的 val IoU 不得低于基线 0.01 以上
T_TOL = 2.0         # 配对 t 阈值


def load_sam():
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    sam.load_state_dict(torch.load(MAIN_CKPT, map_location=DEVICE))
    sam.to(DEVICE)
    sam.eval()
    return sam


def dilate_box(b, f, h, w):
    """按自身宽高比例外扩，并裁剪到图内。"""
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * f, (y2 - y1) * f
    return [max(0.0, x1 - dx), max(0.0, y1 - dy),
            min(float(w), x2 + dx), min(float(h), y2 + dy)]


@torch.no_grad()
def predict_config(sam, img_bgr, min_area, dilate):
    """按指定配置产出二值掩膜（不动权重）。"""
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em, min_area=min_area)
    h, w = img_bgr.shape[:2]
    if not boxes:
        return np.zeros((h, w), bool)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_emb, hh, ww = encode_image(sam, img_rgb)
    combined = np.zeros((hh, ww), bool)
    for b in boxes:
        bb = dilate_box(b, dilate, hh, ww) if dilate > 0 else b
        combined |= predict_box(sam, img_emb, hh, ww, bb)
    return combined


def eval_one(sam, files, cfg):
    """返回逐图指标 list。"""
    rows = []
    for f in files:
        img = cv2.imread(f)
        gt = cv2.imread(str(AISD / "Train412" / "mask" / Path(f).name), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        pred = predict_config(sam, img, **cfg)
        m = compute_metrics((pred * 255).astype(np.uint8), gt)
        m["name"] = Path(f).name
        # FNR = 1 - Recall；FPR 由 BER 定义反解（与 compute_metrics 自洽）
        m["FNR"] = 1.0 - m["Recall"]
        m["FPR"] = 2.0 * m["BER"] - m["FNR"]
        rows.append(m)
    return rows


def agg(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def paired_t(a, b):
    """配对 t：d = a - b（逐图同名对齐）。返回 (mean_d, t, n_win, n_lose)。"""
    d = np.array([x["BER"] - y["BER"] for x, y in zip(a, b)], dtype=float)
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return float(d.mean()), t, int((d > 0).sum()), int((d < 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gate", action="store_true", help="跳过 V0 复现闸门（**不推荐**）")
    args = ap.parse_args()

    for d in (RES_DIR, LOG_DIR):
        d.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ber_reduce_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 压 BER 实验（只动提示框，不动权重）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
    log(f"# 权重: {Path(MAIN_CKPT).name}（seed=0 主结果，40 轮）**单 seed 实验**")
    log(f"# 预注册候选: {[c[0] for c in CONFIGS]}")
    log(f"# 选择规则: val IoU >= V0-{IOU_TOL} 且 val BER 最小 且 配对|t|>{T_TOL}；否则判'无候选稳健优于基线'")

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log(f"划分: 训练 {len(tr_files)}({n_tr}原图) / 验证 {len(val_files)}({n_va}原图) / 测试 {len(test_files)}({n_te}原图)")

    sam = load_sam()

    # ── 闸门：V0 于 test 复现既有主结果 ──
    base_json = {"configs": [{"code": c, **cfg} for c, cfg in CONFIGS],
                 "rule": f"val IoU>=V0-{IOU_TOL} & min val BER & paired |t|>{T_TOL}",
                 "seed": 0, "single_seed": True, "ts": ts}
    if args.skip_gate:
        log("\n[!] 已跳过 V0 复现闸门（--skip-gate）")
        v0_test = eval_one(sam, test_files, dict(CONFIGS[0][1]))
    else:
        log("\n=== 闸门：V0 在 test 上复现既有主结果 ===")
        v0_test = eval_one(sam, test_files, dict(CONFIGS[0][1]))
        a = agg(v0_test, METRIC_KEYS)
        ref = {}
        with open(RES_DIR / "lora_leakfree_test.csv", newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if not r["method"].endswith("_std"):
                    ref = {k: float(r[k]) for k in METRIC_KEYS}
        log(f"  本脚本 V0 : IoU={a['IoU']:.6f}  BER={a['BER']:.6f}")
        log(f"  既有主结果: IoU={ref['IoU']:.6f}  BER={ref['BER']:.6f}")
        dev = max(abs(a[k] - ref[k]) for k in METRIC_KEYS)
        log(f"  六项最大偏差 = {dev:.2e}")
        if dev > 1e-3:
            log("  [中止] 偏差过大 -> 无法区分'配置起效'与'脚本口径漂移'，拒绝继续")
            logf.close()
            raise SystemExit(2)
        log("  [OK] 复现通过（容差 1e-3）")
        base_json["gate_repro_max_dev"] = dev

    # ── val：各候选 ──
    log("\n=== val(58) 各候选 ===")
    keys2 = METRIC_KEYS + ["FPR", "FNR"]
    val_rows, val_store = [], {}
    for code, cfg in CONFIGS:
        rows = eval_one(sam, val_files, dict(cfg))
        val_store[code] = rows
        m = agg(rows, keys2)
        val_rows.append({"config": code, **{k: round(m[k], 6) for k in keys2}})
        log(f"  {code:<5} IoU={m['IoU']:.4f} BER={m['BER']:.4f} "
            f"FPR={m['FPR']:.4f} FNR={m['FNR']:.4f} "
            f"(BER逐图sd={np.std([r['BER'] for r in rows]):.4f})")

    with open(RES_DIR / "ber_reduce_val_grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config"] + keys2)
        w.writeheader(); w.writerows(val_rows)

    # ── 选择（**在跑 test 之前落盘**）──
    v0v = agg(val_store["V0"], keys2)
    cand = []
    for code, _ in CONFIGS[1:]:
        m = agg(val_store[code], keys2)
        if m["IoU"] < v0v["IoU"] - IOU_TOL:
            log(f"  [筛除] {code}: val IoU {m['IoU']:.4f} < V0 {v0v['IoU']:.4f} - {IOU_TOL}")
            continue
        md, t, w_, l_ = paired_t(val_store["V0"], val_store[code])
        cand.append({"config": code, "val_BER": m["BER"], "val_IoU": m["IoU"],
                     "paired_mean_ber_gain": md, "paired_t": t, "win": w_, "lose": l_})
    cand.sort(key=lambda c: c["val_BER"])
    chosen = next((c for c in cand if abs(c["paired_t"]) > T_TOL and c["paired_mean_ber_gain"] > 0), None)

    base_json["val_baseline"] = {k: round(v0v[k], 6) for k in keys2}
    base_json["val_candidates"] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in cand]
    base_json["chosen"] = chosen["config"] if chosen else None
    base_json["verdict"] = ("选中 " + chosen["config"]) if chosen else "无候选稳健优于基线（如实报告）"
    with open(RES_DIR / "ber_reduce_selection.json", "w", encoding="utf-8") as fh:
        json.dump(base_json, fh, ensure_ascii=False, indent=2)
    log(f"\n=== 选择（已落盘 ber_reduce_selection.json）===")
    for c in cand:
        log(f"  {c['config']:<5} val BER={c['val_BER']:.4f}  配对ΔBER={c['paired_mean_ber_gain']:+.4f} "
            f"t={c['paired_t']:+.2f}  {c['win']}胜/{c['lose']}负")
    log(f"  -> {base_json['verdict']}")

    # ── test：V0（已算）+ 选中配置（若与 V0 不同）──
    log("\n=== test(92) ===")
    out_rows = [("V0", v0_test)]
    if chosen and chosen["config"] != "V0":
        cfgd = dict(dict(CONFIGS)[chosen["config"]])
        out_rows.append((chosen["config"], eval_one(sam, test_files, cfgd)))

    summ, perimg = [], []
    for code, rows in out_rows:
        m = agg(rows, keys2)
        std = {k: float(np.std([r[k] for r in rows])) for k in METRIC_KEYS}
        summ.append({"config": code, **{k: round(m[k], 6) for k in keys2}})
        log(f"  {code:<5} IoU={m['IoU']:.6f} BER={m['BER']:.6f} "
            f"FPR={m['FPR']:.4f} FNR={m['FNR']:.4f}")
        for r in rows:
            perimg.append({"config": code, **{k: r[k] for k in ["name"] + METRIC_KEYS}})
        if code == "V0":
            log(f"        (对照既有主结果：IoU 0.787821 / BER 0.064359)")

    hit = any(r["BER"] < 0.05 for r in summ)
    log(f"\n# 成功判据：test BER < 0.05 且 IoU 不低于 V0-{IOU_TOL} -> {'达成' if hit else '未达成'}")
    log("# 限定（强制）：单 seed(seed=0)、超参仅在 val(58张/16原图) 上选、候选预注册共 %d 个" % len(CONFIGS))

    with open(RES_DIR / "ber_reduce_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config"] + keys2)
        w.writeheader(); w.writerows(summ)
    with open(RES_DIR / "ber_reduce_test_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "name"] + METRIC_KEYS)
        w.writeheader(); w.writerows(perimg)

    base_json["test_summary"] = summ
    base_json["success_criterion_met"] = hit
    with open(RES_DIR / "ber_reduce_selection.json", "w", encoding="utf-8") as fh:
        json.dump(base_json, fh, ensure_ascii=False, indent=2)

    log(f"\n汇总 -> {RES_DIR / 'ber_reduce_test.csv'}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
