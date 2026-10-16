"""长任务前置自检（preflight）-- 在启动任何 >1h 的任务前必须先跑这个。

## 为什么要它

最贵的错误不是"跑崩了"，而是"**跑完了才发现基底不一致，全部作废**"。典型：
  - 某个脚本用了**不同的数据划分** -> 结果不可比 -> 两边都要重跑
  - 某个脚本加载了**错误的权重**（如官方口径权重当无泄漏用）-> 数字全错
  - 目标检查点**不存在** -> 静默回退到 ImageNet/随机初始化 -> 结果无意义
  - SAM 源码补丁被覆盖/丢失 -> 方向 Adapter 静默失效

本脚本把这些不变量**自动化检查**，5 秒内给出 PASS/FAIL。退出码非 0 表示**不要启动长任务**。

## 用法

    python preflight_check.py            # 全部检查
    python preflight_check.py --quick    # 只查存在性（不读数据，最快）
"""
import argparse
import ast
import csv
import hashlib
import re
import sys
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
CK = BASE / "checkpoints"
RES = BASE / "results"
RSISD = BASE / "datasets" / "RSISD"

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def warn(name, detail=""):
    RESULTS.append((name, None, detail))
    print(f"  [WARN] {name}" + (f"  -- {detail}" if detail else ""))


def base_name(fp):
    from train_direction_net import base_name as _bn
    return _bn(fp)


def compute_split():
    """**直接调用** train_direction_net.split_by_base（与生产脚本同一个函数）。

    [!] 不要在此重写划分逻辑--重写版若写错，会给出**假的 PASS/FAIL**，
    反而把真正的基底不一致掩盖掉。
    """
    from train_direction_net import split_by_base
    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test, _, _ = split_by_base(files, 0.2, seed=0)
    train, val, _, _ = split_by_base(train_pool, 0.2, seed=1)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("长任务前置自检（preflight）")
    print("=" * 70)

    # ── 1. 数据与权重存在性 ──
    print("\n[1] 数据与权重存在性")
    check("AISD Train412", (AISD / "Train412" / "shadow").exists(),
          f"{len(list((AISD/'Train412'/'shadow').glob('*.tif')))} 张")
    check("RSISD 数据集", RSISD.exists())
    if RSISD.exists():
        for sp in ["test200", "val86", "train1000"]:
            d = RSISD / sp / "shadow"
            if d.exists():
                check(f"  RSISD/{sp}", True, f"{len(list(d.glob('*.png')))} 张")
    for f in ["sam_vit_b_01ec64.pth", "sam_vit_b_shadow_v4_lora_leakfree.pth",
              "sam_vit_b_shadow_v4_lora_leakfree_s1.pth",
              "sam_vit_b_shadow_v4_lora_leakfree_s2.pth"]:
        check(f"权重 {f}", (CK / f).exists())
    check("权重 direction_resnet18_flip_basis.pth", (CK / "direction_resnet18_flip_basis.pth").exists(),
          "[!] 缺失会阻塞 T1.1 及以后全部工作")
    check("权重 direction_resnet18_regress.pth", (CK / "direction_resnet18_regress.pth").exists(),
          "[!] 缺失会阻塞 train_direction_adapter.py 的 regress 模式（主方法口径）")
    check("SARU 权重 best_AISD_ckp.pth", (CK / "saru" / "best_AISD_ckp.pth").exists())

    # ── 2. 划分一致性（基底是否统一）──
    print("\n[2] 划分一致性（基底）")
    if args.quick:
        warn("跳过（--quick）")
    else:
        train, val, test = compute_split()
        check("train 张数 = 262", len(train) == 262, f"实际 {len(train)}")
        check("val   张数 = 58", len(val) == 58, f"实际 {len(val)}")
        check("test  张数 = 92", len(test) == 92, f"实际 {len(test)}")
        bt = {base_name(f) for f in train}
        bv = {base_name(f) for f in val}
        bs = {base_name(f) for f in test}
        check("三集基名交集为 0（无泄漏）",
              len(bt & bv) == 0 and len(bt & bs) == 0 and len(bv & bs) == 0,
              f"train&val={len(bt&bv)} train&test={len(bt&bs)} val&test={len(bv&bs)}")

        # 已落盘 CSV 的测试集是否与复算 test 一致
        test_names = {Path(f).name for f in test}
        for csvname in ["lora_leakfree_test_per_image.csv", "unet_leakfree_test_per_image.csv",
                        "segnet_leakfree_test_per_image.csv"]:
            p = RES / csvname
            if not p.exists():
                continue
            got = {r["name"] for r in csv.DictReader(open(p, encoding="utf-8"))}
            check(f"  {csvname} 的 92 个文件名 == 复算 test 集", got == test_names,
                  f"n={len(got)}")

    # ── 3. 静态扫描：各脚本的 split 调用是否同序 ──
    print("\n[3] 脚本 split 调用一致性（静态扫描）")
    # 形式一：字面量 split_by_base(x, 0.2, seed=0)
    pat_lit = re.compile(r"split_by_base\s*\(\s*[^,]+,\s*([\d.]+)\s*,\s*seed\s*=\s*(\d+)\s*\)")
    # 形式二：来自 argparse split_by_base(x, args.test_frac, seed=0)  <- 潜在陷阱
    pat_arg = re.compile(r"split_by_base\s*\(\s*[^,]+,\s*args\.(\w+)\s*,\s*seed\s*=\s*(\d+)\s*\)")
    expected = [("0.2", "0"), ("0.2", "1")]
    for f in sorted(BASE.glob("*.py")):
        if f.name == "preflight_check.py":
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        if "split_by_base" not in src:
            continue
        lits = pat_lit.findall(src)
        if lits[:2] == expected:
            print(f"  [PASS] {f.name:34s} 字面量 {lits[:2]}")
            RESULTS.append((f"split {f.name}", True, ""))
            continue
        argc = pat_arg.findall(src)
        if argc:
            # 穿透 argparse 默认值（注意属性名 test_frac 对应命令行 --test-frac）
            resolved = []
            for aname, sd in argc[:2]:
                variants = [aname, aname.replace("_", "-")]
                m = None
                for v in variants:
                    m = re.search(r'add_argument\(\s*["\']--' + re.escape(v) + r'["\'][^)]*?default\s*=\s*([\d.]+)', src, re.S)
                    if m:
                        break
                resolved.append((m.group(1) if m else "?", sd))
            ok = resolved == expected
            msg = f"经 argparse 默认值解析后 {resolved}"
            if ok:
                print(f"  [PASS] {f.name:34s} {msg}")
                RESULTS.append((f"split {f.name}", True, ""))
            else:
                warn(f"{f.name} 的划分比例来自命令行，默认值 {resolved} != {expected}",
                     "[!] 若这些默认值被改，基底会静默改变 -> 结果不可比")
            continue
        if lits:
            warn(f"{f.name} 的 split 调用为 {lits[:2]}（非基准 {expected}）", "若产出对外结果需核实")
        else:
            warn(f"{f.name} 含 split_by_base 但未匹配到调用（可能仅 re-export）", "人工确认")

    # ── 4. 权重<->结果自洽 ──
    print("\n[4] 已有结果与权重自洽性")
    p = RES / "lora_leakfree_test.csv"
    if p.exists() and not args.quick:
        row = list(csv.DictReader(open(p, encoding="utf-8")))[0]
        iou = float(row["IoU"])
        check("主结果 lora_leakfree IoU ~= 0.787821", abs(iou - 0.787821) < 1e-4, f"实际 {iou:.6f}")
        pi = RES / "lora_leakfree_test_per_image.csv"
        if pi.exists():
            vals = [float(r["IoU"]) for r in csv.DictReader(open(pi, encoding="utf-8"))]
            check("  逐图重算均值 == 汇总行", abs(float(np.mean(vals)) - iou) < 5e-5,
                  f"重算 {float(np.mean(vals)):.6f}")

    # ── 5. SAM 源码补丁完整性 ──
    print("\n[5] SAM 源码补丁（方向 Adapter 依赖）")
    pe = BASE / "segment-anything-main" / "segment_anything" / "modeling" / "prompt_encoder.py"
    if pe.exists():
        src = pe.read_text(encoding="utf-8", errors="replace")
        check("prompt_encoder.forward 含 direction 参数", "direction: Optional[torch.Tensor] = None" in src)
        check("含 adapter 注入分支", "self.direction_adapter(" in src)
        check("patch 备份存在",
              (BASE / "backups" / "patches" / "prompt_encoder_direction_adapter.patch").exists())
    else:
        check("prompt_encoder.py 存在", False, str(pe))

    # ── 6. 新脚本可导入（语法/依赖）──
    print("\n[6] 新脚本可编译")
    import py_compile
    for f in ["direction_utils.py", "direction_adapter.py", "direction_prompt.py",
              "direction_prompt_eval.py", "train_direction_adapter.py",
              "evaluate_direction_adapter.py", "train_direction_dtict.py",
              "eval_cross_dataset.py", "ber_sweep_eval.py",
              "train_frozen_leakfree.py", "shadow_extract.py"]:
        fp = BASE / f
        if not fp.exists():
            check(f"{f} 存在", False)
            continue
        try:
            py_compile.compile(str(fp), doraise=True)
            print(f"  [PASS] {f}")
            RESULTS.append((f, True, ""))
        except Exception as e:
            check(f"{f} 编译", False, str(e)[:80])

    # ── 汇总 ──
    n_fail = sum(1 for _, ok, _ in RESULTS if ok is False)
    n_warn = sum(1 for _, ok, _ in RESULTS if ok is None)
    print("\n" + "=" * 70)
    print(f"结论：{len(RESULTS)-n_fail-n_warn} PASS / {n_fail} FAIL / {n_warn} WARN")
    if n_fail:
        print("[X] 有 FAIL -- 不要启动长任务，先修好。")
    else:
        print("[OK] 无 FAIL -- 可以启动长任务。")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
