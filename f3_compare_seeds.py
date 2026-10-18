"""F3 H7 3-seed val-only comparison (seed 0/1/2, regress vs random; no test).

For each seed and arm, evaluate the best checkpoint on val (58 images), then:
  - per-seed aggregate IoU/BER and Delta BER (regress - random)
  - 3-seed mean +/- sd of per-seed Delta BER/Delta IoU
  - image-level and base-level paired t of the 3-seed mean Delta BER vs 0
"""
import csv
import json
import math
import re
from pathlib import Path

import env_setup  # noqa: F401
import numpy as np
import torch

from train_direction_adapter import (
    leakfree_split, build_sam, directions_for, eval_box_full, RES_DIR, device,
)
from direction_utils import load_direction_net

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "sprint20261018"
SEEDS = [0, 1, 2]
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]


def ckpt(arm, seed):
    suffix = "" if seed == 0 else f"_s{seed}"
    return BASE / "checkpoints" / f"sam_vit_b_shadow_v4_dir_adapter_h7_cross_{arm}{suffix}_leakfree.pth"


def base_name(name):
    m = re.match(r"^(.+?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def paired_zero(rows_by_seed, key):
    """3-seed mean per-image value; paired t vs 0 (image) and base-level."""
    names = sorted(rows_by_seed[0])
    vals = {}
    for n in names:
        vals[n] = float(np.mean([rows_by_seed[s][n][key] for s in SEEDS]))
    d = np.array(list(vals.values()), dtype=float)
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    groups = {}
    for n, v in vals.items():
        groups.setdefault(base_name(n), []).append(v)
    g = np.array([np.mean(v) for v in groups.values()], dtype=float)
    gsd = g.std(ddof=1)
    gt = float(g.mean() / (gsd / math.sqrt(len(g)))) if gsd > 0 else 0.0
    return {"mean": float(d.mean()), "sd": float(sd), "t": t,
            "win": int((d > 0).sum()), "lose": int((d < 0).sum()), "n": len(d),
            "base_mean": float(g.mean()), "base_sd": float(gsd), "base_t": gt,
            "base_win": int((g > 0).sum()), "base_lose": int((g < 0).sum()),
            "base_n": len(g)}


def main():
    tr, va, te = leakfree_split()
    gts = [str(BASE / "AISD" / "AISD" / "Train412" / "mask" / Path(f).name) for f in va]
    dmodel = load_direction_net(device=device)
    rows = {(s, a): None for s in SEEDS for a in ["regress", "random"]}
    aggregates = []
    for s in SEEDS:
        for arm in ["regress", "random"]:
            path = ckpt(arm, s)
            if not path.exists():
                raise FileNotFoundError(f"missing checkpoint: {path}")
            sam, missing, unexpected = build_sam(str(path), with_adapter=True,
                                                 adapter_mode="cross_attn")
            sam.to(device).eval()
            dirs = directions_for(va, arm, dmodel)
            avg, std, rr = eval_box_full(sam, va, gts, dirs)
            rows[(s, arm)] = {r["name"]: r for r in rr}
            with open(OUT / f"f3_{arm}_s{s}_val_per_image.csv", "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["name"] + METRIC_KEYS)
                w.writeheader()
                for r in rr:
                    w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
            aggregates.append({"seed": s, "arm": arm, "IoU": avg["IoU"],
                               "BER": avg["BER"], "F1": avg["F1"],
                               "missing": len(missing),
                               "unexpected": len(unexpected)})
            del sam
            torch.cuda.empty_cache()

    per_seed = []
    for s in SEEDS:
        reg, rnd = rows[(s, "regress")], rows[(s, "random")]
        names = sorted(set(reg) & set(rnd))
        dber = np.array([reg[n]["BER"] - rnd[n]["BER"] for n in names])
        diou = np.array([reg[n]["IoU"] - rnd[n]["IoU"] for n in names])
        per_seed.append({"seed": s,
                         "regress_BER": float(np.mean([reg[n]["BER"] for n in names])),
                         "random_BER": float(np.mean([rnd[n]["BER"] for n in names])),
                         "delta_BER": float(dber.mean()),
                         "delta_IoU": float(diou.mean()),
                         "n": len(names)})

    # 3-seed mean Delta per image (regress - random)
    names = sorted(rows[(0, "regress")])
    dber_img = {n: float(np.mean([rows[(s, "regress")][n]["BER"] -
                                  rows[(s, "random")][n]["BER"] for s in SEEDS]))
                for n in names}
    diou_img = {n: float(np.mean([rows[(s, "regress")][n]["IoU"] -
                                  rows[(s, "random")][n]["IoU"] for s in SEEDS]))
                for n in names}
    # paired_zero expects rows_by_seed[seed][name][key]; build map for both keys
    map_ber = {s: {n: {"v": rows[(s, "regress")][n]["BER"] -
                       rows[(s, "random")][n]["BER"]} for n in names} for s in SEEDS}
    map_iou = {s: {n: {"v": rows[(s, "regress")][n]["IoU"] -
                       rows[(s, "random")][n]["IoU"]} for n in names} for s in SEEDS}
    # reuse paired_zero with key="v"
    out = {
        "seeds": SEEDS, "n_images": len(names), "n_bases": 16,
        "aggregates": aggregates, "per_seed": per_seed,
        "delta_BER_seed_stats": {
            "mean": float(np.mean([p["delta_BER"] for p in per_seed])),
            "sd": float(np.std([p["delta_BER"] for p in per_seed], ddof=1))},
        "delta_IoU_seed_stats": {
            "mean": float(np.mean([p["delta_IoU"] for p in per_seed])),
            "sd": float(np.std([p["delta_IoU"] for p in per_seed], ddof=1))},
        "delta_BER_3seed": paired_zero(map_ber, "v"),
        "delta_IoU_3seed": paired_zero(map_iou, "v"),
        "note": "val-only; no test; best-val checkpoint per run; fixed 0.5 threshold",
    }
    with open(OUT / "f3_compare_seeds.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
