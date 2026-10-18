"""F3 H7 val-only paired comparison (no test).

Compares:
  base seed-0 LoRA (A50 per-image CSV)
  cross_attn regress (best checkpoint)
  cross_attn random  (best checkpoint)
Image-level n=58 and base-name n=16 paired stats.
"""
import csv
import json
import math
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
CKPT = {
    "regress": BASE / "checkpoints" / "sam_vit_b_shadow_v4_dir_adapter_h7_cross_regress_leakfree.pth",
    "random": BASE / "checkpoints" / "sam_vit_b_shadow_v4_dir_adapter_h7_cross_random_leakfree.pth",
}
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]


def base_name(name):
    import re
    m = re.match(r"^(.+?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def read_base():
    rows = {}
    with open(RES_DIR / "ber_sweep_val_per_image.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["config"] != "A50":
                continue
            rows[r["name"]] = {k: float(r[k]) for k in METRIC_KEYS}
    return rows


def paired(a, b, key):
    names = sorted(set(a) & set(b))
    d = np.array([a[n][key] - b[n][key] for n in names], dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return {"mean_diff": float(d.mean()), "sd": float(sd), "t": t,
            "win": int((d > 0).sum()), "lose": int((d < 0).sum()), "n": len(d)}


def paired_base(a, b, key):
    aa, bb = {}, {}
    for n, v in a.items():
        aa.setdefault(base_name(n), []).append(v[key])
    for n, v in b.items():
        bb.setdefault(base_name(n), []).append(v[key])
    common = sorted(set(aa) & set(bb))
    d = np.array([np.mean(aa[k]) - np.mean(bb[k]) for k in common], dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return {"mean_diff": float(d.mean()), "sd": float(sd), "t": t,
            "win": int((d > 0).sum()), "lose": int((d < 0).sum()), "n": len(d)}


def main():
    tr, va, te = leakfree_split()
    gts = [str(BASE / "AISD" / "AISD" / "Train412" / "mask" / Path(f).name) for f in va]
    base = read_base()
    dmodel = load_direction_net(device=device)
    out = {"n_images": len(va), "base": "seed0_A50", "arms": {}}
    rows_by_arm = {}
    for arm, ckpt in CKPT.items():
        sam, missing, unexpected = build_sam(str(ckpt), with_adapter=True,
                                             adapter_mode="cross_attn")
        sam.to(device).eval()
        dirs = directions_for(va, arm, dmodel)
        avg, std, rows = eval_box_full(sam, va, gts, dirs)
        rows_by_arm[arm] = {r["name"]: r for r in rows}
        out["arms"][arm] = {"val": avg, "std": std,
                            "ckpt": str(ckpt),
                            "missing": len(missing), "unexpected": len(unexpected)}
        with open(OUT / f"f3_{arm}_val_per_image.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["name"] + METRIC_KEYS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in ["name"] + METRIC_KEYS})
        del sam
        torch.cuda.empty_cache()

    for arm in ["regress", "random"]:
        out[f"base_minus_{arm}"] = {
            "IoU": {"image": paired(base, rows_by_arm[arm], "IoU"),
                    "base": paired_base(base, rows_by_arm[arm], "IoU")},
            "BER": {"image": paired(base, rows_by_arm[arm], "BER"),
                    "base": paired_base(base, rows_by_arm[arm], "BER")},
        }
    out["regress_minus_random"] = {
        "IoU": {"image": paired(rows_by_arm["regress"], rows_by_arm["random"], "IoU"),
                "base": paired_base(rows_by_arm["regress"], rows_by_arm["random"], "IoU")},
        "BER": {"image": paired(rows_by_arm["regress"], rows_by_arm["random"], "BER"),
                "base": paired_base(rows_by_arm["regress"], rows_by_arm["random"], "BER")},
    }
    with open(OUT / "f3_compare_val.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
